from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class CGLSTMConfig:
    hrrp_dim: int = 200
    hidden_dim: int = 128
    fusion_dim: int = 128
    num_layers: int = 3
    lambda_forget: float = 0.5
    dropout: float = 0.0


def angles_to_unit_vector(theta_phi: torch.Tensor) -> torch.Tensor:
    """Map [theta, phi] angles in radians to the unit vector used in the paper."""
    theta = theta_phi[..., 0]
    phi = theta_phi[..., 1]
    return torch.stack(
        [
            torch.cos(theta) * torch.cos(phi),
            torch.cos(theta) * torch.sin(phi),
            torch.sin(theta),
        ],
        dim=-1,
    )


def unit_vector_loss(pred_theta_phi: torch.Tensor, target_theta_phi: torch.Tensor) -> torch.Tensor:
    pred_u = angles_to_unit_vector(pred_theta_phi)
    target_u = angles_to_unit_vector(target_theta_phi)
    return torch.mean(torch.sum((pred_u - target_u) ** 2, dim=-1))


class TwoStageFusion(nn.Module):
    """GMU fusion of ASC correlations followed by per-step RLOS fusion."""

    def __init__(self, dim: int):
        super().__init__()
        self.amp_fc = nn.Sequential(nn.Linear(1, dim), nn.Tanh())
        self.phase_fc = nn.Sequential(nn.Linear(1, dim), nn.Tanh())
        self.gate = nn.Linear(dim * 2, dim)

        self.rlos_encoder = nn.Sequential(
            nn.Linear(2, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
            nn.Tanh(),
        )
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        amplitude_corr: torch.Tensor,
        phase_corr: torch.Tensor,
        rlos_delta: torch.Tensor,
    ) -> torch.Tensor:
        amp = self.amp_fc(amplitude_corr.unsqueeze(-1))
        phase = self.phase_fc(phase_corr.unsqueeze(-1))
        beta = torch.sigmoid(self.gate(torch.cat([phase, amp], dim=-1)))
        corr_feature = beta * phase + (1.0 - beta) * amp

        rlos_feature = self.rlos_encoder(rlos_delta)
        q = self.query(corr_feature)
        k = self.key(rlos_feature)
        v = self.value(rlos_feature)

        # Fuse each adjacent HRRP pair with its own RLOS increment instead of
        # mixing all time steps together. This keeps the geometric prior local
        # to the corresponding pair while still using a scaled query-key score.
        attn = torch.sigmoid(torch.sum(q * k, dim=-1, keepdim=True) * self.scale)
        gamma = self.out(attn * v)
        return gamma


class CGLSTMCell(nn.Module):
    """LSTM cell whose gates are modulated by the fused feature Gamma."""

    def __init__(self, input_dim: int, hidden_dim: int, fusion_dim: int, lambda_forget: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lambda_forget = lambda_forget
        self.base = nn.Linear(input_dim + hidden_dim, hidden_dim * 4)
        self.gamma_i = nn.Linear(fusion_dim, hidden_dim, bias=False)
        self.gamma_f = nn.Linear(fusion_dim, hidden_dim, bias=False)
        self.gamma_c = nn.Linear(fusion_dim, hidden_dim, bias=False)
        self.gamma_o = nn.Linear(fusion_dim, hidden_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.base.weight)
        nn.init.zeros_(self.base.bias)

        # A positive forget-gate bias is a standard stabilizing choice for LSTM
        # training and helps retain useful memory at the start of optimization.
        hidden_dim = self.hidden_dim
        self.base.bias.data[hidden_dim:2 * hidden_dim].fill_(1.0)

        for module in (self.gamma_i, self.gamma_f, self.gamma_c, self.gamma_o):
            nn.init.xavier_uniform_(module.weight, gain=0.5)

    def forward(
        self,
        x_t: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
        gamma_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = state
        base_i, base_f, base_c, base_o = self.base(torch.cat([h_prev, x_t], dim=-1)).chunk(4, dim=-1)
        i_t = torch.sigmoid(base_i + self.gamma_i(gamma_t))
        f_t = torch.sigmoid(base_f - self.lambda_forget * self.gamma_f(gamma_t))
        c_hat = torch.tanh(base_c + self.gamma_c(gamma_t))
        o_t = torch.sigmoid(base_o + self.gamma_o(gamma_t))
        c_t = f_t * c_prev + i_t * c_hat
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class StackedCGLSTM(nn.Module):
    def __init__(self, config: CGLSTMConfig):
        super().__init__()
        cells = []
        for layer_idx in range(config.num_layers):
            input_dim = config.hrrp_dim if layer_idx == 0 else config.hidden_dim
            cells.append(
                CGLSTMCell(
                    input_dim=input_dim,
                    hidden_dim=config.hidden_dim,
                    fusion_dim=config.fusion_dim,
                    lambda_forget=config.lambda_forget,
                )
            )
        self.cells = nn.ModuleList(cells)
        self.dropout = nn.Dropout(config.dropout)
        self.hidden_dim = config.hidden_dim

    def forward(self, hrrp: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, _ = hrrp.shape
        device = hrrp.device
        dtype = hrrp.dtype
        h_states = [
            torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
            for _ in self.cells
        ]
        c_states = [
            torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
            for _ in self.cells
        ]
        zero_gamma = torch.zeros(batch_size, gamma.shape[-1], device=device, dtype=dtype)

        for t in range(time_steps):
            layer_input = hrrp[:, t, :].abs()
            gamma_t = zero_gamma if t == 0 else gamma[:, t - 1, :]
            for layer_idx, cell in enumerate(self.cells):
                h_t, c_t = cell(layer_input, (h_states[layer_idx], c_states[layer_idx]), gamma_t)
                h_states[layer_idx] = h_t
                c_states[layer_idx] = c_t
                layer_input = self.dropout(h_t) if layer_idx < len(self.cells) - 1 else h_t
        return h_states[-1]


class CGLSTM(nn.Module):
    """Paper-aligned correlation-guided LSTM for HRRP pose estimation."""

    def __init__(self, config: Optional[CGLSTMConfig] = None):
        super().__init__()
        self.config = config or CGLSTMConfig()
        self.fusion = TwoStageFusion(self.config.fusion_dim)
        self.backbone = StackedCGLSTM(self.config)
        self.regressor = nn.Linear(self.config.hidden_dim, 2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.regressor.weight)
        nn.init.zeros_(self.regressor.bias)

    def forward(
        self,
        hrrp: torch.Tensor,
        amplitude_corr: torch.Tensor,
        phase_corr: torch.Tensor,
        rlos_delta: torch.Tensor,
    ) -> torch.Tensor:
        gamma = self.fusion(amplitude_corr, phase_corr, rlos_delta)
        hidden = self.backbone(hrrp, gamma)
        return self.regressor(hidden)
