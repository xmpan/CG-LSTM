import json
import math
import os
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import scipy.io as scio
    from scipy.optimize import linear_sum_assignment
except ImportError as exc:  # pragma: no cover - exercised only in incomplete envs
    scio = None
    linear_sum_assignment = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None


def _require_scipy() -> None:
    if scio is None:
        raise ImportError(
            "scipy is required to load MATLAB .mat files. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from _SCIPY_IMPORT_ERROR


PathLike = Union[os.PathLike[str], str]


def _load_metadata(path: PathLike) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _crop_last_range_bin(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2 and array.shape[-1] > 1:
        return array[:, :-1]
    return array


def _maybe_add_noise(hrrp: np.ndarray, snr_db: Optional[float]) -> np.ndarray:
    if snr_db is None or snr_db < 0:
        return hrrp
    noisy = hrrp.astype(np.float32, copy=True)
    for row_idx in range(noisy.shape[0]):
        signal = noisy[row_idx]
        signal_power = np.mean(np.abs(signal) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power) * np.random.randn(signal.shape[0])
        noisy[row_idx] = signal + noise
    return noisy


def _topk_asc_from_profile(
    amplitude_profile: np.ndarray,
    phase_profile: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    usable_k = min(top_k, amplitude_profile.shape[0])
    if usable_k <= 0:
        raise ValueError("ASC extraction needs at least one range bin.")
    idx = np.argpartition(np.abs(amplitude_profile), -usable_k)[-usable_k:]
    idx = idx[np.argsort(idx)]
    denom = max(amplitude_profile.shape[0] - 1, 1)
    ranges = idx.astype(np.float32) / float(denom)
    amplitudes = np.abs(amplitude_profile[idx]).astype(np.float32)
    phases = phase_profile[idx].astype(np.float32)
    return ranges, amplitudes, phases


def _hungarian_match(similarity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if linear_sum_assignment is None:
        # Deterministic fallback for environments without scipy.optimize. The
        # released training setup uses SciPy's Hungarian implementation.
        pairs = []
        used_rows = set()
        used_cols = set()
        flat_order = np.argsort(similarity.ravel())[::-1]
        rows, cols = similarity.shape
        for flat_idx in flat_order:
            row = int(flat_idx // cols)
            col = int(flat_idx % cols)
            if row not in used_rows and col not in used_cols:
                used_rows.add(row)
                used_cols.add(col)
                pairs.append((row, col))
            if len(pairs) == min(rows, cols):
                break
        if not pairs:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        row_ind, col_ind = zip(*pairs)
        return np.asarray(row_ind), np.asarray(col_ind)

    row_ind, col_ind = linear_sum_assignment(-similarity)
    return row_ind, col_ind


def compute_asc_correlations(
    hrrp_omp: np.ndarray,
    hrrp_phase: np.ndarray,
    top_k: int = 14,
    sigma_r: float = 0.5,
    sigma_alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the paper's ASC amplitude and phase correlations.

    The dataset stores sparse OMP HRRPs rather than explicit RELAX ASC lists.
    For open-source reproducibility, the strongest OMP range bins are used as
    ASC proxies: range is the normalized bin index, amplitude is the sparse
    magnitude, and phase is read from `hrrp_phase`.
    """
    hrrp_omp = np.asarray(hrrp_omp)
    hrrp_phase = np.asarray(hrrp_phase)
    if hrrp_omp.shape != hrrp_phase.shape:
        raise ValueError(f"hrrp_omp and hrrp_phase must have the same shape, got {hrrp_omp.shape} and {hrrp_phase.shape}.")

    time_steps = hrrp_omp.shape[0]
    amp_corr = []
    phase_corr = []

    asc_cache = [
        _topk_asc_from_profile(hrrp_omp[t], hrrp_phase[t], top_k)
        for t in range(time_steps)
    ]
    for t in range(1, time_steps):
        prev_r, prev_a, prev_p = asc_cache[t - 1]
        curr_r, curr_a, curr_p = asc_cache[t]
        prev_a_norm = prev_a / max(float(np.max(prev_a)), 1e-12)
        curr_a_norm = curr_a / max(float(np.max(curr_a)), 1e-12)

        range_term = np.exp(-((curr_r[:, None] - prev_r[None, :]) ** 2) / (2.0 * sigma_r ** 2))
        amp_term = np.exp(-((curr_a_norm[:, None] - prev_a_norm[None, :]) ** 2) / (2.0 * sigma_alpha ** 2))
        similarity = range_term * amp_term
        curr_idx, prev_idx = _hungarian_match(similarity)
        if curr_idx.size == 0:
            amp_corr.append(0.0)
            phase_corr.append(0.0)
            continue

        matched_similarity = similarity[curr_idx, prev_idx]
        amp_corr.append(float(np.mean(matched_similarity)))

        phase_diff = curr_p[curr_idx] - prev_p[prev_idx]
        circular_mean = np.mean(np.exp(1j * phase_diff))
        phase_corr.append(float(np.abs(circular_mean)))

    return np.asarray(amp_corr, dtype=np.float32), np.asarray(phase_corr, dtype=np.float32)


class HRRPPoseDataset(Dataset):
    """Dataset for the paper-aligned CG-LSTM training pipeline."""

    def __init__(
        self,
        data_dir: PathLike,
        metadata_json: PathLike,
        split: str,
        top_k: int = 14,
        sigma_r: float = 0.5,
        sigma_alpha: float = 0.5,
        snr_db: Optional[float] = None,
    ) -> None:
        _require_scipy()
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'.")
        self.data_dir = Path(data_dir)
        self.metadata = _load_metadata(metadata_json)
        self.split = split
        self.top_k = top_k
        self.sigma_r = sigma_r
        self.sigma_alpha = sigma_alpha
        self.snr_db = snr_db
        self.samples = self._index_samples()

    def _index_samples(self) -> list[tuple[Path, str]]:
        split_key = f"{self.split}_set"
        attitude_names = self.metadata.get(split_key, [])
        samples: list[tuple[Path, str]] = []
        for attitude in attitude_names:
            attitude_dir = self.data_dir / attitude
            if not attitude_dir.exists():
                continue
            blacklist = self._read_blacklist(attitude_dir)
            for mat_path in sorted(attitude_dir.glob("*.mat")):
                rel_name = f"{attitude}\\{mat_path.name}"
                if rel_name not in blacklist:
                    samples.append((mat_path, attitude))
        if not samples:
            raise FileNotFoundError(f"No .mat samples found under {self.data_dir} for split '{self.split}'.")
        return samples

    @staticmethod
    def _read_blacklist(attitude_dir: Path) -> set[str]:
        path = attitude_dir / "blacklist.txt"
        if not path.exists():
            return set()
        return set(path.read_text(encoding="utf-8").splitlines())

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        mat_path, attitude = self.samples[index]
        mat_data = scio.loadmat(mat_path)
        hrrp = self._select_hrrp(mat_data)
        hrrp_omp = self._select_hrrp_omp(mat_data)
        hrrp_phase = _crop_last_range_bin(np.asarray(mat_data["hrrp_phase"], dtype=np.float32))
        rlos = np.asarray(mat_data["theta_phi"], dtype=np.float32).T * np.pi / 180.0

        amplitude_corr, phase_corr = compute_asc_correlations(
            hrrp_omp=hrrp_omp,
            hrrp_phase=hrrp_phase,
            top_k=self.top_k,
            sigma_r=self.sigma_r,
            sigma_alpha=self.sigma_alpha,
        )
        rlos_delta = rlos[1:, :] - rlos[:-1, :]

        target = self.metadata[attitude]
        theta_phi = np.asarray(
            [target["theta"] * math.pi / 180.0, target["phi"] * math.pi / 180.0],
            dtype=np.float32,
        )
        label = int(attitude.split("_")[-1]) - 1

        return {
            "hrrp": torch.as_tensor(hrrp, dtype=torch.float32),
            "target_theta_phi": torch.as_tensor(theta_phi, dtype=torch.float32),
            "rlos": torch.as_tensor(rlos, dtype=torch.float32),
            "amplitude_corr": torch.as_tensor(amplitude_corr, dtype=torch.float32),
            "phase_corr": torch.as_tensor(phase_corr, dtype=torch.float32),
            "rlos_delta": torch.as_tensor(rlos_delta, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "path": str(mat_path),
        }

    def _select_hrrp(self, mat_data: dict[str, Any]) -> np.ndarray:
        key = "hrrp_snr" if self.split == "test" and "hrrp_snr" in mat_data else "hrrp"
        hrrp = _crop_last_range_bin(np.asarray(mat_data[key], dtype=np.float32))
        return _maybe_add_noise(hrrp, self.snr_db)

    def _select_hrrp_omp(self, mat_data: dict[str, Any]) -> np.ndarray:
        key = "hrrp_omp_snr" if self.split == "test" and "hrrp_omp_snr" in mat_data else "hrrp_omp"
        return _crop_last_range_bin(np.asarray(mat_data[key], dtype=np.float32))


# Backward-compatible alias for earlier experiment scripts.
OriginalHRRPLoader = HRRPPoseDataset
