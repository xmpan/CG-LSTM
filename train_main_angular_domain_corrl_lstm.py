import argparse
import csv
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from models import CGLSTM, CGLSTMConfig, unit_vector_loss
from util.dataloader import HRRPPoseDataset


def parse_args() -> argparse.Namespace:
    repo_code_dir = Path(__file__).resolve().parent
    default_data = repo_code_dir / "data"
    parser = argparse.ArgumentParser(
        description="Train the paper-aligned Correlation-Guided LSTM for HRRP pose estimation."
    )
    parser.add_argument("--train-dir", type=Path, default=default_data / "train")
    parser.add_argument("--test-dir", type=Path, default=default_data / "test")
    parser.add_argument("--metadata", type=Path, default=default_data / "attitute_npi_to_pi.json")
    parser.add_argument("--output-dir", type=Path, default=repo_code_dir / "outputs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hrrp-dim", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--fusion-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--lambda-forget", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--top-k-asc", type=int, default=14)
    parser.add_argument("--sigma-r", type=float, default=0.5)
    parser.add_argument("--sigma-alpha", type=float, default=0.5)
    parser.add_argument("--snr-db", type=float, default=-1.0)
    parser.add_argument("--threshold", type=float, default=0.5, help="Qualified-rate threshold in degrees.")
    parser.add_argument(
        "--max-failure-cases",
        type=int,
        default=50,
        help="Number of worst failed test cases to export in the top-N report.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cg_lstm")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_dir / "train.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def angular_error_deg(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = torch.remainder(pred - target + torch.pi, 2.0 * torch.pi) - torch.pi
    return torch.abs(diff) * 180.0 / torch.pi


def move_batch(batch: dict, device: torch.device) -> dict:
    tensor_keys = [
        "hrrp",
        "target_theta_phi",
        "amplitude_corr",
        "phase_corr",
        "rlos_delta",
    ]
    return {
        key: value.to(device) if key in tensor_keys else value
        for key, value in batch.items()
    }


def forward_loss(model: CGLSTM, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    pred = model(
        hrrp=batch["hrrp"],
        amplitude_corr=batch["amplitude_corr"],
        phase_corr=batch["phase_corr"],
        rlos_delta=batch["rlos_delta"],
    )
    loss = unit_vector_loss(pred, batch["target_theta_phi"])
    return pred, loss


def radians_to_degrees(values: torch.Tensor) -> torch.Tensor:
    return values * 180.0 / torch.pi


def save_failure_report(
    output_dir: Path,
    epoch: int,
    failures: list[dict],
    max_failure_cases: int,
) -> tuple[Path, Path, Path]:
    csv_path = output_dir / f"test_failures_epoch_{epoch:03d}.csv"
    latest_csv_path = output_dir / "test_failures_latest.csv"
    top_csv_path = output_dir / f"test_failures_top_{max_failure_cases}_epoch_{epoch:03d}.csv"
    latest_top_csv_path = output_dir / f"test_failures_top_{max_failure_cases}_latest.csv"
    summary_path = output_dir / f"test_failure_summary_epoch_{epoch:03d}.txt"
    latest_summary_path = output_dir / "test_failure_summary_latest.txt"

    fieldnames = [
        "attitude",
        "label",
        "path",
        "avg_err_deg",
        "theta_err_deg",
        "phi_err_deg",
        "pred_theta_deg",
        "pred_phi_deg",
        "target_theta_deg",
        "target_phi_deg",
    ]
    failures_sorted = sorted(failures, key=lambda item: float(item["avg_err_deg"]), reverse=True)
    for path in (csv_path, latest_csv_path):
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(failures_sorted)

    top_failures = failures_sorted[:max_failure_cases]
    for path in (top_csv_path, latest_top_csv_path):
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(top_failures)

    attitude_counts = Counter(item["attitude"] for item in failures_sorted)
    summary_lines = [
        f"Epoch: {epoch}",
        f"Failed cases: {len(failures_sorted)}",
        f"Top-N exported cases: {len(top_failures)}",
        f"Top-N file: {top_csv_path.name}",
        "",
        "Failure count by attitude:",
    ]
    for attitude, count in sorted(attitude_counts.items()):
        summary_lines.append(f"- {attitude}: {count}")

    if top_failures:
        summary_lines.extend(["", "Worst cases by average angular error:"])
        for item in top_failures[: min(10, len(top_failures))]:
            summary_lines.append(
                "- {attitude} | avg={avg_err_deg} deg | theta={theta_err_deg} deg | phi={phi_err_deg} deg | {path}".format(
                    **item
                )
            )

    for path in (summary_path, latest_summary_path):
        path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return csv_path, top_csv_path, summary_path


def run_epoch(
    model: CGLSTM,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    threshold: float,
    output_dir: Optional[Path] = None,
    epoch: Optional[int] = None,
    max_failure_cases: int = 50,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total = 0
    qualified = 0
    loss_sum = 0.0
    theta_err_sum = 0.0
    phi_err_sum = 0.0
    failures: list[dict] = []

    for batch in loader:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(is_train):
            pred, loss = forward_loss(model, batch)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        target = batch["target_theta_phi"]
        err = angular_error_deg(pred.detach(), target)
        batch_size = target.shape[0]
        both_under_threshold = (err[:, 0] < threshold) & (err[:, 1] < threshold)

        total += batch_size
        qualified += int(both_under_threshold.sum().item())
        loss_sum += float(loss.detach().item()) * batch_size
        theta_err_sum += float(err[:, 0].sum().item())
        phi_err_sum += float(err[:, 1].sum().item())

        if not is_train:
            pred_deg = radians_to_degrees(pred.detach()).cpu()
            target_deg = radians_to_degrees(target.detach()).cpu()
            err_cpu = err.cpu()
            labels = batch["label"]
            paths = batch["path"]
            for idx in range(batch_size):
                if bool(both_under_threshold[idx].item()):
                    continue
                sample_path = Path(paths[idx])
                failures.append(
                    {
                        "attitude": sample_path.parent.name,
                        "label": int(labels[idx].item()) if hasattr(labels[idx], "item") else int(labels[idx]),
                        "path": str(sample_path),
                        "avg_err_deg": f"{float(err_cpu[idx].mean().item()):.6f}",
                        "theta_err_deg": f"{float(err_cpu[idx, 0].item()):.6f}",
                        "phi_err_deg": f"{float(err_cpu[idx, 1].item()):.6f}",
                        "pred_theta_deg": f"{float(pred_deg[idx, 0].item()):.6f}",
                        "pred_phi_deg": f"{float(pred_deg[idx, 1].item()):.6f}",
                        "target_theta_deg": f"{float(target_deg[idx, 0].item()):.6f}",
                        "target_phi_deg": f"{float(target_deg[idx, 1].item()):.6f}",
                    }
                )

    metrics = {
        "loss": loss_sum / max(total, 1),
        "qualified_rate": 100.0 * qualified / max(total, 1),
        "theta_err": theta_err_sum / max(total, 1),
        "phi_err": phi_err_sum / max(total, 1),
        "failure_count": len(failures),
    }
    if not is_train and output_dir is not None and epoch is not None:
        csv_path, top_csv_path, summary_path = save_failure_report(
            output_dir,
            epoch,
            failures,
            max_failure_cases=max_failure_cases,
        )
        metrics["failure_csv"] = str(csv_path)
        metrics["failure_top_csv"] = str(top_csv_path)
        metrics["failure_summary"] = str(summary_path)
    return metrics


def build_dataset(args: argparse.Namespace, split: str) -> HRRPPoseDataset:
    data_dir = args.train_dir if split == "train" else args.test_dir
    snr_db = None if args.snr_db < 0 else args.snr_db
    return HRRPPoseDataset(
        data_dir=data_dir,
        metadata_json=args.metadata,
        split=split,
        top_k=args.top_k_asc,
        sigma_r=args.sigma_r,
        sigma_alpha=args.sigma_alpha,
        snr_db=snr_db,
    )


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.output_dir)
    set_seed(args.seed)
    device = resolve_device(args.device)
    logger.info("Using device: %s", device)
    logger.info("Arguments: %s", vars(args))

    train_dataset = build_dataset(args, "train")
    test_dataset = build_dataset(args, "test")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    config = CGLSTMConfig(
        hrrp_dim=args.hrrp_dim,
        hidden_dim=args.hidden_dim,
        fusion_dim=args.fusion_dim,
        num_layers=args.num_layers,
        lambda_forget=args.lambda_forget,
        dropout=args.dropout,
    )
    model = CGLSTM(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_rate = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer, args.threshold)
        with torch.no_grad():
            test_metrics = run_epoch(
                model,
                test_loader,
                device,
                None,
                args.threshold,
                output_dir=args.output_dir,
                epoch=epoch,
                max_failure_cases=args.max_failure_cases,
            )

        logger.info(
            "Epoch %03d | train loss %.6f q-rate %.2f%% theta %.3f phi %.3f | "
            "test loss %.6f q-rate %.2f%% theta %.3f phi %.3f | failed %d",
            epoch,
            train_metrics["loss"],
            train_metrics["qualified_rate"],
            train_metrics["theta_err"],
            train_metrics["phi_err"],
            test_metrics["loss"],
            test_metrics["qualified_rate"],
            test_metrics["theta_err"],
            test_metrics["phi_err"],
            test_metrics["failure_count"],
        )
        logger.info(
            "Epoch %03d failure report: %s | top-N: %s | summary: %s",
            epoch,
            test_metrics["failure_csv"],
            test_metrics["failure_top_csv"],
            test_metrics["failure_summary"],
        )

        if test_metrics["qualified_rate"] > best_rate:
            best_rate = test_metrics["qualified_rate"]
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config.__dict__,
                "metrics": test_metrics,
                "args": vars(args),
            }
            torch.save(checkpoint, args.output_dir / "best_cg_lstm.pt")


if __name__ == "__main__":
    main()
