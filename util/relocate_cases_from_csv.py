import argparse
import csv
import shutil
from pathlib import Path, PureWindowsPath
from typing import Optional


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Copy or move .mat files referenced in a failure CSV to a target directory."
    )
    parser.add_argument("csv_path", type=Path, help="CSV file exported by train_main_angular_domain_corrl_lstm.py")
    parser.add_argument("target_dir", type=Path, help="Directory to store the selected .mat files")
    parser.add_argument(
        "--mode",
        choices=["copy", "move"],
        default="copy",
        help="Whether to copy files or move them. Default: copy",
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        action="append",
        default=None,
        help="Additional root directory used to resolve stale paths in the CSV. Can be passed multiple times.",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Store all files directly under target_dir instead of preserving the attitude subdirectory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be done without copying or moving files.",
    )
    parser.add_argument(
        "--missing-report",
        type=Path,
        default=None,
        help="Optional path to save rows whose files cannot be resolved.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing target file if it already exists.",
    )
    parser.add_argument(
        "--avg-err-threshold",
        type=float,
        default=None,
        help="Only relocate rows whose avg_err_deg is strictly greater than this threshold.",
    )
    parser.set_defaults(default_search_roots=[repo_root / "code" / "data"])
    return parser.parse_args()


def load_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader)


def filter_rows_by_avg_err(rows: list[dict], threshold: Optional[float]) -> list[dict]:
    if threshold is None:
        return rows
    filtered_rows = []
    for row in rows:
        raw_value = row.get("avg_err_deg", "").strip()
        if not raw_value:
            continue
        try:
            avg_err = float(raw_value)
        except ValueError:
            continue
        if avg_err > threshold:
            filtered_rows.append(row)
    return filtered_rows


def candidate_filename(raw_path: str) -> str:
    raw_path = raw_path.strip()
    if "\\" in raw_path or ":" in raw_path:
        return PureWindowsPath(raw_path).name
    return Path(raw_path).name


def resolve_source_path(row: dict, search_roots: list[Path]) -> Optional[Path]:
    raw_path = row.get("path", "").strip()
    if raw_path:
        direct = Path(raw_path)
        if direct.exists():
            return direct

    filename = candidate_filename(raw_path)
    attitude = row.get("attitude", "").strip()

    for root in search_roots:
        if attitude:
            candidate = root / "test" / attitude / filename
            if candidate.exists():
                return candidate
            candidate = root / "train" / attitude / filename
            if candidate.exists():
                return candidate
            candidate = root / attitude / filename
            if candidate.exists():
                return candidate

        matches = list(root.rglob(filename))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1 and attitude:
            filtered = [match for match in matches if match.parent.name == attitude]
            if len(filtered) == 1:
                return filtered[0]
    return None


def target_path_for(row: dict, source_path: Path, target_dir: Path, flat: bool) -> Path:
    if flat:
        return target_dir / source_path.name
    attitude = row.get("attitude", "").strip() or source_path.parent.name
    return target_dir / attitude / source_path.name


def write_missing_report(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def relocate_file(source_path: Path, target_path: Path, mode: str, overwrite: bool, dry_run: bool) -> str:
    if target_path.exists():
        if not overwrite:
            return "skipped_existing"
        if not dry_run:
            target_path.unlink()

    if dry_run:
        return "planned"

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source_path, target_path)
    else:
        shutil.move(str(source_path), str(target_path))
    return mode


def main() -> None:
    args = parse_args()
    all_rows = load_rows(args.csv_path)
    rows = filter_rows_by_avg_err(all_rows, args.avg_err_threshold)
    search_roots = [path.resolve() for path in (args.search_root or args.default_search_roots)]
    target_dir = args.target_dir.resolve()
    if not args.dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)

    missing_rows: list[dict] = []
    resolved = 0
    copied_or_moved = 0
    skipped_existing = 0

    for row in rows:
        source_path = resolve_source_path(row, search_roots)
        if source_path is None:
            missing_rows.append(row)
            continue

        resolved += 1
        target_path = target_path_for(row, source_path, target_dir, args.flat)
        status = relocate_file(
            source_path=source_path,
            target_path=target_path,
            mode=args.mode,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        if status in {"copy", "move", "planned"}:
            copied_or_moved += 1
        elif status == "skipped_existing":
            skipped_existing += 1

    missing_report = args.missing_report or (target_dir / "missing_cases.csv")
    write_missing_report(missing_report, missing_rows)

    print(f"CSV rows: {len(all_rows)}")
    if args.avg_err_threshold is not None:
        print(f"Rows with avg_err_deg > {args.avg_err_threshold}: {len(rows)}")
    print(f"Resolved source files: {resolved}")
    print(f"{'Planned' if args.dry_run else args.mode.title()} files: {copied_or_moved}")
    print(f"Skipped existing targets: {skipped_existing}")
    print(f"Missing files: {len(missing_rows)}")
    if missing_rows:
        print(f"Missing report: {missing_report}")


if __name__ == "__main__":
    main()
