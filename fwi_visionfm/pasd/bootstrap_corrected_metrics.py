"""Paired bootstrap over Phase-3R corrected per-sample metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .corrected_metrics import stable_id_hash
from .phase3_utils import write_json


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _dataset_file(root: Path, target: str) -> Path:
    subdir = target.replace("cross_", "").replace("_a", "_a")
    if target == "cross_curvevel_a":
        return root / "curvevel_a" / "corrected_per_sample_metrics.csv"
    if target == "cross_flatfault_a":
        return root / "flatfault_a" / "corrected_per_sample_metrics.csv"
    return root / subdir / "corrected_per_sample_metrics.csv"


def bootstrap_corrected(args: argparse.Namespace) -> Path:
    root = Path(args.corrected_metrics_root)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    all_summaries: dict[str, list[dict[str, Any]]] = {target: [] for target in args.targets}
    for target in args.targets:
        rows = _read_csv(_dataset_file(root, target))
        by_key = {(row["variant"], int(row["seed"]), int(row["sample_id"])): row for row in rows}
        for seed in args.seeds:
            seed_result: list[dict[str, Any]] = []
            b1_ids = {sample_id for variant, s, sample_id in by_key if variant == args.variants[0] and s == seed}
            pasd_ids = {sample_id for variant, s, sample_id in by_key if variant == args.variants[1] and s == seed}
            ids = np.asarray(sorted(b1_ids.intersection(pasd_ids)), dtype=np.int64)
            if ids.size != len(b1_ids) or ids.size != len(pasd_ids):
                raise ValueError(f"Sample IDs are not fully aligned for {target} seed {seed}")
            rng = np.random.default_rng(seed)
            for metric in args.metrics:
                b1 = np.asarray([float(by_key[(args.variants[0], seed, int(sample_id))][metric]) for sample_id in ids], dtype=np.float64)
                pasd = np.asarray([float(by_key[(args.variants[1], seed, int(sample_id))][metric]) for sample_id in ids], dtype=np.float64)
                diff = pasd - b1
                boot = diff[rng.integers(0, diff.size, size=(args.bootstrap_resamples, diff.size))].mean(axis=1)
                lower_is_better = metric not in {"SSIM", "edge_F1", "edge_precision", "edge_recall"}
                improvement = -diff if lower_is_better else diff
                result = {
                    "target": target,
                    "seed": seed,
                    "metric": metric,
                    "delta": float(diff.mean()),
                    "ci95_low": float(np.quantile(boot, 0.025)),
                    "ci95_high": float(np.quantile(boot, 0.975)),
                    "improvement_probability": float((improvement > 0).mean()),
                    "sample_count": int(ids.size),
                    "sample_id_hash": stable_id_hash(ids),
                    "metric_source": "fresh_prediction_archive_corrected_metrics",
                }
                seed_result.append(result)
                all_summaries[target].append(result)
            write_json(out / f"PASD_Core_locked_vs_B1_raw_unet_seed{seed}_{target}.json", {"comparisons": seed_result})
    for target, rows in all_summaries.items():
        suffix = "curvevel_a" if target == "cross_curvevel_a" else "flatfault_a"
        _write_csv(out / f"bootstrap_summary_{suffix}.csv", rows)
    print(json.dumps({"status": "SUCCESS", "output": str(out)}, ensure_ascii=False))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrected-metrics-root", required=True, type=Path)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    bootstrap_corrected(args)


if __name__ == "__main__":
    main()
