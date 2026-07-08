from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from fwi_visionfm.datasets import discover_npz_samples, load_npz_sample
from fwi_visionfm.shape_utils import infer_npz_dataset_shape


def _stats(values: list[np.ndarray]) -> dict[str, float]:
    flat = np.concatenate([value.astype(np.float32, copy=False).ravel() for value in values], axis=0)
    return {
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
    }


def _flatten_stat_fields(prefix: str, stats: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{name}": float(value) for name, value in stats.items()}


def validate_npz_dataset(data_dir: str | Path, *, max_checks: int | None = None) -> dict[str, Any]:
    root = Path(data_dir)
    paths = discover_npz_samples(root)
    if not paths:
        raise ValueError(f"no npz samples found in {data_dir}")
    if max_checks is not None:
        if max_checks <= 0:
            raise ValueError("max_checks must be positive when provided")
        paths = paths[:max_checks]

    shape_summary = infer_npz_dataset_shape(root, max_checks=max_checks)
    records_shapes: set[tuple[int, ...]] = {tuple(shape) for shape in shape_summary["records_shape_set"]}
    velocity_shapes: set[tuple[int, ...]] = {tuple(shape) for shape in shape_summary["velocity_shape_set"]}
    has_source_positions_count = 0
    records_arrays: list[np.ndarray] = []
    velocity_arrays: list[np.ndarray] = []

    for path in paths:
        with np.load(path) as data:
            if "records" not in data or "velocity" not in data:
                raise ValueError(f"{path} is missing required arrays 'records' and/or 'velocity'")
            records = np.asarray(data["records"], dtype=np.float32)
            velocity = np.asarray(data["velocity"], dtype=np.float32)
            if records.ndim != 3:
                raise ValueError(f"{path} records must be 3D, got {records.shape}")
            if velocity.ndim != 2:
                raise ValueError(f"{path} velocity must be 2D, got {velocity.shape}")
            if "source_positions" in data:
                has_source_positions_count += 1
        sample = load_npz_sample(path)
        records_arrays.append(sample.records)
        velocity_arrays.append(sample.velocity)

    if not shape_summary["is_uniform_records_shape"]:
        error_message = f"inconsistent records shapes detected: {shape_summary['records_shape_set']}"
        records_stats = _stats(records_arrays)
        velocity_stats = _stats(velocity_arrays)
        summary = {
            **shape_summary,
            "has_source_positions_count": int(has_source_positions_count),
            "records_stats": records_stats,
            "velocity_stats": velocity_stats,
            **_flatten_stat_fields("records", records_stats),
            **_flatten_stat_fields("velocity", velocity_stats),
            "error": error_message,
        }
        (root / "validation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        raise ValueError(error_message)
    if not shape_summary["is_uniform_velocity_shape"]:
        error_message = f"inconsistent velocity shapes detected: {shape_summary['velocity_shape_set']}"
        records_stats = _stats(records_arrays)
        velocity_stats = _stats(velocity_arrays)
        summary = {
            **shape_summary,
            "has_source_positions_count": int(has_source_positions_count),
            "records_stats": records_stats,
            "velocity_stats": velocity_stats,
            **_flatten_stat_fields("records", records_stats),
            **_flatten_stat_fields("velocity", velocity_stats),
            "error": error_message,
        }
        (root / "validation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        raise ValueError(error_message)

    records_stats = _stats(records_arrays)
    velocity_stats = _stats(velocity_arrays)
    summary = {
        **shape_summary,
        "has_source_positions_count": int(has_source_positions_count),
        "records_stats": records_stats,
        "velocity_stats": velocity_stats,
        **_flatten_stat_fields("records", records_stats),
        **_flatten_stat_fields("velocity", velocity_stats),
    }
    (root / "validation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证本地逐样本 .npz 数据目录。")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--max-checks", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate_npz_dataset(args.data_dir, max_checks=args.max_checks)
    print(f"样本数: {summary['sample_count']}")
    print(f"records 形状集合: {summary['records_shape_set']}")
    print(f"velocity 形状集合: {summary['velocity_shape_set']}")
    print(f"inferred depth/width: {summary['inferred_depth']}/{summary['inferred_width']}")
    if summary["inferred_depth"] == 70 and summary["inferred_width"] == 70:
        print("Dataset velocity shape verified as 70x70.")
    else:
        print("Warning: this appears to be smoke/tiny synthetic data, not standard OpenFWI 70x70.")
    print(f"validation: {args.data_dir / 'validation_summary.json'}")


if __name__ == "__main__":
    main()
