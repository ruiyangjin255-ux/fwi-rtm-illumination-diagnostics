from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from fwi_visionfm.data.openfwi_npy_dataset import OpenFWINpyDataset


def _sample_values(array: np.ndarray, max_values: int = 200000) -> np.ndarray:
    flat = array.reshape(-1)
    if flat.size <= max_values:
        return flat.astype(np.float32, copy=False)
    step = max(1, flat.size // max_values)
    return flat[::step].astype(np.float32, copy=False)


def _stats_payload(sum_value: float, sq_sum_value: float, count: int, min_value: float, max_value: float, quantile_values: np.ndarray) -> dict[str, float]:
    mean = sum_value / max(count, 1)
    variance = max(sq_sum_value / max(count, 1) - mean * mean, 0.0)
    return {
        "mean": float(mean),
        "std": float(np.sqrt(variance)),
        "min": float(min_value),
        "max": float(max_value),
        "p01": float(np.percentile(quantile_values, 1)),
        "p99": float(np.percentile(quantile_values, 99)),
    }


def compute_openfwi_stats(
    *,
    manifest_path: str | Path,
    split_path: str | Path,
    output_path: str | Path,
    max_samples: int | None = None,
    seed: int = 2026,
) -> dict[str, Any]:
    dataset = OpenFWINpyDataset(
        root=str(Path(manifest_path).parent),
        split_file=str(split_path),
        max_samples=max_samples,
        input_norm="none",
        target_norm="none",
        fit_stats=False,
    )

    seismic_sum = 0.0
    seismic_sq_sum = 0.0
    seismic_count = 0
    seismic_min = float("inf")
    seismic_max = float("-inf")
    seismic_quantiles: list[np.ndarray] = []

    velocity_sum = 0.0
    velocity_sq_sum = 0.0
    velocity_count = 0
    velocity_min = float("inf")
    velocity_max = float("-inf")
    velocity_quantiles: list[np.ndarray] = []

    used_samples = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    for index in range(used_samples):
        sample = dataset[index]
        seismic = sample["seismic"].numpy().astype(np.float32, copy=False)
        velocity = sample["velocity"].numpy().astype(np.float32, copy=False)
        seismic_sum += float(seismic.sum(dtype=np.float64))
        seismic_sq_sum += float(np.square(seismic, dtype=np.float64).sum(dtype=np.float64))
        seismic_count += int(seismic.size)
        seismic_min = min(seismic_min, float(seismic.min()))
        seismic_max = max(seismic_max, float(seismic.max()))
        seismic_quantiles.append(_sample_values(seismic))

        velocity_sum += float(velocity.sum(dtype=np.float64))
        velocity_sq_sum += float(np.square(velocity, dtype=np.float64).sum(dtype=np.float64))
        velocity_count += int(velocity.size)
        velocity_min = min(velocity_min, float(velocity.min()))
        velocity_max = max(velocity_max, float(velocity.max()))
        velocity_quantiles.append(_sample_values(velocity))

    seismic_q = np.concatenate(seismic_quantiles, axis=0) if seismic_quantiles else np.zeros(1, dtype=np.float32)
    velocity_q = np.concatenate(velocity_quantiles, axis=0) if velocity_quantiles else np.zeros(1, dtype=np.float32)

    payload = {
        "manifest_file": str(Path(manifest_path)),
        "split_file": str(Path(split_path)),
        "used_samples": int(used_samples),
        "seed": int(seed),
        "seismic": _stats_payload(seismic_sum, seismic_sq_sum, seismic_count, seismic_min, seismic_max, seismic_q),
        "velocity": _stats_payload(velocity_sum, velocity_sq_sum, velocity_count, velocity_min, velocity_max, velocity_q),
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute train-only OpenFWI normalization stats from a sample-level split CSV.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = compute_openfwi_stats(
        manifest_path=args.manifest,
        split_path=args.split,
        output_path=args.output,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    print(f"写出统计量: {args.output}")
    print(json.dumps({"used_samples": stats["used_samples"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
