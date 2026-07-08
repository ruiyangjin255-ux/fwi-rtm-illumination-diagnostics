# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _per_sample_mae(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    axes = tuple(range(1, prediction.ndim))
    return np.mean(np.abs(prediction - target), axis=axes)


def paired_bootstrap_mae_delta(
    *, candidate_prediction: np.ndarray, candidate_target: np.ndarray, candidate_ids: list[str],
    baseline_prediction: np.ndarray, baseline_target: np.ndarray, baseline_ids: list[str],
    n_bootstrap: int = 2000, seed: int = 0,
) -> dict[str, Any]:
    candidate_map = {str(sample_id): index for index, sample_id in enumerate(candidate_ids)}
    baseline_map = {str(sample_id): index for index, sample_id in enumerate(baseline_ids)}
    common = sorted(set(candidate_map) & set(baseline_map))
    if not common:
        raise ValueError("paired bootstrap has no aligned sample_id values")
    candidate_error = _per_sample_mae(
        np.stack([candidate_prediction[candidate_map[sample_id]] for sample_id in common]),
        np.stack([candidate_target[candidate_map[sample_id]] for sample_id in common]),
    )
    baseline_error = _per_sample_mae(
        np.stack([baseline_prediction[baseline_map[sample_id]] for sample_id in common]),
        np.stack([baseline_target[baseline_map[sample_id]] for sample_id in common]),
    )
    delta = candidate_error - baseline_error
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(int(n_bootstrap), len(delta)))
    means = delta[indices].mean(axis=1)
    return {
        "aligned_sample_count": len(common),
        "mean_difference": float(delta.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "win_probability": float(np.mean(means < 0.0)),
        "candidate_win_count": int(np.sum(delta < 0.0)),
        "paired": True,
    }


def _load_prediction(path: Path) -> dict[str, Any]:
    with np.load(path) as payload:
        return {
            "prediction": payload["velocity_pred_physical"],
            "target": payload["velocity_true_physical"],
            "ids": payload["sample_id"].astype(str).tolist(),
        }


def bootstrap_protocol_v11(*, root: str | Path, n_bootstrap: int, baseline_methods: list[str]) -> list[dict[str, Any]]:
    protocol_root = Path(root)
    output = protocol_root / "bootstrap"; output.mkdir(parents=True, exist_ok=True)
    records: dict[tuple[str, int, str], tuple[Path, dict[str, Any]]] = {}
    for config_path in protocol_root.glob("runs/*/*/seed_*/config.json"):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        prediction_path = config_path.parent / "predictions_cross_family_test.npz"
        if config.get("status") == "SUCCESS" and prediction_path.is_file():
            records[(config["transfer_id"], int(config["seed"]), config["method_key"])] = (prediction_path, config)
    rows = []
    for (transfer, seed, method), (candidate_path, config) in sorted(records.items()):
        for baseline in baseline_methods:
            baseline_record = records.get((transfer, seed, baseline))
            if method == baseline or baseline_record is None:
                continue
            result = paired_bootstrap_mae_delta(
                **{f"candidate_{key}": value for key, value in _load_prediction(candidate_path).items()},
                **{f"baseline_{key}": value for key, value in _load_prediction(baseline_record[0]).items()},
                n_bootstrap=n_bootstrap,
                seed=seed + sum(ord(char) for char in method + baseline),
            )
            rows.append({"transfer_id": transfer, "source_family": config["source_family"], "target_family": config["target_family"], "seed": seed, "method_key": method, "baseline_method": baseline, "n_bootstrap": n_bootstrap, **result})
    fields = ["transfer_id", "source_family", "target_family", "seed", "method_key", "baseline_method", "n_bootstrap", "aligned_sample_count", "mean_difference", "ci_low", "ci_high", "win_probability", "candidate_win_count", "paired"]
    with (output / "protocol_v11_bootstrap_deltas.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    lines = ["# Protocol V11 配对 bootstrap 汇总", "", f"- bootstrap 次数：{n_bootstrap}", "- 所有比较均先按 target `sample_id` 对齐，再计算逐样本 MAE 差值。", "- mean_difference < 0 表示候选方法的逐样本 MAE 更低。", "", "| transfer | seed | method | baseline | mean delta | 95% CI | win probability |", "| --- | ---: | --- | --- | ---: | --- | ---: |"]
    for row in rows:
        lines.append(f"| {row['transfer_id']} | {row['seed']} | {row['method_key']} | {row['baseline_method']} | {row['mean_difference']:.4f} | [{row['ci_low']:.4f}, {row['ci_high']:.4f}] | {row['win_probability']:.3f} |")
    (output / "protocol_v11_bootstrap_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", required=True); parser.add_argument("--n-bootstrap", type=int, default=2000); parser.add_argument("--baseline-methods", nargs="+", default=["cnn_baseline", "random_vit"])
    args = parser.parse_args(); rows = bootstrap_protocol_v11(root=args.root, n_bootstrap=args.n_bootstrap, baseline_methods=args.baseline_methods); print(f"paired_comparisons={len(rows)}")


if __name__ == "__main__": main()
