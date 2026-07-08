# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from fwi_visionfm.evaluation.metrics import compute_velocity_metrics


METRICS = ("mae", "rmse", "gradient_error", "edge_mae")
COMPARISONS = (
    ("M5_vs_M4", "spectrogram_dinov2_lora", "dinov2_lora"),
    ("M5_vs_M2", "spectrogram_dinov2_lora", "random_vit"),
    ("M5_vs_M1", "spectrogram_dinov2_lora", "cnn_baseline"),
    ("M3_vs_M2", "dinov2_frozen", "random_vit"),
    ("M4_vs_M3", "dinov2_lora", "dinov2_frozen"),
)


def _per_sample_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    values = {metric: [] for metric in METRICS}
    for index in range(len(prediction)):
        metrics = compute_velocity_metrics(prediction[index : index + 1], target[index : index + 1])
        for metric in METRICS:
            values[metric].append(float(metrics[metric]))
    return {metric: np.asarray(rows, dtype=np.float64) for metric, rows in values.items()}


def paired_bootstrap_metric_deltas(
    *, candidate_prediction: np.ndarray, candidate_target: np.ndarray, candidate_ids: list[str],
    baseline_prediction: np.ndarray, baseline_target: np.ndarray, baseline_ids: list[str],
    n_bootstrap: int = 2000, seed: int = 0,
) -> dict[str, Any]:
    candidate_map = {str(sample_id): index for index, sample_id in enumerate(candidate_ids)}; baseline_map = {str(sample_id): index for index, sample_id in enumerate(baseline_ids)}
    common = sorted(set(candidate_map) & set(baseline_map))
    if not common:
        raise ValueError("paired bootstrap has no aligned sample_id values")
    candidate_index = [candidate_map[sample_id] for sample_id in common]; baseline_index = [baseline_map[sample_id] for sample_id in common]
    candidate_metrics = _per_sample_metrics(candidate_prediction[candidate_index], candidate_target[candidate_index]); baseline_metrics = _per_sample_metrics(baseline_prediction[baseline_index], baseline_target[baseline_index])
    rng = np.random.default_rng(seed); indices = rng.integers(0, len(common), size=(int(n_bootstrap), len(common)))
    result: dict[str, Any] = {"aligned_sample_count": len(common), "paired": True}
    for metric in METRICS:
        delta = candidate_metrics[metric] - baseline_metrics[metric]; means = delta[indices].mean(axis=1)
        result.update({f"{metric}_mean_difference": float(delta.mean()), f"{metric}_ci_low": float(np.quantile(means, 0.025)), f"{metric}_ci_high": float(np.quantile(means, 0.975)), f"{metric}_win_probability": float(np.mean(means < 0.0))})
    return result


def _load(path: Path) -> dict[str, Any]:
    with np.load(path) as payload:
        return {"prediction": payload["velocity_pred_physical"], "target": payload["velocity_true_physical"], "ids": payload["sample_id"].astype(str).tolist()}


def bootstrap_protocol_v12(*, root: str | Path, n_bootstrap: int) -> list[dict[str, Any]]:
    protocol_root = Path(root); output = protocol_root / "bootstrap"; output.mkdir(parents=True, exist_ok=True); records = {}
    for config_path in protocol_root.glob("runs/*/*/seed_*/config.json"):
        config = json.loads(config_path.read_text(encoding="utf-8")); prediction = config_path.parent / "predictions_cross_family_test.npz"
        if config.get("status") == "SUCCESS" and prediction.is_file(): records[(config["transfer_id"], int(config["seed"]), config["method_key"])] = (prediction, config)
    rows = []
    for comparison_id, candidate, baseline in COMPARISONS:
        keys = sorted((transfer, seed) for transfer, seed, method in records if method == candidate and (transfer, seed, baseline) in records)
        for transfer, seed in keys:
            candidate_path, config = records[(transfer, seed, candidate)]; baseline_path, _ = records[(transfer, seed, baseline)]
            result = paired_bootstrap_metric_deltas(**{f"candidate_{key}": value for key, value in _load(candidate_path).items()}, **{f"baseline_{key}": value for key, value in _load(baseline_path).items()}, n_bootstrap=n_bootstrap, seed=seed + sum(map(ord, comparison_id)))
            rows.append({"comparison_id": comparison_id, "transfer_id": transfer, "source_family": config["source_family"], "target_family": config["target_family"], "seed": seed, "candidate_method": candidate, "baseline_method": baseline, "n_bootstrap": n_bootstrap, **result})
    fields = ["comparison_id", "transfer_id", "source_family", "target_family", "seed", "candidate_method", "baseline_method", "n_bootstrap", "aligned_sample_count", "paired"] + [f"{metric}_{suffix}" for metric in METRICS for suffix in ("mean_difference", "ci_low", "ci_high", "win_probability")]
    with (output / "protocol_v12_bootstrap_deltas.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows: grouped[(row["comparison_id"], row["transfer_id"])].append(row)
    consistency = []
    for (comparison_id, transfer), group in sorted(grouped.items()):
        item = {"comparison_id": comparison_id, "transfer_id": transfer, "seed_count": len(group), "aligned_sample_count_min": min(int(row["aligned_sample_count"]) for row in group)}
        for metric in METRICS:
            item[f"{metric}_improved_seed_count"] = sum(float(row[f"{metric}_mean_difference"]) < 0 for row in group); item[f"{metric}_ci_below_zero_seed_count"] = sum(float(row[f"{metric}_ci_high"]) < 0 for row in group)
        consistency.append(item)
    consistency_fields = list(consistency[0]) if consistency else ["comparison_id", "transfer_id", "seed_count"]
    with (output / "protocol_v12_seed_consistency.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=consistency_fields); writer.writeheader(); writer.writerows(consistency)
    lines = ["# Protocol V12 配对 bootstrap 汇总", "", f"- bootstrap 次数：{n_bootstrap}", "- 所有预测按 target sample_id 对齐后进行配对比较。", "- difference < 0 表示候选方法误差更低。", "", "| 比较 | transfer | seed | MAE delta | MAE 95% CI | win probability |", "| --- | --- | ---: | ---: | --- | ---: |"]
    for row in rows: lines.append(f"| {row['comparison_id']} | {row['transfer_id']} | {row['seed']} | {row['mae_mean_difference']:.3f} | [{row['mae_ci_low']:.3f}, {row['mae_ci_high']:.3f}] | {row['mae_win_probability']:.3f} |")
    (output / "protocol_v12_bootstrap_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8"); return rows


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", required=True); parser.add_argument("--n-bootstrap", type=int, default=2000); parser.add_argument("--baseline-methods", nargs="+", default=["cnn_baseline", "random_vit", "dinov2_lora"])
    args = parser.parse_args(); print(f"paired_comparisons={len(bootstrap_protocol_v12(root=args.root, n_bootstrap=args.n_bootstrap))}")


if __name__ == "__main__":
    main()
