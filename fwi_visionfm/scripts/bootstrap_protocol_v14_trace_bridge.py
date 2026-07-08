# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.bootstrap_protocol_v12_comparisons import METRICS, paired_bootstrap_metric_deltas
except ModuleNotFoundError:  # pragma: no cover
    from bootstrap_protocol_v12_comparisons import METRICS, paired_bootstrap_metric_deltas


COMPARISON_MAP = {
    "M3_B1_vs_B0": (("M3", "B1"), ("M3", "B0")),
    "M3_B2_vs_B1": (("M3", "B2"), ("M3", "B1")),
    "M3_B3_vs_B2": (("M3", "B3"), ("M3", "B2")),
    "M3_B3_vs_B0": (("M3", "B3"), ("M3", "B0")),
    "M6_B1_vs_B0": (("M6", "B1"), ("M6", "B0")),
    "M6_B2_vs_B1": (("M6", "B2"), ("M6", "B1")),
    "M6_B3_vs_B2": (("M6", "B3"), ("M6", "B2")),
    "M6_B3_vs_B0": (("M6", "B3"), ("M6", "B0")),
    "M6_B3_vs_M3_B3": (("M6", "B3"), ("M3", "B3")),
}


def _load_prediction(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as payload:
        return {
            "prediction": np.asarray(payload["velocity_pred_physical"], dtype=np.float32),
            "target": np.asarray(payload["velocity_true_physical"], dtype=np.float32),
            "ids": payload["sample_id"].astype(str).tolist(),
        }


def _collect_records(protocol_root: Path) -> dict[tuple[str, str, int, str, str], dict[str, Any]]:
    records: dict[tuple[str, str, int, str, str], dict[str, Any]] = {}
    for config_path in protocol_root.glob("runs/*/*/seed_*/B*/config.json"):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        prediction_path = config_path.parent / "predictions_cross_family_test.npz"
        if config.get("status") != "SUCCESS" or not prediction_path.is_file():
            continue
        bridge_id = str(config.get("bridge_id") or config_path.parent.name)
        key = (
            str(config["transfer_id"]),
            str(config["method_id"]),
            int(config["seed"]),
            bridge_id,
            str(config["target_family"]),
        )
        records[key] = {
            "prediction_path": prediction_path,
            "config": config,
        }
    return records


def bootstrap_protocol_v14_trace_bridge(*, root: str | Path, n_bootstrap: int, comparisons: list[str]) -> dict[str, Any]:
    protocol_root = Path(root)
    output = protocol_root / "bootstrap"
    output.mkdir(parents=True, exist_ok=True)
    records = _collect_records(protocol_root)
    rows: list[dict[str, Any]] = []
    for comparison_id in comparisons:
        if comparison_id not in COMPARISON_MAP:
            continue
        candidate_spec, baseline_spec = COMPARISON_MAP[comparison_id]
        keys = sorted(
            {
                (transfer_id, seed, target_family)
                for (transfer_id, method_id, seed, bridge_id, target_family) in records
                if (method_id, bridge_id) == candidate_spec
                and (transfer_id, baseline_spec[0], seed, baseline_spec[1], target_family) in records
            }
        )
        for transfer_id, seed, target_family in keys:
            candidate = records[(transfer_id, candidate_spec[0], seed, candidate_spec[1], target_family)]
            baseline = records[(transfer_id, baseline_spec[0], seed, baseline_spec[1], target_family)]
            candidate_payload = _load_prediction(candidate["prediction_path"])
            baseline_payload = _load_prediction(baseline["prediction_path"])
            result = paired_bootstrap_metric_deltas(
                candidate_prediction=candidate_payload["prediction"],
                candidate_target=candidate_payload["target"],
                candidate_ids=candidate_payload["ids"],
                baseline_prediction=baseline_payload["prediction"],
                baseline_target=baseline_payload["target"],
                baseline_ids=baseline_payload["ids"],
                n_bootstrap=int(n_bootstrap),
                seed=int(seed) + sum(map(ord, comparison_id)),
            )
            rows.append(
                {
                    "comparison_id": comparison_id,
                    "transfer_id": transfer_id,
                    "source_family": candidate["config"]["source_family"],
                    "target_family": candidate["config"]["target_family"],
                    "seed": int(seed),
                    "candidate_method_id": candidate_spec[0],
                    "candidate_bridge_id": candidate_spec[1],
                    "baseline_method_id": baseline_spec[0],
                    "baseline_bridge_id": baseline_spec[1],
                    "n_bootstrap": int(n_bootstrap),
                    "aligned_sample_count": int(result["aligned_sample_count"]),
                    "sample_ids_aligned": bool(result["paired"]),
                    "delta_mae_mean": float(result["mae_mean_difference"]),
                    "delta_rmse_mean": float(result["rmse_mean_difference"]),
                    "delta_gradient_error_mean": float(result["gradient_error_mean_difference"]),
                    "delta_edge_mae_mean": float(result["edge_mae_mean_difference"]),
                    "mae_ci_low": float(result["mae_ci_low"]),
                    "mae_ci_high": float(result["mae_ci_high"]),
                    "rmse_ci_low": float(result["rmse_ci_low"]),
                    "rmse_ci_high": float(result["rmse_ci_high"]),
                    "gradient_error_ci_low": float(result["gradient_error_ci_low"]),
                    "gradient_error_ci_high": float(result["gradient_error_ci_high"]),
                    "edge_mae_ci_low": float(result["edge_mae_ci_low"]),
                    "edge_mae_ci_high": float(result["edge_mae_ci_high"]),
                    "mae_win_probability": float(result["mae_win_probability"]),
                    "rmse_win_probability": float(result["rmse_win_probability"]),
                    "gradient_error_win_probability": float(result["gradient_error_win_probability"]),
                    "edge_mae_win_probability": float(result["edge_mae_win_probability"]),
                }
            )
    fields = [
        "comparison_id",
        "transfer_id",
        "source_family",
        "target_family",
        "seed",
        "candidate_method_id",
        "candidate_bridge_id",
        "baseline_method_id",
        "baseline_bridge_id",
        "n_bootstrap",
        "aligned_sample_count",
        "sample_ids_aligned",
        "delta_mae_mean",
        "delta_rmse_mean",
        "delta_gradient_error_mean",
        "delta_edge_mae_mean",
        "mae_ci_low",
        "mae_ci_high",
        "rmse_ci_low",
        "rmse_ci_high",
        "gradient_error_ci_low",
        "gradient_error_ci_high",
        "edge_mae_ci_low",
        "edge_mae_ci_high",
        "mae_win_probability",
        "rmse_win_probability",
        "gradient_error_win_probability",
        "edge_mae_win_probability",
    ]
    with (output / "protocol_v14_bootstrap_deltas.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["comparison_id"], row["transfer_id"])].append(row)
    consistency_rows = []
    for (comparison_id, transfer_id), group in sorted(grouped.items()):
        consistency_rows.append(
            {
                "comparison_id": comparison_id,
                "transfer_id": transfer_id,
                "seed_count": len(group),
                "mae_improved_seed_count": sum(float(row["delta_mae_mean"]) < 0.0 for row in group),
                "rmse_improved_seed_count": sum(float(row["delta_rmse_mean"]) < 0.0 for row in group),
                "structural_nonworse_seed_count": sum(
                    float(row["delta_gradient_error_mean"]) <= 0.0 or float(row["delta_edge_mae_mean"]) <= 0.0
                    for row in group
                ),
                "mae_ci_below_zero_seed_count": sum(float(row["mae_ci_high"]) < 0.0 for row in group),
            }
        )
    with (output / "protocol_v14_seed_bootstrap_consistency.csv").open("w", encoding="utf-8", newline="") as handle:
        fields2 = list(consistency_rows[0].keys()) if consistency_rows else ["comparison_id", "transfer_id", "seed_count"]
        writer = csv.DictWriter(handle, fieldnames=fields2)
        writer.writeheader()
        writer.writerows(consistency_rows)

    lines = [
        "# Protocol V14 配对 bootstrap 汇总",
        "",
        f"- bootstrap 次数：{int(n_bootstrap)}",
        "- 所有比较按相同 target sample_id 对齐。",
        "- delta < 0 表示候选配置误差更低。",
        "",
        "| 比较 | transfer | seed | MAE delta | 95% CI | win probability |",
        "| --- | --- | ---: | ---: | --- | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['comparison_id']} | {row['transfer_id']} | {row['seed']} | "
            f"{row['delta_mae_mean']:.3f} | [{row['mae_ci_low']:.3f}, {row['mae_ci_high']:.3f}] | {row['mae_win_probability']:.3f} |"
        )
    (output / "protocol_v14_bootstrap_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {"comparison_count": len(rows), "n_bootstrap": int(n_bootstrap)}
    (output / "protocol_v14_bootstrap_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--comparisons", nargs="+", required=True)
    args = parser.parse_args()
    print(json.dumps(bootstrap_protocol_v14_trace_bridge(root=args.root, n_bootstrap=args.n_bootstrap, comparisons=args.comparisons), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
