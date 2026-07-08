# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


COMPARISONS = {
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


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean(values: list[float]) -> float | str:
    if not values:
        return ""
    return float(sum(values) / len(values))


def _float(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def _record_map(per_run: list[dict[str, Any]]) -> dict[tuple[str, int, str, str], dict[str, Any]]:
    result = {}
    for row in per_run:
        if row.get("status") != "SUCCESS":
            continue
        result[(str(row["transfer_id"]), int(row["seed"]), str(row["method_id"]), str(row["bridge_id"]))] = row
    return result


def analyze_protocol_v14_geometry_gain(*, root: str | Path) -> dict[str, Any]:
    protocol_root = Path(root)
    per_run = _read_csv(protocol_root / "protocol_v14_per_run_metrics.csv")
    bootstrap_rows = _read_csv(protocol_root / "bootstrap" / "protocol_v14_bootstrap_deltas.csv")
    bootstrap_map = defaultdict(list)
    for row in bootstrap_rows:
        bootstrap_map[(str(row["comparison_id"]), str(row["transfer_id"]))].append(row)
    record_map = _record_map(per_run)

    gain_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for comparison_id, (candidate_spec, baseline_spec) in COMPARISONS.items():
        matched = []
        for (transfer_id, seed, method_id, bridge_id), candidate in record_map.items():
            if (method_id, bridge_id) != candidate_spec:
                continue
            baseline = record_map.get((transfer_id, seed, baseline_spec[0], baseline_spec[1]))
            if baseline is None:
                continue
            delta_mae = _float(candidate, "cross_family_MAE") - _float(baseline, "cross_family_MAE")
            delta_rmse = _float(candidate, "cross_family_RMSE") - _float(baseline, "cross_family_RMSE")
            delta_gradient = _float(candidate, "cross_family_gradient_error") - _float(baseline, "cross_family_gradient_error")
            delta_edge = _float(candidate, "cross_family_edge_MAE") - _float(baseline, "cross_family_edge_MAE")
            matched.append(
                {
                    "transfer_id": transfer_id,
                    "seed": seed,
                    "delta_mae": delta_mae,
                    "delta_rmse": delta_rmse,
                    "delta_gradient_error": delta_gradient,
                    "delta_edge_mae": delta_edge,
                }
            )
        grouped = defaultdict(list)
        for row in matched:
            grouped[row["transfer_id"]].append(row)
        qualifying_transfer_count = 0
        for transfer_id, group in sorted(grouped.items()):
            bootstrap_group = bootstrap_map.get((comparison_id, transfer_id), [])
            numerical_seed_count = sum(item["delta_mae"] < 0.0 and item["delta_rmse"] < 0.0 for item in group)
            structural_seed_count = sum(item["delta_gradient_error"] <= 0.0 or item["delta_edge_mae"] <= 0.0 for item in group)
            bootstrap_seed_count = sum(float(item["mae_ci_high"]) < 0.0 for item in bootstrap_group)
            qualifying = numerical_seed_count >= 2 and structural_seed_count >= 2 and bootstrap_seed_count >= 2
            if qualifying:
                qualifying_transfer_count += 1
            seed_rows.append(
                {
                    "comparison_id": comparison_id,
                    "transfer_id": transfer_id,
                    "seed_count": len(group),
                    "numerical_improved_seed_count": numerical_seed_count,
                    "structural_nonworse_seed_count": structural_seed_count,
                    "mae_ci_below_zero_seed_count": bootstrap_seed_count,
                    "direction_consistent": qualifying,
                }
            )
        all_mae = [row["delta_mae"] for row in matched]
        all_rmse = [row["delta_rmse"] for row in matched]
        all_gradient = [row["delta_gradient_error"] for row in matched]
        all_edge = [row["delta_edge_mae"] for row in matched]
        if qualifying_transfer_count >= 2:
            evidence_level = "一致的方向性证据"
        elif matched and (
            any(value < 0.0 for value in all_mae)
            or any(value < 0.0 for value in all_rmse)
            or any(value < 0.0 for value in all_gradient)
            or any(value < 0.0 for value in all_edge)
        ):
            evidence_level = "部分或混合证据"
        else:
            evidence_level = "未形成一致证据"
        gain_rows.append(
            {
                "comparison_id": comparison_id,
                "evidence_level": evidence_level,
                "qualifying_transfer_count": qualifying_transfer_count,
                "delta_mae": _mean(all_mae),
                "delta_rmse": _mean(all_rmse),
                "delta_gradient_error": _mean(all_gradient),
                "delta_edge_mae": _mean(all_edge),
            }
        )

    out_path = protocol_root / "protocol_v14_geometry_gain.csv"
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gain_rows[0].keys()))
        writer.writeheader()
        writer.writerows(gain_rows)

    seed_path = protocol_root / "protocol_v14_seed_consistency.csv"
    with seed_path.open("w", encoding="utf-8", newline="") as handle:
        fields = list(seed_rows[0].keys()) if seed_rows else ["comparison_id", "transfer_id", "seed_count"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(seed_rows)

    lines = [
        "# Protocol V14 Geometry Gain",
        "",
        "| 比较 | 证据等级 | 合格 transfer 数 | 平均 delta MAE | 平均 delta RMSE | 平均 delta gradient_error | 平均 delta edge_MAE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in gain_rows:
        lines.append(
            f"| {row['comparison_id']} | {row['evidence_level']} | {row['qualifying_transfer_count']} | "
            f"{row['delta_mae']} | {row['delta_rmse']} | {row['delta_gradient_error']} | {row['delta_edge_mae']} |"
        )
    (protocol_root / "protocol_v14_geometry_gain_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"row_count": len(gain_rows), "output_path": str(out_path), "report_path": str(protocol_root / "protocol_v14_geometry_gain_report.md")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    print(json.dumps(analyze_protocol_v14_geometry_gain(root=args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
