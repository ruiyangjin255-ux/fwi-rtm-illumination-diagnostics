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
    from scripts.bootstrap_protocol_v13_pretraining_source import COMPARISONS
    from scripts.analyze_protocol_v13_generalization_gaps import compute_generalization_gaps
except ModuleNotFoundError:  # direct script execution
    from bootstrap_protocol_v13_pretraining_source import COMPARISONS
    from analyze_protocol_v13_generalization_gaps import compute_generalization_gaps


METRICS = ("mae", "rmse", "ssim", "gradient_error", "edge_mae")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _collect(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config_path in sorted(root.glob("runs/*/*/seed_*/config.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        row = {**config, "run_dir": str(config_path.parent)}
        in_path = config_path.parent / "metrics_in_family_test.json"
        cross_path = config_path.parent / "metrics_cross_family_test.json"
        if config.get("status") == "SUCCESS" and in_path.is_file() and cross_path.is_file():
            in_metrics = json.loads(in_path.read_text(encoding="utf-8"))
            cross_metrics = json.loads(cross_path.read_text(encoding="utf-8"))
            for metric in METRICS:
                row[f"in_family_{metric}"] = float(in_metrics[metric])
                row[f"cross_family_{metric}"] = float(cross_metrics[metric])
            row.update(compute_generalization_gaps(in_metrics, cross_metrics))
        rows.append(row)
    return rows


def _comparison_evidence(success: list[dict[str, Any]], bootstrap: list[dict[str, str]]) -> dict[str, Any]:
    by_key = {(row["transfer_id"], int(row["seed"]), row["method_key"]): row for row in success}
    boot = defaultdict(list)
    for row in bootstrap:
        boot[(row["comparison_id"], row["transfer_id"])].append(row)
    transfers = sorted({row["transfer_id"] for row in success})
    evidence: dict[str, Any] = {}
    for comparison_id, (candidate, baseline) in COMPARISONS.items():
        qualifying = 0
        partial = False
        details = []
        for transfer in transfers:
            numerical = structural = 0
            for seed in (0, 1, 2):
                cand = by_key.get((transfer, seed, candidate))
                ref = by_key.get((transfer, seed, baseline))
                if not cand or not ref:
                    continue
                num = cand["cross_family_mae"] < ref["cross_family_mae"] and cand["cross_family_rmse"] < ref["cross_family_rmse"]
                struct = cand["cross_family_gradient_error"] <= ref["cross_family_gradient_error"] or cand["cross_family_edge_mae"] <= ref["cross_family_edge_mae"]
                numerical += int(num)
                structural += int(struct)
                partial = partial or num or struct
            bootstrap_count = sum(
                float(row["mae_ci_high"]) < 0
                and str(row.get("paired", "")).lower() == "true"
                and int(row["aligned_sample_count"]) == 50
                for row in boot.get((comparison_id, transfer), [])
            )
            qualifies = numerical >= 2 and structural >= 2 and bootstrap_count >= 2
            qualifying += int(qualifies)
            details.append({"transfer_id": transfer, "numerical_seed_count": numerical, "structural_seed_count": structural, "bootstrap_ci_below_zero_seed_count": bootstrap_count, "qualifying": qualifies})
        level = "存在一致的方向性证据" if qualifying >= 2 else ("存在部分或混合证据" if partial else "当前未形成一致证据")
        evidence[comparison_id] = {"candidate": candidate, "baseline": baseline, "evidence_level": level, "qualifying_transfer_count": qualifying, "details": details}
    return evidence


def summarize_protocol_v13(root: str | Path) -> dict[str, Any]:
    protocol_root = Path(root)
    rows = _collect(protocol_root)
    success = [row for row in rows if row.get("status") == "SUCCESS"]
    bootstrap = _read_csv(protocol_root / "bootstrap" / "protocol_v13_bootstrap_deltas.csv")
    evidence = _comparison_evidence(success, bootstrap)

    per_fields = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (dict, list))}) or ["run_id"]
    with (protocol_root / "protocol_v13_per_run_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)

    grouped = defaultdict(list)
    for row in success:
        grouped[(row["transfer_id"], row["method_key"])].append(row)
    aggregate = []
    for (transfer, method), group in sorted(grouped.items()):
        item: dict[str, Any] = {"transfer_id": transfer, "source_family": group[0]["source_family"], "target_family": group[0]["target_family"], "method_id": group[0]["method_id"], "method_key": method, "method_name": group[0]["method_name"], "seed_count": len(group)}
        for prefix in ("in_family", "cross_family"):
            for metric in METRICS:
                values = [float(row[f"{prefix}_{metric}"]) for row in group]
                item[f"{prefix}_{metric}_mean"] = float(np.mean(values))
                item[f"{prefix}_{metric}_std"] = float(np.std(values))
        for gap in ("mae", "rmse", "ssim", "gradient", "edge"):
            key = f"{gap}_generalization_gap"
            item[f"{key}_mean"] = float(np.mean([float(row[key]) for row in group]))
        aggregate.append(item)
    aggregate_fields = list(aggregate[0]) if aggregate else ["transfer_id", "method_key"]
    with (protocol_root / "protocol_v13_aggregate_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader(); writer.writerows(aggregate)

    evidence_lines = ["# Protocol V13 预训练来源证据", "", "以下判定只表示当前 CPU 小样本统一协议中的方向性证据。", ""]
    for comparison_id, value in evidence.items():
        evidence_lines.append(f"- **{comparison_id}**（{value['candidate']} vs {value['baseline']}）：{value['evidence_level']}，满足 transfer 数 {value['qualifying_transfer_count']}/3。")
    (protocol_root / "protocol_v13_generalization_evidence.md").write_text("\n".join(evidence_lines) + "\n", encoding="utf-8")

    payload = {"run_count": len(rows), "success": len(success), "failed": sum(str(row.get("status", "")).startswith("FAILED") for row in rows), "skipped": sum(str(row.get("status", "")).startswith("SKIPPED") for row in rows), "reused": sum(row.get("reused_from") == "protocol_v12" for row in rows), "real_ncs_feature_runs": sum(row.get("method_key") == "ncs2d_frozen" and row.get("is_real_feature") is True for row in success), "comparison_evidence": evidence}
    (protocol_root / "protocol_v13_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    print(json.dumps(summarize_protocol_v13(args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
