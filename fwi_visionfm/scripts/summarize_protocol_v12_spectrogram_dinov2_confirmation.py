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
    from scripts.bootstrap_protocol_v12_comparisons import COMPARISONS
except ModuleNotFoundError:  # direct script execution
    from bootstrap_protocol_v12_comparisons import COMPARISONS


METRICS = ("mae", "rmse", "ssim", "gradient_error", "edge_mae")


def _csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file(): return []
    with path.open("r", encoding="utf-8", newline="") as handle: return list(csv.DictReader(handle))


def _collect(root: Path) -> list[dict[str, Any]]:
    rows = []
    for config_path in sorted(root.glob("runs/*/*/seed_*/config.json")):
        config = json.loads(config_path.read_text(encoding="utf-8")); row = dict(config); row["run_dir"] = str(config_path.parent)
        if config.get("status") == "SUCCESS":
            in_metrics = json.loads((config_path.parent / "metrics_in_family_test.json").read_text(encoding="utf-8")); cross = json.loads((config_path.parent / "metrics_cross_family_test.json").read_text(encoding="utf-8"))
            for metric in METRICS:
                row[f"in_family_{metric}"] = float(in_metrics[metric]); row[f"cross_family_{metric}"] = float(cross[metric]); row[f"{metric}_generalization_gap"] = float(cross[metric] - in_metrics[metric])
        rows.append(row)
    return rows


def _comparison_evidence(success: list[dict[str, Any]], bootstrap: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    by_key = {(row["transfer_id"], int(row["seed"]), row["method_key"]): row for row in success}; boot = defaultdict(list)
    for row in bootstrap: boot[(row["comparison_id"], row["transfer_id"])].append(row)
    result = {}
    transfers = sorted({row["transfer_id"] for row in success})
    for comparison_id, candidate, baseline in COMPARISONS:
        qualifying = 0; partial = False; details = []
        for transfer in transfers:
            numerical_count = 0; structural_count = 0
            for seed in (0, 1, 2):
                cand = by_key.get((transfer, seed, candidate)); ref = by_key.get((transfer, seed, baseline))
                if not cand or not ref: continue
                numerical = cand["cross_family_mae"] < ref["cross_family_mae"] and cand["cross_family_rmse"] < ref["cross_family_rmse"]
                structural = cand["cross_family_gradient_error"] <= ref["cross_family_gradient_error"] or cand["cross_family_edge_mae"] <= ref["cross_family_edge_mae"]
                numerical_count += int(numerical); structural_count += int(structural); partial = partial or numerical or structural
            records = boot.get((comparison_id, transfer), []); bootstrap_count = sum(float(row["mae_ci_high"]) < 0 and str(row.get("paired", "")).lower() == "true" and int(row["aligned_sample_count"]) == 50 for row in records)
            is_qualifying = numerical_count >= 2 and structural_count >= 2 and bootstrap_count >= 2
            qualifying += int(is_qualifying); details.append({"transfer_id": transfer, "numerical_seed_count": numerical_count, "structural_seed_count": structural_count, "bootstrap_ci_below_zero_seed_count": bootstrap_count, "qualifying": is_qualifying})
        level = "存在一致的方向性证据" if qualifying >= 2 else ("存在部分或混合证据" if partial else "当前未形成一致证据")
        result[comparison_id] = {"candidate": candidate, "baseline": baseline, "evidence_level": level, "qualifying_transfer_count": qualifying, "details": details}
    return result


def summarize_protocol_v12(root: str | Path) -> dict[str, Any]:
    protocol_root = Path(root); run_rows = _collect(protocol_root); success = [row for row in run_rows if row.get("status") == "SUCCESS"]; bootstrap = _csv(protocol_root / "bootstrap" / "protocol_v12_bootstrap_deltas.csv"); evidence = _comparison_evidence(success, bootstrap)
    per_fields = sorted({key for row in run_rows for key, value in row.items() if not isinstance(value, (dict, list))})
    with (protocol_root / "protocol_v12_per_run_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(run_rows)
    gap_fields = ["transfer_id", "method_key", "seed"] + [f"{metric}_generalization_gap" for metric in METRICS]
    with (protocol_root / "protocol_v12_generalization_gaps.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=gap_fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(success)
    grouped = defaultdict(list)
    for row in success: grouped[(row["transfer_id"], row["method_key"])].append(row)
    aggregate = []
    for (transfer, method), group in sorted(grouped.items()):
        item = {"transfer_id": transfer, "source_family": group[0]["source_family"], "target_family": group[0]["target_family"], "method_id": group[0]["method_id"], "method_key": method, "method_name": group[0]["method_name"], "seed_count": len(group)}
        for prefix in ("in_family", "cross_family"):
            for metric in METRICS:
                values = [float(row[f"{prefix}_{metric}"]) for row in group]; item[f"{prefix}_{metric}_mean"] = float(np.mean(values)); item[f"{prefix}_{metric}_std"] = float(np.std(values))
        for metric in METRICS: item[f"{metric}_generalization_gap_mean"] = float(np.mean([row[f"{metric}_generalization_gap"] for row in group]))
        aggregate.append(item)
    fields = list(aggregate[0]) if aggregate else ["transfer_id", "method_key"]
    with (protocol_root / "protocol_v12_aggregate_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(aggregate)
    lines = ["# Protocol V12 泛化方向性证据", "", "以下结论严格按预注册比较分别判读，不等同于已证明提升 FWI 泛化能力。", ""]
    for comparison_id, value in evidence.items(): lines.append(f"- **{comparison_id}**（{value['candidate']} vs {value['baseline']}）：{value['evidence_level']}，满足 transfer 数 {value['qualifying_transfer_count']}/3。")
    (protocol_root / "protocol_v12_generalization_evidence.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {"run_count": len(run_rows), "success": len(success), "failed": sum(str(row.get("status", "")).startswith("FAILED") for row in run_rows), "skipped": sum(str(row.get("status", "")).startswith("SKIPPED") for row in run_rows), "available_transfer_count": len({row["transfer_id"] for row in success}), "comparison_evidence": evidence}; (protocol_root / "protocol_v12_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"); return payload


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", required=True); args = parser.parse_args(); print(json.dumps(summarize_protocol_v12(args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
