# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from fwi_visionfm.models.protocol_v11_model_registry import METHOD_SPECS


METRICS = ["mae", "rmse", "ssim", "gradient_error", "edge_mae"]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _collect_runs(root: Path) -> list[dict[str, Any]]:
    rows = []
    for config_path in sorted(root.glob("runs/*/*/seed_*/config.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        row = dict(config); row["run_dir"] = str(config_path.parent)
        if config.get("status") == "SUCCESS":
            in_metrics = json.loads((config_path.parent / "metrics_in_family_test.json").read_text(encoding="utf-8"))
            cross_metrics = json.loads((config_path.parent / "metrics_cross_family_test.json").read_text(encoding="utf-8"))
            for metric in METRICS:
                row[f"in_family_{metric}"] = float(in_metrics[metric])
                row[f"cross_family_{metric}"] = float(cross_metrics[metric])
                if metric != "ssim":
                    row[f"{metric}_generalization_gap"] = float(cross_metrics[metric] - in_metrics[metric])
            row["structural_generalization_gap"] = float(cross_metrics["edge_mae"] - in_metrics["edge_mae"])
        rows.append(row)
    return rows


def _evidence_levels(success_rows: list[dict[str, Any]], bootstrap_rows: list[dict[str, str]]) -> dict[str, str]:
    by_key = {(row["transfer_id"], int(row["seed"]), row["method_key"]): row for row in success_rows}
    bootstrap = defaultdict(list)
    for row in bootstrap_rows:
        bootstrap[(row["transfer_id"], row["method_key"], row["baseline_method"])].append(row)
    levels = {spec["method_key"]: "当前未形成一致证据" for spec in METHOD_SPECS}
    for spec in METHOD_SPECS:
        method = spec["method_key"]
        if method in {"cnn_baseline", "random_vit"}:
            levels[method] = "基线参考"
            continue
        qualifying_transfers = 0
        partial = False
        transfers = sorted({key[0] for key in by_key if key[2] == method})
        for transfer in transfers:
            consistent_seeds = 0
            structure_ok_count = 0
            boot_ok = True
            for seed in (0, 1, 2):
                candidate = by_key.get((transfer, seed, method)); cnn = by_key.get((transfer, seed, "cnn_baseline")); random = by_key.get((transfer, seed, "random_vit"))
                if not candidate or not cnn or not random:
                    continue
                numerical = candidate["cross_family_mae"] < min(cnn["cross_family_mae"], random["cross_family_mae"]) and candidate["cross_family_rmse"] < min(cnn["cross_family_rmse"], random["cross_family_rmse"])
                structural = candidate["cross_family_gradient_error"] <= min(cnn["cross_family_gradient_error"], random["cross_family_gradient_error"]) or candidate["cross_family_edge_mae"] <= min(cnn["cross_family_edge_mae"], random["cross_family_edge_mae"])
                consistent_seeds += int(numerical)
                structure_ok_count += int(structural)
                partial = partial or numerical or structural
            for baseline in ("cnn_baseline", "random_vit"):
                records = bootstrap.get((transfer, method, baseline), [])
                if not records or sum(float(record["ci_high"]) < 0 for record in records) < 2:
                    boot_ok = False
            if consistent_seeds >= 2 and structure_ok_count >= 2 and boot_ok:
                qualifying_transfers += 1
        if qualifying_transfers >= 2:
            levels[method] = "存在一致的方向性证据"
        elif partial:
            levels[method] = "存在部分或混合证据"
    return levels


def summarize_protocol_v11(root: str | Path) -> dict[str, Any]:
    protocol_root = Path(root)
    run_rows = _collect_runs(protocol_root)
    success = [row for row in run_rows if row.get("status") == "SUCCESS"]
    bootstrap_rows = _read_csv(protocol_root / "bootstrap" / "protocol_v11_bootstrap_deltas.csv")
    levels = _evidence_levels(success, bootstrap_rows)
    per_run_fields = sorted({key for row in run_rows for key in row if not isinstance(row[key], (dict, list))})
    with (protocol_root / "protocol_v11_per_run_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_run_fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(run_rows)
    grouped = defaultdict(list)
    for row in success:
        grouped[(row["transfer_id"], row["method_key"])].append(row)
    aggregate = []
    for (transfer, method), group in sorted(grouped.items()):
        item: dict[str, Any] = {"transfer_id": transfer, "source_family": group[0]["source_family"], "target_family": group[0]["target_family"], "method_id": group[0]["method_id"], "method_key": method, "method_name": group[0]["method_name"], "seed_count": len(group), "evidence_level": levels[method]}
        for prefix in ("in_family", "cross_family"):
            for metric in METRICS:
                values = [float(row[f"{prefix}_{metric}"]) for row in group]
                item[f"{prefix}_{metric}_mean"] = float(np.mean(values)); item[f"{prefix}_{metric}_std"] = float(np.std(values))
        for metric in ("mae", "rmse", "gradient_error", "edge_mae"):
            values = [float(row[f"{metric}_generalization_gap"]) for row in group]
            item[f"{metric}_generalization_gap_mean"] = float(np.mean(values))
        aggregate.append(item)
    fields = list(aggregate[0]) if aggregate else ["transfer_id", "method_key", "evidence_level"]
    with (protocol_root / "protocol_v11_aggregate_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(aggregate)
    evidence = ["# Protocol V11 泛化方向性证据", "", "结论严格按预注册规则分为三类，不能将方向性证据表述为已经提升泛化能力。", ""]
    for spec in METHOD_SPECS:
        evidence.append(f"- **{spec['method_name']}**：{levels[spec['method_key']]}。")
    available_transfers = len({row["transfer_id"] for row in success})
    if available_transfers < 2:
        evidence.extend(["", "**协议未完整覆盖，不形成跨构造结论。**"])
    (protocol_root / "protocol_v11_generalization_evidence.md").write_text("\n".join(evidence) + "\n", encoding="utf-8")
    payload = {"run_count": len(run_rows), "success": len(success), "failed": sum(row.get("status") == "FAILED" for row in run_rows), "skipped": sum(str(row.get("status", "")).startswith("SKIPPED") for row in run_rows), "available_transfer_count": available_transfers, "evidence_levels": levels}
    (protocol_root / "protocol_v11_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", required=True); args = parser.parse_args(); print(json.dumps(summarize_protocol_v11(args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
