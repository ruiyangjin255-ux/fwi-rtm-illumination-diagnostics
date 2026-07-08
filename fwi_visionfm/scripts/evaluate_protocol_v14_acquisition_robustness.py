# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


METRIC_KEYS = {
    "mae": "mae",
    "rmse": "rmse",
    "ssim": "ssim",
    "gradient_error": "gradient_error",
    "edge_mae": "edge_mae",
}


def write_protocol_v14_robustness_outputs(*, root: str | Path, rows: list[dict[str, Any]], manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    out = Path(root)
    out.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "transfer_id",
        "seed",
        "method_id",
        "bridge_id",
        "perturbation",
        "metric_name",
        "metric_value",
        "degradation",
        "status",
        "note",
    ]
    with (out / "protocol_v14_robustness_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (out / "protocol_v14_robustness_degradation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Protocol V14 鲁棒性评估",
        "",
        "该部分为 evaluation-only，不新增训练，只比较 clean 与扰动条件下指标退化。",
        "",
        f"- 记录行数：{len(rows)}",
        f"- 扰动评估可用状态：{(manifest or {}).get('status', 'UNKNOWN')}",
        "",
        "CPU 小样本统一协议；结果用于检验方向性证据，不构成标准基准级结论。",
    ]
    (out / "robustness_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "perturbation_manifest.json").write_text(json.dumps(manifest or {}, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"row_count": len(rows), "report_path": str(out / "robustness_report.md")}


def evaluate_protocol_v14_acquisition_robustness(
    *,
    root: str | Path,
    methods: list[str],
    bridges: list[str],
    perturbations: list[str],
    device: str,
) -> dict[str, Any]:
    protocol_root = Path(root)
    out = protocol_root / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    example_dir = out / "robustness_prediction_examples"
    example_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    selected = []
    for config_path in protocol_root.glob("runs/*/*/seed_*/B*/config.json"):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("status") != "SUCCESS":
            continue
        if str(config.get("method_id")) not in methods or str(config.get("bridge_id")) not in bridges:
            continue
        metrics_path = config_path.parent / "metrics_cross_family_test.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        selected.append((config, config_path.parent, metrics))
        for metric_name, metric_key in METRIC_KEYS.items():
            rows.append(
                {
                    "transfer_id": config["transfer_id"],
                    "seed": int(config["seed"]),
                    "method_id": config["method_id"],
                    "bridge_id": config["bridge_id"],
                    "perturbation": "clean",
                    "metric_name": metric_name,
                    "metric_value": metrics.get(metric_key, ""),
                    "degradation": 0.0,
                    "status": "AVAILABLE_CLEAN_ONLY",
                    "note": "clean cross-family metrics reused from existing runs",
                }
            )

    for config, run_dir, _ in selected[:8]:
        src = run_dir / "prediction_grid.png"
        if src.exists():
            target = example_dir / f"{config['transfer_id']}__{config['method_id']}__{config['bridge_id']}__seed{config['seed']}.png"
            shutil.copyfile(src, target)

    checkpoint_available = any((run_dir / "checkpoint.pt").exists() for _, run_dir, _ in selected)
    status = "UNAVAILABLE_NO_CHECKPOINT" if not checkpoint_available else "AVAILABLE"
    for perturbation in perturbations:
        if perturbation == "clean":
            continue
        for config, _, _ in selected:
            for metric_name in METRIC_KEYS:
                rows.append(
                    {
                        "transfer_id": config["transfer_id"],
                        "seed": int(config["seed"]),
                        "method_id": config["method_id"],
                        "bridge_id": config["bridge_id"],
                        "perturbation": perturbation,
                        "metric_name": metric_name,
                        "metric_value": "",
                        "degradation": "",
                        "status": status,
                        "note": "现有 V14 输出目录未保存可复评 checkpoint，无法对扰动输入做真实 evaluation-only 重评",
                    }
                )

    manifest = {
        "status": status,
        "device": device,
        "methods": methods,
        "bridges": bridges,
        "perturbations": perturbations,
        "selected_run_count": len(selected),
        "checkpoint_available": checkpoint_available,
    }
    result = write_protocol_v14_robustness_outputs(root=out, rows=rows, manifest=manifest)
    result["status"] = status
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--bridges", nargs="+", required=True)
    parser.add_argument("--perturbations", nargs="+", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(
        evaluate_protocol_v14_acquisition_robustness(
            root=args.root,
            methods=args.methods,
            bridges=args.bridges,
            perturbations=args.perturbations,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()
