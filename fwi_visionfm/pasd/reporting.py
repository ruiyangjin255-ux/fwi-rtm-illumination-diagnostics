"""Aggregation and report utilities for a fixed B1--B4 PASD protocol matrix."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from .plotting import plot_protocol_metric_summary


METRICS = ("mae", "rmse", "ssim", "psnr", "edge_mae", "gradient_error", "laplacian_error", "relative_error")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect_protocol_runs(root: str | Path) -> list[dict[str, object]]:
    root = Path(root)
    rows: list[dict[str, object]] = []
    for summary_path in sorted(root.glob("*/seed_*/metrics_summary.json")):
        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for split_key, split_name in (("in_family", "in_family"), ("cross_family", "cross_family")):
            values = payload.get("metrics", {}).get(split_key)
            if not values:
                continue
            for metric in METRICS:
                if metric in values:
                    rows.append(
                        {
                            "variant": payload["variant"],
                            "seed": int(payload["seed"]),
                            "split": split_name,
                            "metric": metric,
                            "value": float(values[metric]),
                            "source_family": payload.get("source_family"),
                            "target_family": payload.get("target_family"),
                            "run_dir": str(summary_path.parent),
                        }
                    )
    return rows


def summarize_runs(run_rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in run_rows:
        grouped[(str(row["variant"]), str(row["split"]), str(row["metric"]))].append(float(row["value"]))
    summary: list[dict[str, object]] = []
    for (variant, split, metric), values in sorted(grouped.items()):
        summary.append(
            {
                "variant": variant,
                "split": split,
                "metric": metric,
                "n_seeds": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )
    return summary


def write_protocol_report(root: str | Path) -> tuple[Path, Path, Path]:
    """Write raw metric rows, summary CSV, a concise markdown report, and publication metric plots."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    raw_rows = collect_protocol_runs(root)
    summary_rows = summarize_runs(raw_rows)
    raw_path, summary_path, report_path = root / "protocol_runs.csv", root / "protocol_summary.csv", root / "PROTOCOL_REPORT.md"
    _write_csv(raw_path, raw_rows)
    _write_csv(summary_path, summary_rows)
    figures = root / "figures"
    for split in ("in_family", "cross_family"):
        for metric in METRICS:
            plot_protocol_metric_summary(summary_rows, figures / f"{split}_{metric}.png", metric=metric, split=split)
    for metric in ("mae", "rmse", "ssim", "edge_mae", "gradient_error"):
        plot_protocol_metric_summary(summary_rows, figures / f"Figure_cross_family_summary_barplot_{metric}.png", metric=metric, split="cross_family")

    lines = [
        "# PASD Phase-1 Report",
        "",
        "## A. 数据与协议审计",
        "",
        "- Protocol manifest 固定 source train/val/in-family test 与 target cross-family test。",
        "- Velocity scaler 仅由 source train 拟合；target 不参与训练、验证、checkpoint selection 或超参数选择。",
        "",
        "## B. 训练配置",
        "",
        "- B1-B4 使用同一 protocol、epoch、batch size、learning rate 和 seed 集合。",
        "",
        "## C. B1-B4 参数量与运行时间",
        "",
        "- 详见 `variant_audit.json` 与各 run 的 `metrics_summary.json`。",
        "",
        "## D. In-family 结果",
        "",
        "| Variant | Metric | Seeds | Mean | Std |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        if row["split"] == "in_family":
            lines.append(f"| {row['variant']} | {row['metric']} | {row['n_seeds']} | {row['mean']:.6g} | {row['std']:.6g} |")
    lines += ["", "## E. Cross-family 结果", "", "| Variant | Metric | Seeds | Mean | Std |", "|---|---:|---:|---:|---:|"]
    for row in summary_rows:
        if row["split"] == "cross_family":
            lines.append(f"| {row['variant']} | {row['metric']} | {row['n_seeds']} | {row['mean']:.6g} | {row['std']:.6g} |")
    lines += [
        "",
        "## F. B4 vs B1 paired bootstrap",
        "",
        "- Bootstrap JSON 位于 `bootstrap/`；所有比较按相同 `sample_id` 对齐。",
        "",
        "## G. 固定样本成图解释",
        "",
        "- 固定样本由 B1 cross-family median MAE 自动选择，不手工挑样。",
        "",
        "## H. 失败/异常运行",
        "",
        "- 查看 `matrix_status.json` 与缺失 run 目录；本报告不隐藏失败 run。",
        "",
        "## I. 结论边界",
        "",
        "- Phase-1 仅覆盖 FlatVel-A -> CurveVel-A，不能外推为普遍 OOD 泛化结论。",
        "- Smoke 只验证代码路径，不作为科学结论。",
        "- 若 B4 仅改善 MAE/RMSE 而结构指标变差，只能称为 numerical gain。",
        "- 若 B4 同时改善数值与至少两项结构指标并在多数 seed 中成立，才可称为 preliminary cross-family structural gain。",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    phase1_path = root / "PASD_PHASE1_REPORT.md"
    phase1_path.write_text("\n".join(lines), encoding="utf-8")
    return raw_path, summary_path, report_path
