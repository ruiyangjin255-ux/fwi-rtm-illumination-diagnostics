"""Aggregation and report utilities for a fixed B1--B4 PASD protocol matrix."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from .plotting import plot_protocol_metric_summary


METRICS = ("mae", "rmse", "ssim", "edge_mae", "gradient_error")


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

    lines = ["# PASD-FWI Protocol Report", "", "## Aggregated metrics", "", "| Variant | Split | Metric | Seeds | Mean | Std |", "|---|---|---:|---:|---:|---:|"]
    for row in summary_rows:
        lines.append(
            f"| {row['variant']} | {row['split']} | {row['metric']} | {row['n_seeds']} | {row['mean']:.6g} | {row['std']:.6g} |"
        )
    lines += ["", "## Guardrails", "", "- Each run archives aligned `sample_id`, prediction, target, and attention arrays.", "- Target-family data are excluded from source scaler fitting and training.", "- Protocol plots use seed-level mean ± standard deviation; inspect per-sample archives before claiming superiority."]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return raw_path, summary_path, report_path
