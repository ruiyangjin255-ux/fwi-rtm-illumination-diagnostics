"""Phase-3 postprocess with canonical metrics and co-sample paper figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .diagnostics import gradient_magnitude_np
from .phase3_utils import load_json, write_json


CANONICAL_METRICS = ("MAE", "RMSE", "SSIM", "edge_MAE", "gradient_l1_edge", "edge_F1")
METRIC_ALIASES = {
    "MAE": ("MAE", "mae"),
    "RMSE": ("RMSE", "rmse"),
    "SSIM": ("SSIM", "ssim"),
    "edge_MAE": ("edge_MAE", "edge_mae"),
    "gradient_l1_edge": ("gradient_l1_edge", "gradient_error"),
    "edge_F1": ("edge_F1",),
}
DATASET_LABELS = {
    "in_family": "FlatVel-A in-family",
    "cross_curvevel_a": "CurveVel-A target",
    "cross_flatfault_a": "FlatFault-A target",
}
VARIANT_LABELS = {"B1_raw_unet": "B1 raw U-Net", "PASD_Core_locked": "PASD-Core"}
BLUE = (67, 104, 150)
ORANGE = (217, 128, 42)
GRAY = (90, 96, 105)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ["arialbd.ttf", "Arial Bold.ttf"] if bold else ["arial.ttf", "Arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _num(row: dict[str, Any], metric: str) -> float | None:
    for key in METRIC_ALIASES.get(metric, (metric,)):
        value = row.get(key)
        if value not in (None, ""):
            try:
                return float(value)
            except ValueError:
                continue
    return None


def _save_png_pdf(image: Image.Image, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    image.save(base.with_suffix(".png"))
    image.convert("RGB").save(base.with_suffix(".pdf"))


def build_canonical_summary(formal: Path) -> list[dict[str, Any]]:
    run_rows = _read_csv(formal / "protocol_runs.csv")
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in run_rows:
        groups.setdefault((row["variant"], row["dataset"]), []).append(row)
    summary: list[dict[str, Any]] = []
    for (variant, dataset), rows in sorted(groups.items()):
        out: dict[str, Any] = {"variant": variant, "dataset": dataset, "n_seeds": len(rows)}
        for metric in CANONICAL_METRICS:
            values = [value for value in (_num(row, metric) for row in rows) if value is not None]
            if values:
                out[metric] = mean(values)
                out[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        summary.append(out)
    _write_csv(formal / "protocol_summary.csv", summary)
    _write_csv(formal / "tables" / "Table_1_protocol_summary.csv", summary)
    return summary


def _axis_bar(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], title: str, rows: list[dict[str, Any]], metric: str) -> None:
    x0, y0, x1, y1 = xy
    title_font = _font(18, True)
    small = _font(11)
    draw.text((x0, y0 - 32), title, fill=(20, 22, 25), font=title_font)
    plot_h = y1 - y0
    plot_w = x1 - x0
    values = [_num(row, metric) or 0.0 for row in rows]
    vmax = max(values + ([1.0] if metric == "SSIM" else []))
    if vmax <= 0:
        vmax = 1.0
    draw.line((x0, y0, x0, y1), fill=(35, 35, 35), width=2)
    draw.line((x0, y1, x1, y1), fill=(35, 35, 35), width=2)
    step = plot_w / max(1, len(rows))
    bar_w = max(18, int(step * 0.55))
    for i, row in enumerate(rows):
        value = values[i]
        bx0 = int(x0 + i * step + (step - bar_w) / 2)
        bx1 = bx0 + bar_w
        by1 = y1
        by0 = int(y1 - value / vmax * plot_h)
        color = BLUE if row["variant"] == "B1_raw_unet" else ORANGE
        draw.rectangle((bx0, by0, bx1, by1), fill=color)
        draw.text((bx0, max(y0, by0 - 17)), f"{value:.3g}", fill=(20, 22, 25), font=small)
        variant_label = VARIANT_LABELS.get(str(row["variant"]), str(row["variant"]).replace("_", " "))
        dataset_label = DATASET_LABELS.get(str(row["dataset"]), str(row["dataset"]).replace("_", " "))
        label = f"{variant_label}\n{dataset_label.replace(' target', '')}"
        draw.multiline_text((bx0 - 12, y1 + 8), label, fill=(30, 30, 30), font=small, spacing=2)


def _bar_figure(summary: list[dict[str, Any]], metric: str, base: Path, title: str) -> None:
    image = Image.new("RGB", (1500, 760), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 30), title, fill=(15, 18, 22), font=_font(26, True))
    draw.text((70, 64), "Mean across seeds; error bars are reported in the source table.", fill=GRAY, font=_font(14))
    ordered = [
        row
        for dataset in ("in_family", "cross_curvevel_a", "cross_flatfault_a")
        for variant in ("B1_raw_unet", "PASD_Core_locked")
        for row in summary
        if row["dataset"] == dataset and row["variant"] == variant
    ]
    _axis_bar(draw, (90, 140, 1430, 560), metric, ordered, metric)
    _legend(draw, 1140, 36)
    _save_png_pdf(image, base)


def _legend(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    font = _font(14)
    draw.rectangle((x, y, x + 18, y + 18), fill=BLUE)
    draw.text((x + 26, y), "B1 raw U-Net", fill=(20, 22, 25), font=font)
    draw.rectangle((x, y + 28, x + 18, y + 46), fill=ORANGE)
    draw.text((x + 26, y + 28), "PASD-Core", fill=(20, 22, 25), font=font)


def _improve_figure(summary: list[dict[str, Any]], base: Path) -> None:
    image = Image.new("RGB", (1200, 720), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 30), "PASD-Core improvement relative to B1", fill=(15, 18, 22), font=_font(26, True))
    rows: list[dict[str, Any]] = []
    for dataset in ("in_family", "cross_curvevel_a", "cross_flatfault_a"):
        b1 = next(row for row in summary if row["variant"] == "B1_raw_unet" and row["dataset"] == dataset)
        pasd = next(row for row in summary if row["variant"] == "PASD_Core_locked" and row["dataset"] == dataset)
        rows.append(
            {
                "variant": "PASD_Core_locked",
                "dataset": dataset,
                "MAE": 100.0 * ((_num(b1, "MAE") or 0.0) - (_num(pasd, "MAE") or 0.0)) / max(_num(b1, "MAE") or 1.0, 1e-9),
            }
        )
    _axis_bar(draw, (100, 135, 1080, 545), "MAE reduction (%)", rows, "MAE")
    _save_png_pdf(image, base)


def _bootstrap_figure(formal: Path, base: Path) -> None:
    rows = _read_csv(formal / "tables" / "Table_2_paired_bootstrap.csv")
    selected = [row for row in rows if row["metric"] == "mae"]
    image = Image.new("RGB", (1300, 740), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 30), "Paired bootstrap: PASD-Core minus B1 MAE", fill=(15, 18, 22), font=_font(25, True))
    draw.text((70, 62), "Negative values indicate lower error for PASD-Core.", fill=GRAY, font=_font(14))
    x0, y0, x1, y1 = 120, 135, 1180, 585
    values = [float(row["candidate_minus_baseline_mean"]) for row in selected]
    lows = [float(row["candidate_minus_baseline_ci95"].strip("[]").split(",")[0]) for row in selected]
    highs = [float(row["candidate_minus_baseline_ci95"].strip("[]").split(",")[1]) for row in selected]
    lo, hi = min(lows + values), max(highs + values)
    if lo == hi:
        hi = lo + 1.0
    zero_x = x0 + int((0 - lo) / (hi - lo) * (x1 - x0))
    draw.line((zero_x, y0, zero_x, y1), fill=(160, 160, 160), width=2)
    font = _font(12)
    for i, row in enumerate(selected):
        y = y0 + int((i + 0.5) * (y1 - y0) / len(selected))
        mean_x = x0 + int((float(row["candidate_minus_baseline_mean"]) - lo) / (hi - lo) * (x1 - x0))
        low_x = x0 + int((float(row["candidate_minus_baseline_ci95"].strip("[]").split(",")[0]) - lo) / (hi - lo) * (x1 - x0))
        high_x = x0 + int((float(row["candidate_minus_baseline_ci95"].strip("[]").split(",")[1]) - lo) / (hi - lo) * (x1 - x0))
        draw.line((low_x, y, high_x, y), fill=ORANGE, width=3)
        draw.ellipse((mean_x - 5, y - 5, mean_x + 5, y + 5), fill=ORANGE)
        draw.text((20, y - 8), f"{row['dataset']} s{row['seed']}", fill=(20, 22, 25), font=font)
    _save_png_pdf(image, base)


def _colorize(array: np.ndarray, palette: str = "velocity", vmin: float | None = None, vmax: float | None = None) -> Image.Image:
    a = np.asarray(array, dtype=np.float32)
    if vmin is None:
        vmin = float(np.nanmin(a))
    if vmax is None:
        vmax = float(np.nanmax(a))
    x = np.clip((a - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    if palette == "error":
        colors = np.stack([255 * x, 48 + 110 * x, 38 + 20 * (1 - x)], axis=-1)
    elif palette == "gradient":
        colors = np.stack([35 + 220 * x, 45 + 130 * x, 70 + 70 * (1 - x)], axis=-1)
    else:
        colors = np.stack([45 + 190 * x, 65 + 115 * x, 115 + 55 * (1 - x)], axis=-1)
    return Image.fromarray(colors.astype(np.uint8)).resize((170, 170), Image.Resampling.BILINEAR)


def _load_pair(formal: Path, dataset: str, seed: int = 0) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    archives = {}
    ids = None
    target = None
    for variant in ("B1_raw_unet", "PASD_Core_locked"):
        with np.load(formal / "prediction_archives" / f"{variant}_seed{seed}_{dataset}.npz") as payload:
            sample_ids = payload["sample_id"].astype(int)
            if ids is None:
                ids = sample_ids
                target = payload["target"]
            shared_id = int(ids[0])
            idx = int(np.where(sample_ids == shared_id)[0][0])
            archives[variant] = payload["prediction"][idx]
            if target is None:
                target = payload["target"][idx]
            else:
                target = payload["target"][0]
    return np.asarray(target[0] if target.ndim == 3 else target), archives


def _cosample_figure(formal: Path, dataset: str, base: Path) -> None:
    truth, predictions = _load_pair(formal, dataset, seed=0)
    values = [truth, *predictions.values()]
    vmin = min(float(v.min()) for v in values)
    vmax = max(float(v.max()) for v in values)
    emax = max(float(np.abs(pred - truth).max()) for pred in predictions.values())
    image = Image.new("RGB", (1250, 560), "white")
    draw = ImageDraw.Draw(image)
    draw.text((50, 28), f"Co-sample prediction plate: {DATASET_LABELS[dataset]}", fill=(15, 18, 22), font=_font(24, True))
    panels = [("Ground truth", truth, "velocity"), ("B1 raw U-Net", predictions["B1_raw_unet"], "velocity"), ("PASD-Core", predictions["PASD_Core_locked"], "velocity")]
    panels += [
        ("B1 |error|", np.abs(predictions["B1_raw_unet"] - truth), "error"),
        ("PASD |error|", np.abs(predictions["PASD_Core_locked"] - truth), "error"),
    ]
    x = 50
    for title, array, palette in panels:
        im = _colorize(array, palette, vmin if palette == "velocity" else 0.0, vmax if palette == "velocity" else emax)
        image.paste(im, (x, 135))
        draw.text((x, 108), title, fill=(20, 22, 25), font=_font(14, True))
        x += 230
    draw.text((50, 335), "Shared sample and shared color scale within velocity/error groups.", fill=GRAY, font=_font(13))
    _save_png_pdf(image, base)


def _gradient_figure(formal: Path, base: Path) -> None:
    image = Image.new("RGB", (1250, 620), "white")
    draw = ImageDraw.Draw(image)
    draw.text((50, 28), "Co-sample gradient evidence across targets", fill=(15, 18, 22), font=_font(24, True))
    y = 120
    for dataset in ("cross_curvevel_a", "cross_flatfault_a"):
        truth, predictions = _load_pair(formal, dataset, seed=0)
        maps = [
            ("Truth grad", gradient_magnitude_np(truth)),
            ("B1 grad err", np.abs(gradient_magnitude_np(predictions["B1_raw_unet"]) - gradient_magnitude_np(truth))),
            ("PASD grad err", np.abs(gradient_magnitude_np(predictions["PASD_Core_locked"]) - gradient_magnitude_np(truth))),
        ]
        draw.text((50, y - 28), DATASET_LABELS[dataset], fill=(20, 22, 25), font=_font(15, True))
        x = 220
        vmax = max(float(m.max()) for _, m in maps)
        for title, array in maps:
            im = _colorize(array, "gradient", 0.0, vmax)
            image.paste(im, (x, y))
            draw.text((x, y - 22), title, fill=(20, 22, 25), font=_font(12, True))
            x += 240
        y += 235
    _save_png_pdf(image, base)


def _source_selection_figure(root: Path, base: Path) -> None:
    rows = _read_csv(root / "source_aggregation_selection" / "source_aggregation_selection.csv")
    means = [row for row in rows if row.get("seed", "") and row.get("seed_std", "")]
    image = Image.new("RGB", (1100, 650), "white")
    draw = ImageDraw.Draw(image)
    draw.text((60, 30), "Source-val-only aggregation selection", fill=(15, 18, 22), font=_font(24, True))
    ordered = []
    for row in means:
        row = dict(row)
        row["variant"] = row["candidate"]
        row["dataset"] = "source_val"
        ordered.append(row)
    _axis_bar(draw, (100, 140, 980, 500), "Source validation MAE", ordered, "MAE")
    _save_png_pdf(image, base)


def _scatter_figure(formal: Path, base: Path) -> None:
    rows = _read_csv(formal / "diagnostics" / "phase3_corrected_sample_metrics.csv")
    xs = [float(row["MAE"]) for row in rows]
    ys = [float(row["gradient_l1_edge"]) for row in rows]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    image = Image.new("RGB", (1100, 720), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 28), "Metric consistency: MAE vs edge-gradient error", fill=(15, 18, 22), font=_font(24, True))
    x0, y0, x1, y1 = 110, 110, 1010, 610
    draw.line((x0, y0, x0, y1), fill=(35, 35, 35), width=2)
    draw.line((x0, y1, x1, y1), fill=(35, 35, 35), width=2)
    for row, x, y in zip(rows, xs, ys):
        px = x0 + int((x - xmin) / max(xmax - xmin, 1e-9) * (x1 - x0))
        py = y1 - int((y - ymin) / max(ymax - ymin, 1e-9) * (y1 - y0))
        color = BLUE if row["variant"] == "B1_raw_unet" else ORANGE
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)
    draw.text((520, 650), "MAE", fill=(20, 22, 25), font=_font(15))
    draw.text((15, 350), "gradient_l1_edge", fill=(20, 22, 25), font=_font(15))
    _legend(draw, 820, 35)
    _save_png_pdf(image, base)


def _write_report(root: Path, protocol: dict[str, Any], locked: dict[str, Any], summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# PASD-FWI Phase-3 Paper Report",
        "",
        "本报告按 Phase-3 locked protocol 重新训练 B1_raw_unet 与 PASD_Core_locked，并在 CurveVel-A 与 FlatFault-A 上做双目标评估。",
        "",
        f"- PASD-Core selected candidate: `{locked['selected_candidate']}`",
        f"- PASD-Core mapped variant: `{locked['selected_variant']}`",
        f"- Source family: `{protocol['source']['family']}`",
        f"- Targets: `{', '.join(protocol['targets'].keys())}`",
        "- Target role during model/aggregation selection: evaluation only",
        "",
        "## Summary",
        "",
        "| variant | dataset | MAE | RMSE | SSIM | edge_MAE | gradient_l1_edge | edge_F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['variant']} | {row['dataset']} | {_num(row, 'MAE'):.6g} | {_num(row, 'RMSE'):.6g} | "
            f"{_num(row, 'SSIM'):.6g} | {_num(row, 'edge_MAE'):.6g} | {_num(row, 'gradient_l1_edge'):.6g} | "
            f"{(_num(row, 'edge_F1') or 0.0):.6g} |"
        )
    lines.extend(
        [
            "",
            "## Figure Package",
            "",
            "Figure_1 为主性能柱状图；Figure_2 为 MAE 相对改进；Figure_3 为 paired bootstrap；Figure_6-8 为共样本预测板；Figure_9 为梯度共样本证据；Figure_10 为 source-val-only 聚合选择。",
        ]
    )
    text = "\n".join(lines)
    (root / "PASD_PHASE3_PAPER_REPORT.md").write_text(text, encoding="utf-8")
    (root / "dual_target_formal" / "PASD_PHASE3_PAPER_REPORT.md").write_text(text, encoding="utf-8")


def postprocess(root: Path, protocol_path: Path, config_path: Path) -> Path:
    formal = root / "dual_target_formal"
    summary = build_canonical_summary(formal)
    fig = formal / "figures"
    _bar_figure(summary, "MAE", fig / "Figure_1", "Phase-3 formal MAE across targets")
    _improve_figure(summary, fig / "Figure_2")
    _bootstrap_figure(formal, fig / "Figure_3")
    _bar_figure(summary, "RMSE", fig / "Figure_4", "Phase-3 formal RMSE across targets")
    _bar_figure(summary, "SSIM", fig / "Figure_5", "Phase-3 formal SSIM across targets")
    _cosample_figure(formal, "in_family", fig / "Figure_6")
    _cosample_figure(formal, "cross_curvevel_a", fig / "Figure_7")
    _cosample_figure(formal, "cross_flatfault_a", fig / "Figure_8")
    _gradient_figure(formal, fig / "Figure_9")
    _source_selection_figure(root, fig / "Figure_10")
    _scatter_figure(formal, fig / "Figure_A1")
    protocol = load_json(protocol_path)
    config = load_json(config_path)
    _write_report(root, protocol, config, summary)
    manifest = {
        "status": "SUCCESS",
        "root": str(root),
        "dual_target_formal": str(formal),
        "protocol": str(protocol_path),
        "pasd_core_config": str(config_path),
        "figures_png": len(list(fig.glob("*.png"))),
        "figures_pdf": len(list(fig.glob("*.pdf"))),
        "report": str(root / "PASD_PHASE3_PAPER_REPORT.md"),
        "metric_fix": "canonical summary coalesces legacy mae/rmse/ssim and Phase-3 corrected MAE/RMSE/SSIM columns before plotting.",
    }
    write_json(formal / "phase3_dual_target_manifest.json", manifest)
    write_json(root / "phase3_completion_manifest.json", manifest)
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--pasd-core-config", required=True, type=Path)
    args = parser.parse_args()
    root = postprocess(args.root, args.protocol, args.pasd_core_config)
    print(json.dumps({"status": "SUCCESS", "root": str(root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
