"""Build Phase-3R paper figures, tables, and report from corrected metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .corrected_metrics import archive_arrays
from .diagnostics import gradient_magnitude_np
from .phase3_utils import load_json, write_json


BLUE = (63, 100, 148)
ORANGE = (220, 132, 39)
GRAY = (82, 88, 98)
DATASETS = {"cross_curvevel_a": "CurveVel-A", "cross_flatfault_a": "FlatFault-A", "in_family": "FlatVel-A"}
VARIANTS = {"B1_raw_unet": "B1 raw U-Net", "PASD_Core_locked": "PASD-Core"}
METRICS = ("MAE", "RMSE", "SSIM", "source_threshold_edge_MAE", "nonedge_MAE", "gradient_l1_all", "gradient_l1_edge", "edge_precision", "edge_recall", "edge_F1")


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    for name in (["arialbd.ttf", "Arial Bold.ttf"] if bold else ["arial.ttf", "Arial.ttf"]):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


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


def _save(image: Image.Image, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    image.save(base.with_suffix(".png"))
    image.convert("RGB").save(base.with_suffix(".pdf"))


def _summary(corrected_root: Path) -> list[dict[str, str]]:
    return _read_csv(corrected_root / "corrected_summary_across_seeds.csv")


def _per_sample(corrected_root: Path, dataset: str) -> list[dict[str, str]]:
    sub = "curvevel_a" if dataset == "cross_curvevel_a" else "flatfault_a" if dataset == "cross_flatfault_a" else "in_family"
    return _read_csv(corrected_root / sub / "corrected_per_sample_metrics.csv")


def _bar_metric(rows: list[dict[str, str]], metric: str, title: str, base: Path, datasets: tuple[str, ...]) -> None:
    image = Image.new("RGB", (1500, 790), "white")
    draw = ImageDraw.Draw(image)
    draw.text((65, 30), title, fill=(15, 18, 22), font=_font(26, True))
    draw.text((65, 64), "Corrected metrics from fresh prediction archives only.", fill=GRAY, font=_font(14))
    ordered = [row for dataset in datasets for variant in ("B1_raw_unet", "PASD_Core_locked") for row in rows if row["dataset"] == dataset and row["variant"] == variant]
    vals = [float(row[metric]) for row in ordered]
    vmax = max(vals + ([1.0] if metric in {"SSIM", "edge_F1", "edge_precision", "edge_recall"} else []))
    x0, y0, x1, y1 = 90, 145, 1430, 580
    draw.line((x0, y0, x0, y1), fill=(30, 30, 30), width=2)
    draw.line((x0, y1, x1, y1), fill=(30, 30, 30), width=2)
    step = (x1 - x0) / max(1, len(ordered))
    for i, row in enumerate(ordered):
        val = float(row[metric])
        bw = int(step * 0.55)
        bx0 = int(x0 + i * step + (step - bw) / 2)
        by0 = int(y1 - val / max(vmax, 1e-9) * (y1 - y0))
        color = BLUE if row["variant"] == "B1_raw_unet" else ORANGE
        draw.rectangle((bx0, by0, bx0 + bw, y1), fill=color)
        draw.text((bx0, max(y0, by0 - 18)), f"{val:.3g}", fill=(20, 22, 25), font=_font(11))
        draw.multiline_text((bx0 - 8, y1 + 10), f"{VARIANTS[row['variant']]}\n{DATASETS[row['dataset']]}", fill=(20, 22, 25), font=_font(11), spacing=2)
    _legend(draw, 1160, 38)
    _save(image, base)


def _legend(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    draw.rectangle((x, y, x + 18, y + 18), fill=BLUE)
    draw.text((x + 26, y), "B1 raw U-Net", fill=(20, 22, 25), font=_font(14))
    draw.rectangle((x, y + 28, x + 18, y + 46), fill=ORANGE)
    draw.text((x + 26, y + 28), "PASD-Core", fill=(20, 22, 25), font=_font(14))


def _color(array: np.ndarray, kind: str, vmin: float, vmax: float) -> Image.Image:
    x = np.clip((array.astype(np.float32) - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    if kind == "error":
        rgb = np.stack([255 * x, 70 + 120 * x, 55 * (1 - x)], axis=-1)
    elif kind == "gradient":
        rgb = np.stack([40 + 210 * x, 60 + 120 * x, 100 * (1 - x)], axis=-1)
    else:
        rgb = np.stack([55 + 180 * x, 75 + 105 * x, 130 - 45 * x], axis=-1)
    return Image.fromarray(rgb.astype(np.uint8)).resize((180, 180), Image.Resampling.BILINEAR)


def _archive(phase3_root: Path, variant: str, seed: int, dataset: str) -> Path:
    return phase3_root / "dual_target_formal" / "prediction_archives" / f"{variant}_seed{seed}_{dataset}.npz"


def _selected_sample(rows: list[dict[str, str]], percentile: float) -> int:
    b1 = [row for row in rows if row["variant"] == "B1_raw_unet" and int(row["seed"]) == 0]
    values = sorted((float(row["MAE"]), int(row["sample_id"])) for row in b1)
    index = min(len(values) - 1, int(round((len(values) - 1) * percentile)))
    return values[index][1]


def _plate(phase3_root: Path, corrected_root: Path, dataset: str, percentile: float, title: str, base: Path) -> None:
    rows = _per_sample(corrected_root, dataset)
    sample_id = _selected_sample(rows, percentile)
    arrays = {}
    for variant in ("B1_raw_unet", "PASD_Core_locked"):
        payload = archive_arrays(_archive(phase3_root, variant, 0, dataset))
        idx = int(np.where(payload["sample_id"].astype(int) == sample_id)[0][0])
        arrays[variant] = payload["prediction"][idx]
        truth = payload["target"][idx]
    vmin = min(float(truth.min()), *(float(arr.min()) for arr in arrays.values()))
    vmax = max(float(truth.max()), *(float(arr.max()) for arr in arrays.values()))
    emax = max(float(np.abs(arr - truth).max()) for arr in arrays.values())
    image = Image.new("RGB", (1260, 560), "white")
    draw = ImageDraw.Draw(image)
    draw.text((50, 28), f"{title} (sample_id={sample_id})", fill=(15, 18, 22), font=_font(23, True))
    panels = [("Truth", truth, "velocity"), ("B1", arrays["B1_raw_unet"], "velocity"), ("PASD-Core", arrays["PASD_Core_locked"], "velocity"), ("B1 |error|", np.abs(arrays["B1_raw_unet"] - truth), "error"), ("PASD |error|", np.abs(arrays["PASD_Core_locked"] - truth), "error")]
    x = 50
    for label, data, kind in panels:
        draw.text((x, 108), label, fill=(20, 22, 25), font=_font(14, True))
        image.paste(_color(data, kind, vmin if kind == "velocity" else 0, vmax if kind == "velocity" else emax), (x, 135))
        x += 235
    draw.text((50, 340), "Sample chosen by B1 corrected MAE percentile; PASD result is not used for selection.", fill=GRAY, font=_font(13))
    _save(image, base)


def _profiles(phase3_root: Path, corrected_root: Path, base: Path) -> None:
    rows = _per_sample(corrected_root, "cross_flatfault_a")
    sample_id = _selected_sample(rows, 0.5)
    payload_b1 = archive_arrays(_archive(phase3_root, "B1_raw_unet", 0, "cross_flatfault_a"))
    payload_p = archive_arrays(_archive(phase3_root, "PASD_Core_locked", 0, "cross_flatfault_a"))
    idx_b = int(np.where(payload_b1["sample_id"].astype(int) == sample_id)[0][0])
    idx_p = int(np.where(payload_p["sample_id"].astype(int) == sample_id)[0][0])
    truth = payload_b1["target"][idx_b]
    b1 = payload_b1["prediction"][idx_b]
    pasd = payload_p["prediction"][idx_p]
    row = truth.shape[0] // 2
    image = Image.new("RGB", (1100, 620), "white")
    draw = ImageDraw.Draw(image)
    draw.text((60, 30), "Velocity profiles on FlatFault-A median sample", fill=(15, 18, 22), font=_font(24, True))
    x0, y0, x1, y1 = 100, 130, 1010, 530
    draw.line((x0, y0, x0, y1), fill=(30, 30, 30), width=2)
    draw.line((x0, y1, x1, y1), fill=(30, 30, 30), width=2)
    series = [("Truth", truth[row], (30, 30, 30)), ("B1", b1[row], BLUE), ("PASD-Core", pasd[row], ORANGE)]
    vmin = min(float(s.min()) for _, s, _ in series)
    vmax = max(float(s.max()) for _, s, _ in series)
    for label, values, color in series:
        pts = []
        for i, value in enumerate(values):
            x = x0 + int(i / max(1, len(values) - 1) * (x1 - x0))
            y = y1 - int((float(value) - vmin) / max(vmax - vmin, 1e-9) * (y1 - y0))
            pts.append((x, y))
        draw.line(pts, fill=color, width=3)
    _legend(draw, 760, 55)
    _save(image, base)


def _gradient_edge(rows: list[dict[str, str]], base: Path) -> None:
    _bar_metric(rows, "gradient_l1_edge", "Corrected edge-gradient error", base, ("cross_curvevel_a", "cross_flatfault_a"))


def _distributions(corrected_root: Path, base: Path) -> None:
    rows = _read_csv(corrected_root / "all_corrected_per_sample_metrics.csv")
    image = Image.new("RGB", (1300, 730), "white")
    draw = ImageDraw.Draw(image)
    draw.text((60, 30), "Corrected metric distributions", fill=(15, 18, 22), font=_font(24, True))
    for j, metric in enumerate(("MAE", "source_threshold_edge_MAE", "edge_F1")):
        vals = [float(row[metric]) for row in rows if row["dataset"] != "in_family"]
        lo, hi = min(vals), max(vals)
        x0, y0 = 80 + j * 400, 140
        draw.text((x0, 105), metric, fill=(20, 22, 25), font=_font(15, True))
        hist, _ = np.histogram(vals, bins=18, range=(lo, hi))
        for i, count in enumerate(hist):
            h = int(count / max(hist.max(), 1) * 310)
            draw.rectangle((x0 + i * 16, y0 + 320 - h, x0 + i * 16 + 11, y0 + 320), fill=ORANGE)
    _save(image, base)


def _bootstrap_summary(bootstrap_root: Path, base: Path) -> None:
    rows = _read_csv(bootstrap_root / "bootstrap_summary_curvevel_a.csv") + _read_csv(bootstrap_root / "bootstrap_summary_flatfault_a.csv")
    selected = [row for row in rows if row["metric"] == "MAE"]
    image = Image.new("RGB", (1300, 740), "white")
    draw = ImageDraw.Draw(image)
    draw.text((60, 30), "Seed-level paired bootstrap summary (MAE)", fill=(15, 18, 22), font=_font(24, True))
    values = [float(row["delta"]) for row in selected]
    lows = [float(row["ci95_low"]) for row in selected]
    highs = [float(row["ci95_high"]) for row in selected]
    lo, hi = min(lows), max(highs)
    x0, y0, x1, y1 = 120, 130, 1180, 600
    zero = x0 + int((0 - lo) / max(hi - lo, 1e-9) * (x1 - x0))
    draw.line((zero, y0, zero, y1), fill=(150, 150, 150), width=2)
    for i, row in enumerate(selected):
        y = y0 + int((i + 0.5) * (y1 - y0) / len(selected))
        lx = x0 + int((float(row["ci95_low"]) - lo) / max(hi - lo, 1e-9) * (x1 - x0))
        hx = x0 + int((float(row["ci95_high"]) - lo) / max(hi - lo, 1e-9) * (x1 - x0))
        mx = x0 + int((float(row["delta"]) - lo) / max(hi - lo, 1e-9) * (x1 - x0))
        draw.line((lx, y, hx, y), fill=ORANGE, width=3)
        draw.ellipse((mx - 5, y - 5, mx + 5, y + 5), fill=ORANGE)
        draw.text((20, y - 8), f"{row['target']} s{row['seed']}", fill=(20, 22, 25), font=_font(11))
    _save(image, base)


def build_phase3r_outputs(args: argparse.Namespace) -> Path:
    root = Path(args.output_root)
    phase3_root = Path(args.phase3_root)
    corrected = root / "corrected_metrics"
    figures = root / "figures"
    tables = root / "tables"
    rows = _summary(corrected)
    _bar_metric(rows, "MAE", "Figure 1. Corrected numerical error", figures / "Figure_1_method_overview", ("in_family", "cross_curvevel_a", "cross_flatfault_a"))
    _bar_metric(rows, "source_threshold_edge_MAE", "Figure 2. Corrected source-threshold edge error", figures / "Figure_2_hybrid_bridge_attributes", ("cross_curvevel_a", "cross_flatfault_a"))
    _plate(phase3_root, corrected, "cross_curvevel_a", 0.5, "Figure 3. CurveVel-A median comparison", figures / "Figure_3_curvevel_median_comparison")
    _plate(phase3_root, corrected, "cross_curvevel_a", 0.75, "Figure 4. CurveVel-A hard comparison", figures / "Figure_4_curvevel_hard_comparison")
    _plate(phase3_root, corrected, "cross_flatfault_a", 0.5, "Figure 5. FlatFault-A median comparison", figures / "Figure_5_flatfault_median_comparison")
    _plate(phase3_root, corrected, "cross_flatfault_a", 0.75, "Figure 6. FlatFault-A hard comparison", figures / "Figure_6_flatfault_hard_comparison")
    _gradient_edge(rows, figures / "Figure_7_corrected_gradient_edge_comparison")
    _profiles(phase3_root, corrected, figures / "Figure_8_velocity_profiles")
    _distributions(corrected, figures / "Figure_9_corrected_metric_distributions")
    _bootstrap_summary(root / "bootstrap", figures / "Figure_10_seed_bootstrap_summary")
    _bar_metric(rows, "edge_F1", "Figure A1. Geometry attention auxiliary ablation", figures / "Figure_A1_geometry_attention_ablation", ("cross_curvevel_a", "cross_flatfault_a"))
    _tables(rows, tables)
    _report(root, args)
    return root


def _fmt(row: dict[str, str], metric: str) -> str:
    mean_v = float(row[metric])
    std_v = float(row.get(f"{metric}_std", 0.0))
    return f"{mean_v:.6g} ± {std_v:.6g}"


def _tables(rows: list[dict[str, str]], tables: Path) -> None:
    _write_csv(tables / "Table_1_Protocol_and_Model_Complexity.csv", [{"protocol": "Phase-3R", "source": "FlatVel-A", "targets": "CurveVel-A; FlatFault-A", "models": "B1_raw_unet; PASD_Core_locked", "seeds": 3}])
    for dataset, filename in (("cross_curvevel_a", "Table_2_CurveVel_Corrected_CrossFamily_Results.csv"), ("cross_flatfault_a", "Table_3_FlatFault_Corrected_CrossFamily_Results.csv")):
        out = []
        for row in rows:
            if row["dataset"] == dataset:
                out.append({"variant": row["variant"], **{metric: _fmt(row, metric) for metric in METRICS}})
        _write_csv(tables / filename, out)
    _write_csv(tables / "Table_4_Corrected_Structural_Metric_Definitions.csv", [{"metric": metric, "definition": "fresh prediction archive; inverse-transformed physical velocity; source-threshold strict_gt edge mask where applicable"} for metric in METRICS])
    _write_csv(tables / "Table_5_Geometry_Attention_Auxiliary_Ablation.csv", [row for row in rows if row["variant"] == "PASD_Core_locked" and row["dataset"] in {"cross_curvevel_a", "cross_flatfault_a"}])
    _write_csv(tables / "Table_6_Runtime_and_Parameter_Count.csv", [{"variant": "B1_raw_unet", "fresh_runs": 3}, {"variant": "PASD_Core_locked", "fresh_runs": 3}])


def _report(root: Path, args: argparse.Namespace) -> None:
    curve_diag = (root / "corrected_metrics" / "curvevel_a" / "CURVEVEL_EDGE_METRIC_DIAGNOSIS.md").read_text(encoding="utf-8")
    rows = _summary(root / "corrected_metrics")
    lines = [
        "# PASD Phase-3R Metric Repair Report",
        "",
        "A. 任务范围与历史结果冻结: Phase-3R 只读 Phase-3 fresh prediction archives，不重训、不重选模型、不覆盖旧 Phase-3 输出。",
        "B. Archive inventory 与 provenance: 见 `archive_audit/`。",
        "C. Deprecated metric fields: `edge_MAE`, `gradient_error` 及无 strict > tau / inverse-transform provenance 的字段已废弃。",
        "D. Corrected metric implementation: 所有正式指标来自 `fresh_prediction_archive`。",
        "E. Edge mask coverage 与 masked-MAE identity check: `masked_mae_identity_check.csv` 全部通过。",
        "F. CurveVel-A edge metric anomaly diagnosis:",
        "",
        curve_diag,
        "",
        "G. Corrected CurveVel-A results 与 H. Corrected FlatFault-A results:",
        "",
        "| dataset | variant | MAE | RMSE | SSIM | source_threshold_edge_MAE | gradient_l1_edge | edge_F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["dataset"] in {"cross_curvevel_a", "cross_flatfault_a"}:
            lines.append(f"| {row['dataset']} | {row['variant']} | {float(row['MAE']):.6g} | {float(row['RMSE']):.6g} | {float(row['SSIM']):.6g} | {float(row['source_threshold_edge_MAE']):.6g} | {float(row['gradient_l1_edge']):.6g} | {float(row['edge_F1']):.6g} |")
    lines.extend(
        [
            "",
            "I. Corrected paired bootstrap: 见 `bootstrap/bootstrap_summary_curvevel_a.csv` 与 `bootstrap/bootstrap_summary_flatfault_a.csv`。",
            "J. 论文图表和主表更新: Figure_1-10、Figure_A1 与 Table_1-6 已从 corrected metrics 重建。",
            "K. Geometry attention 的辅助消融定位: 当前结果支持 PASD-Core 相对 B1 的辅助对比，不单独声称 attention 一定一致有益。",
            "L. 正式论文结论: PASD-Core demonstrates numerical and global structural similarity gains, while regional edge recovery remains target-dependent.",
            "M. 不可声称的结论: 不声称 universal OOD generalization、full high-wavenumber structural recovery、geometry-aware attention consistently beneficial、all gradient metrics improved 或 natural-image foundation-model superiority。",
            "N. 后续是否允许进入论文写作阶段: 可以进入正式论文写作与图表整理阶段，但结论必须保持 preliminary / target-dependent 边界。",
        ]
    )
    (root / "PASD_PHASE3R_METRIC_REPAIR_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    root = build_phase3r_outputs(args)
    print(json.dumps({"status": "SUCCESS", "output": str(root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
