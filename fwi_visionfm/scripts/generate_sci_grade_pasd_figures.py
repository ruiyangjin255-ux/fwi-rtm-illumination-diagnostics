from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.7
plt.rcParams["legend.frameon"] = False


ROOT = Path(__file__).resolve().parents[1]
PHASE3 = ROOT / "outputs" / "pasd_phase3_paper" / "dual_target_formal"
PHASE3R = ROOT / "outputs" / "pasd_phase3r_metric_repair"
FREEZE = ROOT / "outputs" / "pasd_phase4_paper_freeze"
OUT = FREEZE / "paper_figures_sci"
SOURCE_DATA = OUT / "source_data"

BLUE = "#4B5F97"
ORANGE = "#D9892B"
GREEN = "#2E9E44"
RED = "#B64342"
NEUTRAL = "#606060"
LIGHT = "#EAEAF2"

DATASET_LABEL = {
    "in_family": "FlatVel-A\nin-family",
    "cross_curvevel_a": "CurveVel-A\ncross-family",
    "cross_flatfault_a": "FlatFault-A\ncross-family",
}

DATASET_SHORT = {
    "in_family": "FlatVel-A",
    "cross_curvevel_a": "CurveVel-A",
    "cross_flatfault_a": "FlatFault-A",
}

METRIC_LABEL = {
    "MAE": "MAE",
    "RMSE": "RMSE",
    "SSIM": "SSIM",
    "source_threshold_edge_MAE": "Edge MAE",
    "gradient_l1_edge": "Edge gradient",
    "edge_F1": "Edge F1",
}

LOWER_BETTER = {"MAE", "RMSE", "source_threshold_edge_MAE", "gradient_l1_edge"}
HIGHER_BETTER = {"SSIM", "edge_F1"}


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE_DATA.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, stem: str) -> None:
    for ext in ("svg", "pdf", "tiff", "png"):
        kwargs = {"bbox_inches": "tight"}
        if ext in {"tiff", "png"}:
            kwargs["dpi"] = 600
        fig.savefig(OUT / f"{stem}.{ext}", **kwargs)
    plt.close(fig)


def add_panel(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def clean_axes(ax: plt.Axes) -> None:
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(width=0.7, length=2.5, color="#444444")


def load_summary() -> pd.DataFrame:
    return pd.read_csv(PHASE3R / "corrected_metrics" / "corrected_summary_across_seeds.csv")


def load_bootstrap() -> pd.DataFrame:
    parts = [
        pd.read_csv(PHASE3R / "bootstrap" / "bootstrap_summary_curvevel_a.csv"),
        pd.read_csv(PHASE3R / "bootstrap" / "bootstrap_summary_flatfault_a.csv"),
    ]
    return pd.concat(parts, ignore_index=True)


def wide_summary_for_metric(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for dataset in ["in_family", "cross_curvevel_a", "cross_flatfault_a"]:
        sub = summary[summary["dataset"] == dataset]
        b1 = float(sub[sub["variant"] == "B1_raw_unet"][metric].iloc[0])
        pasd = float(sub[sub["variant"] == "PASD_Core_locked"][metric].iloc[0])
        rows.append({"dataset": dataset, "B1_raw_unet": b1, "PASD_Core": pasd})
    return pd.DataFrame(rows)


def directional_gain(b1: float, pasd: float, metric: str) -> float:
    if metric in LOWER_BETTER:
        return (b1 - pasd) / abs(b1) * 100.0
    if metric in HIGHER_BETTER:
        return (pasd - b1) / max(abs(b1), 1e-9) * 100.0
    raise ValueError(metric)


def figure_sci_1(summary: pd.DataFrame, boot: pd.DataFrame) -> None:
    """Asymmetric evidence summary: one claim, three evidence layers."""
    metrics = ["MAE", "RMSE", "SSIM", "source_threshold_edge_MAE", "edge_F1"]
    targets = ["cross_curvevel_a", "cross_flatfault_a"]

    heat = np.zeros((len(metrics), len(targets)))
    for i, metric in enumerate(metrics):
        for j, target in enumerate(targets):
            sub = summary[summary["dataset"] == target]
            b1 = float(sub[sub["variant"] == "B1_raw_unet"][metric].iloc[0])
            pasd = float(sub[sub["variant"] == "PASD_Core_locked"][metric].iloc[0])
            heat[i, j] = directional_gain(b1, pasd, metric)

    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(heat, index=[METRIC_LABEL[m] for m in metrics], columns=[DATASET_LABEL[t].replace("\n", " ") for t in targets]).to_csv(
        SOURCE_DATA / "Figure_SCI_1_directional_gain_heatmap.csv"
    )

    fig = plt.figure(figsize=(7.2, 6.05))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.26, 1.15, 1.62], height_ratios=[1.0, 1.08], wspace=0.78, hspace=0.48)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1:])
    ax_c = fig.add_subplot(gs[1, 1:])

    # Panel a: compact percent-reduction bars for the two primary lower-is-better errors.
    bar_rows = []
    for target in ["in_family", *targets]:
        for metric in ["MAE", "source_threshold_edge_MAE"]:
            sub = summary[summary["dataset"] == target]
            b1 = float(sub[sub["variant"] == "B1_raw_unet"][metric].iloc[0])
            pasd = float(sub[sub["variant"] == "PASD_Core_locked"][metric].iloc[0])
            bar_rows.append(
                {
                    "label": f"{DATASET_SHORT[target]}\n{METRIC_LABEL[metric]}",
                    "gain": directional_gain(b1, pasd, metric),
                    "metric": metric,
                }
            )
    y_positions = np.arange(len(bar_rows))[::-1]
    gains = [row["gain"] for row in bar_rows]
    colors = [BLUE if row["metric"] == "MAE" else ORANGE for row in bar_rows]
    ax_a.barh(y_positions, gains, color=colors, height=0.56, alpha=0.92)
    for y, gain in zip(y_positions, gains):
        ax_a.text(gain + 1.2, y, f"{gain:.1f}%", ha="left", va="center", fontsize=5.9, color="#222222")
    ax_a.set_xlim(0, max(gains) + 10)
    ax_a.set_ylim(-0.6, len(bar_rows) - 0.4)
    ax_a.set_xlabel("Reduction vs. B1 (%)", fontsize=6.5)
    ax_a.set_yticks(y_positions)
    ax_a.set_yticklabels([row["label"] for row in bar_rows], fontsize=5.7)
    ax_a.set_title("Primary error reductions", loc="left", fontsize=8, pad=6)
    ax_a.grid(axis="x", color="#E0E0E0", lw=0.5)
    clean_axes(ax_a)
    ax_a.tick_params(axis="y", length=0)
    add_panel(ax_a, "a")

    # Panel b: directional gain heatmap.
    norm = TwoSlopeNorm(vmin=-10, vcenter=0, vmax=max(90, np.nanmax(heat)))
    im = ax_b.imshow(heat, cmap="Greens", aspect="auto", vmin=0, vmax=max(90, np.nanmax(heat)))
    ax_b.set_xticks(range(len(targets)))
    ax_b.set_xticklabels(["CurveVel-A", "FlatFault-A"])
    ax_b.set_yticks(range(len(metrics)))
    ax_b.set_yticklabels([METRIC_LABEL[m] for m in metrics])
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            txt_color = "white" if heat[i, j] > 85 else "#1E3B20"
            ax_b.text(j, i, f"{heat[i, j]:.1f}%", ha="center", va="center", fontsize=7, color=txt_color)
    ax_b.set_title("Directional improvement over B1", loc="left", fontsize=8, pad=6)
    for s in ax_b.spines.values():
        s.set_visible(False)
    cbar = fig.colorbar(im, ax=ax_b, fraction=0.035, pad=0.02)
    cbar.set_label("Improvement (%)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    add_panel(ax_b, "b", x=-0.12)

    # Panel c: paired bootstrap forest plot for key cross-family metrics.
    focus = boot[boot["metric"].isin(["MAE", "source_threshold_edge_MAE", "edge_F1"])].copy()
    row_specs = []
    for target in targets:
        for metric in ["MAE", "source_threshold_edge_MAE", "edge_F1"]:
            sub = focus[(focus["target"] == target) & (focus["metric"] == metric)]
            row_specs.append(
                {
                    "target": target,
                    "metric": metric,
                    "label": f"{'Curve' if target == 'cross_curvevel_a' else 'Fault'} | {METRIC_LABEL[metric]}",
                    "lo": sub["ci95_low"].min(),
                    "hi": sub["ci95_high"].max(),
                    "mean": sub["delta"].mean(),
                    "seed_values": sub.sort_values("seed")["delta"].to_numpy(),
                }
            )
    y = np.arange(len(row_specs))[::-1]
    for yi, row in zip(y, row_specs):
        color = GREEN if row["metric"] == "edge_F1" else RED
        ax_c.plot([row["lo"], row["hi"]], [yi, yi], color=color, lw=1.4, alpha=0.45)
        jitter = np.linspace(-0.12, 0.12, len(row["seed_values"]))
        ax_c.scatter(row["seed_values"], yi + jitter, s=14, color=color, edgecolor="white", linewidth=0.4, zorder=3)
        ax_c.plot(row["mean"], yi, marker="D", ms=4, color="black", zorder=4)
    ax_c.axvline(0, color=NEUTRAL, ls="--", lw=0.8)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([r["label"] for r in row_specs], fontsize=6.3)
    ax_c.set_xlabel("PASD-Core minus B1 (seed points; diamond = seed mean)")
    ax_c.set_title("Bootstrap direction is consistent but target-dependent", loc="left", fontsize=8, pad=6)
    clean_axes(ax_c)
    add_panel(ax_c, "c", x=-0.12)

    fig.suptitle("PASD-Core improves cross-family error while preserving target-specific uncertainty", x=0.01, y=0.995, ha="left", fontsize=10)
    save(fig, "Figure_SCI_1_evidence_summary")


def archive_path(variant: str, seed: int, dataset: str) -> Path:
    return PHASE3 / "prediction_archives" / f"{variant}_seed{seed}_{dataset}.npz"


def select_sample(per_sample: pd.DataFrame, dataset: str, seed: int, percentile: float) -> int:
    sub = per_sample[
        (per_sample["dataset"] == dataset)
        & (per_sample["seed"] == seed)
        & (per_sample["variant"] == "B1_raw_unet")
    ].copy()
    target_value = np.percentile(sub["MAE"].to_numpy(), percentile)
    row = sub.iloc[(sub["MAE"] - target_value).abs().argsort().iloc[0]]
    return int(row["sample_id"])


def get_sample_arrays(dataset: str, seed: int, sample_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b1 = np.load(archive_path("B1_raw_unet", seed, dataset))
    pasd = np.load(archive_path("PASD_Core_locked", seed, dataset))
    ids = b1["sample_id"]
    idx = int(np.where(ids == sample_id)[0][0])
    return b1["target"][idx], b1["prediction"][idx], pasd["prediction"][idx]


def figure_sci_2() -> None:
    """Image plate with shared colorbars and explicit error maps."""
    per_sample = pd.read_csv(PHASE3R / "corrected_metrics" / "all_corrected_per_sample_metrics.csv")
    rowspec = [
        ("cross_curvevel_a", 0, 50, "CurveVel-A median"),
        ("cross_curvevel_a", 0, 75, "CurveVel-A hard"),
        ("cross_flatfault_a", 0, 50, "FlatFault-A median"),
        ("cross_flatfault_a", 0, 75, "FlatFault-A hard"),
    ]

    selected = []
    data = []
    for dataset, seed, percentile, title in rowspec:
        sid = select_sample(per_sample, dataset, seed, percentile)
        target, b1, pasd = get_sample_arrays(dataset, seed, sid)
        selected.append({"dataset": dataset, "seed": seed, "percentile": percentile, "sample_id": sid, "row_title": title})
        data.append((title, sid, target, b1, pasd))
    pd.DataFrame(selected).to_csv(SOURCE_DATA / "Figure_SCI_2_selected_samples.csv", index=False)

    fig = plt.figure(figsize=(7.2, 7.6))
    gs = fig.add_gridspec(4, 6, width_ratios=[1, 1, 1, 1, 1, 1.05], wspace=0.08, hspace=0.18)
    vmin, vmax = 1500, 4500
    err_max = 1200
    imp_lim = 900
    col_titles = ["Target", "B1", "PASD-Core", "|B1 - target|", "|PASD - target|", "Error reduction"]
    axes = []
    for r, (row_title, sid, target, b1, pasd) in enumerate(data):
        row_arrays = [
            target,
            b1,
            pasd,
            np.abs(b1 - target),
            np.abs(pasd - target),
            np.abs(b1 - target) - np.abs(pasd - target),
        ]
        for c, arr in enumerate(row_arrays):
            ax = fig.add_subplot(gs[r, c])
            axes.append(ax)
            if c < 3:
                im = ax.imshow(arr, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
            elif c < 5:
                im = ax.imshow(arr, cmap="magma", vmin=0, vmax=err_max, aspect="equal")
            else:
                im = ax.imshow(arr, cmap="BrBG", vmin=-imp_lim, vmax=imp_lim, aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=7, pad=4)
            if c == 0:
                ax.set_ylabel(f"{row_title}\nID {sid}", fontsize=7, rotation=0, labelpad=34, va="center")

    # Shared colorbars as slim insets at the bottom.
    cax1 = fig.add_axes([0.17, 0.055, 0.23, 0.012])
    cax2 = fig.add_axes([0.48, 0.055, 0.20, 0.012])
    cax3 = fig.add_axes([0.75, 0.055, 0.17, 0.012])
    cb1 = fig.colorbar(plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin, vmax)), cax=cax1, orientation="horizontal")
    cb2 = fig.colorbar(plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(0, err_max)), cax=cax2, orientation="horizontal")
    cb3 = fig.colorbar(plt.cm.ScalarMappable(cmap="BrBG", norm=plt.Normalize(-imp_lim, imp_lim)), cax=cax3, orientation="horizontal")
    cb1.set_label("Velocity (m/s)", fontsize=6)
    cb2.set_label("Absolute error (m/s)", fontsize=6)
    cb3.set_label("B1 error - PASD error (m/s)", fontsize=6)
    for cb in (cb1, cb2, cb3):
        cb.ax.tick_params(labelsize=5, length=2)

    fig.suptitle("Co-sample velocity plates show whether lower error occurs at interfaces or only in the background", x=0.01, y=0.995, ha="left", fontsize=10)
    fig.text(0.01, 0.965, "Samples are selected by B1 MAE percentile only; PASD-Core is not used for sample selection.", fontsize=7, color=NEUTRAL)
    save(fig, "Figure_SCI_2_velocity_error_plate")


def figure_sci_3(summary: pd.DataFrame, boot: pd.DataFrame) -> None:
    """Metric provenance + distribution figure replacing repetitive bars."""
    per = pd.read_csv(PHASE3R / "corrected_metrics" / "all_corrected_per_sample_metrics.csv")
    targets = ["cross_curvevel_a", "cross_flatfault_a"]

    fig = plt.figure(figsize=(7.2, 4.85))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.72, 1.18], wspace=0.34, hspace=0.36)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])

    # Panel a: compact protocol/provenance lane with explicit leakage barriers.
    ax_a.axis("off")
    steps = [
        ("1 Source-only data", "FlatVel-A\ntrain/val/test"),
        ("2 Lock model", "source validation\naggregation"),
        ("3 Hold-out targets", "CurveVel-A\nFlatFault-A"),
        ("4 Repair metrics", "inverse velocity\nsource edge mask"),
        ("5 Freeze package", "tables, figures\nsource data"),
    ]
    xs = np.linspace(0.08, 0.92, len(steps))
    box_w = 0.15
    box_h = 0.46
    for i, (head, body) in enumerate(steps):
        x = xs[i]
        face = "#EEF1F8" if i % 2 == 0 else "#F7F0E8"
        rect = FancyBboxPatch(
            (x - box_w / 2, 0.25),
            box_w,
            box_h,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            facecolor=face,
            edgecolor="#909090",
            lw=0.8,
        )
        ax_a.add_patch(rect)
        ax_a.text(x, 0.58, head, ha="center", va="center", fontsize=6.4, fontweight="bold")
        ax_a.text(x, 0.40, body, ha="center", va="center", fontsize=5.7, linespacing=1.05)
        if i < len(steps) - 1:
            ax_a.annotate(
                "",
                xy=(xs[i + 1] - box_w / 2 - 0.013, 0.48),
                xytext=(x + box_w / 2 + 0.013, 0.48),
                arrowprops=dict(arrowstyle="->", lw=0.85, color=NEUTRAL, shrinkA=0, shrinkB=0),
            )
    ax_a.text(0.0, 0.86, "a", fontsize=9, fontweight="bold", transform=ax_a.transAxes)
    ax_a.text(
        0.035,
        0.86,
        "Target-family leakage control is built into the evidence chain",
        transform=ax_a.transAxes,
        ha="left",
        va="center",
        fontsize=8,
    )
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)

    # Lower panels: boxplots with deterministic jittered samples, avoiding over-smoothed violin shapes.
    panel_specs = [
        (ax_b, "MAE", "Velocity MAE (m/s)", True),
        (ax_c, "source_threshold_edge_MAE", "Source-threshold edge MAE (m/s)", True),
        (ax_d, "edge_F1", "Edge F1", False),
    ]
    rng = np.random.default_rng(20260704)
    for ax, metric, title, lower in panel_specs:
        positions = []
        data = []
        colors = []
        labels = []
        pos = 0
        for target in targets:
            for variant, color in [("B1_raw_unet", BLUE), ("PASD_Core_locked", ORANGE)]:
                vals = per[(per["dataset"] == target) & (per["variant"] == variant)][metric].dropna().to_numpy()
                data.append(vals)
                positions.append(pos)
                colors.append(color)
                labels.append(f"{'Curve' if target == targets[0] else 'Fault'}\n{'B1' if variant.startswith('B1') else 'PASD'}")
                pos += 1
            pos += 0.45
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.54,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.0},
            boxprops={"linewidth": 0.7, "color": "#333333"},
            whiskerprops={"linewidth": 0.7, "color": "#555555"},
            capprops={"linewidth": 0.7, "color": "#555555"},
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.52)
        for vals, xpos, color in zip(data, positions, colors):
            jitter = rng.normal(0, 0.055, size=len(vals))
            ax.scatter(np.full_like(vals, xpos, dtype=float) + jitter, vals, s=4, color=color, alpha=0.22, linewidth=0, zorder=2)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=5.5)
        ax.set_title(title, loc="left", fontsize=8, pad=5)
        ax.grid(axis="y", color="#E0E0E0", lw=0.5)
        ax.set_xlim(min(positions) - 0.55, max(positions) + 0.55)
        clean_axes(ax)
    add_panel(ax_b, "b", x=-0.18)
    add_panel(ax_c, "c", x=-0.18)
    add_panel(ax_d, "d", x=-0.18)
    fig.suptitle("Metric provenance and full-sample distributions expose both gain and residual spread", x=0.01, y=0.995, ha="left", fontsize=10)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.08, top=0.93)
    save(fig, "Figure_SCI_3_provenance_distributions")


def main() -> None:
    ensure_dirs()
    summary = load_summary()
    boot = load_bootstrap()
    figure_sci_1(summary, boot)
    figure_sci_2()
    figure_sci_3(summary, boot)


if __name__ == "__main__":
    main()
