from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from admit_fwi.acoustic_rtm import preprocess_migration_section
from admit_fwi.build_jge_innovation_framework import build as build_innovation_framework
from admit_fwi.build_spatial_update_gate import build as build_spatial_update_gate
from admit_fwi.build_target_zone_illumination_diagnostics import build as build_target_zone_diagnostics


ROOT = Path(__file__).resolve().parents[1]
RTM_ROOT = ROOT / "admit_fwi"
FIG_DIR = RTM_ROOT / "docs" / "jge_main_figures"
FWI_DIR = RTM_ROOT / "outputs" / "FWI" / "full_salt_fwi_cg_allshots_v2"
PIPELINE_DIR = FWI_DIR / "optimized_fwi_rtm_pipeline"
SCALE_DIR = PIPELINE_DIR / "update_scale_optimization"
RTM_COMPARE_DIR = RTM_ROOT / "outputs" / "RTM" / "before_after_fwi_alpha010_nt1200_shots12"
SCHEME2_DIR = RTM_ROOT / "outputs" / "RTM" / "seg_salt_scheme2_full30m_nt4001_workers4"
LOCAL_FWI_DIR = RTM_ROOT / "outputs" / "FWI影响因素" / "small_salt_fwi_adaptive_line_search"

NX = 676
NZ = 230
DX_KM = 0.01
DZ_KM = 0.01
EXTENT_KM = [0.0, NX * DX_KM, NZ * DZ_KM, 0.0]


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "figure.dpi": 160,
    }
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig: plt.Figure, stem: str, *, dpi: int = 600) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    base = FIG_DIR / stem
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight")


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.015,
        0.985,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.8),
    )


def _imshow_velocity(ax: plt.Axes, model: np.ndarray, title: str, *, vmin: float, vmax: float) -> None:
    image = ax.imshow(model, cmap="turbo", vmin=vmin, vmax=vmax, aspect="auto", extent=EXTENT_KM)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Depth (km)")
    return image


def _imshow_section(
    ax: plt.Axes,
    section: np.ndarray,
    title: str,
    *,
    cmap: str = "gray",
    percentile: float = 99.0,
    colorbar: bool = False,
) -> None:
    if cmap in {"gray", "seismic"}:
        clip = float(np.percentile(np.abs(section), percentile)) or 1.0
        image = ax.imshow(section, cmap=cmap, vmin=-clip, vmax=clip, aspect="auto", extent=EXTENT_KM)
    else:
        image = ax.imshow(section, cmap=cmap, aspect="auto", extent=EXTENT_KM)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Depth (km)")
    if colorbar:
        cbar = ax.figure.colorbar(image, ax=ax, fraction=0.035, pad=0.018)
        cbar.ax.tick_params(labelsize=6)


def _line(ax: plt.Axes, x: Iterable[float], y: Iterable[float], label: str, **kwargs) -> None:
    ax.plot(list(x), list(y), marker="o", linewidth=1.2, markersize=3, label=label, **kwargs)


def make_figure1_fwi_quality_gate() -> None:
    pipeline = _load_json(PIPELINE_DIR / "optimized_fwi_rtm_pipeline_report.json")
    scale = _load_json(SCALE_DIR / "update_scale_optimization.json")
    local = _load_json(LOCAL_FWI_DIR / "adaptive_line_search_summary.json")
    selected_alpha = scale["selected_alpha"]
    selected_candidate = next(row for row in scale["candidates"] if abs(row["alpha"] - selected_alpha) < 1.0e-12)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.1), constrained_layout=True)

    ax = axes[0, 0]
    ax.set_axis_off()
    rows = [
        ("FWI challenge", "cycle skipping;\nweak salt edges", "model-quality\ngate"),
        ("Regularization", "structure\npreservation", "edge/gradient\nmetrics"),
        ("Optimization", "step-length\nsensitivity", "alpha scan +\nline search"),
        ("RTM/LSRTM", "image condition\nbias", "RTM before/after\nvalidation"),
        ("Benchmarks", "reproducibility", "SEG/Salt +\nfixed scripts"),
    ]
    table = ax.table(
        cellText=rows,
        colLabels=["Literature need", "Risk in this result", "Implemented response"],
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.30, 0.34, 0.36],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(5.8)
    table.scale(1.0, 1.55)
    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_facecolor("#23395b")
            cell.set_text_props(color="white", weight="bold")
        elif c == 2:
            cell.set_facecolor("#e8f3ee")
        else:
            cell.set_facecolor("#f7f7f7")
    ax.set_title("Literature-guided method synthesis", fontsize=8)
    _panel_label(ax, "a")

    candidates = scale["candidates"]
    alphas = [row["alpha"] for row in candidates]
    mae = [row["mae_improvement_fraction"] * 100.0 for row in candidates]
    edge = [row["edge_mae_improvement_fraction"] * 100.0 for row in candidates]
    accepted = [row["accepted"] for row in candidates]
    ax = axes[0, 1]
    colors = ["#2a9d8f" if ok else "#b7b7b7" for ok in accepted]
    ax.bar([str(a) for a in alphas], mae, color=colors, label="Model MAE")
    ax.plot([str(a) for a in alphas], edge, color="#c0392b", marker="o", linewidth=1.1, label="Edge MAE")
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.axvline([str(a) for a in alphas].index(str(selected_alpha)), color="#264653", linestyle="--", linewidth=1.0)
    ax.set_title("Update-scale quality gate", fontsize=8)
    ax.set_xlabel("FWI update scale alpha")
    ax.set_ylabel("Improvement (%)")
    ax.legend(frameon=False, fontsize=6)
    _panel_label(ax, "b")

    ax = axes[1, 0]
    evidence_names = [
        "FWI data\nmisfit",
        "Local adaptive\nline search",
        "Local illum.\npreconditioner",
        "Damped edge\nMAE",
        "RTM RMSE\nchange",
        "Full edge\nMAE",
    ]
    evidence_values = [
        pipeline["full_update_quality"]["misfit_reduction_fraction"] * 100.0,
        local["baseline"]["misfit_reduction_fraction"] * 100.0,
        local["illumination_preconditioned"]["misfit_reduction_fraction"] * 100.0,
        selected_candidate["edge_mae_improvement_fraction"] * 100.0,
        pipeline["rtm_validation"]["filtered_rmse_improvement_fraction"] * 100.0,
        pipeline["full_update_quality"]["edge_mae_improvement_fraction"] * 100.0,
    ]
    evidence_colors = ["#2a9d8f" if value >= 0 else "#c0392b" for value in evidence_values]
    y_pos = np.arange(len(evidence_names))
    ax.barh(y_pos, evidence_values, color=evidence_colors)
    ax.axvline(0.0, color="black", linewidth=0.7)
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xticks([-1.0, -0.1, 0.0, 0.1, 1.0, 10.0, 50.0])
    ax.set_xticklabels(["-1", "-0.1", "0", "0.1", "1", "10", "50"])
    ax.set_title("Evidence strength and limitation", fontsize=8)
    ax.set_xlabel("Improvement or reduction (%)")
    ax.set_yticks(y_pos, evidence_names)
    ax.tick_params(axis="y", labelsize=5.7)
    for y, value in zip(y_pos, evidence_values):
        ha = "left" if value >= 0 else "right"
        offset = 1.1 if value >= 0 else 0.9
        label_x = value * offset if abs(value) > 0.08 else (0.075 if value >= 0 else -0.075)
        ax.text(label_x, y, f"{value:.3f}%", ha=ha, va="center", fontsize=5.8)
    _panel_label(ax, "c")

    ax = axes[1, 1]
    ax.set_axis_off()
    claim_rows = [
        ("High-quality FWI velocity recovery", "No", "#f7d7d2"),
        ("Quality-gated FWI-to-RTM workflow", "Yes", "#dff0e6"),
        ("RTM imaging-condition diagnostics", "Yes", "#dff0e6"),
        ("Illumination scaling as primary FWI optimizer", "No", "#f7d7d2"),
        ("ML/foundation-model prior as main result", "Future", "#fff1c2"),
    ]
    y0 = 0.84
    for idx, (claim, status, color) in enumerate(claim_rows):
        y = y0 - idx * 0.17
        ax.text(
            0.02,
            y,
            "\n".join(textwrap.wrap(claim, width=32)),
            ha="left",
            va="center",
            fontsize=6.4,
        )
        ax.text(
            0.91,
            y,
            status,
            ha="center",
            va="center",
            fontsize=6.4,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.18", facecolor=color, edgecolor="#505050", linewidth=0.5),
        )
    ax.set_title("Publishable claim boundary", fontsize=8)
    _panel_label(ax, "d")
    _save(fig, "figure1_fwi_quality_gate")
    plt.close(fig)


def make_figure2_rtm_validation() -> None:
    summary = _load_json(RTM_COMPARE_DIR / "rtm_before_after_summary.json")
    reference = np.load(RTM_COMPARE_DIR / "reference_true_velocity" / "rtm_display.npy")
    before = np.load(RTM_COMPARE_DIR / "before_initial_velocity" / "rtm_display.npy")
    after = np.load(RTM_COMPARE_DIR / "after_fwi_velocity" / "rtm_display.npy")
    difference = after - before
    before_rmse = summary["cases"]["before_initial_velocity"]["filtered"]["reference_rmse"]
    after_rmse = summary["cases"]["after_fwi_velocity"]["filtered"]["reference_rmse"]
    gain = summary["filtered_reference_rmse_improvement_fraction"] * 100.0

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), constrained_layout=True)
    panels = [
        ("a", "Reference RTM using true velocity", reference, "gray", False),
        ("b", "RTM using initial velocity", before, "gray", False),
        ("c", "RTM using quality-gated FWI velocity", after, "gray", False),
        ("d", "After-minus-before display difference", difference, "seismic", True),
    ]
    for ax, (label, title, data, cmap, colorbar) in zip(axes.ravel(), panels):
        _imshow_section(ax, data, title, cmap=cmap, colorbar=colorbar)
        _panel_label(ax, label)
    fig.suptitle(
        f"RTM validation: RMSE {before_rmse:.5f} -> {after_rmse:.5f}; improvement {gain:.3f}%",
        fontsize=8,
    )
    _save(fig, "figure2_rtm_before_after_validation")
    plt.close(fig)


def make_figure3_imaging_conditions() -> None:
    source = preprocess_migration_section(np.load(SCHEME2_DIR / "scheme2_source_normalized.npy"))
    source_receiver = preprocess_migration_section(np.load(SCHEME2_DIR / "scheme2_source_receiver_normalized.npy"))
    laplacian = preprocess_migration_section(np.load(SCHEME2_DIR / "scheme2_laplacian_source_normalized.npy"))
    illumination = np.log10(np.load(SCHEME2_DIR / "scheme2_receiver_illumination.npy") + 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), constrained_layout=True)
    panels = [
        ("a", "Source-normalized RTM", source, "gray", False),
        ("b", "Source-receiver-normalized RTM", source_receiver, "gray", False),
        ("c", "Laplacian source-normalized RTM", laplacian, "gray", False),
        ("d", "Receiver illumination (log10)", illumination, "magma", True),
    ]
    for ax, (label, title, data, cmap, colorbar) in zip(axes.ravel(), panels):
        _imshow_section(ax, data, title, cmap=cmap, colorbar=colorbar)
        _panel_label(ax, label)
    fig.suptitle("Full-aperture imaging-condition diagnostics (224 shots, nt=4001)", fontsize=8)
    _save(fig, "figure3_imaging_condition_diagnostics")
    plt.close(fig)


def make_figure4_local_fwi_and_claim_boundary() -> None:
    local = _load_json(LOCAL_FWI_DIR / "adaptive_line_search_summary.json")
    pipeline = _load_json(PIPELINE_DIR / "optimized_fwi_rtm_pipeline_report.json")
    base = local["baseline"]
    pre = local["illumination_preconditioned"]
    reduction_labels = ["Baseline\nadaptive", "Illumination\npreconditioned"]
    reductions = [base["misfit_reduction_fraction"] * 100.0, pre["misfit_reduction_fraction"] * 100.0]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
    ax = axes[0, 0]
    _line(ax, range(len(base["misfit_history"])), base["misfit_history"], "Baseline", color="#2a9d8f")
    _line(ax, range(len(pre["misfit_history"])), pre["misfit_history"], "Illumination-preconditioned", color="#c0392b")
    ax.set_title("Local FWI misfit histories", fontsize=8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Misfit")
    ax.legend(frameon=False, fontsize=6)
    _panel_label(ax, "a")

    ax = axes[0, 1]
    bars = ax.bar(reduction_labels, reductions, color=["#2a9d8f", "#c0392b"])
    ax.set_title("Local-window reduction", fontsize=8)
    ax.set_ylabel("Misfit reduction (%)")
    for bar, value in zip(bars, reductions):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.15, f"{value:.2f}%", ha="center", va="bottom", fontsize=7)
    _panel_label(ax, "b")

    ax = axes[1, 0]
    names = ["Full update\nedge MAE", "Damped update\nedge MAE", "RTM RMSE\nchange"]
    selected_alpha = pipeline["update_scale"]["selected_alpha"]
    selected_candidate = next(
        row for row in pipeline["update_scale"]["candidates"] if abs(row["alpha"] - selected_alpha) < 1.0e-12
    )
    values = [
        pipeline["full_update_quality"]["edge_mae_improvement_fraction"] * 100.0,
        selected_candidate["edge_mae_improvement_fraction"] * 100.0,
        pipeline["rtm_validation"]["filtered_rmse_improvement_fraction"] * 100.0,
    ]
    colors = ["#c0392b" if value < 0 else "#2a9d8f" for value in values]
    ax.bar(names, values, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_title("Claim boundary metrics", fontsize=8)
    ax.set_ylabel("Improvement (%)")
    _panel_label(ax, "c")

    ax = axes[1, 1]
    ax.set_axis_off()
    steps = [
        ("FWI misfit\nreduction", 0.10),
        ("Model-quality\ngate", 0.38),
        ("Damped update\nalpha=0.1", 0.66),
        ("RTM\nvalidation", 0.90),
    ]
    for text, x in steps:
        ax.text(
            x,
            0.55,
            text,
            ha="center",
            va="center",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#edf4ea", edgecolor="#303030", linewidth=0.7),
        )
    for (_, x0), (_, x1) in zip(steps[:-1], steps[1:]):
        ax.annotate("", xy=(x1 - 0.08, 0.55), xytext=(x0 + 0.11, 0.55), arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_title("Defensible SCI claim path", fontsize=8)
    _panel_label(ax, "d")
    _save(fig, "figure4_local_fwi_claim_boundary")
    plt.close(fig)


def write_captions() -> None:
    captions = """# JGE Main Figure Captions

## Figure 1. Literature-guided FWI-RTM synthesis and claim boundary

The figure reframes the weak FWI image result into a defensible integrated method. It maps recent FWI/RTM literature needs to the implemented response, ranks update-scale candidates, summarizes evidence strength and explicitly separates publishable claims from unsupported claims.

## Figure 2. RTM validation before and after quality-gated FWI updating

RTM is performed with the true velocity, the initial smoothed velocity, and the selected quality-gated FWI velocity. The damped FWI update slightly reduces the Laplacian-filtered image RMSE relative to the true-velocity reference from 0.027130 to 0.027109 in the 12-shot nt=1200 validation.

## Figure 3. Full-aperture RTM imaging-condition diagnostics

Source-normalized, source-receiver-normalized, Laplacian-enhanced, and receiver-illumination panels summarize the full-aperture imaging-condition behavior. The source-receiver normalized image remains close to the source-normalized image, while the Laplacian condition changes the spectral emphasis and reflector expression.

## Figure 4. Illumination-trust spatial FWI update gate

The figure replaces a single global FWI update scale with a spatially varying alpha field controlled by source-receiver illumination. Candidate gates are accepted only when MAE, RMSE, and edge-MAE all improve relative to the initial model. The selected gate improves model error while avoiding the edge degradation observed for the global alpha=0.1 update.

## Figure 5. Target-zone illumination, RTM response, and FWI update diagnostics

Salt-top, salt-flank, and subsalt zones are derived from the high-velocity salt body. The figure evaluates source-receiver illumination, RTM response, and FWI update energy over the same target zones, showing that the subsalt shadow zone has weaker illumination and RTM response while receiving negligible FWI update energy in the current gated workflow.
"""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (FIG_DIR / "jge_main_figure_captions.md").write_text(captions, encoding="utf-8")


def main() -> None:
    build_innovation_framework()
    make_figure1_fwi_quality_gate()
    make_figure2_rtm_validation()
    make_figure3_imaging_conditions()
    build_spatial_update_gate()
    build_target_zone_diagnostics()
    write_captions()


if __name__ == "__main__":
    main()
