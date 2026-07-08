from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rtm_acoustic.acoustic_rtm import preprocess_migration_section


FIG_DIR = Path(__file__).resolve().parent
SCHEME1_DIR = (
    ROOT
    / "rtm_acoustic"
    / "outputs"
    / "RTM"
    / "seg_salt_multishot_rtm_padded60_full30m_workers4"
    / "optimization_compare"
)
SCHEME2_DIR = ROOT / "rtm_acoustic" / "outputs" / "RTM" / "seg_salt_scheme2_full30m_nt4001_workers4"
BEFORE_AFTER_DIR = ROOT / "rtm_acoustic" / "outputs" / "RTM" / "before_after_fwi_alpha010_nt1200_shots12"

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
    }
)


def save_figure(fig: plt.Figure, stem: str, *, dpi: int = 600) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    base = FIG_DIR / stem
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight")


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def imshow_section(
    ax: plt.Axes,
    section: np.ndarray,
    *,
    title: str,
    percentile: float = 99.0,
    cmap: str = "gray",
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
    ax.set_xlim(EXTENT_KM[0], EXTENT_KM[1])
    ax.set_ylim(EXTENT_KM[2], EXTENT_KM[3])
    if colorbar:
        cbar = ax.figure.colorbar(image, ax=ax, fraction=0.035, pad=0.018)
        cbar.ax.tick_params(labelsize=6)


def make_figure1_workflow() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    boxes = [
        (0.03, 0.62, 0.16, 0.20, "SEG/Salt\nvelocity model"),
        (0.24, 0.62, 0.16, 0.20, "Padding +\nsmoothing"),
        (0.45, 0.62, 0.16, 0.20, "Forward\nmodeling"),
        (0.66, 0.62, 0.16, 0.20, "Reverse-time\npropagation"),
        (0.45, 0.22, 0.16, 0.20, "Zero-lag\ncorrelation"),
        (0.64, 0.22, 0.15, 0.20, "Illumination\nnormalization"),
        (0.84, 0.22, 0.13, 0.20, "Display +\nfigures"),
    ]
    colors = ["#e8eef5", "#e8eef5", "#edf4ea", "#edf4ea", "#f4efe3", "#f4efe3", "#eee8f4"]
    for (x, y, w, h, text), color in zip(boxes, colors):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                linewidth=0.8,
                edgecolor="#303030",
                facecolor=color,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7)

    arrows = [
        ((0.19, 0.72), (0.24, 0.72)),
        ((0.40, 0.72), (0.45, 0.72)),
        ((0.61, 0.72), (0.66, 0.72)),
        ((0.74, 0.62), (0.53, 0.42)),
        ((0.61, 0.32), (0.64, 0.32)),
        ((0.79, 0.32), (0.84, 0.32)),
    ]
    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=0.8,
                color="#303030",
            )
        )

    ax.text(
        0.03,
        0.08,
        "Outputs: source-normalized image, source-receiver-normalized image, "
        "Laplacian-enhanced image and checkpointed full-shot accumulation.",
        fontsize=7,
        ha="left",
        va="center",
    )
    save_figure(fig, "fig1_workflow")
    plt.close(fig)


def make_figure2_scheme1() -> None:
    section = np.load(SCHEME1_DIR / "candidate_paper_recommended.npy")
    fig, ax = plt.subplots(figsize=(7.2, 3.4), constrained_layout=True)
    imshow_section(ax, section, title="Conservative display-optimized RTM image", percentile=99.0)
    save_figure(fig, "fig2_scheme1_main_result")
    plt.close(fig)


def make_figure3_scheme2() -> None:
    source = np.load(SCHEME2_DIR / "scheme2_source_normalized.npy")
    source_receiver = np.load(SCHEME2_DIR / "scheme2_source_receiver_normalized.npy")
    laplacian = np.load(SCHEME2_DIR / "scheme2_laplacian_source_normalized.npy")
    illumination = np.log10(np.load(SCHEME2_DIR / "scheme2_receiver_illumination.npy") + 1.0)

    source = preprocess_migration_section(source)
    source_receiver = preprocess_migration_section(source_receiver)
    laplacian = preprocess_migration_section(laplacian)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5), constrained_layout=True)
    panels = [
        ("a", "Source-normalized", source, "gray", False),
        ("b", "Source-receiver-normalized", source_receiver, "gray", False),
        ("c", "Laplacian source-normalized", laplacian, "gray", False),
        ("d", "Receiver illumination (log10)", illumination, "magma", True),
    ]
    for ax, (label, title, data, cmap, colorbar) in zip(axes.ravel(), panels):
        imshow_section(ax, data, title=title, cmap=cmap, colorbar=colorbar)
        panel_label(ax, label)
    save_figure(fig, "fig3_scheme2_imaging_condition")
    plt.close(fig)


def make_figure3_before_after_fwi() -> None:
    summary = json.loads((BEFORE_AFTER_DIR / "rtm_before_after_summary.json").read_text(encoding="utf-8"))
    reference = np.load(BEFORE_AFTER_DIR / "reference_true_velocity" / "rtm_display.npy")
    before = np.load(BEFORE_AFTER_DIR / "before_initial_velocity" / "rtm_display.npy")
    after = np.load(BEFORE_AFTER_DIR / "after_fwi_velocity" / "rtm_display.npy")
    difference = after - before

    before_metrics = summary["cases"]["before_initial_velocity"]["filtered"]
    after_metrics = summary["cases"]["after_fwi_velocity"]["filtered"]
    improvement = summary["filtered_reference_rmse_improvement_fraction"] * 100.0
    subtitle = (
        f"12 shots, nt=1200; filtered RMSE change after FWI = {improvement:.2f}% "
        f"(before {before_metrics['reference_rmse']:.5f}, after {after_metrics['reference_rmse']:.5f})"
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5), constrained_layout=True)
    panels = [
        ("a", "Reference RTM using true velocity", reference, "gray", False),
        ("b", "RTM using initial velocity", before, "gray", False),
        ("c", "RTM using damped FWI velocity", after, "gray", False),
        ("d", "After-minus-before display difference", difference, "seismic", True),
    ]
    for ax, (label, title, data, cmap, colorbar) in zip(axes.ravel(), panels):
        imshow_section(ax, data, title=title, percentile=99.0, cmap=cmap, colorbar=colorbar)
        panel_label(ax, label)
    fig.suptitle(subtitle, fontsize=8)
    save_figure(fig, "fig3_rtm_before_after_fwi")
    plt.close(fig)


def write_caption_file() -> None:
    captions = """# SCI figure package

## Figure 1. Acoustic RTM workflow

The workflow starts from the SEG/Salt velocity model, applies model padding and smoothing, performs finite-difference source modeling and receiver reverse-time propagation, forms a zero-lag cross-correlation image, and generates illumination-normalized and Laplacian-enhanced image candidates. Checkpointed accumulation records completed shots for recoverable full-shot computation.

## Figure 2. Conservative scheme-1 RTM display

The scheme-1 result is generated from the existing full RTM image by conservative display optimization only. The acoustic propagator and zero-lag cross-correlation imaging condition are unchanged. This panel is recommended as the stable main result because it improves readability without aggressive edge enhancement.

## Figure 3. RTM before and after FWI velocity updating

The reference image is migrated with the true velocity, while the two test images use the initial smoothed velocity and the model-quality-gated FWI velocity, respectively. In the 12-shot, nt=1200 comparison, the selected damped update scale is alpha=0.1; this slightly reduces the Laplacian-filtered RTM image RMSE relative to the true-velocity reference, while the full alpha=1.0 update is rejected by the model-quality gate.

## Figure 4. Scheme-2 imaging-condition comparison

Source-receiver normalization preserves the main image structure relative to source-only normalization, whereas Laplacian source-normalized imaging suppresses low-wavenumber background and highlights salt-boundary and reflector detail. Receiver illumination is shown on a log10 scale to document the full-aperture illumination distribution.
"""
    (FIG_DIR / "figure_captions.md").write_text(captions, encoding="utf-8")


def main() -> None:
    make_figure1_workflow()
    make_figure2_scheme1()
    make_figure3_before_after_fwi()
    make_figure3_scheme2()
    write_caption_file()


if __name__ == "__main__":
    main()
