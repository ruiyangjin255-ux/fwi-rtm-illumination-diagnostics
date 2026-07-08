from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from rtm_acoustic.acoustic_rtm import preprocess_migration_section


ROOT = Path(__file__).resolve().parents[1]
RTM_ROOT = ROOT / "rtm_acoustic"
FIG_DIR = RTM_ROOT / "docs" / "jge_main_figures"
OUT_DIR = RTM_ROOT / "docs" / "jge_revision"
FWI_DIR = RTM_ROOT / "outputs" / "FWI" / "full_salt_fwi_cg_allshots_v2"
SCALE_DIR = FWI_DIR / "optimized_fwi_rtm_pipeline" / "update_scale_optimization"
SCHEME2_DIR = RTM_ROOT / "outputs" / "RTM" / "seg_salt_scheme2_full30m_nt4001_workers4"

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
    }
)


@dataclass(frozen=True)
class DiagnosticArrays:
    true_velocity: np.ndarray
    source_illumination: np.ndarray
    receiver_illumination: np.ndarray
    source_receiver_illumination: np.ndarray
    source_normalized_rtm: np.ndarray
    laplacian_rtm: np.ndarray
    full_update: np.ndarray
    damped_update: np.ndarray


def _normalize_positive(values: np.ndarray) -> np.ndarray:
    finite = np.nan_to_num(values.astype(float), copy=False)
    scale = float(np.nanmax(finite))
    if scale <= 0.0:
        return np.zeros_like(finite, dtype=float)
    return finite / scale


def _normalize_abs(values: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    finite = np.nan_to_num(values.astype(float), copy=False)
    scale = float(np.percentile(np.abs(finite), percentile))
    if scale <= 0.0:
        return np.zeros_like(finite, dtype=float)
    return np.clip(np.abs(finite) / scale, 0.0, 1.0)


def load_diagnostic_arrays() -> DiagnosticArrays:
    true_velocity = np.load(FWI_DIR / "full_salt_true_model.npy")
    initial = np.load(FWI_DIR / "full_salt_initial_model.npy")
    inverted = np.load(FWI_DIR / "full_salt_inverted_model.npy")
    damped = np.load(SCALE_DIR / "selected_fwi_model.npy")

    source = _normalize_positive(np.load(SCHEME2_DIR / "scheme2_source_illumination.npy"))
    receiver = _normalize_positive(np.load(SCHEME2_DIR / "scheme2_receiver_illumination.npy"))
    source_receiver = np.sqrt(source * receiver)

    source_normalized_rtm = _normalize_abs(
        preprocess_migration_section(np.load(SCHEME2_DIR / "scheme2_source_normalized.npy"))
    )
    laplacian_rtm = _normalize_abs(
        preprocess_migration_section(np.load(SCHEME2_DIR / "scheme2_laplacian_source_normalized.npy"))
    )
    return DiagnosticArrays(
        true_velocity=true_velocity,
        source_illumination=source,
        receiver_illumination=receiver,
        source_receiver_illumination=source_receiver,
        source_normalized_rtm=source_normalized_rtm,
        laplacian_rtm=laplacian_rtm,
        full_update=np.abs(inverted - initial),
        damped_update=np.abs(damped - initial),
    )


def build_target_zone_masks(true_velocity: np.ndarray) -> dict[str, np.ndarray]:
    salt = true_velocity >= 4000.0
    if not salt.any():
        raise ValueError("salt mask is empty; expected SEG/Salt velocity values above 4000 m/s")

    nz, nx = true_velocity.shape
    salt_columns = np.flatnonzero(salt.any(axis=0))
    x_mid = float(np.mean(salt_columns))
    masks = {
        "salt_top": np.zeros_like(salt, dtype=bool),
        "salt_flanks": np.zeros_like(salt, dtype=bool),
        "subsalt_shadow": np.zeros_like(salt, dtype=bool),
    }

    for x in salt_columns:
        z_idx = np.flatnonzero(salt[:, x])
        z_top = int(z_idx.min())
        z_bottom = int(z_idx.max())
        masks["salt_top"][max(0, z_top - 6) : min(nz, z_top + 10), x] = True
        masks["subsalt_shadow"][min(nz, z_bottom + 8) : min(nz, z_bottom + 58), x] = True

        is_left_or_right = abs(float(x) - x_mid) > 0.24 * max(len(salt_columns), 1)
        if is_left_or_right:
            masks["salt_flanks"][max(0, z_top - 2) : min(nz, z_bottom + 8), x] = True

    for name, mask in masks.items():
        if not mask.any():
            raise ValueError(f"target zone mask is empty: {name}")
    return masks


def _zone_stats(arrays: DiagnosticArrays, name: str, mask: np.ndarray) -> dict[str, Any]:
    src = arrays.source_illumination[mask]
    rec = arrays.receiver_illumination[mask]
    bi = arrays.source_receiver_illumination[mask]
    src_rtm = arrays.source_normalized_rtm[mask]
    lap = arrays.laplacian_rtm[mask]
    full_update = arrays.full_update[mask]
    damped_update = arrays.damped_update[mask]
    bi_mean = float(np.mean(bi))
    bi_std = float(np.std(bi))
    return {
        "zone": name,
        "pixel_count": int(mask.sum()),
        "mean_source_illumination_norm": float(np.mean(src)),
        "mean_receiver_illumination_norm": float(np.mean(rec)),
        "mean_source_receiver_illumination_norm": bi_mean,
        "source_receiver_illumination_cv": bi_std / bi_mean if bi_mean > 0.0 else None,
        "low_source_receiver_illumination_fraction": float(np.mean(bi < 0.01)),
        "mean_source_normalized_rtm_abs_norm": float(np.mean(src_rtm)),
        "mean_laplacian_rtm_abs_norm": float(np.mean(lap)),
        "mean_full_update_abs_ms": float(np.mean(full_update)),
        "mean_damped_update_abs_ms": float(np.mean(damped_update)),
        "damped_to_full_update_ratio": float(np.mean(damped_update) / np.mean(full_update)) if np.mean(full_update) > 0 else None,
    }


def compute_target_zone_metrics(arrays: DiagnosticArrays) -> list[dict[str, Any]]:
    masks = build_target_zone_masks(arrays.true_velocity)
    return [_zone_stats(arrays, name, masks[name]) for name in ["salt_top", "salt_flanks", "subsalt_shadow"]]


def write_target_zone_outputs(rows: list[dict[str, Any]], output_dir: Path = OUT_DIR) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "target_zone_illumination_metrics.csv"
    json_path = output_dir / "target_zone_illumination_metrics.json"
    md_path = output_dir / "target_zone_illumination_metrics.md"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = [
        "# Target-zone illumination and FWI-RTM diagnostics",
        "",
        "The zones are derived from the SEG/Salt high-velocity body: salt top, salt flanks, and subsalt shadow. Metrics connect illumination, RTM image response, and FWI update energy.",
        "",
        "| Zone | Pixels | Src illum | Rec illum | Src-rec illum | Low illum frac | Src-norm RTM | Lap RTM | Full update | Damped update |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {zone} | {pixel_count} | {mean_source_illumination_norm:.4f} | {mean_receiver_illumination_norm:.4f} | {mean_source_receiver_illumination_norm:.4f} | {low_source_receiver_illumination_fraction:.4f} | {mean_source_normalized_rtm_abs_norm:.4f} | {mean_laplacian_rtm_abs_norm:.4f} | {mean_full_update_abs_ms:.3f} | {mean_damped_update_abs_ms:.3f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The table makes the paper framework operational: RTM illumination, migrated-image response, and FWI update energy are evaluated over the same target zones.",
            "- The damped update should not be read as high-quality velocity recovery; it is a controlled update passed through quality gates before RTM validation.",
            "- Subsalt and flank metrics are the most relevant zones for illumination compensation claims.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "markdown": md_path}


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
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.8),
    )


def write_target_zone_figure(
    arrays: DiagnosticArrays,
    rows: list[dict[str, Any]],
    fig_dir: Path = FIG_DIR,
    stem: str = "figure5_target_zone_illumination_diagnostics",
) -> dict[str, Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    masks = build_target_zone_masks(arrays.true_velocity)
    zone_index = np.full(arrays.true_velocity.shape, np.nan)
    for idx, name in enumerate(["salt_top", "salt_flanks", "subsalt_shadow"], start=1):
        zone_index[masks[name]] = idx

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    ax = axes[0, 0]
    ax.imshow(arrays.true_velocity, cmap="turbo", aspect="auto", extent=EXTENT_KM)
    overlay = ax.imshow(zone_index, cmap="Set2", vmin=1, vmax=3, alpha=np.where(np.isnan(zone_index), 0.0, 0.58), aspect="auto", extent=EXTENT_KM)
    ax.set_title("Target zones on SEG/Salt velocity", fontsize=8)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Depth (km)")
    cbar = fig.colorbar(overlay, ax=ax, fraction=0.035, pad=0.02, ticks=[1, 2, 3])
    cbar.ax.set_yticklabels(["top", "flank", "subsalt"])
    cbar.ax.tick_params(labelsize=6)
    _panel_label(ax, "a")

    ax = axes[0, 1]
    illum = np.log10(arrays.source_receiver_illumination + 1.0e-6)
    img = ax.imshow(illum, cmap="magma", aspect="auto", extent=EXTENT_KM)
    ax.contour(zone_index, levels=[0.5, 1.5, 2.5, 3.5], colors="white", linewidths=0.45, extent=EXTENT_KM)
    ax.set_title("Source-receiver illumination (log10)", fontsize=8)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Depth (km)")
    fig.colorbar(img, ax=ax, fraction=0.035, pad=0.02)
    _panel_label(ax, "b")

    zones = [row["zone"] for row in rows]
    labels = ["Top", "Flanks", "Subsalt"]
    x = np.arange(len(zones))
    width = 0.34

    ax = axes[1, 0]
    ax.bar(x - width / 2, [row["mean_source_receiver_illumination_norm"] for row in rows], width, label="Src-rec illum", color="#2a9d8f")
    ax.bar(x + width / 2, [row["mean_laplacian_rtm_abs_norm"] for row in rows], width, label="Laplacian RTM", color="#6d597a")
    ax.set_title("Target-zone illumination and RTM response", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Normalized mean")
    ax.legend(frameon=False, fontsize=6)
    _panel_label(ax, "c")

    ax = axes[1, 1]
    ax.bar(x - width / 2, [row["mean_full_update_abs_ms"] for row in rows], width, label="Full update", color="#c0392b")
    ax.bar(x + width / 2, [row["mean_damped_update_abs_ms"] for row in rows], width, label="Damped update", color="#2a9d8f")
    ax.set_title("FWI update energy passed through gate", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean |update| (m/s)")
    ax.legend(frameon=False, fontsize=6)
    _panel_label(ax, "d")

    paths = {
        "png": fig_dir / f"{stem}.png",
        "pdf": fig_dir / f"{stem}.pdf",
        "svg": fig_dir / f"{stem}.svg",
        "tiff": fig_dir / f"{stem}.tiff",
    }
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["svg"], bbox_inches="tight")
    fig.savefig(paths["tiff"], dpi=600, bbox_inches="tight")
    plt.close(fig)
    return paths


def build() -> dict[str, Path]:
    arrays = load_diagnostic_arrays()
    rows = compute_target_zone_metrics(arrays)
    outputs = write_target_zone_outputs(rows)
    outputs.update({f"figure_{key}": path for key, path in write_target_zone_figure(arrays, rows).items()})
    return outputs


def main() -> None:
    paths = build()
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
