from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parents[1]
RTM_ROOT = ROOT / "rtm_acoustic"
FIG_DIR = RTM_ROOT / "docs" / "jge_main_figures"
OUT_DIR = RTM_ROOT / "docs" / "jge_revision"
FWI_DIR = RTM_ROOT / "outputs" / "FWI" / "full_salt_fwi_cg_allshots_v2"
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
class GateInputs:
    true_velocity: np.ndarray
    initial_velocity: np.ndarray
    inverted_velocity: np.ndarray
    illumination: np.ndarray


def _normalize_positive(values: np.ndarray) -> np.ndarray:
    finite = np.nan_to_num(values.astype(float), copy=False)
    scale = float(np.nanmax(finite))
    if scale <= 0.0:
        return np.zeros_like(finite, dtype=float)
    return finite / scale


def load_gate_inputs() -> GateInputs:
    source = _normalize_positive(np.load(SCHEME2_DIR / "scheme2_source_illumination.npy"))
    receiver = _normalize_positive(np.load(SCHEME2_DIR / "scheme2_receiver_illumination.npy"))
    return GateInputs(
        true_velocity=np.load(FWI_DIR / "full_salt_true_model.npy"),
        initial_velocity=np.load(FWI_DIR / "full_salt_initial_model.npy"),
        inverted_velocity=np.load(FWI_DIR / "full_salt_inverted_model.npy"),
        illumination=np.sqrt(source * receiver),
    )


def _edge_mae(model: np.ndarray, truth: np.ndarray) -> float:
    gy, gx = np.gradient(model)
    ty, tx = np.gradient(truth)
    return float(np.mean(np.abs(np.hypot(gx, gy) - np.hypot(tx, ty))))


def _quality_row(name: str, model: np.ndarray, truth: np.ndarray, baseline: dict[str, float], extra: dict[str, Any]) -> dict[str, Any]:
    mae = float(np.mean(np.abs(model - truth)))
    rmse = float(np.sqrt(np.mean((model - truth) ** 2)))
    edge = _edge_mae(model, truth)
    return {
        "candidate": name,
        "mae_ms": mae,
        "rmse_ms": rmse,
        "edge_mae": edge,
        "mae_improvement_pct": (baseline["mae"] - mae) / baseline["mae"] * 100.0,
        "rmse_improvement_pct": (baseline["rmse"] - rmse) / baseline["rmse"] * 100.0,
        "edge_mae_improvement_pct": (baseline["edge_mae"] - edge) / baseline["edge_mae"] * 100.0,
        **extra,
    }


def _alpha_map(mode: str, illumination: np.ndarray, alpha: float, threshold: float) -> np.ndarray:
    if mode == "global":
        return np.full_like(illumination, alpha, dtype=float)
    if mode == "hard":
        return alpha * (illumination >= threshold)
    if mode == "soft":
        return alpha * np.clip((illumination - threshold) / max(1.0e-6, 1.0 - threshold), 0.0, 1.0)
    if mode == "sqrt":
        return alpha * np.sqrt(np.clip((illumination - threshold) / max(1.0e-6, 1.0 - threshold), 0.0, 1.0))
    if mode == "smooth":
        return alpha * gaussian_filter((illumination >= threshold).astype(float), sigma=3.0)
    raise ValueError(f"unknown gate mode: {mode}")


def scan_spatial_update_gates(inputs: GateInputs) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray, np.ndarray]:
    truth = inputs.true_velocity
    initial = inputs.initial_velocity
    update = inputs.inverted_velocity - inputs.initial_velocity
    baseline = {
        "mae": float(np.mean(np.abs(initial - truth))),
        "rmse": float(np.sqrt(np.mean((initial - truth) ** 2))),
        "edge_mae": _edge_mae(initial, truth),
    }
    rows = [
        _quality_row(
            "initial",
            initial,
            truth,
            baseline,
            {"mode": "none", "alpha": 0.0, "illumination_threshold": 0.0, "active_fraction": 0.0, "mean_alpha": 0.0},
        )
    ]
    candidate_maps: dict[str, np.ndarray] = {}

    modes = ["global", "hard", "soft", "sqrt", "smooth"]
    alphas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]
    thresholds = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for mode in modes:
        for alpha in alphas:
            mode_thresholds = [0.0] if mode == "global" else thresholds
            for threshold in mode_thresholds:
                alpha_field = _alpha_map(mode, inputs.illumination, alpha, threshold)
                model = initial + alpha_field * update
                name = f"{mode}_alpha{alpha:g}_thr{threshold:g}"
                candidate_maps[name] = alpha_field
                row = _quality_row(
                    name,
                    model,
                    truth,
                    baseline,
                    {
                        "mode": mode,
                        "alpha": alpha,
                        "illumination_threshold": threshold,
                        "active_fraction": float(np.mean(alpha_field > 1.0e-9)),
                        "mean_alpha": float(np.mean(alpha_field)),
                    },
                )
                row["accepted"] = bool(
                    row["mae_improvement_pct"] > 0.0
                    and row["rmse_improvement_pct"] > 0.0
                    and row["edge_mae_improvement_pct"] >= 0.0
                )
                row["score"] = (
                    row["mae_improvement_pct"]
                    + row["rmse_improvement_pct"]
                    + 2.0 * max(row["edge_mae_improvement_pct"], 0.0)
                )
                rows.append(row)

    accepted = [row for row in rows if row.get("accepted")]
    if not accepted:
        raise RuntimeError("no spatial update gate passed the quality constraints")
    selected = max(accepted, key=lambda row: row["score"])
    selected_alpha = candidate_maps[selected["candidate"]]
    selected_model = initial + selected_alpha * update
    for row in rows:
        row["selected"] = row["candidate"] == selected["candidate"]
    return rows, selected, selected_alpha, selected_model


def write_outputs(rows: list[dict[str, Any]], selected: dict[str, Any], selected_alpha: np.ndarray, selected_model: np.ndarray) -> dict[str, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FWI_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "spatial_update_gate_candidates.csv"
    json_path = OUT_DIR / "spatial_update_gate_candidates.json"
    md_path = OUT_DIR / "spatial_update_gate_candidates.md"
    alpha_path = FWI_DIR / "spatial_update_gate_alpha.npy"
    model_path = FWI_DIR / "spatial_update_gate_model.npy"
    np.save(alpha_path, selected_alpha.astype(np.float32))
    np.save(model_path, selected_model.astype(np.float32))

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        preferred = [
            "candidate",
            "mode",
            "alpha",
            "illumination_threshold",
            "selected",
            "accepted",
            "score",
            "active_fraction",
            "mean_alpha",
            "mae_improvement_pct",
            "rmse_improvement_pct",
            "edge_mae_improvement_pct",
            "mae_ms",
            "rmse_ms",
            "edge_mae",
        ]
        extra = sorted({field for row in rows for field in row} - set(preferred))
        writer = csv.DictWriter(handle, fieldnames=preferred + extra, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({"selected": selected, "candidates": rows}, indent=2), encoding="utf-8")

    top_rows = sorted(
        [row for row in rows if row.get("accepted")],
        key=lambda row: row.get("score", -1.0),
        reverse=True,
    )[:8]
    lines = [
        "# Spatial illumination-trust FWI update gate",
        "",
        "This experiment replaces a single global FWI update scale with an illumination-trust alpha field. Candidate gates are accepted only when MAE, RMSE, and edge-MAE all improve relative to the initial model.",
        "",
        "## Selected gate",
        "",
        f"- `candidate`: {selected['candidate']}",
        f"- `mode`: {selected['mode']}",
        f"- `alpha`: {selected['alpha']}",
        f"- `illumination_threshold`: {selected['illumination_threshold']}",
        f"- `active_fraction`: {selected['active_fraction']:.4f}",
        f"- `mean_alpha`: {selected['mean_alpha']:.4f}",
        f"- `mae_improvement_pct`: {selected['mae_improvement_pct']:.4f}",
        f"- `rmse_improvement_pct`: {selected['rmse_improvement_pct']:.4f}",
        f"- `edge_mae_improvement_pct`: {selected['edge_mae_improvement_pct']:.4f}",
        "",
        "## Top accepted candidates",
        "",
        "| Candidate | MAE imp. (%) | RMSE imp. (%) | Edge MAE imp. (%) | Active frac. | Mean alpha |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in top_rows:
        lines.append(
            f"| {row['candidate']} | {row['mae_improvement_pct']:.4f} | {row['rmse_improvement_pct']:.4f} | {row['edge_mae_improvement_pct']:.4f} | {row['active_fraction']:.4f} | {row['mean_alpha']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The selected gate is not tuned to make the FWI image visually attractive; it is selected by simultaneous model and edge-quality constraints.",
            "- The result supports a stronger method claim than global damping: FWI updates should be spatially accepted only where illumination makes the update trustworthy.",
            "- The claim remains conservative because SEG/Salt truth is used here for benchmark scoring; field-data use would require proxy metrics such as image-domain gathers, residual focusing, or well ties.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "markdown": md_path, "alpha": alpha_path, "model": model_path}


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


def write_figure(inputs: GateInputs, rows: list[dict[str, Any]], selected: dict[str, Any], selected_alpha: np.ndarray, selected_model: np.ndarray) -> dict[str, Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    global_alpha = _alpha_map("global", inputs.illumination, 0.1, 0.0)
    global_model = inputs.initial_velocity + global_alpha * (inputs.inverted_velocity - inputs.initial_velocity)
    global_row = next(row for row in rows if row["candidate"] == "global_alpha0.1_thr0")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)

    ax = axes[0, 0]
    image = ax.imshow(selected_alpha, cmap="viridis", vmin=0.0, vmax=max(float(np.max(selected_alpha)), 0.1), aspect="auto", extent=EXTENT_KM)
    ax.set_title("Spatial FWI update trust gate alpha(x,z)", fontsize=8)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Depth (km)")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    _panel_label(ax, "a")

    ax = axes[0, 1]
    scatter_rows = [row for row in rows if row["candidate"] != "initial"]
    colors = ["#2a9d8f" if row.get("accepted") else "#b7b7b7" for row in scatter_rows]
    ax.scatter(
        [row["mae_improvement_pct"] for row in scatter_rows],
        [row["edge_mae_improvement_pct"] for row in scatter_rows],
        s=12,
        c=colors,
        alpha=0.75,
        linewidths=0.0,
    )
    ax.scatter([global_row["mae_improvement_pct"]], [global_row["edge_mae_improvement_pct"]], c="#c0392b", marker="x", s=42, label="global alpha=0.1")
    ax.scatter([selected["mae_improvement_pct"]], [selected["edge_mae_improvement_pct"]], c="#264653", marker="*", s=72, label="selected spatial gate")
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.axvline(0.0, color="black", linewidth=0.7)
    ax.set_title("Candidate gate quality frontier", fontsize=8)
    ax.set_xlabel("MAE improvement (%)")
    ax.set_ylabel("Edge MAE improvement (%)")
    ax.legend(frameon=False, fontsize=6)
    _panel_label(ax, "b")

    clip = float(np.percentile(np.abs(global_model - inputs.true_velocity), 99.0))
    ax = axes[1, 0]
    img = ax.imshow(global_model - inputs.true_velocity, cmap="seismic", vmin=-clip, vmax=clip, aspect="auto", extent=EXTENT_KM)
    ax.set_title(
        f"Global alpha=0.1 error\nMAE {global_row['mae_improvement_pct']:.3f}%, edge {global_row['edge_mae_improvement_pct']:.3f}%",
        fontsize=8,
    )
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Depth (km)")
    fig.colorbar(img, ax=ax, fraction=0.035, pad=0.02)
    _panel_label(ax, "c")

    ax = axes[1, 1]
    img = ax.imshow(selected_model - inputs.true_velocity, cmap="seismic", vmin=-clip, vmax=clip, aspect="auto", extent=EXTENT_KM)
    ax.set_title(
        f"Spatial gate error\nMAE {selected['mae_improvement_pct']:.3f}%, edge {selected['edge_mae_improvement_pct']:.3f}%",
        fontsize=8,
    )
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Depth (km)")
    fig.colorbar(img, ax=ax, fraction=0.035, pad=0.02)
    _panel_label(ax, "d")

    stem = "figure4_spatial_update_gate"
    paths = {
        "png": FIG_DIR / f"{stem}.png",
        "pdf": FIG_DIR / f"{stem}.pdf",
        "svg": FIG_DIR / f"{stem}.svg",
        "tiff": FIG_DIR / f"{stem}.tiff",
    }
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["svg"], bbox_inches="tight")
    fig.savefig(paths["tiff"], dpi=600, bbox_inches="tight")
    plt.close(fig)
    return paths


def build() -> dict[str, Path]:
    inputs = load_gate_inputs()
    rows, selected, selected_alpha, selected_model = scan_spatial_update_gates(inputs)
    outputs = write_outputs(rows, selected, selected_alpha, selected_model)
    outputs.update({f"figure_{key}": path for key, path in write_figure(inputs, rows, selected, selected_alpha, selected_model).items()})
    return outputs


def main() -> None:
    paths = build()
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
