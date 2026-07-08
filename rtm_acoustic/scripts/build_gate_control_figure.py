from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))


def _load(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32, copy=False)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).ravel()
    bv = np.asarray(b, dtype=np.float64).ravel()
    if av.size != bv.size or av.size == 0 or np.std(av) == 0.0 or np.std(bv) == 0.0:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def _edge_mae(model: np.ndarray, true: np.ndarray) -> float:
    gz_model, gx_model = np.gradient(model.astype(np.float64))
    gz_true, gx_true = np.gradient(true.astype(np.float64))
    return float(np.mean(np.abs(np.hypot(gz_model, gx_model) - np.hypot(gz_true, gx_true))))


def _model_metrics(model: np.ndarray, true: np.ndarray) -> dict[str, float]:
    diff = model.astype(np.float64) - true.astype(np.float64)
    return {
        "model_mae": float(np.mean(np.abs(diff))),
        "model_rmse": float(np.sqrt(np.mean(diff * diff))),
        "edge_mae": _edge_mae(model, true),
    }


def _gate_metrics(name: str, gate: np.ndarray, delta: np.ndarray, initial: np.ndarray, true: np.ndarray, ecg: np.ndarray) -> dict[str, float | str]:
    gated_model = initial + gate * delta
    update = gate * delta
    metrics = {
        "name": name,
        "update_l2": float(np.linalg.norm(update.astype(np.float64))),
        "alpha_min": float(np.min(gate)),
        "alpha_mean": float(np.mean(gate)),
        "alpha_max": float(np.max(gate)),
        "active_fraction": float(np.mean(gate > 1.0e-6)),
        "corr_with_ecg_gate": _safe_corr(gate, ecg),
    }
    metrics.update(_model_metrics(gated_model, true))
    return metrics


def _imshow(ax, array: np.ndarray, title: str, cmap: str, vmin: float | None = None, vmax: float | None = None) -> None:
    im = ax.imshow(array, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-ready ECG gate/control comparison figure and metrics.")
    parser.add_argument("--result-dir", type=Path, default=ROOT / "rtm_acoustic" / "outputs" / "salt_reliability_gate_v1")
    parser.add_argument("--fwi-dir", type=Path, default=ROOT / "rtm_acoustic" / "outputs" / "FWI" / "full_salt_fwi_cg_allshots_ecg_v1")
    parser.add_argument("--label", default="allshots")
    args = parser.parse_args()

    result_dir = args.result_dir
    diagnostics_dir = result_dir / "diagnostics"
    gates_dir = result_dir / "gates"
    figures_dir = result_dir / "figures"
    tables_dir = result_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    true = _load(args.fwi_dir / "full_salt_true_model.npy")
    initial = _load(args.fwi_dir / "full_salt_initial_model.npy")
    inverted = _load(args.fwi_dir / "full_salt_inverted_model.npy")
    delta = _load(diagnostics_dir / "delta_model.npy")
    illumination = _load(diagnostics_dir / "illumination_score.npy")
    consensus = _load(diagnostics_dir / "gradient_consensus.npy")
    descent = _load(diagnostics_dir / "descent_alignment.npy")
    reliability = _load(diagnostics_dir / "ecg_reliability_score.npy")

    gate_names = [
        "global_matched",
        "illumination_only_matched",
        "gradient_consensus_only_matched",
        "ecg_reliability_gate",
        "inverse_illumination_negative_control",
        "depth_matched",
        "random_matched_seed_0",
        "random_matched_seed_1",
        "random_matched_seed_2",
        "random_matched_seed_3",
        "random_matched_seed_4",
    ]
    gates = {name: _load(gates_dir / f"{name}.npy") for name in gate_names}
    ecg = gates["ecg_reliability_gate"]

    rows = [
        {
            "name": "initial_model",
            "update_l2": 0.0,
            "alpha_min": 0.0,
            "alpha_mean": 0.0,
            "alpha_max": 0.0,
            "active_fraction": 0.0,
            "corr_with_ecg_gate": float("nan"),
            **_model_metrics(initial, true),
        },
        {
            "name": "full_fwi_model",
            "update_l2": float(np.linalg.norm((inverted - initial).astype(np.float64))),
            "alpha_min": 1.0,
            "alpha_mean": 1.0,
            "alpha_max": 1.0,
            "active_fraction": 1.0,
            "corr_with_ecg_gate": float("nan"),
            **_model_metrics(inverted, true),
        },
    ]
    rows.extend(_gate_metrics(name, gate, delta, initial, true, ecg) for name, gate in gates.items())

    csv_path = tables_dir / f"gate_control_metrics_{args.label}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_lines = ["# Gate Control Metrics", "", "| name | update_l2 | alpha_max | active_fraction | model_mae | model_rmse | edge_mae | corr_with_ecg_gate |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        md_lines.append(
            f"| {row['name']} | {row['update_l2']:.6g} | {row['alpha_max']:.6g} | {row['active_fraction']:.4f} | "
            f"{row['model_mae']:.6g} | {row['model_rmse']:.6g} | {row['edge_mae']:.6g} | {row['corr_with_ecg_gate']:.4f} |"
        )
    md_path = tables_dir / f"gate_control_metrics_{args.label}.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(3, 4, figsize=(15.5, 9.2), constrained_layout=True)
    panels = [
        (illumination, "a Illumination", "viridis"),
        (consensus, "b Cross-shot consensus", "viridis"),
        (descent, "c Descent alignment", "viridis"),
        (reliability, "d ECG reliability Q", "viridis"),
        (gates["global_matched"], "e Global matched", "magma"),
        (gates["illumination_only_matched"], "f Illumination-only", "magma"),
        (gates["gradient_consensus_only_matched"], "g Consensus-only", "magma"),
        (gates["ecg_reliability_gate"], "h ECG gate", "magma"),
        (gates["inverse_illumination_negative_control"], "i Inverse illumination", "magma"),
        (gates["depth_matched"], "j Depth matched", "magma"),
        (np.mean([gates[f"random_matched_seed_{i}"] for i in range(5)], axis=0), "k Random mean", "magma"),
        (initial + gates["ecg_reliability_gate"] * delta - initial, "l ECG accepted update", "seismic"),
    ]
    for ax, (array, title, cmap) in zip(axes.ravel(), panels):
        if cmap == "seismic":
            vmax = float(np.percentile(np.abs(array), 99.0))
            _imshow(ax, array, title, cmap, -vmax, vmax)
        else:
            _imshow(ax, array, title, cmap)
    fig.suptitle("Evidence-calibrated gate components and matched controls", fontsize=13)
    fig_path = figures_dir / f"figure_gate_controls_{args.label}.png"
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    summary = {
        "status": "READY",
        "label": args.label,
        "figure": str(fig_path),
        "csv": str(csv_path),
        "markdown": str(md_path),
        "gate_count": len(gates),
        "note": "Metrics use initial + alpha(x,z) * final-iteration delta_model for matched gate comparison.",
    }
    (result_dir / "figures" / f"gate_control_summary_{args.label}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
