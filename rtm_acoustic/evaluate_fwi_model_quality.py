from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "rtm_acoustic" / "outputs" / "FWI" / "full_salt_fwi_cg_allshots_v2"
DEFAULT_OUTPUT = DEFAULT_RUN_DIR / "fwi_model_quality.json"


def _gradient_magnitude(model: np.ndarray) -> np.ndarray:
    dz, dx = np.gradient(np.asarray(model, dtype=np.float64))
    return np.sqrt(dx * dx + dz * dz)


def _edge_mask(model: np.ndarray, percentile: float) -> np.ndarray:
    grad = _gradient_magnitude(model)
    threshold = float(np.percentile(grad, percentile))
    return grad >= threshold


def _mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    if mask is not None:
        diff = diff[mask]
    return float(np.mean(diff)) if diff.size else 0.0


def _rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if mask is not None:
        diff = diff[mask]
    return float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0


def _improvement(initial_value: float, final_value: float) -> float:
    return float((initial_value - final_value) / max(abs(initial_value), 1.0e-20))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).ravel()
    bv = np.asarray(b, dtype=np.float64).ravel()
    if av.size != bv.size or av.size == 0 or np.std(av) == 0.0 or np.std(bv) == 0.0:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def evaluate_model_quality(
    *,
    true_model: np.ndarray,
    initial_model: np.ndarray,
    inverted_model: np.ndarray,
    update: np.ndarray | None = None,
    edge_percentile: float = 90.0,
) -> dict[str, Any]:
    true_model = np.asarray(true_model, dtype=np.float32)
    initial_model = np.asarray(initial_model, dtype=np.float32)
    inverted_model = np.asarray(inverted_model, dtype=np.float32)
    if true_model.shape != initial_model.shape or true_model.shape != inverted_model.shape:
        raise ValueError("true_model, initial_model and inverted_model must have the same shape")
    if update is None:
        update = inverted_model - initial_model
    update = np.asarray(update, dtype=np.float32)
    if update.shape != true_model.shape:
        raise ValueError("update must have the same shape as the models")

    true_grad = _gradient_magnitude(true_model)
    initial_grad = _gradient_magnitude(initial_model)
    inverted_grad = _gradient_magnitude(inverted_model)
    edge_mask = _edge_mask(true_model, edge_percentile)
    non_edge_mask = ~edge_mask
    update_abs = np.abs(update.astype(np.float64))
    total_update_l1 = float(np.sum(update_abs))
    edge_update_l1 = float(np.sum(update_abs[edge_mask]))

    initial_mae = _mae(initial_model, true_model)
    inverted_mae = _mae(inverted_model, true_model)
    initial_rmse = _rmse(initial_model, true_model)
    inverted_rmse = _rmse(inverted_model, true_model)
    initial_edge_mae = _mae(initial_model, true_model, edge_mask)
    inverted_edge_mae = _mae(inverted_model, true_model, edge_mask)
    initial_non_edge_mae = _mae(initial_model, true_model, non_edge_mask)
    inverted_non_edge_mae = _mae(inverted_model, true_model, non_edge_mask)
    initial_gradient_mae = _mae(initial_grad, true_grad)
    inverted_gradient_mae = _mae(inverted_grad, true_grad)

    verdict = "improved" if inverted_mae < initial_mae and inverted_rmse < initial_rmse else "not_improved"
    if inverted_mae < initial_mae and inverted_gradient_mae > initial_gradient_mae:
        verdict = "numerical_improvement_without_gradient_improvement"

    return {
        "model_shape": [int(true_model.shape[0]), int(true_model.shape[1])],
        "edge_percentile": float(edge_percentile),
        "edge_fraction": float(np.mean(edge_mask)),
        "initial_mae": initial_mae,
        "inverted_mae": inverted_mae,
        "mae_improvement_fraction": _improvement(initial_mae, inverted_mae),
        "initial_rmse": initial_rmse,
        "inverted_rmse": inverted_rmse,
        "rmse_improvement_fraction": _improvement(initial_rmse, inverted_rmse),
        "initial_edge_mae": initial_edge_mae,
        "inverted_edge_mae": inverted_edge_mae,
        "edge_mae_improvement_fraction": _improvement(initial_edge_mae, inverted_edge_mae),
        "initial_non_edge_mae": initial_non_edge_mae,
        "inverted_non_edge_mae": inverted_non_edge_mae,
        "non_edge_mae_improvement_fraction": _improvement(initial_non_edge_mae, inverted_non_edge_mae),
        "initial_gradient_mae": initial_gradient_mae,
        "inverted_gradient_mae": inverted_gradient_mae,
        "gradient_mae_improvement_fraction": _improvement(initial_gradient_mae, inverted_gradient_mae),
        "update_l1_total": total_update_l1,
        "update_l1_edge_fraction": float(edge_update_l1 / max(total_update_l1, 1.0e-20)),
        "update_true_error_correlation": _safe_corr(update, true_model - initial_model),
        "verdict": verdict,
    }


def evaluate_run_dir(run_dir: Path, edge_percentile: float = 90.0) -> dict[str, Any]:
    true_model = np.load(run_dir / "full_salt_true_model.npy")
    initial_model = np.load(run_dir / "full_salt_initial_model.npy")
    inverted_model = np.load(run_dir / "full_salt_inverted_model.npy")
    update_path = run_dir / "full_salt_model_update.npy"
    update = np.load(update_path) if update_path.exists() else inverted_model - initial_model
    metrics = evaluate_model_quality(
        true_model=true_model,
        initial_model=initial_model,
        inverted_model=inverted_model,
        update=update,
        edge_percentile=edge_percentile,
    )
    metrics["run_dir"] = str(run_dir)
    summary_path = run_dir / "full_salt_fwi_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics["misfit_reduction_fraction"] = summary.get("misfit_reduction_fraction")
        metrics["optimizer"] = summary.get("config", {}).get("optimizer")
        metrics["shot_count"] = summary.get("shot_count")
        metrics["iterations"] = summary.get("iterations")
    return metrics


def write_outputs(metrics: dict[str, Any], output: Path) -> dict[str, Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    md_path = output.with_suffix(".md")
    lines = ["# FWI model quality metrics", ""]
    for key, value in metrics.items():
        lines.append(f"- `{key}`: {value}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": output, "csv": csv_path, "markdown": md_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FWI model quality beyond the misfit curve.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--edge-percentile", type=float, default=90.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_run_dir(args.run_dir, edge_percentile=args.edge_percentile)
    written = write_outputs(metrics, args.output)
    for label, path in written.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
