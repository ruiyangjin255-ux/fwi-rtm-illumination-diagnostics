from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from rtm_acoustic.evaluate_fwi_model_quality import evaluate_model_quality


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "rtm_acoustic" / "outputs" / "FWI" / "full_salt_fwi_cg_allshots_v2"
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "update_scale_optimization"
DEFAULT_ALPHAS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


def _score(metrics: dict[str, Any]) -> float:
    return float(
        metrics["mae_improvement_fraction"]
        + metrics["rmse_improvement_fraction"]
        + metrics["edge_mae_improvement_fraction"]
        + 0.25 * metrics["gradient_mae_improvement_fraction"]
    )


def optimize_update_scale(
    *,
    true_model: np.ndarray,
    initial_model: np.ndarray,
    inverted_model: np.ndarray,
    alphas: list[float],
    edge_tolerance: float = 0.0,
    gradient_tolerance: float = 0.05,
) -> dict[str, Any]:
    update = np.asarray(inverted_model, dtype=np.float32) - np.asarray(initial_model, dtype=np.float32)
    candidates: list[dict[str, Any]] = []
    for alpha in alphas:
        scaled_model = initial_model + np.float32(alpha) * update
        metrics = evaluate_model_quality(
            true_model=true_model,
            initial_model=initial_model,
            inverted_model=scaled_model,
            update=np.float32(alpha) * update,
        )
        edge_ok = metrics["edge_mae_improvement_fraction"] >= -abs(edge_tolerance)
        gradient_ok = metrics["gradient_mae_improvement_fraction"] >= -abs(gradient_tolerance)
        model_ok = metrics["mae_improvement_fraction"] > 0.0 and metrics["rmse_improvement_fraction"] > 0.0
        row = {
            "alpha": float(alpha),
            "score": _score(metrics),
            "accepted": bool(model_ok and edge_ok and gradient_ok),
            **metrics,
        }
        candidates.append(row)

    accepted = [row for row in candidates if row["accepted"]]
    pool = accepted if accepted else candidates
    best = max(pool, key=lambda row: (row["score"], row["mae_improvement_fraction"], -abs(row["alpha"])))
    selected_model = initial_model + np.float32(best["alpha"]) * update
    return {
        "selected_alpha": best["alpha"],
        "selected_score": best["score"],
        "selection_rule": (
            "maximize model-quality score with positive MAE/RMSE improvement, "
            "non-degraded edge MAE within tolerance, and bounded gradient degradation"
        ),
        "edge_tolerance": float(edge_tolerance),
        "gradient_tolerance": float(gradient_tolerance),
        "candidates": candidates,
        "selected_model": selected_model.astype(np.float32),
    }


def optimize_run_dir(
    run_dir: Path,
    *,
    output_dir: Path,
    alphas: list[float],
    edge_tolerance: float = 0.0,
    gradient_tolerance: float = 0.05,
) -> dict[str, Path]:
    true_model = np.load(run_dir / "full_salt_true_model.npy")
    initial_model = np.load(run_dir / "full_salt_initial_model.npy")
    inverted_model = np.load(run_dir / "full_salt_inverted_model.npy")
    result = optimize_update_scale(
        true_model=true_model,
        initial_model=initial_model,
        inverted_model=inverted_model,
        alphas=alphas,
        edge_tolerance=edge_tolerance,
        gradient_tolerance=gradient_tolerance,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "selected_fwi_model.npy"
    np.save(model_path, result.pop("selected_model"))

    json_path = output_dir / "update_scale_optimization.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    csv_path = output_dir / "update_scale_optimization.csv"
    rows = result["candidates"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_dir / "update_scale_optimization.md"
    lines = [
        "# FWI update-scale optimization",
        "",
        f"- `selected_alpha`: {result['selected_alpha']}",
        f"- `selected_score`: {result['selected_score']}",
        f"- `selected_model`: {model_path}",
        "",
        "| alpha | accepted | score | mae_improvement_pct | rmse_improvement_pct | edge_mae_improvement_pct | gradient_mae_improvement_pct | verdict |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{row['alpha']:.4g}",
                    str(row["accepted"]),
                    f"{row['score']:.8f}",
                    f"{row['mae_improvement_fraction'] * 100:.4f}",
                    f"{row['rmse_improvement_fraction'] * 100:.4f}",
                    f"{row['edge_mae_improvement_fraction'] * 100:.4f}",
                    f"{row['gradient_mae_improvement_fraction'] * 100:.4f}",
                    str(row["verdict"]),
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"json": json_path, "csv": csv_path, "markdown": md_path, "model": model_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a damped FWI update scale using model-quality diagnostics.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    parser.add_argument("--edge-tolerance", type=float, default=0.0)
    parser.add_argument("--gradient-tolerance", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = optimize_run_dir(
        args.run_dir,
        output_dir=args.output_dir,
        alphas=list(args.alphas),
        edge_tolerance=args.edge_tolerance,
        gradient_tolerance=args.gradient_tolerance,
    )
    for label, path in written.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
