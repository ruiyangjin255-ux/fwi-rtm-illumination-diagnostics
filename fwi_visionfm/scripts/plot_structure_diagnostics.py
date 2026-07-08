from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def _load_matplotlib():
    try:
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("需要安装 matplotlib 才能生成结构诊断图") from exc


def _gradient_magnitude(array: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(array.astype(np.float32), axis=(-2, -1))
    return np.sqrt(gx * gx + gy * gy)


def _load_prediction_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as payload:
        prediction = np.asarray(payload["velocity_pred_physical"] if "velocity_pred_physical" in payload else payload["prediction"], dtype=np.float32)
        target = np.asarray(payload["velocity_true_physical"] if "velocity_true_physical" in payload else payload["target"], dtype=np.float32)
        seismic = None
        for key in ("seismic_preview", "records_preview", "shot_gather", "records"):
            if key in payload:
                seismic = np.asarray(payload[key], dtype=np.float32)
                break
    return {"prediction": prediction, "target": target, "seismic": seismic}


def _select_indices(prediction: np.ndarray, target: np.ndarray) -> list[tuple[str, int]]:
    sample_mae = np.mean(np.abs(prediction - target), axis=(-2, -1))
    order = np.argsort(sample_mae)
    return [
        ("best", int(order[0])),
        ("median", int(order[len(order) // 2])),
        ("worst", int(order[-1])),
    ]


def _extract_shot(seismic: np.ndarray | None, index: int, fallback_shape: tuple[int, int]) -> np.ndarray:
    if seismic is None:
        return np.zeros(fallback_shape, dtype=np.float32)
    array = np.asarray(seismic[index], dtype=np.float32)
    if array.ndim == 3:
        return array[0]
    if array.ndim == 2:
        return array
    return np.zeros(fallback_shape, dtype=np.float32)


def _plot_prediction_grid(path: Path, *, shot: np.ndarray, true: np.ndarray, pred: np.ndarray, error: np.ndarray) -> None:
    plt = _load_matplotlib()
    vmin = float(min(np.min(true), np.min(pred)))
    vmax = float(max(np.max(true), np.max(pred)))
    evmax = float(np.max(error)) if float(np.max(error)) > 0.0 else 1.0
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    panels = [
        (shot, "shot gather", "seismic", None, None),
        (true, "true velocity", "viridis", vmin, vmax),
        (pred, "predicted velocity", "viridis", vmin, vmax),
        (error, "absolute error", "magma", 0.0, evmax),
    ]
    for ax, (array, title, cmap, pvmin, pvmax) in zip(axes, panels):
        im = ax.imshow(array, aspect="auto", cmap=cmap, vmin=pvmin, vmax=pvmax)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_gradient_grid(path: Path, *, grad_true: np.ndarray, grad_pred: np.ndarray, grad_error: np.ndarray) -> None:
    plt = _load_matplotlib()
    vmax = float(max(np.max(grad_true), np.max(grad_pred)))
    evmax = float(np.max(grad_error)) if float(np.max(grad_error)) > 0.0 else 1.0
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    panels = [
        (grad_true, "true gradient magnitude", "viridis", 0.0, vmax),
        (grad_pred, "predicted gradient magnitude", "viridis", 0.0, vmax),
        (grad_error, "gradient error map", "magma", 0.0, evmax),
    ]
    for ax, (array, title, cmap, pvmin, pvmax) in zip(axes, panels):
        im = ax.imshow(array, aspect="auto", cmap=cmap, vmin=pvmin, vmax=pvmax)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_structure_diagnostics(
    *,
    predictions_path: str | Path,
    metrics_path: str | Path,
    output_dir: str | Path,
    prefix: str,
) -> dict[str, Any]:
    payload = _load_prediction_payload(Path(predictions_path))
    prediction = payload["prediction"]
    target = payload["target"]
    seismic = payload["seismic"]
    if prediction.ndim == 4 and prediction.shape[1] == 1:
        prediction = prediction[:, 0]
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    indices = _select_indices(prediction, target)
    output_root = Path(output_dir)
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8")) if Path(metrics_path).exists() else {}
    prediction_grids: list[str] = []
    gradient_grids: list[str] = []
    for label, index in indices:
        true = np.asarray(target[index], dtype=np.float32)
        pred = np.asarray(prediction[index], dtype=np.float32)
        shot = _extract_shot(seismic, index, (true.shape[-2], true.shape[-1]))
        error = np.abs(pred - true).astype(np.float32)
        grad_true = _gradient_magnitude(true)
        grad_pred = _gradient_magnitude(pred)
        grad_error = np.abs(grad_pred - grad_true).astype(np.float32)
        pred_path = output_root / f"{prefix}_{label}_prediction_grid.png"
        grad_path = output_root / f"{prefix}_{label}_gradient_grid.png"
        _plot_prediction_grid(pred_path, shot=shot, true=true, pred=pred, error=error)
        _plot_gradient_grid(grad_path, grad_true=grad_true, grad_pred=grad_pred, grad_error=grad_error)
        prediction_grids.append(str(pred_path))
        gradient_grids.append(str(grad_path))
    result = {
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "metric_space": metrics.get("metric_space", ""),
        "prediction_grids": prediction_grids,
        "gradient_grids": gradient_grids,
    }
    (output_root / f"{prefix}_structure_diagnostics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot structure-aware FWI diagnostics from prediction npz.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = plot_structure_diagnostics(
        predictions_path=args.predictions,
        metrics_path=args.metrics,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )
    print(f"Wrote {len(result['prediction_grids']) + len(result['gradient_grids'])} structure diagnostic plots")


if __name__ == "__main__":
    main()
