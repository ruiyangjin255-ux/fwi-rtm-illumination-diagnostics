from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _squeeze_velocity(array: np.ndarray) -> np.ndarray:
    if array.ndim == 4 and array.shape[1] == 1:
        return array[:, 0]
    return array


def plot_openfwi_predictions(*, prediction_npz: str | Path, output_path: str | Path, max_items: int = 4) -> Path:
    prediction_npz = Path(prediction_npz)
    output_path = Path(output_path)
    payload = np.load(prediction_npz, allow_pickle=True)
    seismic = payload["seismic_preview"] if "seismic_preview" in payload else payload.get("seismic")
    true = _squeeze_velocity(np.asarray(payload["velocity_true"], dtype=np.float32))
    pred = _squeeze_velocity(np.asarray(payload["velocity_pred"], dtype=np.float32))
    error = _squeeze_velocity(np.asarray(payload["error_map"], dtype=np.float32)) if "error_map" in payload else np.abs(pred - true)
    count = min(int(max_items), true.shape[0])
    has_seismic = seismic is not None and np.asarray(seismic).size > 0
    cols = 4 if has_seismic else 3
    fig, axes = plt.subplots(count, cols, figsize=(4 * cols, 3 * count), squeeze=False)
    for row in range(count):
        col = 0
        if has_seismic:
            shot = np.asarray(seismic[row], dtype=np.float32)
            if shot.ndim == 4:
                shot = shot[0]
            if shot.ndim == 3:
                shot = shot[0]
            axes[row, col].imshow(shot, aspect="auto", cmap="seismic")
            axes[row, col].set_title(f"shot {row}")
            col += 1
        axes[row, col].imshow(true[row], aspect="auto", cmap="viridis")
        axes[row, col].set_title("true velocity")
        axes[row, col + 1].imshow(pred[row], aspect="auto", cmap="viridis")
        axes[row, col + 1].set_title("predicted velocity")
        axes[row, col + 2].imshow(error[row], aspect="auto", cmap="magma")
        axes[row, col + 2].set_title("absolute error")
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制 OpenFWI 预测结果网格图。")
    parser.add_argument("--prediction-npz", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-items", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = plot_openfwi_predictions(
        prediction_npz=args.prediction_npz,
        output_path=args.output,
        max_items=args.max_items,
    )
    print(f"写出预测图: {output}")


if __name__ == "__main__":
    main()
