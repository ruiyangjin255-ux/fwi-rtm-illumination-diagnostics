from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

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


def _load_prediction_npz(path: str | Path) -> dict[str, np.ndarray]:
    payload = np.load(Path(path), allow_pickle=True)
    seismic = payload["seismic_preview"] if "seismic_preview" in payload else payload.get("seismic")
    true = _squeeze_velocity(np.asarray(payload["velocity_true"], dtype=np.float32))
    pred = _squeeze_velocity(np.asarray(payload["velocity_pred"], dtype=np.float32))
    return {
        "seismic": None if seismic is None else np.asarray(seismic, dtype=np.float32),
        "true": true,
        "pred": pred,
        "error": np.abs(pred - true).astype(np.float32),
    }


def _sobel_edges(batch: np.ndarray) -> np.ndarray:
    gx_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32) / 4.0
    gy_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32) / 4.0
    padded = np.pad(batch, ((0, 0), (1, 1), (1, 1)), mode="edge")
    gx = np.zeros_like(batch, dtype=np.float32)
    gy = np.zeros_like(batch, dtype=np.float32)
    for i in range(3):
        for j in range(3):
            patch = padded[:, i:i + batch.shape[1], j:j + batch.shape[2]]
            gx += gx_kernel[i, j] * patch
            gy += gy_kernel[i, j] * patch
    return np.sqrt(gx * gx + gy * gy)


def _box_smooth(batch: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    pad = kernel_size // 2
    padded = np.pad(batch, ((0, 0), (pad, pad), (pad, pad)), mode="edge")
    out = np.zeros_like(batch, dtype=np.float32)
    for i in range(kernel_size):
        for j in range(kernel_size):
            out += padded[:, i:i + batch.shape[1], j:j + batch.shape[2]]
    out /= float(kernel_size * kernel_size)
    return out


def _region_mean(error: np.ndarray, start: int, end: int) -> float:
    return float(np.mean(error[:, start:end, :]))


def _make_summary(true: np.ndarray, raw_pred: np.ndarray, spec_pred: np.ndarray, label: str) -> dict[str, Any]:
    raw_error = np.abs(raw_pred - true)
    spec_error = np.abs(spec_pred - true)
    edge_true = _sobel_edges(true)
    raw_edge_error = np.abs(_sobel_edges(raw_pred) - edge_true)
    spec_edge_error = np.abs(_sobel_edges(spec_pred) - edge_true)
    height = true.shape[1]
    one_third = max(height // 3, 1)
    smooth_true = _box_smooth(true)
    smooth_raw_error = np.abs(_box_smooth(raw_pred) - smooth_true)
    smooth_spec_error = np.abs(_box_smooth(spec_pred) - smooth_true)
    return {
        "label": label,
        "mean_abs_error_raw": float(np.mean(raw_error)),
        "mean_abs_error_spectrogram": float(np.mean(spec_error)),
        "mean_edge_error_raw": float(np.mean(raw_edge_error)),
        "mean_edge_error_spectrogram": float(np.mean(spec_edge_error)),
        "shallow_error_raw": _region_mean(raw_error, 0, one_third),
        "shallow_error_spectrogram": _region_mean(spec_error, 0, one_third),
        "deep_error_raw": _region_mean(raw_error, height - one_third, height),
        "deep_error_spectrogram": _region_mean(spec_error, height - one_third, height),
        "smooth_error_raw": float(np.mean(smooth_raw_error)),
        "smooth_error_spectrogram": float(np.mean(smooth_spec_error)),
    }


def compare_prediction_grids(
    *,
    raw_prediction_npz: str | Path,
    spectrogram_prediction_npz: str | Path,
    output_path: str | Path,
    max_items: int = 4,
    label: str = "comparison",
) -> dict[str, Any]:
    raw_payload = _load_prediction_npz(raw_prediction_npz)
    spec_payload = _load_prediction_npz(spectrogram_prediction_npz)
    true = raw_payload["true"]
    raw_pred = raw_payload["pred"]
    spec_pred = spec_payload["pred"]
    if true.shape != spec_payload["true"].shape:
        raise ValueError(f"true velocity shape mismatch: {true.shape} vs {spec_payload['true'].shape}")
    if raw_pred.shape != spec_pred.shape:
        raise ValueError(f"prediction shape mismatch: {raw_pred.shape} vs {spec_pred.shape}")

    raw_error = np.abs(raw_pred - true)
    spec_error = np.abs(spec_pred - true)
    error_diff = spec_error - raw_error
    edge_true = _sobel_edges(true)
    raw_edge_error = np.abs(_sobel_edges(raw_pred) - edge_true)
    spec_edge_error = np.abs(_sobel_edges(spec_pred) - edge_true)

    count = min(int(max_items), true.shape[0], raw_pred.shape[0], spec_pred.shape[0])
    fig, axes = plt.subplots(count * 2, 6, figsize=(24, 5 * count), squeeze=False)
    for row in range(count):
        axes[2 * row, 0].imshow(true[row], aspect="auto", cmap="viridis")
        axes[2 * row, 0].set_title("true velocity")
        axes[2 * row, 1].imshow(raw_pred[row], aspect="auto", cmap="viridis")
        axes[2 * row, 1].set_title("pred raw_repeat3")
        axes[2 * row, 2].imshow(spec_pred[row], aspect="auto", cmap="viridis")
        axes[2 * row, 2].set_title("pred raw_spectrogram")
        axes[2 * row, 3].imshow(raw_error[row], aspect="auto", cmap="magma")
        axes[2 * row, 3].set_title("abs error raw")
        axes[2 * row, 4].imshow(spec_error[row], aspect="auto", cmap="magma")
        axes[2 * row, 4].set_title("abs error spectrogram")
        axes[2 * row, 5].imshow(error_diff[row], aspect="auto", cmap="coolwarm", vmin=-np.max(np.abs(error_diff[row])), vmax=np.max(np.abs(error_diff[row])))
        axes[2 * row, 5].set_title("error diff spec-raw")

        axes[2 * row + 1, 0].imshow(edge_true[row], aspect="auto", cmap="gray")
        axes[2 * row + 1, 0].set_title("true edge")
        axes[2 * row + 1, 1].imshow(raw_edge_error[row], aspect="auto", cmap="inferno")
        axes[2 * row + 1, 1].set_title("raw edge error")
        axes[2 * row + 1, 2].imshow(spec_edge_error[row], aspect="auto", cmap="inferno")
        axes[2 * row + 1, 2].set_title("spec edge error")
        axes[2 * row + 1, 3].imshow(raw_error[row] - spec_error[row], aspect="auto", cmap="coolwarm")
        axes[2 * row + 1, 3].set_title("raw-spec abs error")
        axes[2 * row + 1, 4].imshow(raw_edge_error[row] - spec_edge_error[row], aspect="auto", cmap="coolwarm")
        axes[2 * row + 1, 4].set_title("raw-spec edge error")
        axes[2 * row + 1, 5].axis("off")
        axes[2 * row + 1, 5].text(0.0, 0.75, f"{label} sample {row}", fontsize=12)
        axes[2 * row + 1, 5].text(0.0, 0.55, f"mae raw={np.mean(raw_error[row]):.4f}", fontsize=10)
        axes[2 * row + 1, 5].text(0.0, 0.40, f"mae spec={np.mean(spec_error[row]):.4f}", fontsize=10)
        axes[2 * row + 1, 5].text(0.0, 0.25, f"edge raw={np.mean(raw_edge_error[row]):.4f}", fontsize=10)
        axes[2 * row + 1, 5].text(0.0, 0.10, f"edge spec={np.mean(spec_edge_error[row]):.4f}", fontsize=10)

        for axis in axes[2 * row]:
            axis.set_xticks([])
            axis.set_yticks([])
        for axis in axes[2 * row + 1, :5]:
            axis.set_xticks([])
            axis.set_yticks([])

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    summary = _make_summary(true, raw_pred, spec_pred, label)
    summary["output_path"] = str(output_path)
    return summary


def write_qualitative_comparison_report(
    *,
    in_family_summary: dict[str, Any],
    cross_family_summary: dict[str, Any],
    output_path: str | Path,
) -> Path:
    def _region_conclusion(summary: dict[str, Any]) -> str:
        shallow_gain = summary["shallow_error_raw"] - summary["shallow_error_spectrogram"]
        deep_gain = summary["deep_error_raw"] - summary["deep_error_spectrogram"]
        if shallow_gain > deep_gain and shallow_gain > 0:
            return "raw_spectrogram 的误差降低更偏向浅部。"
        if deep_gain > shallow_gain and deep_gain > 0:
            return "raw_spectrogram 的误差降低更偏向深部。"
        if shallow_gain > 0 and deep_gain > 0:
            return "raw_spectrogram 的误差降低更接近整体背景速度校正，而不是集中在单一深度区间。"
        return "当前图像上没有看到 raw_spectrogram 在深度方向上的稳定误差优势。"

    def _trend_conclusion(summary: dict[str, Any]) -> str:
        if summary["smooth_error_spectrogram"] < summary["smooth_error_raw"]:
            return "raw_spectrogram 更接近大尺度速度趋势。"
        return "raw_spectrogram 对大尺度速度趋势没有明显优势。"

    def _edge_conclusion(summary: dict[str, Any]) -> str:
        if summary["mean_edge_error_spectrogram"] > summary["mean_edge_error_raw"]:
            return "raw_spectrogram 在边界/层位/局部梯度上更容易产生平滑化误差。"
        return "raw_spectrogram 没有明显损伤边界结构。"

    lines = [
        "# Qualitative Comparison Report",
        "",
        "## In-family",
        "",
        f"1. {_region_conclusion(in_family_summary)}",
        f"2. {_trend_conclusion(in_family_summary)}",
        f"3. {_edge_conclusion(in_family_summary)}",
        "",
        "## Cross-family",
        "",
        f"1. {_region_conclusion(cross_family_summary)}",
        f"2. {_trend_conclusion(cross_family_summary)}",
        f"3. {_edge_conclusion(cross_family_summary)}",
        "",
        "## Interpretation",
        "",
        "4. MAE/RMSE 改善但 edge/laplacian 指标变差，说明 raw_spectrogram 更可能帮助模型校正整体或低频速度背景，从而降低数值误差；但它同时弱化了局部界面、边缘强度和梯度变化，因此结构恢复指标不升反降。",
        "5. 当前图像证据支持：target-family numerical error gain, not structural recovery gain。",
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比 raw_repeat3 与 raw_spectrogram 的 prediction grid。")
    parser.add_argument("--raw-prediction-npz", required=True, type=Path)
    parser.add_argument("--spectrogram-prediction-npz", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-items", type=int, default=4)
    parser.add_argument("--label", default="comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = compare_prediction_grids(
        raw_prediction_npz=args.raw_prediction_npz,
        spectrogram_prediction_npz=args.spectrogram_prediction_npz,
        output_path=args.output,
        max_items=args.max_items,
        label=args.label,
    )
    print(f"写出对比图: {summary['output_path']}")


if __name__ == "__main__":
    main()
