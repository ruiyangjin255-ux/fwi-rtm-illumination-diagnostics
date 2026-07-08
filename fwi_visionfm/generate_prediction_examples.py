from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from fwi_visionfm.split_utils import load_split_paths


MODEL_DIR_NAMES = ("torch_cnn_baseline", "dummy_dinov2_frozen", "dummy_dinov2_lora")


def _load_matplotlib():
    try:
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:
        raise RuntimeError("需要安装 matplotlib 才能生成预测图: python -m pip install matplotlib") from exc


def _metric_row(index: int, true: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    error = pred - true
    return {
        "sample_index": index,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "error_mean": float(np.mean(error)),
        "error_std": float(np.std(error)),
    }


def _first_shot_from_sample(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    with np.load(path) as data:
        if "records" not in data:
            return None
        records = np.asarray(data["records"], dtype=np.float32)
    if records.ndim != 3:
        return None
    return records[0]


def _plot_sample(path: Path, *, shot: np.ndarray | None, true: np.ndarray, pred: np.ndarray, title: str) -> None:
    plt = _load_matplotlib()
    error = np.abs(pred - true)
    vmin = float(min(np.min(true), np.min(pred)))
    vmax = float(max(np.max(true), np.max(pred)))
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 3.8), constrained_layout=True)
    panels = [
        (axes[0], shot if shot is not None else np.zeros_like(true), "Shot gather", "seismic", None, None),
        (axes[1], true, "True velocity", "viridis", vmin, vmax),
        (axes[2], pred, "Predicted velocity", "viridis", vmin, vmax),
        (axes[3], error, "Absolute error", "coolwarm", None, None),
    ]
    for ax, array, panel_title, cmap, panel_vmin, panel_vmax in panels:
        im = ax.imshow(array, aspect="auto", cmap=cmap, vmin=panel_vmin, vmax=panel_vmax)
        ax.set_title(panel_title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_grid(path: Path, arrays: list[np.ndarray], titles: list[str], *, cmap: str, vmin=None, vmax=None) -> None:
    plt = _load_matplotlib()
    count = max(1, len(arrays))
    fig, axes = plt.subplots(1, count, figsize=(4.2 * count, 3.8), constrained_layout=True)
    if count == 1:
        axes = [axes]
    for ax, array, title in zip(axes, arrays, titles):
        im = ax.imshow(array, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_velocity_profiles(path: Path, true_arrays: list[np.ndarray], pred_arrays: list[np.ndarray], titles: list[str]) -> None:
    plt = _load_matplotlib()
    fig, axes = plt.subplots(1, len(true_arrays), figsize=(4.2 * len(true_arrays), 4.0), constrained_layout=True)
    if len(true_arrays) == 1:
        axes = [axes]
    for ax, true, pred, title in zip(axes, true_arrays, pred_arrays, titles):
        width = true.shape[1]
        positions = sorted(set([0, width // 4, width // 2, max(0, width - 1)]))
        depth_axis = np.arange(true.shape[0], dtype=np.float32)
        for pos in positions:
            ax.plot(true[:, pos], depth_axis, label=f"true x={pos}", linewidth=1.5)
            ax.plot(pred[:, pos], depth_axis, linestyle="--", label=f"pred x={pos}", linewidth=1.2)
        ax.set_title(title)
        ax.invert_yaxis()
        ax.set_xlabel("velocity")
        ax.set_ylabel("depth")
        ax.legend(fontsize=7)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _is_mean_biased(true_arrays: list[np.ndarray], pred_arrays: list[np.ndarray]) -> bool:
    true_std = float(np.mean([np.std(array) for array in true_arrays]))
    pred_std = float(np.mean([np.std(array) for array in pred_arrays]))
    return pred_std < max(1.0e-6, true_std * 0.2)


def generate_prediction_examples(
    *,
    split_manifest: str | Path,
    experiment_dir: str | Path,
    model_type: str,
    output_dir: str | Path,
    num_samples: int = 3,
    device: str = "cpu",
) -> dict[str, Any]:
    split_paths = load_split_paths(split_manifest)
    test_paths = split_paths["test"]
    prediction_paths = sorted((Path(experiment_dir) / "predictions").glob("sample_*.npz"))
    if not prediction_paths:
        raise FileNotFoundError(f"缺少预测 npz: {Path(experiment_dir) / 'predictions'}")
    count = min(int(num_samples), len(prediction_paths), len(test_paths) if test_paths else len(prediction_paths))
    if count <= 0:
        raise ValueError("没有可用于绘图的预测样本")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    pred_arrays: list[np.ndarray] = []
    true_arrays: list[np.ndarray] = []
    error_arrays: list[np.ndarray] = []
    titles: list[str] = []
    for index in range(count):
        with np.load(prediction_paths[index]) as data:
            true = np.asarray(data["velocity_true"], dtype=np.float32)
            pred = np.asarray(data["velocity_pred"], dtype=np.float32)
            error = np.abs(np.asarray(data["velocity_error"], dtype=np.float32) if "velocity_error" in data else pred - true)
        shot = _first_shot_from_sample(Path(test_paths[index])) if index < len(test_paths) else None
        row = _metric_row(index, true, pred)
        row["prediction_file"] = str(prediction_paths[index])
        row["test_sample"] = str(test_paths[index]) if index < len(test_paths) else ""
        rows.append(row)
        true_arrays.append(true)
        pred_arrays.append(pred)
        error_arrays.append(error)
        titles.append(f"sample {index:03d}")
        _plot_sample(
            output / f"sample_{index:03d}_comparison.png",
            shot=shot,
            true=true,
            pred=pred,
            title=f"{model_type} sample {index:03d}",
        )

    metrics_csv = output / "test_metrics.csv"
    with metrics_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    vmin = float(min(np.min(array) for array in true_arrays + pred_arrays))
    vmax = float(max(np.max(array) for array in true_arrays + pred_arrays))
    comparison_grid = output / "prediction_grid.png"
    error_grid = output / "error_grid.png"
    velocity_profile_grid = output / "velocity_profile_grid.png"
    _plot_grid(comparison_grid, pred_arrays, titles, cmap="viridis", vmin=vmin, vmax=vmax)
    _plot_grid(error_grid, error_arrays, titles, cmap="coolwarm")
    _plot_velocity_profiles(velocity_profile_grid, true_arrays, pred_arrays, titles)

    avg_mae = float(np.mean([row["mae"] for row in rows]))
    avg_rmse = float(np.mean([row["rmse"] for row in rows]))
    summary_path = output / "prediction_summary.md"
    summary_lines = [
        f"# Prediction Examples: {model_type}",
        "",
        f"- `experiment_dir`: `{Path(experiment_dir)}`",
        f"- `split_manifest`: `{Path(split_manifest)}`",
        f"- `device`: `{device}`",
        f"- `sample_count`: `{count}`",
        f"- `mean_mae`: `{avg_mae}`",
        f"- `mean_rmse`: `{avg_rmse}`",
    ]
    if _is_mean_biased(true_arrays, pred_arrays):
        summary_lines.append('- `notice`: `"prediction appears over-smoothed / mean-biased"`')
    summary_lines.extend(["", "这些图用于检查预测输出、误差结构和数据流，不作为最终科研结论。"])
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    result = {
        "output_dir": str(output),
        "summary_path": str(summary_path),
        "metrics_csv": str(metrics_csv),
        "comparison_grid": str(comparison_grid),
        "error_grid": str(error_grid),
        "velocity_profile_grid": str(velocity_profile_grid),
        "sample_count": count,
        "mean_mae": avg_mae,
        "mean_rmse": avg_rmse,
    }
    (output / "prediction_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def generate_prediction_examples_batch(
    *,
    protocol_dir: str | Path,
    outputs_root: str | Path,
    output_subdir: str,
    num_samples: int = 8,
    device: str = "cpu",
) -> dict[str, Any]:
    protocol_dir = Path(protocol_dir)
    outputs_root = Path(outputs_root)
    rows: list[dict[str, Any]] = []
    for split_manifest in sorted(path for path in protocol_dir.glob("protocol_v1_*.json") if path.name != "protocol_v1_summary.json"):
        experiment_name = split_manifest.stem
        for model_name in MODEL_DIR_NAMES:
            experiment_dir = outputs_root / experiment_name / model_name
            output_dir = experiment_dir / output_subdir / model_name
            try:
                result = generate_prediction_examples(
                    split_manifest=split_manifest,
                    experiment_dir=experiment_dir,
                    model_type=model_name,
                    output_dir=output_dir,
                    num_samples=num_samples,
                    device=device,
                )
                rows.append(
                    {
                        "experiment": experiment_name,
                        "model_name": model_name,
                        "status": "completed",
                        "output_dir": str(output_dir),
                        "sample_count": result["sample_count"],
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "experiment": experiment_name,
                        "model_name": model_name,
                        "status": "failed",
                        "output_dir": str(output_dir),
                        "sample_count": 0,
                        "error": str(exc),
                    }
                )
    return {"count": len(rows), "rows": rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从已保存 prediction npz 生成速度预测对比图。")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--model-type")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--protocol-dir", type=Path)
    parser.add_argument("--outputs-root", type=Path)
    parser.add_argument("--output-subdir", default="prediction_examples")
    parser.add_argument("--batch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch:
        if args.protocol_dir is None or args.outputs_root is None:
            raise SystemExit("错误: --batch 模式需要提供 --protocol-dir 和 --outputs-root。")
        result = generate_prediction_examples_batch(
            protocol_dir=args.protocol_dir,
            outputs_root=args.outputs_root,
            output_subdir=args.output_subdir,
            num_samples=args.num_samples,
            device=args.device,
        )
        print(f"批量预测图记录数: {result['count']}")
        return
    if args.split_manifest is None or args.experiment_dir is None or args.model_type is None or args.output_dir is None:
        raise SystemExit("错误: 单实验模式需要提供 --split-manifest --experiment-dir --model-type --output-dir。")
    result = generate_prediction_examples(
        split_manifest=args.split_manifest,
        experiment_dir=args.experiment_dir,
        model_type=args.model_type,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        device=args.device,
    )
    print(f"写出预测示例: {result['output_dir']}")
    print(f"mean_mae/mean_rmse: {result['mean_mae']:.6f}/{result['mean_rmse']:.6f}")


if __name__ == "__main__":
    main()
