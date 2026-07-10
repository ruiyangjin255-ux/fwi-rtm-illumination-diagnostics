from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT_DIR = (
    Path(__file__).resolve().parent
    / "outputs"
    / "seg_salt_multishot_rtm_padded60_full30m_workers4"
)


def _as_float32(image: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(image, dtype=np.float32), copy=False)


def _robust_scale(image: np.ndarray, percentile: float) -> float:
    values = np.abs(_as_float32(image))
    scale = float(np.percentile(values, percentile))
    return scale if np.isfinite(scale) and scale > 0.0 else 1.0


def robust_symmetric_display(
    image: np.ndarray,
    percentile: float = 99.0,
    output_clip: float = 0.95,
) -> np.ndarray:
    """Return a symmetric, clipped display image in a stable amplitude range."""
    data = _as_float32(image).copy()
    scale = _robust_scale(data, percentile)
    data = data / np.float32(scale)
    np.clip(data, -np.float32(output_clip), np.float32(output_clip), out=data)
    return data.astype(np.float32, copy=False)


def mask_low_illumination(
    image: np.ndarray,
    illumination: np.ndarray,
    fraction: float = 0.02,
) -> np.ndarray:
    """Zero RTM samples where source illumination is below a max-relative floor."""
    if fraction < 0.0:
        raise ValueError("fraction must be non-negative")
    data = _as_float32(image).copy()
    illum = _as_float32(illumination)
    max_illum = float(np.max(illum)) if illum.size else 0.0
    if max_illum <= 0.0:
        return np.zeros_like(data, dtype=np.float32)
    data[illum < np.float32(max_illum * fraction)] = 0.0
    return data.astype(np.float32, copy=False)


def depth_balanced_display(
    image: np.ndarray,
    percentile: float = 99.0,
    output_clip: float = 0.95,
    max_gain: float = 3.0,
    eps: float = 1.0e-6,
) -> np.ndarray:
    """Balance each depth row before robust clipping for visual comparison."""
    if max_gain < 1.0:
        raise ValueError("max_gain must be at least 1")
    data = _as_float32(image).copy()
    row_scale = np.percentile(np.abs(data), percentile, axis=1).astype(np.float32)
    positive = row_scale[row_scale > eps]
    reference = np.float32(np.median(positive)) if positive.size else np.float32(1.0)
    row_scale = np.maximum(row_scale, np.float32(eps))
    gain = np.clip(reference / row_scale, 1.0 / np.float32(max_gain), np.float32(max_gain))
    balanced = data * gain[:, None]
    return robust_symmetric_display(balanced, percentile=percentile, output_clip=output_clip)


def soft_threshold_display(
    image: np.ndarray,
    threshold_percentile: float = 45.0,
    clip_percentile: float = 99.2,
    output_clip: float = 0.9,
) -> np.ndarray:
    """Suppress low-amplitude background while keeping reflector polarity."""
    if threshold_percentile < 0.0 or threshold_percentile >= 100.0:
        raise ValueError("threshold_percentile must be in [0, 100)")
    data = _as_float32(image)
    abs_data = np.abs(data)
    nonzero = abs_data[abs_data > 0.0]
    if nonzero.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    threshold = np.float32(np.percentile(nonzero, threshold_percentile))
    enhanced = np.sign(data) * np.maximum(abs_data - threshold, np.float32(0.0))
    return robust_symmetric_display(enhanced, percentile=clip_percentile, output_clip=output_clip)


def make_paper_ready_products(
    filtered: np.ndarray,
    current_display: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build conservative paper-display variants from the existing filtered RTM."""
    conservative = robust_symmetric_display(filtered, percentile=99.4, output_clip=0.9)
    enhanced = soft_threshold_display(filtered, threshold_percentile=42.0, clip_percentile=99.2)
    # Keep the recommendation conservative so the paper figure does not invent
    # apparent continuity through aggressive row-by-row gain.
    recommended = conservative
    return {
        "paper_current_display": _as_float32(current_display),
        "paper_conservative": conservative,
        "paper_enhanced": enhanced,
        "paper_recommended": recommended,
    }


def _zone_energy(image: np.ndarray, axis: int, labels: tuple[str, str, str]) -> dict[str, float]:
    data = _as_float32(image)
    chunks = np.array_split(data, 3, axis=axis)
    return {
        label: float(np.mean(chunk * chunk)) if chunk.size else 0.0
        for label, chunk in zip(labels, chunks)
    }


def compute_diagnostics(
    raw: np.ndarray,
    normalized: np.ndarray,
    filtered: np.ndarray,
    illumination: np.ndarray,
    low_illumination_fraction: float = 0.02,
) -> dict[str, Any]:
    raw = _as_float32(raw)
    normalized = _as_float32(normalized)
    filtered = _as_float32(filtered)
    illumination = _as_float32(illumination)
    max_illum = float(np.max(illumination)) if illumination.size else 0.0
    illum_threshold = max_illum * low_illumination_fraction
    low_mask = illumination < np.float32(illum_threshold) if max_illum > 0.0 else np.ones_like(illumination, dtype=bool)
    norm_energy = float(np.mean(normalized * normalized)) if normalized.size else 0.0
    filt_energy = float(np.mean(filtered * filtered)) if filtered.size else 0.0
    retention = filt_energy / norm_energy if norm_energy > 0.0 else 0.0
    return {
        "shape": [int(raw.shape[0]), int(raw.shape[1])],
        "raw_abs_p99": _robust_scale(raw, 99.0),
        "normalized_abs_p99": _robust_scale(normalized, 99.0),
        "filtered_abs_p99": _robust_scale(filtered, 99.0),
        "low_illumination_threshold": float(illum_threshold),
        "low_illumination_fraction": float(np.mean(low_mask)) if low_mask.size else 0.0,
        "lateral_energy": _zone_energy(normalized, axis=1, labels=("left", "center", "right")),
        "depth_energy": _zone_energy(normalized, axis=0, labels=("shallow", "middle", "deep")),
        "laplacian_energy_retention": float(retention),
    }


def _select_decision(metrics: dict[str, Any]) -> str:
    low_illum = float(metrics["low_illumination_fraction"])
    lateral = metrics["lateral_energy"]
    values = np.array([lateral["left"], lateral["center"], lateral["right"]], dtype=np.float64)
    lateral_ratio = float(values.max() / max(values.min(), 1.0e-12))
    if low_illum > 0.08 or lateral_ratio > 6.0:
        return "imaging_condition_limited"
    return "display_dominated"


def build_candidate_products(
    velocity: np.ndarray,
    raw: np.ndarray,
    illumination: np.ndarray,
    normalized: np.ndarray,
    filtered: np.ndarray,
    current_display: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any], str]:
    del velocity
    metrics = compute_diagnostics(raw=raw, normalized=normalized, filtered=filtered, illumination=illumination)
    improved_display = robust_symmetric_display(filtered, percentile=99.4, output_clip=0.9)
    masked = robust_symmetric_display(
        mask_low_illumination(normalized, illumination, fraction=0.02),
        percentile=99.2,
        output_clip=0.9,
    )
    depth_balanced = depth_balanced_display(filtered, percentile=99.0, output_clip=0.9)
    decision = _select_decision(metrics)
    recommended = improved_display if decision == "display_dominated" else masked
    products = {
        "current_display": _as_float32(current_display),
        "improved_display": improved_display,
        "low_illumination_masked": masked,
        "depth_balanced": depth_balanced,
        "recommended": recommended,
    }
    products.update(make_paper_ready_products(filtered=filtered, current_display=current_display))
    return products, metrics, decision


def _load_inputs(input_dir: Path) -> dict[str, np.ndarray]:
    names = {
        "velocity": "migration_velocity_smooth.npy",
        "raw": "multishot_rtm_image_raw.npy",
        "illumination": "multishot_rtm_illumination.npy",
        "normalized": "multishot_rtm_source_normalized.npy",
        "filtered": "multishot_rtm_laplacian_filtered.npy",
        "current_display": "multishot_rtm_display.npy",
    }
    missing = [filename for filename in names.values() if not (input_dir / filename).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required RTM outputs in {input_dir}: {', '.join(missing)}")
    return {key: np.load(input_dir / filename) for key, filename in names.items()}


def _save_compare_figure(
    output_path: Path,
    velocity: np.ndarray,
    raw: np.ndarray,
    illumination: np.ndarray,
    normalized: np.ndarray,
    filtered: np.ndarray,
    products: dict[str, np.ndarray],
) -> None:
    panels = [
        ("Migration velocity", velocity, "viridis", False),
        ("Raw RTM", raw, "gray", True),
        ("Source illumination", illumination, "magma", False),
        ("Source-normalized", normalized, "gray", True),
        ("Current Laplacian", filtered, "gray", True),
        ("Current display", products["current_display"], "gray", True),
        ("Improved display", products["improved_display"], "gray", True),
        ("Low-illumination masked", products["low_illumination_masked"], "gray", True),
        ("Depth-balanced candidate", products["depth_balanced"], "gray", True),
        ("Recommended conservative display", products["recommended"], "gray", True),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(18, 11), constrained_layout=True)
    axes_flat = axes.ravel()
    for ax, (title, data, cmap, symmetric) in zip(axes_flat, panels):
        data = _as_float32(data)
        if symmetric:
            vmax = _robust_scale(data, 99.0)
            vmin = -vmax
        else:
            vmin = float(np.percentile(data, 1.0))
            vmax = float(np.percentile(data, 99.0))
        im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x index")
        ax.set_ylabel("z index")
        fig.colorbar(im, ax=ax, shrink=0.75)
    for ax in axes_flat[len(panels):]:
        ax.axis("off")
    fig.suptitle("SEG/Salt RTM existing-result optimization comparison", fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_report(path: Path, input_dir: Path, metrics: dict[str, Any], decision: str) -> None:
    next_step = (
        "收敛到方案 1：优先优化显示与论文图后处理。"
        if decision == "display_dominated"
        else "进入方案 2：优先检查成像条件、照明补偿和低照明区处理。"
    )
    text = f"""# SEG/Salt RTM 优化诊断报告

输入目录：`{input_dir}`

## 诊断结论

`{decision}`

{next_step}

## 关键指标

- 低照明区占比：{metrics["low_illumination_fraction"]:.4f}
- Laplacian 后能量保留比例：{metrics["laplacian_energy_retention"]:.4f}
- 横向归一化能量：
  - 左部：{metrics["lateral_energy"]["left"]:.6g}
  - 中部：{metrics["lateral_energy"]["center"]:.6g}
  - 右部：{metrics["lateral_energy"]["right"]:.6g}
- 深度分层归一化能量：
  - 浅部：{metrics["depth_energy"]["shallow"]:.6g}
  - 中部：{metrics["depth_energy"]["middle"]:.6g}
  - 深部：{metrics["depth_energy"]["deep"]:.6g}

## 生成结果

- `candidate_current_display.npy`
- `candidate_improved_display.npy`
- `candidate_low_illumination_masked.npy`
- `candidate_depth_balanced.npy`
- `candidate_recommended.npy`
- `candidate_paper_recommended.npy`
- `paper_figures/paper_ready_migration.png`
- `paper_figures/paper_ready_comparison.png`
- `optimization_compare.png`
- `metrics.json`
"""
    path.write_text(text, encoding="utf-8")


def _save_single_paper_figure(
    output_path: Path,
    migration: np.ndarray,
    *,
    title: str,
    dx: float,
    dz: float,
) -> None:
    data = _as_float32(migration)
    x_km = np.arange(data.shape[1], dtype=np.float32) * np.float32(dx / 1000.0)
    z_km = np.arange(data.shape[0], dtype=np.float32) * np.float32(dz / 1000.0)
    clip = _robust_scale(data, 99.0)
    fig, ax = plt.subplots(figsize=(10.8, 4.8), constrained_layout=True)
    ax.imshow(
        data,
        cmap="gray",
        vmin=-clip,
        vmax=clip,
        aspect="auto",
        extent=[float(x_km[0]), float(x_km[-1]), float(z_km[-1]), float(z_km[0])],
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Depth (km)")
    ax.grid(False)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _save_paper_comparison_figure(
    output_path: Path,
    current_display: np.ndarray,
    recommended: np.ndarray,
    enhanced: np.ndarray,
    *,
    dx: float,
    dz: float,
) -> None:
    panels = [
        ("Current display", current_display),
        ("Recommended paper display", recommended),
        ("Enhanced candidate", enhanced),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), constrained_layout=True)
    for ax, (title, data) in zip(axes, panels):
        data = _as_float32(data)
        x_km = np.arange(data.shape[1], dtype=np.float32) * np.float32(dx / 1000.0)
        z_km = np.arange(data.shape[0], dtype=np.float32) * np.float32(dz / 1000.0)
        clip = _robust_scale(data, 99.0)
        ax.imshow(
            data,
            cmap="gray",
            vmin=-clip,
            vmax=clip,
            aspect="auto",
            extent=[float(x_km[0]), float(x_km[-1]), float(z_km[-1]), float(z_km[0])],
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Depth (km)")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _write_paper_report(path: Path, metrics: dict[str, Any], decision: str) -> None:
    text = f"""# 论文图优化说明

## 处理结论

本轮按方案 1 执行，只做显示与后处理优化，不改变声波方程正演、反传和零延迟互相关成像条件。

方案 3 的诊断结论为 `{decision}`，说明当前盐丘模型偏移结果主要受显示压缩和 Laplacian 后处理影响；低照明区占比为 {metrics["low_illumination_fraction"]:.4f}，不是主导问题。

## 推荐使用

- 推荐论文图数组：`candidate_paper_recommended.npy`
- 推荐论文图：`paper_ready_migration.png`
- 对比图：`paper_ready_comparison.png`

## 注意事项

- `paper_enhanced` 会更突出局部同相轴，但比推荐版本更激进，写论文时应作为辅助对比，不建议作为唯一主图。
- `paper_recommended` 使用保守对称裁剪，目标是增强可读性，同时避免逐深度强增益造成虚假连续性。
"""
    path.write_text(text, encoding="utf-8")


def run_optimization(input_dir: Path, output_dir: Path | None = None) -> Path:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir is not None else input_dir / "optimization_compare"
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = _load_inputs(input_dir)
    products, metrics, decision = build_candidate_products(**arrays)
    for name, data in products.items():
        np.save(output_dir / f"candidate_{name}.npy", data.astype(np.float32, copy=False))
    (output_dir / "metrics.json").write_text(
        json.dumps({"decision": decision, **metrics}, indent=2),
        encoding="utf-8",
    )
    _write_report(output_dir / "optimization_report.md", input_dir, metrics, decision)
    paper_dir = output_dir / "paper_figures"
    paper_dir.mkdir(parents=True, exist_ok=True)
    _save_single_paper_figure(
        paper_dir / "paper_ready_migration.png",
        products["paper_recommended"],
        title="SEG/Salt RTM migration section",
        dx=10.0,
        dz=10.0,
    )
    _save_paper_comparison_figure(
        paper_dir / "paper_ready_comparison.png",
        products["paper_current_display"],
        products["paper_recommended"],
        products["paper_enhanced"],
        dx=10.0,
        dz=10.0,
    )
    _write_paper_report(paper_dir / "paper_optimization_report.md", metrics, decision)
    _save_compare_figure(
        output_dir / "optimization_compare.png",
        velocity=arrays["velocity"],
        raw=arrays["raw"],
        illumination=arrays["illumination"],
        normalized=arrays["normalized"],
        filtered=arrays["filtered"],
        products=products,
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize existing SEG/Salt RTM output displays.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = run_optimization(args.input_dir, args.output_dir)
    print(f"Saved optimization comparison to {output_dir}")


if __name__ == "__main__":
    main()
