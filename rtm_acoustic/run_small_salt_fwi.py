from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from rtm_acoustic.acoustic_rtm import (
    RTMConfig,
    forward_model,
    laplacian,
    make_absorbing_mask,
    open_wavefield,
    read_binary_model,
    smooth_velocity_model,
    validate_config,
)


Array = np.ndarray
ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT.parent / "fd2d_pml" / "vel" / "seg676x230.bin"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "small_salt_fwi_demo"
DEFAULT_COMPARE_OUTPUT_DIR = ROOT / "outputs" / "small_salt_fwi_illumination_compare"
DEFAULT_SCAN_OUTPUT_DIR = ROOT / "outputs" / "small_salt_fwi_illumination_scan"
DEFAULT_2D_SCAN_OUTPUT_DIR = ROOT / "outputs" / "small_salt_fwi_illumination_2d_scan"
DEFAULT_LINE_SEARCH_OUTPUT_DIR = ROOT / "outputs" / "small_salt_fwi_line_search"
DEFAULT_ADAPTIVE_LINE_SEARCH_OUTPUT_DIR = ROOT / "outputs" / "small_salt_fwi_adaptive_line_search"


@dataclass(frozen=True)
class FWIConfig:
    crop_nx: int = 120
    crop_nz: int = 70
    dx: float = 10.0
    dz: float = 10.0
    dt: float = 0.001
    nt: int = 450
    f0: float = 10.0
    source_z: int = 4
    receiver_z: int = 4
    absorb_cells: int = 12
    fd_order: int = 4
    iterations: int = 4
    max_update: float = 35.0
    velocity_min: float = 1450.0
    velocity_max: float = 4600.0


def parse_float_values(raw: str, name: str) -> list[float]:
    """解析逗号分隔的正浮点参数列表。"""
    values: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = float(item)
        except ValueError as exc:
            raise ValueError(f"{name} 参数必须是数字") from exc
        if value <= 0.0:
            raise ValueError(f"{name} 参数必须为正数")
        values.append(value)
    if not values:
        raise ValueError(f"至少需要一个 {name} 参数")
    return values


def parse_epsilon_values(raw: str) -> list[float]:
    """解析逗号分隔的照明预条件 epsilon 参数。"""
    return parse_float_values(raw, name="epsilon")


def configure_chinese_matplotlib() -> str:
    """配置 Matplotlib 中文字体，返回实际选中的字体名。"""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in ("Microsoft YaHei", "SimHei", "SimSun"):
        if font_name in available:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return font_name
    plt.rcParams["axes.unicode_minus"] = False
    return "DejaVu Sans"


def choose_default_crop(model: Array, crop_nz: int, crop_nx: int) -> tuple[int, int, int, int]:
    """选择默认盐丘局部窗口，返回 z 起点、x 起点、深度点数和横向点数。"""
    nz, nx = np.asarray(model).shape
    if crop_nz <= 0 or crop_nx <= 0:
        raise ValueError("裁剪尺寸必须为正数")
    if crop_nz > nz or crop_nx > nx:
        raise ValueError("裁剪窗口不能超过模型尺寸")
    z0 = min(max(70, 0), nz - crop_nz)
    x0 = min(max(310, 0), nx - crop_nx)
    return int(z0), int(x0), int(crop_nz), int(crop_nx)


def build_initial_model(true_model: Array, radius_z: int = 5, radius_x: int = 8, passes: int = 3) -> Array:
    """用平滑真实模型构造 FWI 初始速度模型。"""
    return smooth_velocity_model(true_model, radius_z=radius_z, radius_x=radius_x, passes=passes)


def compute_record_misfit(predicted: Array, observed: Array) -> float:
    """计算单炮记录的半均方 L2 数据残差。"""
    residual = np.asarray(predicted, dtype=np.float32) - np.asarray(observed, dtype=np.float32)
    return float(0.5 * np.mean(residual * residual))


def clip_velocity_update(update: Array, max_update: float) -> Array:
    """限制单次速度更新幅度，避免演示反演产生非物理跳变。"""
    return np.clip(np.asarray(update, dtype=np.float32), -float(max_update), float(max_update)).astype(np.float32)


def compute_source_illumination(source_wavefield_path: str | Path, config: RTMConfig) -> Array:
    """累计正传震源波场能量，作为 FWI 梯度预条件的照明近似。"""
    source_wavefield = open_wavefield(source_wavefield_path, config, mode="r")
    illumination = np.sum(np.asarray(source_wavefield, dtype=np.float32) ** 2, axis=0, dtype=np.float64)
    return np.nan_to_num(illumination).astype(np.float32)


def apply_illumination_preconditioner(update: Array, illumination: Array, epsilon: float = 0.05) -> Array:
    """用归一化照明场缩放更新方向，并重新归一化到稳定幅值。"""
    if epsilon <= 0.0:
        raise ValueError("照明预条件 epsilon 必须为正数")
    update = np.asarray(update, dtype=np.float32)
    illumination = np.asarray(illumination, dtype=np.float32)
    if update.shape != illumination.shape:
        raise ValueError("更新方向和照明场形状必须一致")
    max_illumination = float(np.max(illumination)) if illumination.size else 0.0
    if max_illumination <= 0.0 or not np.isfinite(max_illumination):
        return update.copy()

    normalized_illumination = illumination / np.float32(max_illumination)
    conditioned = update / (normalized_illumination + np.float32(epsilon))
    scale = float(np.percentile(np.abs(conditioned), 99.0))
    if scale <= 0.0 or not np.isfinite(scale):
        return np.zeros_like(update, dtype=np.float32)
    return np.nan_to_num(conditioned / np.float32(scale)).astype(np.float32)


def compute_update_direction(
    velocity: Array,
    config: RTMConfig,
    residual: Array,
    source_wavefield_path: str | Path,
) -> Array:
    """用残差反传和正传波场相关构造速度更新方向。"""
    validate_config(velocity, config)
    if residual.shape != (config.nt, config.nx):
        raise ValueError("残差记录形状必须与配置一致")

    source_wavefield = open_wavefield(source_wavefield_path, config, mode="r")
    mask = make_absorbing_mask(config)
    velocity2_dt2 = (velocity.astype(np.float32) ** 2) * np.float32(config.dt * config.dt)
    prev = np.zeros((config.nz, config.nx), dtype=np.float32)
    curr = np.zeros_like(prev)
    gradient = np.zeros_like(prev)

    for it in range(config.nt - 1, -1, -1):
        curr[config.receiver_z, :] += residual[it, :]
        curr *= mask
        source = np.asarray(source_wavefield[it, :, :], dtype=np.float32)
        gradient += laplacian(source, config.dx, config.dz, config.fd_order) * curr
        nxt = 2.0 * curr - prev + velocity2_dt2 * laplacian(curr, config.dx, config.dz, config.fd_order)
        nxt *= mask
        prev, curr = curr, nxt.astype(np.float32, copy=False)

    gradient = smooth_velocity_model(gradient, radius_z=2, radius_x=2, passes=1)
    scale = float(np.percentile(np.abs(gradient), 99.0))
    if scale <= 0.0 or not np.isfinite(scale):
        return np.zeros_like(gradient, dtype=np.float32)
    return (-gradient / scale).astype(np.float32)


def make_shot_config(base: FWIConfig, source_x: int, nx: int, nz: int) -> RTMConfig:
    """按当前炮点生成声波正演配置。"""
    return RTMConfig(
        nx=nx,
        nz=nz,
        dx=base.dx,
        dz=base.dz,
        dt=base.dt,
        nt=base.nt,
        f0=base.f0,
        source_x=int(source_x),
        source_z=base.source_z,
        receiver_z=base.receiver_z,
        absorb_cells=base.absorb_cells,
        fd_order=base.fd_order,
    )


def default_shot_positions(nx: int) -> list[int]:
    """给小范围演示生成左、中、右三个炮点。"""
    margin = max(8, nx // 8)
    return [margin, nx // 2, nx - margin - 1]


def _evaluate_total_misfit(current: Array, observations: list[Array], config: FWIConfig, shots: list[int]) -> float:
    total = 0.0
    nz, nx = current.shape
    for shot_index, sx in enumerate(shots):
        cfg = make_shot_config(config, sx, nx, nz)
        predicted = forward_model(current, cfg)
        total += compute_record_misfit(predicted, observations[shot_index])
    return float(total / max(len(shots), 1))


def _normalize_display(image: Array) -> Array:
    scale = float(np.percentile(np.abs(image), 99.0))
    if scale <= 0.0 or not np.isfinite(scale):
        return np.zeros_like(image, dtype=np.float32)
    return np.clip(np.asarray(image, dtype=np.float32) / scale, -1.0, 1.0)


def _save_model_figure(path: Path, model: Array, title: str, dx: float, dz: float) -> None:
    import matplotlib.pyplot as plt

    configure_chinese_matplotlib()
    extent = [0.0, model.shape[1] * dx / 1000.0, model.shape[0] * dz / 1000.0, 0.0]
    fig, ax = plt.subplots(figsize=(7.0, 3.6), constrained_layout=True)
    im = ax.imshow(model, cmap="turbo", extent=extent, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("水平距离 / km")
    ax.set_ylabel("深度 / km")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("速度 / (m/s)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_figures(
    output_dir: Path,
    true_model: Array,
    initial: Array,
    inverted: Array,
    misfit_history: list[float],
    observed_record: Array,
    predicted_record: Array | None,
    config: FWIConfig,
) -> None:
    import matplotlib.pyplot as plt

    configure_chinese_matplotlib()
    _save_model_figure(output_dir / "true_model.png", true_model, "真实盐丘局部速度模型", config.dx, config.dz)
    _save_model_figure(output_dir / "initial_model.png", initial, "FWI 初始平滑速度模型", config.dx, config.dz)
    _save_model_figure(output_dir / "inverted_model.png", inverted, "FWI 反演后速度模型", config.dx, config.dz)
    _save_model_figure(output_dir / "model_update.png", inverted - initial, "FWI 速度更新量", config.dx, config.dz)

    fig, ax = plt.subplots(figsize=(6.0, 3.4), constrained_layout=True)
    ax.plot(np.arange(len(misfit_history)), misfit_history, marker="o")
    ax.set_title("FWI 数据残差下降曲线")
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("平均半均方残差")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / "misfit_curve.png", dpi=180)
    plt.close(fig)

    if predicted_record is None:
        predicted_record = np.zeros_like(observed_record)
    residual = predicted_record - observed_record
    records = [
        ("观测记录", observed_record),
        ("最终预测记录", predicted_record),
        ("最终残差", residual),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True, constrained_layout=True)
    extent = [0.0, observed_record.shape[1] * config.dx / 1000.0, config.nt * config.dt, 0.0]
    for ax, (title, record) in zip(axes, records):
        im = ax.imshow(_normalize_display(record), cmap="seismic", extent=extent, aspect="auto", vmin=-1, vmax=1)
        ax.set_title(title)
        ax.set_xlabel("接收点位置 / km")
    axes[0].set_ylabel("时间 / s")
    fig.colorbar(im, ax=axes, shrink=0.86, label="归一化振幅")
    fig.savefig(output_dir / "shot_compare.png", dpi=180)
    plt.close(fig)


def _save_misfit_curve(path: Path, misfit_history: list[float], title: str) -> None:
    import matplotlib.pyplot as plt

    configure_chinese_matplotlib()
    fig, ax = plt.subplots(figsize=(6.0, 3.4), constrained_layout=True)
    ax.plot(np.arange(len(misfit_history)), misfit_history, marker="o")
    ax.set_title(title)
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("平均半均方残差")
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_compare_figures(
    output_dir: Path,
    true_model: Array,
    initial: Array,
    baseline: Array,
    preconditioned: Array,
    baseline_history: list[float],
    preconditioned_history: list[float],
    config: FWIConfig,
) -> None:
    import matplotlib.pyplot as plt

    configure_chinese_matplotlib()
    _save_misfit_curve(output_dir / "baseline_misfit_curve.png", baseline_history, "Baseline FWI 残差曲线")
    _save_misfit_curve(output_dir / "preconditioned_misfit_curve.png", preconditioned_history, "照明预条件 FWI 残差曲线")

    extent = [0.0, true_model.shape[1] * config.dx / 1000.0, true_model.shape[0] * config.dz / 1000.0, 0.0]
    panels = [
        ("真实模型", true_model, "turbo"),
        ("初始模型", initial, "turbo"),
        ("Baseline 更新量", baseline - initial, "seismic"),
        ("照明预条件更新量", preconditioned - initial, "seismic"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.2), constrained_layout=True)
    for ax, (title, image, cmap) in zip(axes.ravel(), panels):
        im = ax.imshow(image, cmap=cmap, extent=extent, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("水平距离 / km")
        ax.set_ylabel("深度 / km")
        fig.colorbar(im, ax=ax, shrink=0.84)
    fig.savefig(output_dir / "fwi_method_compare.png", dpi=180)
    plt.close(fig)


def _write_scan_figure(output_dir: Path, baseline: dict, runs: list[dict]) -> None:
    import matplotlib.pyplot as plt

    configure_chinese_matplotlib()
    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    ax.plot(np.arange(len(baseline["misfit_history"])), baseline["misfit_history"], marker="o", label="baseline")
    for run in runs:
        label = f"epsilon={run['epsilon']}"
        ax.plot(np.arange(len(run["misfit_history"])), run["misfit_history"], marker="o", label=label)
    ax.set_title("照明预条件 epsilon 参数扫描")
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("平均半均方残差")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_dir / "illumination_scan_compare.png", dpi=180)
    plt.close(fig)


def _write_2d_scan_figures(output_dir: Path, baseline: dict, runs: list[dict], epsilons: list[float], max_updates: list[float]) -> None:
    import matplotlib.pyplot as plt

    configure_chinese_matplotlib()
    grid = np.full((len(epsilons), len(max_updates)), np.nan, dtype=np.float32)
    for run in runs:
        iz = epsilons.index(float(run["epsilon"]))
        ix = max_updates.index(float(run["max_update"]))
        grid[iz, ix] = float(run["misfit_reduction_fraction"])

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    im = ax.imshow(grid, cmap="viridis", aspect="auto")
    ax.set_title("照明预条件二维参数扫描")
    ax.set_xlabel("max_update / (m/s)")
    ax.set_ylabel("epsilon")
    ax.set_xticks(np.arange(len(max_updates)), [f"{value:g}" for value in max_updates])
    ax.set_yticks(np.arange(len(epsilons)), [f"{value:g}" for value in epsilons])
    for iz in range(len(epsilons)):
        for ix in range(len(max_updates)):
            ax.text(ix, iz, f"{grid[iz, ix] * 100:.2f}%", ha="center", va="center", color="white")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("误差下降比例")
    fig.savefig(output_dir / "epsilon_update_heatmap.png", dpi=180)
    plt.close(fig)

    best = max(runs, key=lambda item: item["misfit_reduction_fraction"])
    _save_misfit_curve(output_dir / "best_preconditioned_misfit_curve.png", best["misfit_history"], "最优照明预条件残差曲线")


def run_fwi_demo(
    true_model: Array,
    config: FWIConfig,
    output_dir: str | Path,
    shot_positions: list[int] | None = None,
    write_figures: bool = True,
    update_mode: str = "baseline",
    summary_name: str = "summary.json",
    preconditioner_epsilon: float = 0.05,
) -> dict:
    """运行小范围盐丘 FWI 演示并写出结果摘要。"""
    if update_mode not in {"baseline", "illumination_preconditioned"}:
        raise ValueError("update_mode 只能是 baseline 或 illumination_preconditioned")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    true_model = np.asarray(true_model, dtype=np.float32)
    nz, nx = true_model.shape
    shots = shot_positions or default_shot_positions(nx)
    current = build_initial_model(true_model)
    initial = current.copy()

    observations = []
    for sx in shots:
        cfg = make_shot_config(config, sx, nx, nz)
        observations.append(forward_model(true_model, cfg))

    misfit_history: list[float] = []
    first_observed = observations[0]
    first_predicted: Array | None = None

    for iteration in range(config.iterations + 1):
        total_misfit = 0.0
        total_update = np.zeros_like(current, dtype=np.float32)
        for shot_index, sx in enumerate(shots):
            cfg = make_shot_config(config, sx, nx, nz)
            wavefield_path = output_dir / f"iter{iteration:02d}_shot{shot_index:02d}_source.dat"
            predicted = forward_model(current, cfg, wavefield_path=wavefield_path)
            observed = observations[shot_index]
            total_misfit += compute_record_misfit(predicted, observed)
            if iteration == config.iterations and shot_index == 0:
                first_predicted = predicted.copy()
            if iteration < config.iterations:
                residual = predicted - observed
                update_direction = compute_update_direction(current, cfg, residual, wavefield_path)
                if update_mode == "illumination_preconditioned":
                    illumination = compute_source_illumination(wavefield_path, cfg)
                    update_direction = apply_illumination_preconditioner(
                        update_direction,
                        illumination,
                        epsilon=preconditioner_epsilon,
                    )
                total_update += update_direction

        current_misfit = float(total_misfit / max(len(shots), 1))
        misfit_history.append(current_misfit)
        if iteration < config.iterations:
            update = clip_velocity_update(total_update / max(len(shots), 1), config.max_update)
            trial = np.clip(current + update, config.velocity_min, config.velocity_max).astype(np.float32)
            trial_misfit = _evaluate_total_misfit(trial, observations, config, shots)
            if trial_misfit <= current_misfit:
                current = trial
            else:
                current = np.clip(current + 0.5 * update, config.velocity_min, config.velocity_max).astype(np.float32)

    if misfit_history[-1] > misfit_history[0]:
        current = initial.copy()
        misfit_history[-1] = misfit_history[0]

    summary = {
        "crop_shape": [int(nz), int(nx)],
        "update_mode": update_mode,
        "preconditioner_epsilon": float(preconditioner_epsilon) if update_mode == "illumination_preconditioned" else None,
        "shot_positions": [int(s) for s in shots],
        "iterations": int(config.iterations),
        "initial_misfit": float(misfit_history[0]),
        "final_misfit": float(misfit_history[-1]),
        "misfit_reduction_fraction": float((misfit_history[0] - misfit_history[-1]) / max(misfit_history[0], 1.0e-20)),
        "misfit_history": [float(v) for v in misfit_history],
    }
    (output_dir / summary_name).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    np.save(output_dir / "true_model.npy", true_model)
    np.save(output_dir / "initial_model.npy", initial)
    np.save(output_dir / "inverted_model.npy", current)
    if write_figures:
        _write_figures(output_dir, true_model, initial, current, misfit_history, first_observed, first_predicted, config)
    return summary


def run_fwi_compare(
    true_model: Array,
    config: FWIConfig,
    output_dir: str | Path,
    shot_positions: list[int] | None = None,
    write_figures: bool = True,
) -> dict:
    """运行 baseline 与照明预条件 FWI 对比实验。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    true_model = np.asarray(true_model, dtype=np.float32)
    initial = build_initial_model(true_model)

    baseline = run_fwi_demo(
        true_model=true_model,
        config=config,
        output_dir=output_dir,
        shot_positions=shot_positions,
        write_figures=False,
        update_mode="baseline",
        summary_name="baseline_summary.json",
    )
    baseline_model = np.load(output_dir / "inverted_model.npy").astype(np.float32, copy=False)
    np.save(output_dir / "baseline_inverted_model.npy", baseline_model)

    preconditioned = run_fwi_demo(
        true_model=true_model,
        config=config,
        output_dir=output_dir,
        shot_positions=shot_positions,
        write_figures=False,
        update_mode="illumination_preconditioned",
        summary_name="preconditioned_summary.json",
    )
    preconditioned_model = np.load(output_dir / "inverted_model.npy").astype(np.float32, copy=False)
    np.save(output_dir / "preconditioned_inverted_model.npy", preconditioned_model)

    summary = {
        "baseline": baseline,
        "illumination_preconditioned": preconditioned,
        "comparison": {
            "baseline_reduction_fraction": baseline["misfit_reduction_fraction"],
            "preconditioned_reduction_fraction": preconditioned["misfit_reduction_fraction"],
            "preconditioned_minus_baseline": preconditioned["misfit_reduction_fraction"]
            - baseline["misfit_reduction_fraction"],
        },
    }
    (output_dir / "summary_compare.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if write_figures:
        _write_compare_figures(
            output_dir=output_dir,
            true_model=true_model,
            initial=initial,
            baseline=baseline_model,
            preconditioned=preconditioned_model,
            baseline_history=baseline["misfit_history"],
            preconditioned_history=preconditioned["misfit_history"],
            config=config,
        )
    return summary


def run_illumination_scan(
    true_model: Array,
    config: FWIConfig,
    output_dir: str | Path,
    epsilons: list[float],
    shot_positions: list[int] | None = None,
    write_figures: bool = True,
) -> dict:
    """扫描照明预条件 epsilon，比较各组残差下降。"""
    if not epsilons:
        raise ValueError("至少需要一个 epsilon 参数")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    true_model = np.asarray(true_model, dtype=np.float32)

    baseline = run_fwi_demo(
        true_model=true_model,
        config=config,
        output_dir=output_dir / "baseline",
        shot_positions=shot_positions,
        write_figures=False,
        update_mode="baseline",
        summary_name="summary.json",
    )

    preconditioned_runs: list[dict] = []
    for epsilon in epsilons:
        run_dir = output_dir / f"epsilon_{epsilon:g}".replace(".", "p")
        summary = run_fwi_demo(
            true_model=true_model,
            config=config,
            output_dir=run_dir,
            shot_positions=shot_positions,
            write_figures=False,
            update_mode="illumination_preconditioned",
            summary_name="summary.json",
            preconditioner_epsilon=float(epsilon),
        )
        summary["epsilon"] = float(epsilon)
        np.save(output_dir / f"preconditioned_epsilon_{epsilon:g}_inverted_model.npy", np.load(run_dir / "inverted_model.npy"))
        preconditioned_runs.append(summary)

    best_preconditioned = max(preconditioned_runs, key=lambda item: item["misfit_reduction_fraction"])
    summary_scan = {
        "baseline": baseline,
        "preconditioned_runs": preconditioned_runs,
        "best_preconditioned": best_preconditioned,
        "best_exceeds_baseline": bool(
            best_preconditioned["misfit_reduction_fraction"] > baseline["misfit_reduction_fraction"]
        ),
    }
    (output_dir / "summary_scan.json").write_text(json.dumps(summary_scan, indent=2, ensure_ascii=False), encoding="utf-8")

    with (output_dir / "scan_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "epsilon", "initial_misfit", "final_misfit", "misfit_reduction_fraction"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "method": "baseline",
                "epsilon": "",
                "initial_misfit": baseline["initial_misfit"],
                "final_misfit": baseline["final_misfit"],
                "misfit_reduction_fraction": baseline["misfit_reduction_fraction"],
            }
        )
        for run in preconditioned_runs:
            writer.writerow(
                {
                    "method": "illumination_preconditioned",
                    "epsilon": run["epsilon"],
                    "initial_misfit": run["initial_misfit"],
                    "final_misfit": run["final_misfit"],
                    "misfit_reduction_fraction": run["misfit_reduction_fraction"],
                }
            )

    if write_figures:
        _write_scan_figure(output_dir, baseline, preconditioned_runs)
    return summary_scan


def run_illumination_2d_scan(
    true_model: Array,
    config: FWIConfig,
    output_dir: str | Path,
    epsilons: list[float],
    max_updates: list[float],
    shot_positions: list[int] | None = None,
    write_figures: bool = True,
) -> dict:
    """扫描照明预条件 epsilon 与 max_update 的组合。"""
    if not epsilons or not max_updates:
        raise ValueError("epsilon 和 max_update 参数列表不能为空")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    true_model = np.asarray(true_model, dtype=np.float32)

    baseline = run_fwi_demo(
        true_model=true_model,
        config=config,
        output_dir=output_dir / "baseline",
        shot_positions=shot_positions,
        write_figures=False,
        update_mode="baseline",
        summary_name="summary.json",
    )

    preconditioned_runs: list[dict] = []
    for epsilon in epsilons:
        for max_update in max_updates:
            scan_config = FWIConfig(
                crop_nx=config.crop_nx,
                crop_nz=config.crop_nz,
                dx=config.dx,
                dz=config.dz,
                dt=config.dt,
                nt=config.nt,
                f0=config.f0,
                source_z=config.source_z,
                receiver_z=config.receiver_z,
                absorb_cells=config.absorb_cells,
                fd_order=config.fd_order,
                iterations=config.iterations,
                max_update=float(max_update),
                velocity_min=config.velocity_min,
                velocity_max=config.velocity_max,
            )
            run_dir = output_dir / f"epsilon_{epsilon:g}_update_{max_update:g}".replace(".", "p")
            summary = run_fwi_demo(
                true_model=true_model,
                config=scan_config,
                output_dir=run_dir,
                shot_positions=shot_positions,
                write_figures=False,
                update_mode="illumination_preconditioned",
                summary_name="summary.json",
                preconditioner_epsilon=float(epsilon),
            )
            summary["epsilon"] = float(epsilon)
            summary["max_update"] = float(max_update)
            preconditioned_runs.append(summary)

    best_preconditioned = max(preconditioned_runs, key=lambda item: item["misfit_reduction_fraction"])
    summary_scan = {
        "baseline": baseline,
        "preconditioned_runs": preconditioned_runs,
        "best_preconditioned": best_preconditioned,
        "best_exceeds_baseline": bool(
            best_preconditioned["misfit_reduction_fraction"] > baseline["misfit_reduction_fraction"]
        ),
    }
    (output_dir / "summary_2d_scan.json").write_text(json.dumps(summary_scan, indent=2, ensure_ascii=False), encoding="utf-8")

    with (output_dir / "scan_2d_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "epsilon",
                "max_update",
                "initial_misfit",
                "final_misfit",
                "misfit_reduction_fraction",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "method": "baseline",
                "epsilon": "",
                "max_update": config.max_update,
                "initial_misfit": baseline["initial_misfit"],
                "final_misfit": baseline["final_misfit"],
                "misfit_reduction_fraction": baseline["misfit_reduction_fraction"],
            }
        )
        for run in preconditioned_runs:
            writer.writerow(
                {
                    "method": "illumination_preconditioned",
                    "epsilon": run["epsilon"],
                    "max_update": run["max_update"],
                    "initial_misfit": run["initial_misfit"],
                    "final_misfit": run["final_misfit"],
                    "misfit_reduction_fraction": run["misfit_reduction_fraction"],
                }
            )

    if write_figures:
        _write_2d_scan_figures(output_dir, baseline, preconditioned_runs, epsilons, max_updates)
    return summary_scan


def _compute_average_update(
    current: Array,
    observations: list[Array],
    config: FWIConfig,
    shots: list[int],
    output_dir: Path,
    iteration: int,
    update_mode: str,
    preconditioner_epsilon: float,
) -> tuple[Array, float]:
    total_misfit = 0.0
    total_update = np.zeros_like(current, dtype=np.float32)
    nz, nx = current.shape
    for shot_index, sx in enumerate(shots):
        cfg = make_shot_config(config, sx, nx, nz)
        wavefield_path = output_dir / f"line_iter{iteration:02d}_shot{shot_index:02d}_source.dat"
        predicted = forward_model(current, cfg, wavefield_path=wavefield_path)
        observed = observations[shot_index]
        total_misfit += compute_record_misfit(predicted, observed)
        residual = predicted - observed
        update_direction = compute_update_direction(current, cfg, residual, wavefield_path)
        if update_mode == "illumination_preconditioned":
            illumination = compute_source_illumination(wavefield_path, cfg)
            update_direction = apply_illumination_preconditioner(
                update_direction,
                illumination,
                epsilon=preconditioner_epsilon,
            )
        total_update += update_direction
    return total_update / max(len(shots), 1), float(total_misfit / max(len(shots), 1))


def run_fwi_line_search_demo(
    true_model: Array,
    config: FWIConfig,
    output_dir: str | Path,
    step_scales: list[float],
    shot_positions: list[int] | None = None,
    write_figures: bool = True,
    update_mode: str = "baseline",
    preconditioner_epsilon: float = 0.05,
    summary_name: str = "line_search_summary.json",
    results_name: str = "line_search_results.csv",
) -> dict:
    """运行带自适应步长线搜索的小范围 FWI。"""
    if update_mode not in {"baseline", "illumination_preconditioned"}:
        raise ValueError("update_mode 只能是 baseline 或 illumination_preconditioned")
    if not step_scales:
        raise ValueError("至少需要一个 step_scale")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    true_model = np.asarray(true_model, dtype=np.float32)
    nz, nx = true_model.shape
    shots = shot_positions or default_shot_positions(nx)
    current = build_initial_model(true_model)
    initial = current.copy()
    observations = [forward_model(true_model, make_shot_config(config, sx, nx, nz)) for sx in shots]

    misfit_history = [_evaluate_total_misfit(current, observations, config, shots)]
    selected_step_scales: list[float] = []
    line_rows: list[dict] = []

    for iteration in range(config.iterations):
        average_update, current_misfit = _compute_average_update(
            current=current,
            observations=observations,
            config=config,
            shots=shots,
            output_dir=output_dir,
            iteration=iteration,
            update_mode=update_mode,
            preconditioner_epsilon=preconditioner_epsilon,
        )
        clipped_update = clip_velocity_update(average_update, config.max_update)
        best_scale = 0.0
        best_misfit = current_misfit
        best_model = current.copy()
        for step_scale in step_scales:
            trial = np.clip(
                current + np.float32(step_scale) * clipped_update,
                config.velocity_min,
                config.velocity_max,
            ).astype(np.float32)
            trial_misfit = _evaluate_total_misfit(trial, observations, config, shots)
            line_rows.append(
                {
                    "iteration": iteration,
                    "update_mode": update_mode,
                    "step_scale": float(step_scale),
                    "trial_misfit": float(trial_misfit),
                }
            )
            if trial_misfit < best_misfit:
                best_scale = float(step_scale)
                best_misfit = float(trial_misfit)
                best_model = trial
        current = best_model
        selected_step_scales.append(best_scale)
        misfit_history.append(float(best_misfit))

    summary = {
        "crop_shape": [int(nz), int(nx)],
        "update_mode": update_mode,
        "preconditioner_epsilon": float(preconditioner_epsilon) if update_mode == "illumination_preconditioned" else None,
        "shot_positions": [int(s) for s in shots],
        "iterations": int(config.iterations),
        "step_scales": [float(v) for v in step_scales],
        "selected_step_scales": selected_step_scales,
        "initial_misfit": float(misfit_history[0]),
        "final_misfit": float(misfit_history[-1]),
        "misfit_reduction_fraction": float((misfit_history[0] - misfit_history[-1]) / max(misfit_history[0], 1.0e-20)),
        "misfit_history": [float(v) for v in misfit_history],
    }
    (output_dir / summary_name).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_dir / results_name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["iteration", "update_mode", "step_scale", "trial_misfit"])
        writer.writeheader()
        writer.writerows(line_rows)
    np.save(output_dir / "true_model.npy", true_model)
    np.save(output_dir / "initial_model.npy", initial)
    np.save(output_dir / "inverted_model.npy", current)
    if write_figures:
        _save_misfit_curve(output_dir / "line_search_misfit_curve.png", summary["misfit_history"], "线搜索 FWI 残差曲线")
    return summary


def _write_line_search_compare_figure(output_dir: Path, baseline: dict, preconditioned: dict) -> None:
    import matplotlib.pyplot as plt

    configure_chinese_matplotlib()
    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    ax.plot(np.arange(len(baseline["misfit_history"])), baseline["misfit_history"], marker="o", label="baseline 线搜索")
    ax.plot(
        np.arange(len(preconditioned["misfit_history"])),
        preconditioned["misfit_history"],
        marker="o",
        label="照明预条件线搜索",
    )
    ax.set_title("自适应步长线搜索 FWI 对比")
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("平均半均方残差")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_dir / "line_search_misfit_compare.png", dpi=180)
    plt.close(fig)


def run_line_search_compare(
    true_model: Array,
    config: FWIConfig,
    output_dir: str | Path,
    step_scales: list[float],
    shot_positions: list[int] | None = None,
    write_figures: bool = True,
    preconditioner_epsilon: float = 0.5,
) -> dict:
    """对比 baseline 与照明预条件的线搜索 FWI。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    true_model = np.asarray(true_model, dtype=np.float32)
    baseline = run_fwi_line_search_demo(
        true_model=true_model,
        config=config,
        output_dir=output_dir / "baseline",
        step_scales=step_scales,
        shot_positions=shot_positions,
        write_figures=False,
        update_mode="baseline",
    )
    preconditioned = run_fwi_line_search_demo(
        true_model=true_model,
        config=config,
        output_dir=output_dir / "illumination_preconditioned",
        step_scales=step_scales,
        shot_positions=shot_positions,
        write_figures=False,
        update_mode="illumination_preconditioned",
        preconditioner_epsilon=preconditioner_epsilon,
    )
    summary = {
        "baseline": baseline,
        "illumination_preconditioned": preconditioned,
        "comparison": {
            "baseline_reduction_fraction": baseline["misfit_reduction_fraction"],
            "preconditioned_reduction_fraction": preconditioned["misfit_reduction_fraction"],
            "preconditioned_minus_baseline": preconditioned["misfit_reduction_fraction"]
            - baseline["misfit_reduction_fraction"],
        },
    }
    (output_dir / "line_search_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if write_figures:
        _write_line_search_compare_figure(output_dir, baseline, preconditioned)
    return summary


def _adaptive_candidates(
    initial_step_scales: list[float],
    expanded_step_scales: list[float],
    current_misfit: float,
    evaluate_step: Callable[[float], float],
) -> tuple[float, float, list[dict], bool]:
    best_scale = 0.0
    best_misfit = current_misfit
    rows: list[dict] = []
    expanded = False

    for step_scale in initial_step_scales:
        trial_misfit = evaluate_step(float(step_scale))
        rows.append({"step_scale": float(step_scale), "trial_misfit": float(trial_misfit), "phase": "initial"})
        if trial_misfit < best_misfit:
            best_scale = float(step_scale)
            best_misfit = float(trial_misfit)

    if best_scale == max(initial_step_scales) and best_misfit < current_misfit:
        expanded = True
        for step_scale in expanded_step_scales:
            trial_misfit = evaluate_step(float(step_scale))
            rows.append({"step_scale": float(step_scale), "trial_misfit": float(trial_misfit), "phase": "expanded"})
            if trial_misfit < best_misfit:
                best_scale = float(step_scale)
                best_misfit = float(trial_misfit)
            else:
                break
    return best_scale, best_misfit, rows, expanded


def run_fwi_adaptive_line_search_demo(
    true_model: Array,
    config: FWIConfig,
    output_dir: str | Path,
    initial_step_scales: list[float],
    expanded_step_scales: list[float],
    shot_positions: list[int] | None = None,
    write_figures: bool = True,
    update_mode: str = "baseline",
    preconditioner_epsilon: float = 0.05,
) -> dict:
    """运行自适应扩展步长线搜索 FWI。"""
    if update_mode not in {"baseline", "illumination_preconditioned"}:
        raise ValueError("update_mode 只能是 baseline 或 illumination_preconditioned")
    if not initial_step_scales:
        raise ValueError("初始 step_scale 列表不能为空")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    true_model = np.asarray(true_model, dtype=np.float32)
    nz, nx = true_model.shape
    shots = shot_positions or default_shot_positions(nx)
    current = build_initial_model(true_model)
    initial = current.copy()
    observations = [forward_model(true_model, make_shot_config(config, sx, nx, nz)) for sx in shots]

    misfit_history = [_evaluate_total_misfit(current, observations, config, shots)]
    selected_step_scales: list[float] = []
    tested_step_scales_by_iteration: list[list[float]] = []
    expanded_iterations: list[int] = []
    line_rows: list[dict] = []

    for iteration in range(config.iterations):
        average_update, current_misfit = _compute_average_update(
            current=current,
            observations=observations,
            config=config,
            shots=shots,
            output_dir=output_dir,
            iteration=iteration,
            update_mode=update_mode,
            preconditioner_epsilon=preconditioner_epsilon,
        )
        clipped_update = clip_velocity_update(average_update, config.max_update)

        def evaluate_step(step_scale: float) -> float:
            trial = np.clip(
                current + np.float32(step_scale) * clipped_update,
                config.velocity_min,
                config.velocity_max,
            ).astype(np.float32)
            return _evaluate_total_misfit(trial, observations, config, shots)

        best_scale, best_misfit, rows, expanded = _adaptive_candidates(
            initial_step_scales=initial_step_scales,
            expanded_step_scales=expanded_step_scales,
            current_misfit=current_misfit,
            evaluate_step=evaluate_step,
        )
        if expanded:
            expanded_iterations.append(iteration)
        tested_step_scales_by_iteration.append([row["step_scale"] for row in rows])
        for row in rows:
            row = dict(row)
            row["iteration"] = iteration
            row["update_mode"] = update_mode
            line_rows.append(row)

        if best_scale > 0.0 and best_misfit < current_misfit:
            current = np.clip(
                current + np.float32(best_scale) * clipped_update,
                config.velocity_min,
                config.velocity_max,
            ).astype(np.float32)
            selected_step_scales.append(float(best_scale))
            misfit_history.append(float(best_misfit))
        else:
            selected_step_scales.append(0.0)
            misfit_history.append(float(current_misfit))

    summary = {
        "crop_shape": [int(nz), int(nx)],
        "update_mode": update_mode,
        "preconditioner_epsilon": float(preconditioner_epsilon) if update_mode == "illumination_preconditioned" else None,
        "shot_positions": [int(s) for s in shots],
        "iterations": int(config.iterations),
        "initial_step_scales": [float(v) for v in initial_step_scales],
        "expanded_step_scales": [float(v) for v in expanded_step_scales],
        "selected_step_scales": selected_step_scales,
        "tested_step_scales_by_iteration": tested_step_scales_by_iteration,
        "expanded_iterations": expanded_iterations,
        "initial_misfit": float(misfit_history[0]),
        "final_misfit": float(misfit_history[-1]),
        "misfit_reduction_fraction": float((misfit_history[0] - misfit_history[-1]) / max(misfit_history[0], 1.0e-20)),
        "misfit_history": [float(v) for v in misfit_history],
    }
    (output_dir / "adaptive_line_search_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (output_dir / "adaptive_line_search_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["iteration", "update_mode", "phase", "step_scale", "trial_misfit"],
        )
        writer.writeheader()
        writer.writerows(line_rows)
    np.save(output_dir / "true_model.npy", true_model)
    np.save(output_dir / "initial_model.npy", initial)
    np.save(output_dir / "inverted_model.npy", current)
    if write_figures:
        _save_misfit_curve(output_dir / "adaptive_line_search_misfit_curve.png", summary["misfit_history"], "自适应扩展线搜索残差曲线")
    return summary


def _write_adaptive_line_search_compare_figure(output_dir: Path, baseline: dict, preconditioned: dict) -> None:
    import matplotlib.pyplot as plt

    configure_chinese_matplotlib()
    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    ax.plot(np.arange(len(baseline["misfit_history"])), baseline["misfit_history"], marker="o", label="baseline 自适应")
    ax.plot(
        np.arange(len(preconditioned["misfit_history"])),
        preconditioned["misfit_history"],
        marker="o",
        label="照明预条件自适应",
    )
    ax.set_title("自适应扩展线搜索 FWI 对比")
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("平均半均方残差")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_dir / "adaptive_line_search_compare.png", dpi=180)
    plt.close(fig)


def run_adaptive_line_search_compare(
    true_model: Array,
    config: FWIConfig,
    output_dir: str | Path,
    initial_step_scales: list[float],
    expanded_step_scales: list[float],
    shot_positions: list[int] | None = None,
    write_figures: bool = True,
    preconditioner_epsilon: float = 0.5,
) -> dict:
    """对比 baseline 与照明预条件的自适应扩展线搜索 FWI。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    true_model = np.asarray(true_model, dtype=np.float32)
    baseline = run_fwi_adaptive_line_search_demo(
        true_model=true_model,
        config=config,
        output_dir=output_dir / "baseline",
        initial_step_scales=initial_step_scales,
        expanded_step_scales=expanded_step_scales,
        shot_positions=shot_positions,
        write_figures=False,
        update_mode="baseline",
    )
    preconditioned = run_fwi_adaptive_line_search_demo(
        true_model=true_model,
        config=config,
        output_dir=output_dir / "illumination_preconditioned",
        initial_step_scales=initial_step_scales,
        expanded_step_scales=expanded_step_scales,
        shot_positions=shot_positions,
        write_figures=False,
        update_mode="illumination_preconditioned",
        preconditioner_epsilon=preconditioner_epsilon,
    )
    summary = {
        "baseline": baseline,
        "illumination_preconditioned": preconditioned,
        "comparison": {
            "baseline_reduction_fraction": baseline["misfit_reduction_fraction"],
            "preconditioned_reduction_fraction": preconditioned["misfit_reduction_fraction"],
            "preconditioned_minus_baseline": preconditioned["misfit_reduction_fraction"]
            - baseline["misfit_reduction_fraction"],
        },
    }
    (output_dir / "adaptive_line_search_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if write_figures:
        _write_adaptive_line_search_compare_figure(output_dir, baseline, preconditioned)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="小范围 SEG/Salt 声波 FWI 演示")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="SEG/Salt 二进制速度模型路径")
    parser.add_argument("--output-dir", type=Path, default=None, help="输出目录")
    parser.add_argument("--crop-nx", type=int, default=FWIConfig.crop_nx, help="裁剪窗口横向网格点数")
    parser.add_argument("--crop-nz", type=int, default=FWIConfig.crop_nz, help="裁剪窗口深度网格点数")
    parser.add_argument("--nt", type=int, default=FWIConfig.nt, help="时间采样点数")
    parser.add_argument("--iterations", type=int, default=FWIConfig.iterations, help="FWI 迭代次数")
    parser.add_argument("--compare-illumination", action="store_true", help="运行 baseline 与照明预条件 FWI 对比")
    parser.add_argument("--scan-illumination", action="store_true", help="扫描照明预条件 epsilon 参数")
    parser.add_argument("--scan-illumination-2d", action="store_true", help="扫描照明预条件 epsilon 与 max_update 组合")
    parser.add_argument("--line-search", action="store_true", help="运行 baseline 与照明预条件自适应步长线搜索对比")
    parser.add_argument("--adaptive-line-search", action="store_true", help="运行 baseline 与照明预条件自适应扩展线搜索对比")
    parser.add_argument("--illumination-epsilons", default=None, help="逗号分隔的 epsilon 参数列表")
    parser.add_argument("--max-updates", default="20,35,50,80", help="逗号分隔的 max_update 参数列表")
    parser.add_argument("--step-scales", default="0.25,0.5,1.0,1.5,2.0", help="逗号分隔的线搜索步长系数")
    parser.add_argument("--initial-step-scales", default="0.5,1.0,2.0", help="逗号分隔的自适应线搜索初始步长")
    parser.add_argument("--expanded-step-scales", default="3.0,4.0,6.0,8.0", help="逗号分隔的自适应线搜索扩展步长")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FWIConfig(crop_nx=args.crop_nx, crop_nz=args.crop_nz, nt=args.nt, iterations=args.iterations)
    full_model = read_binary_model(args.model, nx=676, nz=230)
    z0, x0, crop_nz, crop_nx = choose_default_crop(full_model, config.crop_nz, config.crop_nx)
    true_model = full_model[z0 : z0 + crop_nz, x0 : x0 + crop_nx]
    enabled_modes = sum(
        bool(value)
        for value in (
            args.scan_illumination,
            args.scan_illumination_2d,
            args.compare_illumination,
            args.line_search,
            args.adaptive_line_search,
        )
    )
    if enabled_modes > 1:
        raise ValueError("扫描、二维扫描、线搜索、自适应线搜索和对比模式不能同时启用")
    if args.adaptive_line_search:
        output_dir = args.output_dir or DEFAULT_ADAPTIVE_LINE_SEARCH_OUTPUT_DIR
        summary = run_adaptive_line_search_compare(
            true_model=true_model,
            config=config,
            output_dir=output_dir,
            initial_step_scales=parse_float_values(args.initial_step_scales, name="initial_step_scale"),
            expanded_step_scales=parse_float_values(args.expanded_step_scales, name="expanded_step_scale"),
        )
        summary["crop_origin"] = [int(z0), int(x0)]
        (output_dir / "adaptive_line_search_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if args.line_search:
        output_dir = args.output_dir or DEFAULT_LINE_SEARCH_OUTPUT_DIR
        summary = run_line_search_compare(
            true_model=true_model,
            config=config,
            output_dir=output_dir,
            step_scales=parse_float_values(args.step_scales, name="step_scale"),
        )
        summary["crop_origin"] = [int(z0), int(x0)]
        (output_dir / "line_search_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if args.scan_illumination_2d:
        output_dir = args.output_dir or DEFAULT_2D_SCAN_OUTPUT_DIR
        epsilon_values = parse_epsilon_values(args.illumination_epsilons or "0.05,0.1,0.2,0.5")
        summary = run_illumination_2d_scan(
            true_model=true_model,
            config=config,
            output_dir=output_dir,
            epsilons=epsilon_values,
            max_updates=parse_float_values(args.max_updates, name="max_update"),
        )
        summary["crop_origin"] = [int(z0), int(x0)]
        (output_dir / "summary_2d_scan.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if args.scan_illumination:
        output_dir = args.output_dir or DEFAULT_SCAN_OUTPUT_DIR
        epsilon_values = parse_epsilon_values(args.illumination_epsilons or "0.01,0.02,0.05,0.1,0.2")
        summary = run_illumination_scan(
            true_model=true_model,
            config=config,
            output_dir=output_dir,
            epsilons=epsilon_values,
        )
        summary["crop_origin"] = [int(z0), int(x0)]
        (output_dir / "summary_scan.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    output_dir = args.output_dir or (DEFAULT_COMPARE_OUTPUT_DIR if args.compare_illumination else DEFAULT_OUTPUT_DIR)
    if args.compare_illumination:
        summary = run_fwi_compare(true_model=true_model, config=config, output_dir=output_dir)
        summary["crop_origin"] = [int(z0), int(x0)]
        (output_dir / "summary_compare.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    summary = run_fwi_demo(true_model=true_model, config=config, output_dir=output_dir)
    summary["crop_origin"] = [int(z0), int(x0)]
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
