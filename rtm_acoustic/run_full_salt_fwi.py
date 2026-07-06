from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT.parent))

from rtm_acoustic.acoustic_rtm import (
    crop_padded_model,
    forward_model,
    pad_velocity_model,
    read_binary_model,
    shot_positions_from_spacing,
)
from rtm_acoustic.run_small_salt_fwi import (
    Array,
    apply_illumination_preconditioner,
    build_initial_model,
    clip_velocity_update,
    compute_source_illumination,
    compute_record_misfit,
    compute_update_direction,
    configure_chinese_matplotlib,
    make_shot_config,
)


DEFAULT_MODEL = ROOT.parent / "fd2d_pml" / "vel" / "seg676x230.bin"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "full_salt_fwi"


@dataclass(frozen=True)
class FullSaltFWIConfig:
    nx: int = 676
    nz: int = 230
    dx: float = 10.0
    dz: float = 10.0
    dt: float = 0.001
    nt: int = 900
    f0: float = 8.0
    source_z: int = 4
    receiver_z: int = 4
    absorb_cells: int = 40
    fd_order: int = 8
    iterations: int = 3
    shot_spacing: float = 30.0
    shot_margin_cells: int = 4
    max_shots: int = 0
    step_scale: float = 4.0
    max_update: float = 25.0
    velocity_min: float = 1450.0
    velocity_max: float = 4600.0
    smooth_radius_z: int = 10
    smooth_radius_x: int = 16
    smooth_passes: int = 3
    optimizer: str = "steepest"
    preconditioner_epsilon: float = 0.5
    pad_x: int = 0
    pad_top: int = 0
    pad_bottom: int = 0


def limited_shots(shots: list[int], max_shots: int) -> list[int]:
    """从完整炮集均匀抽取指定炮数，max_shots<=0 表示使用全部炮。"""
    if max_shots <= 0 or max_shots >= len(shots):
        return [int(value) for value in shots]
    indices = np.linspace(0, len(shots) - 1, max_shots).round().astype(int)
    return [int(shots[i]) for i in sorted(set(indices.tolist()))]


def select_fwi_shots(config: FullSaltFWIConfig) -> list[int]:
    """根据炮间距和最大炮数生成全范围 FWI 炮点。"""
    all_shots = shot_positions_from_spacing(
        nx=config.nx,
        dx=config.dx,
        spacing_m=config.shot_spacing,
        margin_cells=config.shot_margin_cells,
    )
    return limited_shots(all_shots, config.max_shots)


def _runtime_config(config: FullSaltFWIConfig) -> FullSaltFWIConfig:
    """Return the padded grid configuration used internally for wave propagation."""
    return replace(
        config,
        nx=config.nx + 2 * int(config.pad_x),
        nz=config.nz + int(config.pad_top) + int(config.pad_bottom),
        source_z=config.source_z + int(config.pad_top),
        receiver_z=config.receiver_z + int(config.pad_top),
    )


def _runtime_source_x(config: FullSaltFWIConfig, physical_source_x: int) -> int:
    return int(physical_source_x) + int(config.pad_x)


def _json_dump(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _save_array(path: Path, array: Array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("wb") as handle:
        np.save(handle, np.asarray(array, dtype=np.float32))
    tmp.replace(path)


def _observation_path(output_dir: Path, source_x: int) -> Path:
    return output_dir / "observations" / f"shot_{source_x:05d}.npy"


def _load_initial_model_override(
    path: str | Path,
    expected_shape: tuple[int, int],
    *,
    pad_x: int = 0,
    pad_top: int = 0,
    pad_bottom: int = 0,
) -> Array:
    """读取外部初始模型，用于长记录反演的 warm start。"""
    model = np.load(Path(path)).astype(np.float32, copy=False)
    padded_shape = (
        expected_shape[0] + int(pad_top) + int(pad_bottom),
        expected_shape[1] + 2 * int(pad_x),
    )
    if model.shape == expected_shape:
        if pad_x > 0 or pad_top > 0 or pad_bottom > 0:
            return pad_velocity_model(model, pad_x=pad_x, pad_top=pad_top, pad_bottom=pad_bottom)
        return model
    if model.shape != padded_shape:
        raise ValueError(f"外部初始模型尺寸必须为 {expected_shape} 或 {padded_shape}，当前为 {model.shape}")
    return model


def _load_or_create_observation(true_model: Array, config: FullSaltFWIConfig, output_dir: Path, source_x: int) -> Array:
    path = _observation_path(output_dir, source_x)
    if path.exists():
        return np.load(path).astype(np.float32, copy=False)
    runtime = _runtime_config(config)
    cfg = make_shot_config(runtime, _runtime_source_x(config, source_x), runtime.nx, runtime.nz)
    observed = forward_model(true_model, cfg)
    _save_array(path, observed)
    return observed


def _checkpoint_path(output_dir: Path, iteration: int) -> Path:
    return output_dir / "checkpoint" / f"iteration_{iteration:03d}.json"


def _write_iteration_checkpoint(
    output_dir: Path,
    *,
    iteration: int,
    completed_shots: list[int],
    misfit_sum: float,
    gradient_sum: Array,
    current_model: Array,
    config: FullSaltFWIConfig,
    shots: list[int],
) -> None:
    checkpoint_dir = output_dir / "checkpoint"
    _save_array(checkpoint_dir / f"gradient_sum_iter_{iteration:03d}.npy", gradient_sum)
    _save_array(checkpoint_dir / f"current_model_iter_{iteration:03d}.npy", current_model)
    manifest = {
        "version": 1,
        "iteration": int(iteration),
        "completed_count": len(completed_shots),
        "completed_shots": [int(value) for value in completed_shots],
        "misfit_sum": float(misfit_sum),
        "config": asdict(config),
        "shots": [int(value) for value in shots],
        "arrays": {
            "gradient_sum": f"gradient_sum_iter_{iteration:03d}.npy",
            "current_model": f"current_model_iter_{iteration:03d}.npy",
        },
    }
    _json_dump(_checkpoint_path(output_dir, iteration), manifest)


def _checkpoint_config_matches(saved: dict[str, Any], current: FullSaltFWIConfig) -> bool:
    """检查 checkpoint 是否可续跑；允许仅追加总迭代次数。"""
    current_config = asdict(current)
    ignored_keys = {"iterations"}
    for key, value in current_config.items():
        if key in ignored_keys:
            continue
        if saved.get(key) != value:
            return False
    return True


def _load_iteration_checkpoint(
    output_dir: Path,
    iteration: int,
    config: FullSaltFWIConfig,
    shots: list[int],
) -> tuple[list[int], float, Array, Array] | None:
    path = _checkpoint_path(output_dir, iteration)
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not _checkpoint_config_matches(manifest.get("config", {}), config) or manifest.get("shots") != [int(value) for value in shots]:
        raise ValueError("checkpoint 与当前全范围 FWI 参数不匹配")
    checkpoint_dir = output_dir / "checkpoint"
    gradient_sum = np.load(checkpoint_dir / manifest["arrays"]["gradient_sum"]).astype(np.float32, copy=False)
    current_model = np.load(checkpoint_dir / manifest["arrays"]["current_model"]).astype(np.float32, copy=False)
    return (
        [int(value) for value in manifest.get("completed_shots", [])],
        float(manifest.get("misfit_sum", 0.0)),
        gradient_sum,
        current_model,
    )


def _write_misfit_csv(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    with (output_dir / "fwi_iteration_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "iteration",
                "completed_shots",
                "mean_misfit",
                "step_scale",
                "optimizer",
                "cg_beta",
                "max_abs_update",
                "model_min",
                "model_max",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _normalize_history_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按迭代号去重历史记录，保留每个迭代首次完整记录。"""
    seen: set[int] = set()
    normalized: list[dict[str, Any]] = []
    for row in rows:
        iteration = int(row["iteration"])
        if iteration in seen:
            continue
        seen.add(iteration)
        normalized.append(row)
    return normalized


def _rtm_style_display_window(config: FullSaltFWIConfig) -> tuple[slice, slice, list[float]]:
    """生成与 RTM 一致的成图窗口：保留顶部，裁掉左右和底部吸收层。"""
    cells = max(0, int(config.absorb_cells))
    z_stop = config.nz if config.pad_bottom >= cells else (config.nz - cells if cells < config.nz else config.nz)
    if config.pad_x >= cells:
        x_start = 0
        x_stop = config.nx
    else:
        x_start = cells if cells * 2 < config.nx else 0
        x_stop = config.nx - cells if cells * 2 < config.nx else config.nx
    z_slice = slice(0, z_stop)
    x_slice = slice(x_start, x_stop)
    extent = [
        x_start * config.dx / 1000.0,
        x_stop * config.dx / 1000.0,
        z_stop * config.dz / 1000.0,
        0.0,
    ]
    return z_slice, x_slice, extent


def _write_figures(output_dir: Path, true_model: Array, initial: Array, current: Array, history: list[float], config: FullSaltFWIConfig) -> None:
    import matplotlib.pyplot as plt

    configure_chinese_matplotlib()
    z_slice, x_slice, extent = _rtm_style_display_window(config)
    panels = [
        ("真实盐丘速度模型", true_model, "turbo"),
        ("FWI 初始平滑模型", initial, "turbo"),
        ("FWI 反演速度模型", current, "turbo"),
        ("速度更新量", current - initial, "seismic"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.8), constrained_layout=True)
    for ax, (title, image, cmap) in zip(axes.ravel(), panels):
        im = ax.imshow(image[z_slice, x_slice], cmap=cmap, extent=extent, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("水平距离 / km")
        ax.set_ylabel("深度 / km")
        fig.colorbar(im, ax=ax, shrink=0.82)
    fig.savefig(output_dir / "full_salt_fwi_model_compare.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    ax.plot(np.arange(len(history)), history, marker="o")
    ax.set_title("全范围盐丘 FWI 残差曲线")
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("平均半均方残差")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / "full_salt_fwi_misfit_curve.png", dpi=180)
    plt.close(fig)


def _write_report(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# 全范围盐丘模型 FWI 结果",
        "",
        "## 实验定义",
        "",
        "本结果为完整 `676 x 230` 盐丘模型上的声波全波形反演。观测记录由真实速度模型正演生成，初始模型由真实模型平滑得到。",
        "程序按炮流式计算，每完成一炮保存 checkpoint，因此中断后可用 `--resume` 继续。",
        "",
        "## 关键参数",
        "",
        f"- 模型尺寸: {summary['model_shape']}",
        f"- 炮数: {summary['shot_count']}",
        f"- 迭代次数: {summary['iterations']}",
        f"- nt: {summary['config']['nt']}, f0: {summary['config']['f0']} Hz",
        f"- step_scale: {summary['config']['step_scale']}, max_update: {summary['config']['max_update']} m/s",
        f"- 优化方法: {summary['config']['optimizer']}",
        f"- 外部 padding: x={summary['padding']['pad_x']} cells, top={summary['padding']['pad_top']} cells, bottom={summary['padding']['pad_bottom']} cells；padded 尺寸: {summary['padded_model_shape']}",
        f"- 图件显示裁剪: z={summary['figure_display_crop']['z_start']}:{summary['figure_display_crop']['z_stop']}, x={summary['figure_display_crop']['x_start']}:{summary['figure_display_crop']['x_stop']}；保留顶部，裁掉左右和底部吸收层",
        "",
        "## 结果",
        "",
        f"- 初始残差: {summary['initial_misfit']:.6e}",
        f"- 最终残差: {summary['final_misfit']:.6e}",
        f"- 残差下降比例: {summary['misfit_reduction_fraction'] * 100:.4f}%",
        "",
        "## 输出文件",
        "",
        "- `full_salt_true_model.npy`",
        "- `full_salt_initial_model.npy`",
        "- `full_salt_inverted_model.npy`",
        "- `full_salt_model_update.npy`",
        "- `full_salt_fwi_model_compare.png`",
        "- `full_salt_fwi_misfit_curve.png`",
        "- `full_salt_fwi_summary.json`",
        "",
    ]
    (output_dir / "full_salt_fwi_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _optimizer_state_paths(output_dir: Path) -> dict[str, Path]:
    state_dir = output_dir / "optimizer_state"
    return {
        "previous_gradient": state_dir / "previous_gradient.npy",
        "previous_direction": state_dir / "previous_direction.npy",
    }


def _load_optimizer_state(output_dir: Path) -> tuple[Array | None, Array | None]:
    paths = _optimizer_state_paths(output_dir)
    previous_gradient = (
        np.load(paths["previous_gradient"]).astype(np.float32, copy=False)
        if paths["previous_gradient"].exists()
        else None
    )
    previous_direction = (
        np.load(paths["previous_direction"]).astype(np.float32, copy=False)
        if paths["previous_direction"].exists()
        else None
    )
    return previous_gradient, previous_direction


def _save_optimizer_state(output_dir: Path, gradient: Array, direction: Array) -> None:
    paths = _optimizer_state_paths(output_dir)
    _save_array(paths["previous_gradient"], gradient)
    _save_array(paths["previous_direction"], direction)


def _compute_cg_direction(
    gradient_direction: Array,
    previous_gradient: Array | None,
    previous_direction: Array | None,
) -> tuple[Array, float]:
    """使用 Fletcher-Reeves 公式构造 CG 方向。"""
    current = np.asarray(gradient_direction, dtype=np.float32)
    if previous_gradient is None or previous_direction is None:
        return current.copy(), 0.0
    current64 = current.astype(np.float64, copy=False)
    previous64 = np.asarray(previous_gradient, dtype=np.float32).astype(np.float64, copy=False)
    numerator = float(np.sum(current64 * current64, dtype=np.float64))
    denominator = float(np.sum(previous64 * previous64, dtype=np.float64))
    if denominator <= 0.0 or not np.isfinite(denominator):
        return current.copy(), 0.0
    beta = max(0.0, numerator / denominator)
    direction = current + np.float32(beta) * np.asarray(previous_direction, dtype=np.float32)
    if not np.isfinite(direction).all():
        return current.copy(), 0.0
    return direction.astype(np.float32, copy=False), float(beta)


def run_full_salt_fwi(
    *,
    model_path: str | Path,
    output_dir: str | Path,
    config: FullSaltFWIConfig,
    initial_model_path: str | Path | None = None,
    resume: bool = False,
    write_figures: bool = True,
) -> dict[str, Any]:
    """运行完整盐丘模型 FWI，并在每炮后保存可恢复 checkpoint。"""
    if config.optimizer not in {"steepest", "cg", "p-cg"}:
        raise ValueError("optimizer 必须是 steepest、cg 或 p-cg")
    if config.pad_x < 0 or config.pad_top < 0 or config.pad_bottom < 0:
        raise ValueError("pad_x、pad_top 和 pad_bottom 必须为非负整数")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    true_physical = read_binary_model(model_path, nx=config.nx, nz=config.nz)
    physical_shape = true_physical.shape
    true_model = pad_velocity_model(
        true_physical,
        pad_x=config.pad_x,
        pad_top=config.pad_top,
        pad_bottom=config.pad_bottom,
    )
    shots = select_fwi_shots(config)
    if not shots:
        raise ValueError("没有可用炮点")
    if initial_model_path is None:
        initial = build_initial_model(
            true_model,
            radius_z=config.smooth_radius_z,
            radius_x=config.smooth_radius_x,
            passes=config.smooth_passes,
        )
    else:
        initial = _load_initial_model_override(
            initial_model_path,
            physical_shape,
            pad_x=config.pad_x,
            pad_top=config.pad_top,
            pad_bottom=config.pad_bottom,
        )
    current = initial.copy()
    history_rows: list[dict[str, Any]] = []
    history: list[float] = []
    completed_history_iterations: set[int] = set()
    if resume and (output_dir / "fwi_iteration_history.csv").exists():
        with (output_dir / "fwi_iteration_history.csv").open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                history_rows.append(dict(row))
        history_rows = _normalize_history_rows(history_rows)
        _write_misfit_csv(output_dir, history_rows)
        completed_history_iterations = {int(row["iteration"]) for row in history_rows}
        history = [float(row["mean_misfit"]) for row in history_rows]
        if history_rows:
            last_completed = max(completed_history_iterations)
            model_path = output_dir / f"model_iter_{last_completed + 1:03d}.npy"
            if model_path.exists():
                current = np.load(model_path).astype(np.float32, copy=False)
    elif not resume:
        _save_array(output_dir / "full_salt_true_model.npy", true_physical)
        _save_array(
            output_dir / "full_salt_initial_model.npy",
            crop_padded_model(initial, physical_shape, pad_x=config.pad_x, pad_top=config.pad_top),
        )
        if config.pad_x > 0 or config.pad_top > 0 or config.pad_bottom > 0:
            _save_array(output_dir / "full_salt_true_model_padded.npy", true_model)
            _save_array(output_dir / "full_salt_initial_model_padded.npy", initial)
    previous_gradient, previous_direction = _load_optimizer_state(output_dir) if resume else (None, None)

    for iteration in range(config.iterations):
        if iteration in completed_history_iterations and (output_dir / f"model_iter_{iteration + 1:03d}.npy").exists():
            current = np.load(output_dir / f"model_iter_{iteration + 1:03d}.npy").astype(np.float32, copy=False)
            continue
        loaded = _load_iteration_checkpoint(output_dir, iteration, config, shots) if resume else None
        if loaded is None:
            completed_shots: list[int] = []
            misfit_sum = 0.0
            gradient_sum = np.zeros_like(current, dtype=np.float32)
        else:
            completed_shots, misfit_sum, gradient_sum, current = loaded
        completed_set = set(completed_shots)

        for shot_index, source_x in enumerate(shots):
            if source_x in completed_set:
                continue
            runtime = _runtime_config(config)
            cfg = make_shot_config(runtime, _runtime_source_x(config, source_x), runtime.nx, runtime.nz)
            observed = _load_or_create_observation(true_model, config, output_dir, source_x)
            wavefield_path = output_dir / "wavefields" / f"iter{iteration:03d}_shot{shot_index:05d}_source.dat"
            predicted = forward_model(current, cfg, wavefield_path=wavefield_path)
            residual = predicted - observed
            misfit_sum += compute_record_misfit(predicted, observed)
            update_direction = compute_update_direction(current, cfg, residual, wavefield_path)
            if config.optimizer == "p-cg":
                illumination = compute_source_illumination(wavefield_path, cfg)
                update_direction = apply_illumination_preconditioner(
                    update_direction,
                    illumination,
                    epsilon=config.preconditioner_epsilon,
                )
            gradient_sum += update_direction
            completed_shots.append(int(source_x))
            completed_set.add(int(source_x))
            _write_iteration_checkpoint(
                output_dir,
                iteration=iteration,
                completed_shots=completed_shots,
                misfit_sum=misfit_sum,
                gradient_sum=gradient_sum,
                current_model=current,
                config=config,
                shots=shots,
            )
            wavefield_path.unlink(missing_ok=True)

        mean_misfit = float(misfit_sum / max(len(shots), 1))
        average_update = gradient_sum / np.float32(max(len(shots), 1))
        if config.optimizer in {"cg", "p-cg"}:
            search_direction, cg_beta = _compute_cg_direction(
                average_update,
                previous_gradient,
                previous_direction,
            )
        else:
            search_direction, cg_beta = average_update, 0.0
        clipped_update = clip_velocity_update(search_direction, config.max_update)
        scaled_update = np.float32(config.step_scale) * clipped_update
        current = np.clip(current + scaled_update, config.velocity_min, config.velocity_max).astype(np.float32)
        _save_optimizer_state(output_dir, average_update, search_direction)
        previous_gradient = average_update.copy()
        previous_direction = search_direction.copy()
        _save_array(output_dir / f"model_iter_{iteration + 1:03d}.npy", current)
        _save_array(output_dir / f"update_iter_{iteration + 1:03d}.npy", scaled_update)
        history.append(mean_misfit)
        history_rows.append(
            {
                "iteration": iteration,
                "completed_shots": len(shots),
                "mean_misfit": mean_misfit,
                "step_scale": config.step_scale,
                "optimizer": config.optimizer,
                "cg_beta": cg_beta,
                "max_abs_update": float(np.max(np.abs(scaled_update))),
                "model_min": float(np.min(current)),
                "model_max": float(np.max(current)),
            }
        )
        _write_misfit_csv(output_dir, history_rows)

    final_misfit = history[-1]
    initial_misfit = history[0]
    _, _, figure_extent = _rtm_style_display_window(config)
    display_crop = {
        "mode": "rtm_style_absorbing_boundary_crop",
        "z_start": 0,
        "z_stop": int(config.nz - config.absorb_cells if config.absorb_cells < config.nz else config.nz),
        "x_start": int(config.absorb_cells if config.absorb_cells * 2 < config.nx else 0),
        "x_stop": int(config.nx - config.absorb_cells if config.absorb_cells * 2 < config.nx else config.nx),
        "extent_km": [float(value) for value in figure_extent],
        "note": "npy 结果保留全模型；png 成果图按 RTM 口径裁掉左右和底部吸收层，顶部不裁剪。",
    }
    current_physical = crop_padded_model(current, physical_shape, pad_x=config.pad_x, pad_top=config.pad_top)
    initial_physical = crop_padded_model(initial, physical_shape, pad_x=config.pad_x, pad_top=config.pad_top)
    update_physical = current_physical - initial_physical
    summary = {
        "model_shape": [int(config.nz), int(config.nx)],
        "padded_model_shape": [int(true_model.shape[0]), int(true_model.shape[1])],
        "padding": {
            "pad_x": int(config.pad_x),
            "pad_top": int(config.pad_top),
            "pad_bottom": int(config.pad_bottom),
            "physical_depth_m": float(config.nz * config.dz),
            "padded_depth_m": float(true_model.shape[0] * config.dz),
            "bottom_absorbing_boundary_starts_m": float((true_model.shape[0] - config.absorb_cells) * config.dz),
            "physical_width_m": float(config.nx * config.dx),
            "padded_width_m": float(true_model.shape[1] * config.dx),
        },
        "config": asdict(config),
        "figure_display_crop": display_crop,
        "shot_positions": [int(value) for value in shots],
        "shot_count": len(shots),
        "iterations": config.iterations,
        "initial_misfit": float(initial_misfit),
        "final_misfit": float(final_misfit),
        "misfit_reduction_fraction": float((initial_misfit - final_misfit) / max(initial_misfit, 1.0e-20)),
        "misfit_history": [float(value) for value in history],
        "outputs": {
            "true_model": "full_salt_true_model.npy",
            "initial_model": "full_salt_initial_model.npy",
            "inverted_model": "full_salt_inverted_model.npy",
            "model_update": "full_salt_model_update.npy",
            "history": "fwi_iteration_history.csv",
            "checkpoint_dir": "checkpoint",
        },
    }
    _save_array(output_dir / "full_salt_inverted_model.npy", current_physical)
    _save_array(output_dir / "full_salt_model_update.npy", update_physical)
    if config.pad_x > 0 or config.pad_top > 0 or config.pad_bottom > 0:
        _save_array(output_dir / "full_salt_inverted_model_padded.npy", current)
        _save_array(output_dir / "full_salt_model_update_padded.npy", current - initial)
    _json_dump(output_dir / "full_salt_fwi_summary.json", summary)
    if write_figures:
        _write_figures(output_dir, true_physical, initial_physical, current_physical, history, config)
    _write_report(output_dir, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行完整盐丘模型声波 FWI")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--initial-model", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--nt", type=int, default=FullSaltFWIConfig.nt)
    parser.add_argument("--f0", type=float, default=FullSaltFWIConfig.f0)
    parser.add_argument("--iterations", type=int, default=FullSaltFWIConfig.iterations)
    parser.add_argument("--shot-spacing", type=float, default=FullSaltFWIConfig.shot_spacing)
    parser.add_argument("--max-shots", type=int, default=FullSaltFWIConfig.max_shots)
    parser.add_argument("--step-scale", type=float, default=FullSaltFWIConfig.step_scale)
    parser.add_argument("--max-update", type=float, default=FullSaltFWIConfig.max_update)
    parser.add_argument("--optimizer", choices=["steepest", "cg", "p-cg"], default=FullSaltFWIConfig.optimizer)
    parser.add_argument("--preconditioner-epsilon", type=float, default=FullSaltFWIConfig.preconditioner_epsilon)
    parser.add_argument("--pad-x", type=int, default=FullSaltFWIConfig.pad_x)
    parser.add_argument("--pad-top", type=int, default=FullSaltFWIConfig.pad_top)
    parser.add_argument("--pad-bottom", type=int, default=FullSaltFWIConfig.pad_bottom)
    parser.add_argument("--fd-order", type=int, default=FullSaltFWIConfig.fd_order)
    parser.add_argument("--absorb-cells", type=int, default=FullSaltFWIConfig.absorb_cells)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FullSaltFWIConfig(
        nt=args.nt,
        f0=args.f0,
        iterations=args.iterations,
        shot_spacing=args.shot_spacing,
        max_shots=args.max_shots,
        step_scale=args.step_scale,
        max_update=args.max_update,
        optimizer=args.optimizer,
        preconditioner_epsilon=args.preconditioner_epsilon,
        pad_x=args.pad_x,
        pad_top=args.pad_top,
        pad_bottom=args.pad_bottom,
        fd_order=args.fd_order,
        absorb_cells=args.absorb_cells,
    )
    summary = run_full_salt_fwi(
        model_path=args.model,
        output_dir=args.output_dir,
        config=config,
        initial_model_path=args.initial_model,
        resume=args.resume,
        write_figures=not args.no_figures,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
