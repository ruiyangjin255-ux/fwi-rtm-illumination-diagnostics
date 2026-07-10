from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT.parent))

from admit_fwi.acoustic_rtm import (
    RTMConfig,
    multishot_reverse_time_migrate,
    mute_direct_arrivals,
    read_binary_model,
    smooth_velocity_model,
)


Array = np.ndarray
DEFAULT_MODEL = ROOT.parent / "fd2d_pml" / "vel" / "seg676x230.bin"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "small_salt_full_waveform_migration"
DEFAULT_MUTE_SCAN_OUTPUT_DIR = ROOT / "outputs" / "small_salt_full_waveform_mute_scan"


@dataclass(frozen=True)
class FullWaveformMigrationConfig:
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
    min_illumination_fraction: float = 0.02
    laplacian_power: int = 2
    smooth_radius_z: int = 5
    smooth_radius_x: int = 8
    smooth_passes: int = 3


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


def default_shot_positions(nx: int) -> list[int]:
    """给小范围全波形偏移生成左、中、右三个炮点。"""
    margin = max(8, nx // 8)
    return [margin, nx // 2, nx - margin - 1]


def parse_int_values(raw: str, name: str) -> list[int]:
    """解析逗号分隔的正整数列表。"""
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError as exc:
            raise ValueError(f"{name} 参数必须是整数") from exc
        if value < 0:
            raise ValueError(f"{name} 参数不能为负数")
        values.append(value)
    if not values:
        raise ValueError(f"至少需要一个 {name} 参数")
    return values


def parse_float_values(raw: str, name: str, allow_zero: bool = False) -> list[float]:
    """解析逗号分隔的浮点参数列表。"""
    values: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = float(item)
        except ValueError as exc:
            raise ValueError(f"{name} 参数必须是数字") from exc
        if allow_zero:
            if value < 0.0:
                raise ValueError(f"{name} 参数不能为负数")
        elif value <= 0.0:
            raise ValueError(f"{name} 参数必须为正数")
        values.append(value)
    if not values:
        raise ValueError(f"至少需要一个 {name} 参数")
    return values


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


def make_rtm_config(config: FullWaveformMigrationConfig, nx: int, nz: int, source_x: int) -> RTMConfig:
    """根据裁剪模型尺寸生成声波 RTM 配置。"""
    return RTMConfig(
        nx=nx,
        nz=nz,
        dx=config.dx,
        dz=config.dz,
        dt=config.dt,
        nt=config.nt,
        f0=config.f0,
        source_x=int(source_x),
        source_z=config.source_z,
        receiver_z=config.receiver_z,
        absorb_cells=config.absorb_cells,
        fd_order=config.fd_order,
    )


def _normalize_display(image: Array) -> Array:
    scale = float(np.percentile(np.abs(image), 99.0))
    if scale <= 0.0 or not np.isfinite(scale):
        return np.zeros_like(image, dtype=np.float32)
    return np.clip(np.asarray(image, dtype=np.float32) / np.float32(scale), -1.0, 1.0)


def _image_metrics(image: Array, illumination: Array) -> dict[str, float]:
    image = np.nan_to_num(np.asarray(image, dtype=np.float32))
    illumination = np.nan_to_num(np.asarray(illumination, dtype=np.float32))
    max_illumination = float(np.max(illumination)) if illumination.size else 0.0
    if max_illumination > 0.0 and np.isfinite(max_illumination):
        low_fraction = float(np.mean(illumination < max_illumination * 0.02))
    else:
        low_fraction = 1.0
    return {
        "image_abs_p99": float(np.percentile(np.abs(image), 99.0)),
        "image_rms": float(np.sqrt(np.mean(image * image))),
        "image_mean_abs": float(np.mean(np.abs(image))),
        "shallow_abs_mean": float(np.mean(np.abs(image[: max(1, image.shape[0] // 4), :]))),
        "middle_abs_mean": float(np.mean(np.abs(image[image.shape[0] // 3 : 2 * image.shape[0] // 3, :]))),
        "illumination_max": max_illumination,
        "illumination_low_fraction": low_fraction,
    }


def _run_case(
    name: str,
    true_model: Array,
    migration_velocity: Array,
    config: FullWaveformMigrationConfig,
    shots: list[int],
    output_dir: Path,
    *,
    subtract_direct_wave: bool,
) -> dict[str, Any]:
    nz, nx = true_model.shape
    rtm_config = make_rtm_config(config, nx=nx, nz=nz, source_x=shots[0])
    result = multishot_reverse_time_migrate(
        true_model,
        rtm_config,
        shots,
        wavefield_path=output_dir / f"{name}_source_wavefield.dat",
        laplacian_power=config.laplacian_power,
        migration_velocity=migration_velocity,
        subtract_direct_wave=subtract_direct_wave,
        min_illumination_fraction=config.min_illumination_fraction,
    )
    prefix = "full_waveform" if name == "full_waveform" else "reflection_only"
    arrays = {
        "image": result.image,
        "normalized_image": result.normalized_image,
        "source_receiver_normalized_image": result.source_receiver_normalized_image,
        "laplacian_image": result.laplacian_image,
        "laplacian_normalized_image": result.laplacian_normalized_image,
        "filtered_image": result.filtered_image,
        "source_illumination": result.illumination,
        "receiver_illumination": result.receiver_illumination,
        "stacked_record": result.stacked_record,
    }
    for suffix, array in arrays.items():
        np.save(output_dir / f"{prefix}_{suffix}.npy", np.asarray(array, dtype=np.float32))
    metrics = _image_metrics(result.image, result.illumination)
    metrics["normalized_abs_p99"] = float(np.percentile(np.abs(result.normalized_image), 99.0))
    metrics["source_receiver_normalized_abs_p99"] = float(
        np.percentile(np.abs(result.source_receiver_normalized_image), 99.0)
    )
    metrics["stacked_record_rms"] = float(np.sqrt(np.mean(result.stacked_record * result.stacked_record)))
    metrics["shot_count"] = int(result.shot_count)
    return {"result": result, "metrics": metrics}


def _write_figures(
    output_dir: Path,
    true_model: Array,
    migration_velocity: Array,
    full_image: Array,
    reflection_image: Array,
    full_record: Array,
    reflection_record: Array,
    config: FullWaveformMigrationConfig,
) -> None:
    import matplotlib.pyplot as plt

    configure_chinese_matplotlib()
    extent_model = [0.0, true_model.shape[1] * config.dx / 1000.0, true_model.shape[0] * config.dz / 1000.0, 0.0]
    extent_record = [
        0.0,
        full_record.shape[1] * config.dx / 1000.0,
        full_record.shape[0] * config.dt,
        0.0,
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.2), constrained_layout=True)
    model_im = axes[0, 0].imshow(true_model, cmap="turbo", extent=extent_model, aspect="auto")
    axes[0, 0].set_title("真实盐丘局部速度")
    axes[0, 0].set_xlabel("水平距离 / km")
    axes[0, 0].set_ylabel("深度 / km")
    fig.colorbar(model_im, ax=axes[0, 0], label="速度 / (m/s)")

    mig_im = axes[0, 1].imshow(migration_velocity, cmap="turbo", extent=extent_model, aspect="auto")
    axes[0, 1].set_title("平滑迁移速度")
    axes[0, 1].set_xlabel("水平距离 / km")
    axes[0, 1].set_ylabel("深度 / km")
    fig.colorbar(mig_im, ax=axes[0, 1], label="速度 / (m/s)")

    axes[1, 0].imshow(_normalize_display(full_image), cmap="seismic", extent=extent_model, aspect="auto", vmin=-1, vmax=1)
    axes[1, 0].set_title("全波形偏移")
    axes[1, 0].set_xlabel("水平距离 / km")
    axes[1, 0].set_ylabel("深度 / km")

    axes[1, 1].imshow(
        _normalize_display(reflection_image),
        cmap="seismic",
        extent=extent_model,
        aspect="auto",
        vmin=-1,
        vmax=1,
    )
    axes[1, 1].set_title("反射波偏移")
    axes[1, 1].set_xlabel("水平距离 / km")
    axes[1, 1].set_ylabel("深度 / km")
    fig.savefig(output_dir / "migration_compare.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), constrained_layout=True)
    axes[0].imshow(_normalize_display(full_record), cmap="gray", extent=extent_record, aspect="auto", vmin=-1, vmax=1)
    axes[0].set_title("全波形叠加记录")
    axes[0].set_xlabel("水平距离 / km")
    axes[0].set_ylabel("时间 / s")
    axes[1].imshow(
        _normalize_display(reflection_record),
        cmap="gray",
        extent=extent_record,
        aspect="auto",
        vmin=-1,
        vmax=1,
    )
    axes[1].set_title("反射波叠加记录")
    axes[1].set_xlabel("水平距离 / km")
    axes[1].set_ylabel("时间 / s")
    fig.savefig(output_dir / "stacked_record_compare.png", dpi=180)
    plt.close(fig)


def _write_report(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# 盐丘模型小范围全波形偏移结果",
        "",
        "## 实验定义",
        "",
        "本实验中的全波形偏移指使用未扣除直达波的完整炮记录进行叠前声波 RTM。",
        "反射波偏移指在相同迁移速度和炮点条件下扣除平滑模型直达波后的 RTM 参考结果。",
        "该实验不是 LSRTM，也不是弹性全波场偏移，主要用于比较完整记录进入零延迟互相关成像条件后的响应。",
        "",
        "## 关键参数",
        "",
        f"- 裁剪窗口: z0={summary['crop']['z0']}, x0={summary['crop']['x0']}, "
        f"nz={summary['crop']['nz']}, nx={summary['crop']['nx']}",
        f"- 炮点: {summary['shots']}",
        f"- 时间采样: nt={summary['config']['nt']}, dt={summary['config']['dt']} s",
        "",
        "## 指标对比",
        "",
        "| 指标 | 全波形偏移 | 反射波偏移 |",
        "| --- | ---: | ---: |",
        f"| 原始成像 99% 振幅 | {summary['full_waveform']['image_abs_p99']:.6e} | "
        f"{summary['reflection_only']['image_abs_p99']:.6e} |",
        f"| 源照明归一化 99% 振幅 | {summary['full_waveform']['normalized_abs_p99']:.6e} | "
        f"{summary['reflection_only']['normalized_abs_p99']:.6e} |",
        f"| 源-检波照明归一化 99% 振幅 | "
        f"{summary['full_waveform']['source_receiver_normalized_abs_p99']:.6e} | "
        f"{summary['reflection_only']['source_receiver_normalized_abs_p99']:.6e} |",
        f"| 低照明比例 | {summary['full_waveform']['illumination_low_fraction']:.6f} | "
        f"{summary['reflection_only']['illumination_low_fraction']:.6f} |",
        f"| 叠加记录 RMS | {summary['full_waveform']['stacked_record_rms']:.6e} | "
        f"{summary['reflection_only']['stacked_record_rms']:.6e} |",
        "",
        "## 初步解释",
        "",
        f"全波形与反射波成像差异 L2 指标为 {summary['comparison']['image_difference_l2']:.6e}。",
        "若全波形偏移浅部振幅更强，这是直达波和潜水波能量参与互相关的正常表现。",
        "后续若要提升构造成像清晰度，应继续比较直达波静音、Laplacian 增强、照明归一化和迭代式 LSRTM。",
        "",
    ]
    (output_dir / "full_waveform_migration_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _direct_velocity_from_model(migration_velocity: Array, source_z: int) -> float:
    """用近地表迁移速度中位数估计直达波静音速度。"""
    upper = np.asarray(migration_velocity[: max(source_z + 2, 4), :], dtype=np.float32)
    velocity = float(np.median(upper))
    if velocity <= 0.0 or not np.isfinite(velocity):
        raise ValueError("无法从迁移速度估计有效直达波速度")
    return velocity


def _run_muted_full_waveform_case(
    case_name: str,
    true_model: Array,
    migration_velocity: Array,
    config: FullWaveformMigrationConfig,
    shots: list[int],
    output_dir: Path,
    *,
    direct_velocity: float,
    padding_time: float,
    taper_time: float,
) -> dict[str, Any]:
    nz, nx = true_model.shape
    rtm_config = make_rtm_config(config, nx=nx, nz=nz, source_x=shots[0])

    def provider(source_x: int, record: Array) -> Array:
        return mute_direct_arrivals(
            record,
            rtm_config,
            source_x=source_x,
            direct_velocity=direct_velocity,
            padding_time=padding_time,
            taper_time=taper_time,
        )

    result = multishot_reverse_time_migrate(
        true_model,
        rtm_config,
        shots,
        wavefield_path=output_dir / f"{case_name}_source_wavefield.dat",
        laplacian_power=config.laplacian_power,
        record_provider=provider,
        migration_velocity=migration_velocity,
        subtract_direct_wave=False,
        min_illumination_fraction=config.min_illumination_fraction,
    )
    for suffix, array in {
        "image": result.image,
        "normalized_image": result.normalized_image,
        "source_receiver_normalized_image": result.source_receiver_normalized_image,
        "laplacian_image": result.laplacian_image,
        "laplacian_normalized_image": result.laplacian_normalized_image,
        "filtered_image": result.filtered_image,
        "stacked_record": result.stacked_record,
    }.items():
        np.save(output_dir / f"{case_name}_{suffix}.npy", np.asarray(array, dtype=np.float32))
    metrics = _image_metrics(result.image, result.illumination)
    metrics["normalized_abs_p99"] = float(np.percentile(np.abs(result.normalized_image), 99.0))
    metrics["source_receiver_normalized_abs_p99"] = float(
        np.percentile(np.abs(result.source_receiver_normalized_image), 99.0)
    )
    metrics["laplacian_abs_p99"] = float(np.percentile(np.abs(result.laplacian_image), 99.0))
    metrics["stacked_record_rms"] = float(np.sqrt(np.mean(result.stacked_record * result.stacked_record)))
    return {
        "case_name": case_name,
        "padding_time": float(padding_time),
        "taper_time": float(taper_time),
        "direct_velocity": float(direct_velocity),
        "result": result,
        "metrics": metrics,
    }


def _score_mute_case(metrics: dict[str, float]) -> float:
    """用浅部能量占比和整体振幅构造静音参数排序分数，数值越小越稳健。"""
    shallow = metrics["shallow_abs_mean"]
    middle = max(metrics["middle_abs_mean"], 1.0e-20)
    return float((shallow / middle) + 0.01 * metrics["image_abs_p99"])


def _write_mute_scan_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "case_name",
        "padding_time",
        "taper_time",
        "direct_velocity",
        "score",
        "image_abs_p99",
        "normalized_abs_p99",
        "source_receiver_normalized_abs_p99",
        "laplacian_abs_p99",
        "shallow_abs_mean",
        "middle_abs_mean",
        "stacked_record_rms",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_mute_scan_figure(
    output_dir: Path,
    true_model: Array,
    full_image: Array,
    best_image: Array,
    reflection_image: Array,
    config: FullWaveformMigrationConfig,
) -> None:
    import matplotlib.pyplot as plt

    configure_chinese_matplotlib()
    extent = [0.0, true_model.shape[1] * config.dx / 1000.0, true_model.shape[0] * config.dz / 1000.0, 0.0]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), constrained_layout=True)
    panels = [
        ("未静音全波形偏移", full_image),
        ("最佳静音全波形偏移", best_image),
        ("反射波偏移参考", reflection_image),
    ]
    for ax, (title, image) in zip(axes, panels):
        ax.imshow(_normalize_display(image), cmap="seismic", extent=extent, aspect="auto", vmin=-1, vmax=1)
        ax.set_title(title)
        ax.set_xlabel("水平距离 / km")
        ax.set_ylabel("深度 / km")
    fig.savefig(output_dir / "mute_scan_compare.png", dpi=180)
    plt.close(fig)


def _write_mute_scan_report(output_dir: Path, summary: dict[str, Any]) -> None:
    best = summary["best_case"]
    lines = [
        "# 盐丘全波形偏移直达波静音扫描结果",
        "",
        "## 最佳静音参数",
        "",
        f"- padding_time: {best['padding_time']:.3f} s",
        f"- taper_time: {best['taper_time']:.3f} s",
        f"- direct_velocity: {best['direct_velocity']:.2f} m/s",
        f"- 综合分数: {best['score']:.6e}",
        "",
        "## 对比解释",
        "",
        "直达波静音作用于输入炮记录，主要压制早至强能量，避免其在零延迟互相关成像中形成浅部强背景。",
        "照明归一化作用于成像结果，主要补偿震源和检波照明不均衡，不能替代记录静音。",
        "Laplacian 增强作用于成像剖面，主要压制低波数背景并突出界面响应。",
        "三者处理对象不同，应组合比较而不是互相替代。",
        "",
        "## 指标",
        "",
        f"- 未静音全波形 99% 振幅: {summary['full_waveform']['image_abs_p99']:.6e}",
        f"- 最佳静音 99% 振幅: {best['image_abs_p99']:.6e}",
        f"- 反射波参考 99% 振幅: {summary['reflection_only']['image_abs_p99']:.6e}",
        f"- 最佳静音浅部/中部能量比: "
        f"{best['shallow_abs_mean'] / max(best['middle_abs_mean'], 1.0e-20):.6e}",
        "",
    ]
    (output_dir / "mute_scan_best_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_full_waveform_migration_demo(
    *,
    true_model: Array,
    output_dir: str | Path,
    config: FullWaveformMigrationConfig,
    shot_positions: list[int] | None = None,
    crop_origin: tuple[int, int] = (0, 0),
) -> dict[str, Any]:
    """运行小范围全波形偏移和反射波偏移对比实验。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    true_model = np.asarray(true_model, dtype=np.float32)
    if true_model.ndim != 2:
        raise ValueError("true_model 必须是二维速度模型")
    nz, nx = true_model.shape
    shots = default_shot_positions(nx) if shot_positions is None else [int(value) for value in shot_positions]
    if any(source_x < 0 or source_x >= nx for source_x in shots):
        raise ValueError("炮点必须位于裁剪模型横向范围内")

    migration_velocity = smooth_velocity_model(
        true_model,
        radius_z=config.smooth_radius_z,
        radius_x=config.smooth_radius_x,
        passes=config.smooth_passes,
    )
    np.save(output_dir / "true_model.npy", true_model)
    np.save(output_dir / "migration_velocity.npy", migration_velocity)

    full = _run_case(
        "full_waveform",
        true_model,
        migration_velocity,
        config,
        shots,
        output_dir,
        subtract_direct_wave=False,
    )
    reflection = _run_case(
        "reflection_only",
        true_model,
        migration_velocity,
        config,
        shots,
        output_dir,
        subtract_direct_wave=True,
    )
    full_result = full["result"]
    reflection_result = reflection["result"]
    image_difference = np.asarray(full_result.image - reflection_result.image, dtype=np.float32)
    normalized_difference = np.asarray(
        full_result.normalized_image - reflection_result.normalized_image,
        dtype=np.float32,
    )
    np.save(output_dir / "image_difference_full_minus_reflection.npy", image_difference)
    np.save(output_dir / "normalized_difference_full_minus_reflection.npy", normalized_difference)

    summary: dict[str, Any] = {
        "config": asdict(config),
        "crop": {
            "z0": int(crop_origin[0]),
            "x0": int(crop_origin[1]),
            "nz": int(nz),
            "nx": int(nx),
        },
        "shots": shots,
        "shot_count": len(shots),
        "full_waveform": full["metrics"],
        "reflection_only": reflection["metrics"],
        "comparison": {
            "image_difference_l2": float(np.sqrt(np.mean(image_difference * image_difference))),
            "normalized_difference_l2": float(np.sqrt(np.mean(normalized_difference * normalized_difference))),
            "full_to_reflection_image_p99_ratio": float(
                full["metrics"]["image_abs_p99"] / max(reflection["metrics"]["image_abs_p99"], 1.0e-20)
            ),
        },
        "outputs": {
            "migration_compare": "migration_compare.png",
            "stacked_record_compare": "stacked_record_compare.png",
            "report": "full_waveform_migration_summary.md",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_figures(
        output_dir,
        true_model,
        migration_velocity,
        full_result.image,
        reflection_result.image,
        full_result.stacked_record,
        reflection_result.stacked_record,
        config,
    )
    _write_report(output_dir, summary)
    return summary


def run_direct_mute_scan_demo(
    *,
    true_model: Array,
    output_dir: str | Path,
    config: FullWaveformMigrationConfig,
    shot_positions: list[int] | None = None,
    padding_times: list[float] | None = None,
    taper_times: list[float] | None = None,
    crop_origin: tuple[int, int] = (0, 0),
) -> dict[str, Any]:
    """扫描直达波静音参数，并比较照明归一化和 Laplacian 成像指标。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    true_model = np.asarray(true_model, dtype=np.float32)
    if true_model.ndim != 2:
        raise ValueError("true_model 必须是二维速度模型")
    nz, nx = true_model.shape
    shots = default_shot_positions(nx) if shot_positions is None else [int(value) for value in shot_positions]
    if any(source_x < 0 or source_x >= nx for source_x in shots):
        raise ValueError("炮点必须位于裁剪模型横向范围内")
    padding_times = [0.0, 0.01, 0.02, 0.03] if padding_times is None else [float(value) for value in padding_times]
    taper_times = [0.01, 0.02] if taper_times is None else [float(value) for value in taper_times]
    if any(value < 0.0 for value in padding_times):
        raise ValueError("padding_time 不能为负数")
    if any(value <= 0.0 for value in taper_times):
        raise ValueError("taper_time 必须为正数")

    migration_velocity = smooth_velocity_model(
        true_model,
        radius_z=config.smooth_radius_z,
        radius_x=config.smooth_radius_x,
        passes=config.smooth_passes,
    )
    direct_velocity = _direct_velocity_from_model(migration_velocity, config.source_z)
    np.save(output_dir / "true_model.npy", true_model)
    np.save(output_dir / "migration_velocity.npy", migration_velocity)

    full = _run_case(
        "full_waveform",
        true_model,
        migration_velocity,
        config,
        shots,
        output_dir,
        subtract_direct_wave=False,
    )
    reflection = _run_case(
        "reflection_only",
        true_model,
        migration_velocity,
        config,
        shots,
        output_dir,
        subtract_direct_wave=True,
    )

    scan_rows: list[dict[str, Any]] = []
    scan_cases: list[dict[str, Any]] = []
    for padding_time in padding_times:
        for taper_time in taper_times:
            case_name = f"mute_pad{int(round(padding_time * 1000)):03d}_tap{int(round(taper_time * 1000)):03d}"
            case = _run_muted_full_waveform_case(
                case_name,
                true_model,
                migration_velocity,
                config,
                shots,
                output_dir,
                direct_velocity=direct_velocity,
                padding_time=padding_time,
                taper_time=taper_time,
            )
            row = {
                "case_name": case_name,
                "padding_time": float(padding_time),
                "taper_time": float(taper_time),
                "direct_velocity": direct_velocity,
                **case["metrics"],
            }
            row["score"] = _score_mute_case(case["metrics"])
            scan_rows.append(row)
            scan_cases.append(case)

    best_row = min(scan_rows, key=lambda row: float(row["score"]))
    best_case = next(case for case in scan_cases if case["case_name"] == best_row["case_name"])
    _write_mute_scan_csv(output_dir / "mute_scan_metrics.csv", scan_rows)
    _write_mute_scan_figure(
        output_dir,
        true_model,
        full["result"].image,
        best_case["result"].image,
        reflection["result"].image,
        config,
    )
    summary: dict[str, Any] = {
        "config": asdict(config),
        "crop": {
            "z0": int(crop_origin[0]),
            "x0": int(crop_origin[1]),
            "nz": int(nz),
            "nx": int(nx),
        },
        "shots": shots,
        "shot_count": len(shots),
        "padding_times": padding_times,
        "taper_times": taper_times,
        "scan_count": len(scan_rows),
        "full_waveform": full["metrics"],
        "reflection_only": reflection["metrics"],
        "best_case": best_row,
        "scan_rows": scan_rows,
        "outputs": {
            "metrics_csv": "mute_scan_metrics.csv",
            "summary_json": "mute_scan_summary.json",
            "compare_png": "mute_scan_compare.png",
            "report": "mute_scan_best_summary.md",
        },
    }
    (output_dir / "mute_scan_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_mute_scan_report(output_dir, summary)
    return summary


def run_from_model_file(
    *,
    model_path: str | Path = DEFAULT_MODEL,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config: FullWaveformMigrationConfig = FullWaveformMigrationConfig(),
    shot_positions: list[int] | None = None,
) -> dict[str, Any]:
    """读取完整盐丘模型并运行默认局部窗口全波形偏移实验。"""
    full_model = read_binary_model(model_path, nx=676, nz=230)
    z0, x0, crop_nz, crop_nx = choose_default_crop(full_model, config.crop_nz, config.crop_nx)
    true_model = full_model[z0 : z0 + crop_nz, x0 : x0 + crop_nx]
    return run_full_waveform_migration_demo(
        true_model=true_model,
        output_dir=output_dir,
        config=config,
        shot_positions=shot_positions,
        crop_origin=(z0, x0),
    )


def run_mute_scan_from_model_file(
    *,
    model_path: str | Path = DEFAULT_MODEL,
    output_dir: str | Path = DEFAULT_MUTE_SCAN_OUTPUT_DIR,
    config: FullWaveformMigrationConfig = FullWaveformMigrationConfig(),
    shot_positions: list[int] | None = None,
    padding_times: list[float] | None = None,
    taper_times: list[float] | None = None,
) -> dict[str, Any]:
    """读取完整盐丘模型并运行默认局部窗口直达波静音扫描。"""
    full_model = read_binary_model(model_path, nx=676, nz=230)
    z0, x0, crop_nz, crop_nx = choose_default_crop(full_model, config.crop_nz, config.crop_nx)
    true_model = full_model[z0 : z0 + crop_nz, x0 : x0 + crop_nx]
    return run_direct_mute_scan_demo(
        true_model=true_model,
        output_dir=output_dir,
        config=config,
        shot_positions=shot_positions,
        padding_times=padding_times,
        taper_times=taper_times,
        crop_origin=(z0, x0),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行盐丘模型小范围全波形偏移对比实验")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL, help="SEG 盐丘速度模型二进制文件")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--crop-nx", type=int, default=FullWaveformMigrationConfig.crop_nx, help="裁剪窗口横向点数")
    parser.add_argument("--crop-nz", type=int, default=FullWaveformMigrationConfig.crop_nz, help="裁剪窗口深度点数")
    parser.add_argument("--nt", type=int, default=FullWaveformMigrationConfig.nt, help="时间采样点数")
    parser.add_argument("--dt", type=float, default=FullWaveformMigrationConfig.dt, help="时间采样间隔")
    parser.add_argument("--f0", type=float, default=FullWaveformMigrationConfig.f0, help="震源主频")
    parser.add_argument("--shots", type=str, default="", help="逗号分隔炮点位置；为空时使用默认左中右三炮")
    parser.add_argument("--mute-scan", action="store_true", help="运行直达波静音参数扫描")
    parser.add_argument("--padding-times", type=str, default="0,0.01,0.02,0.03", help="逗号分隔静音延迟时间")
    parser.add_argument("--taper-times", type=str, default="0.01,0.02", help="逗号分隔静音余弦渐变时间")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FullWaveformMigrationConfig(
        crop_nx=args.crop_nx,
        crop_nz=args.crop_nz,
        nt=args.nt,
        dt=args.dt,
        f0=args.f0,
    )
    shots = None if not args.shots.strip() else parse_int_values(args.shots, name="shots")
    if args.mute_scan:
        padding_times = parse_float_values(args.padding_times, name="padding_time", allow_zero=True)
        taper_times = parse_float_values(args.taper_times, name="taper_time")
        summary = run_mute_scan_from_model_file(
            model_path=args.model_path,
            output_dir=args.output_dir,
            config=config,
            shot_positions=shots,
            padding_times=padding_times,
            taper_times=taper_times,
        )
    else:
        summary = run_from_model_file(
            model_path=args.model_path,
            output_dir=args.output_dir,
            config=config,
            shot_positions=shots,
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
