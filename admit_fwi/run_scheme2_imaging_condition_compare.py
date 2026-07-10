from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from admit_fwi.acoustic_rtm import (
    RTMConfig,
    crop_padded_model,
    multishot_reverse_time_migrate_parallel,
    mute_direct_arrivals,
    pad_rtm_config,
    pad_velocity_model,
    preprocess_migration_section,
    read_binary_model,
    shot_positions_from_spacing,
    smooth_velocity_model,
)
from admit_fwi.run_multishot_rtm import limited_shots


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "fd2d_pml" / "vel" / "seg676x230.bin"
DEFAULT_OUTPUT = ROOT / "admit_fwi" / "outputs" / "seg_salt_scheme2_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small Scheme 2 RTM imaging-condition comparison on the SEG salt model."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--nx", type=int, default=676)
    parser.add_argument("--nz", type=int, default=230)
    parser.add_argument("--dx", type=float, default=10.0)
    parser.add_argument("--dz", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--nt", type=int, default=500)
    parser.add_argument("--f0", type=float, default=20.0)
    parser.add_argument("--source-z", type=int, default=4)
    parser.add_argument("--receiver-z", type=int, default=4)
    parser.add_argument("--shot-spacing", type=float, default=300.0)
    parser.add_argument("--shot-margin-cells", type=int, default=4)
    parser.add_argument("--max-shots", type=int, default=6)
    parser.add_argument("--fd-order", type=int, default=8)
    parser.add_argument("--absorb-cells", type=int, default=40)
    parser.add_argument("--absorb-strength", type=float, default=3.0)
    parser.add_argument("--laplacian-power", type=int, default=1)
    parser.add_argument("--min-illumination-fraction", type=float, default=0.01)
    parser.add_argument("--smooth-radius-x", type=int, default=10)
    parser.add_argument("--smooth-radius-z", type=int, default=10)
    parser.add_argument("--smooth-passes", type=int, default=2)
    parser.add_argument("--pad-x", type=int, default=60)
    parser.add_argument("--pad-top", type=int, default=0)
    parser.add_argument("--pad-bottom", type=int, default=60)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--no-direct-subtract", action="store_true")
    parser.add_argument("--no-direct-mute", action="store_true")
    parser.add_argument("--direct-velocity", type=float, default=2000.0)
    parser.add_argument("--direct-mute-padding", type=float, default=0.04)
    parser.add_argument("--direct-mute-taper", type=float, default=0.02)
    parser.add_argument("--display-depth-power", type=float, default=0.15)
    parser.add_argument("--display-clip-percentile", type=float, default=99.5)
    parser.add_argument("--display-trace-balance", type=float, default=0.25)
    parser.add_argument("--display-output-clip", type=float, default=0.80)
    return parser.parse_args()


def crop_result_arrays(result, original_shape: tuple[int, int], pad_x: int, pad_top: int) -> dict[str, np.ndarray]:
    return {
        "raw": crop_padded_model(result.image, original_shape, pad_x=pad_x, pad_top=pad_top).astype(np.float32),
        "source_illumination": crop_padded_model(
            result.illumination,
            original_shape,
            pad_x=pad_x,
            pad_top=pad_top,
        ).astype(np.float32),
        "receiver_illumination": crop_padded_model(
            result.receiver_illumination,
            original_shape,
            pad_x=pad_x,
            pad_top=pad_top,
        ).astype(np.float32),
        "source_normalized": crop_padded_model(
            result.normalized_image,
            original_shape,
            pad_x=pad_x,
            pad_top=pad_top,
        ).astype(np.float32),
        "source_receiver_normalized": crop_padded_model(
            result.source_receiver_normalized_image,
            original_shape,
            pad_x=pad_x,
            pad_top=pad_top,
        ).astype(np.float32),
        "laplacian_image": crop_padded_model(
            result.laplacian_image,
            original_shape,
            pad_x=pad_x,
            pad_top=pad_top,
        ).astype(np.float32),
        "laplacian_source_normalized": crop_padded_model(
            result.laplacian_normalized_image,
            original_shape,
            pad_x=pad_x,
            pad_top=pad_top,
        ).astype(np.float32),
    }


def display_image(image: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    return preprocess_migration_section(
        image,
        depth_power=args.display_depth_power,
        clip_percentile=args.display_clip_percentile,
        trace_balance=args.display_trace_balance,
        output_clip=args.display_output_clip,
    )


def normalized_correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    a -= np.mean(a)
    b -= np.mean(b)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def image_metrics(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    src = arrays["source_illumination"]
    rec = arrays["receiver_illumination"]
    geom = np.sqrt(np.maximum(src * rec, 0.0))
    geom_max = float(np.max(geom))
    low_fraction = 0.0 if geom_max == 0.0 else float(np.mean(geom < geom_max * 0.01))
    return {
        "source_receiver_correlation": normalized_correlation(
            arrays["source_normalized"],
            arrays["source_receiver_normalized"],
        ),
        "laplacian_source_correlation": normalized_correlation(
            arrays["source_normalized"],
            arrays["laplacian_source_normalized"],
        ),
        "receiver_illumination_max": float(np.max(rec)),
        "geometric_low_illumination_fraction_1pct": low_fraction,
        "source_receiver_abs_p99": float(np.percentile(np.abs(arrays["source_receiver_normalized"]), 99.0)),
        "laplacian_source_abs_p99": float(np.percentile(np.abs(arrays["laplacian_source_normalized"]), 99.0)),
    }


def save_arrays(output_dir: Path, arrays: dict[str, np.ndarray]) -> dict[str, str]:
    names = {
        "raw": "scheme2_raw.npy",
        "source_illumination": "scheme2_source_illumination.npy",
        "receiver_illumination": "scheme2_receiver_illumination.npy",
        "source_normalized": "scheme2_source_normalized.npy",
        "source_receiver_normalized": "scheme2_source_receiver_normalized.npy",
        "laplacian_image": "scheme2_laplacian_image.npy",
        "laplacian_source_normalized": "scheme2_laplacian_source_normalized.npy",
    }
    for key, filename in names.items():
        np.save(output_dir / filename, arrays[key])
    return names


def save_compare_figure(output_path: Path, arrays: dict[str, np.ndarray], args: argparse.Namespace) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        ("Source normalized", display_image(arrays["source_normalized"], args), "gray"),
        ("Source-receiver normalized", display_image(arrays["source_receiver_normalized"], args), "gray"),
        ("Laplacian source normalized", display_image(arrays["laplacian_source_normalized"], args), "gray"),
        ("Receiver illumination log10", np.log10(arrays["receiver_illumination"] + 1.0), "magma"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.4), constrained_layout=True)
    extent = [0.0, args.nx * args.dx, args.nz * args.dz, 0.0]
    for ax, (title, data, cmap) in zip(axes.ravel(), panels):
        if cmap == "gray":
            clip = float(np.percentile(np.abs(data), 99.0)) or 1.0
            im = ax.imshow(data, cmap=cmap, vmin=-clip, vmax=clip, aspect="auto", extent=extent)
        else:
            im = ax.imshow(data, cmap=cmap, aspect="auto", extent=extent)
        ax.set_title(title)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Depth (m)")
        fig.colorbar(im, ax=ax, shrink=0.82)
    fig.savefig(output_path, dpi=260)
    plt.close(fig)


def write_report(
    output_path: Path,
    *,
    args: argparse.Namespace,
    shots: list[int],
    padded_shots: list[int],
    arrays: dict[str, str],
    metrics: dict[str, float],
) -> None:
    report = f"""# 方案 2 成像条件对比 smoke 运行报告

## 运行结论

- 本次运行用于快速比较源照明归一化、源-检几何照明归一化和 Laplacian 成像条件，不覆盖既有 full 结果。
- 源-检归一化与源归一化相关系数：`{metrics["source_receiver_correlation"]:.4f}`。
- Laplacian 源归一化与源归一化相关系数：`{metrics["laplacian_source_correlation"]:.4f}`。
- 源-检几何照明低于最大值 1% 的网格比例：`{metrics["geometric_low_illumination_fraction_1pct"]:.4f}`。

## 关键参数

- 模型：`{args.model}`
- 输出目录：`{args.output_dir}`
- 网格：`nx={args.nx}, nz={args.nz}, dx={args.dx}, dz={args.dz}`
- 时间：`dt={args.dt}, nt={args.nt}, f0={args.f0}`
- 炮点数量：`{len(shots)}`，物理炮点：`{shots}`
- padding 后炮点：`{padded_shots}`
- padding：`pad_x={args.pad_x}, pad_top={args.pad_top}, pad_bottom={args.pad_bottom}`
- worker：`{args.workers}`

## 输出文件

- 对比图：`scheme2_compare.png`
- 参数：`scheme2_parameters.json`
- 数组：
{chr(10).join(f"  - `{filename}`" for filename in arrays.values())}

## 判读建议

- 如果源-检归一化图与源归一化图主体结构接近，只是局部振幅更均衡，可优先保留方案 1 的论文展示图，并把方案 2 作为附录或方法对照。
- 如果 Laplacian 源归一化明显削弱低频噪声并提升盐体边界连续性，再扩大 `nt` 和炮数做正式 full 复算。
"""
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_dir is None:
        args.checkpoint_dir = args.output_dir / "scheme2_checkpoints"

    velocity = read_binary_model(args.model, nx=args.nx, nz=args.nz)
    original_shape = velocity.shape
    cfg = RTMConfig(
        nx=args.nx,
        nz=args.nz,
        dx=args.dx,
        dz=args.dz,
        dt=args.dt,
        nt=args.nt,
        f0=args.f0,
        source_x=args.nx // 2,
        source_z=args.source_z,
        receiver_z=args.receiver_z,
        absorb_cells=args.absorb_cells,
        absorb_strength=args.absorb_strength,
        fd_order=args.fd_order,
    )
    padded_cfg = pad_rtm_config(cfg, pad_x=args.pad_x, pad_top=args.pad_top, pad_bottom=args.pad_bottom)
    rtm_velocity = pad_velocity_model(velocity, pad_x=args.pad_x, pad_top=args.pad_top, pad_bottom=args.pad_bottom)
    migration_velocity = smooth_velocity_model(
        rtm_velocity,
        radius_z=args.smooth_radius_z,
        radius_x=args.smooth_radius_x,
        passes=args.smooth_passes,
    )
    all_shots = shot_positions_from_spacing(
        nx=args.nx,
        dx=args.dx,
        spacing_m=args.shot_spacing,
        margin_cells=args.shot_margin_cells,
    )
    shots = limited_shots(all_shots, args.max_shots)
    if not shots:
        raise ValueError("no shots selected")
    padded_shots = [shot + args.pad_x for shot in shots]

    def receiver_record(source_x: int, synthetic_record: np.ndarray) -> np.ndarray:
        if args.no_direct_mute:
            return synthetic_record
        return mute_direct_arrivals(
            synthetic_record,
            padded_cfg,
            source_x=source_x,
            direct_velocity=args.direct_velocity,
            padding_time=args.direct_mute_padding,
            taper_time=args.direct_mute_taper,
        )

    print(
        f"Running Scheme 2 imaging-condition smoke test for {len(shots)} shots, nt={args.nt}...",
        flush=True,
    )
    direct_mute_params = None
    if not args.no_direct_mute:
        direct_mute_params = {
            "direct_velocity": args.direct_velocity,
            "padding_time": args.direct_mute_padding,
            "taper_time": args.direct_mute_taper,
        }

    def progress(completed: int, total: int, source_x: int) -> None:
        print(f"Finished shot {completed}/{total}: padded source_x={source_x}", flush=True)

    result = multishot_reverse_time_migrate_parallel(
        rtm_velocity,
        padded_cfg,
        shot_positions=padded_shots,
        work_dir=args.output_dir / "scheme2_parallel_wavefields",
        workers=args.workers,
        laplacian_power=args.laplacian_power,
        migration_velocity=migration_velocity,
        subtract_direct_wave=not args.no_direct_subtract,
        min_illumination_fraction=args.min_illumination_fraction,
        direct_mute_params=direct_mute_params,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        checkpoint_interval=args.checkpoint_interval,
        progress_callback=progress,
    )

    arrays = crop_result_arrays(result, original_shape, pad_x=args.pad_x, pad_top=args.pad_top)
    output_arrays = save_arrays(args.output_dir, arrays)
    metrics = image_metrics(arrays)
    save_compare_figure(args.output_dir / "scheme2_compare.png", arrays, args)

    params = {
        "model": str(args.model),
        "output_dir": str(args.output_dir),
        "config": cfg.__dict__,
        "padded_config": padded_cfg.__dict__,
        "shot_positions": shots,
        "padded_shot_positions": padded_shots,
        "workers": args.workers,
        "checkpoint_dir": str(args.checkpoint_dir),
        "resume": args.resume,
        "checkpoint_interval": args.checkpoint_interval,
        "laplacian_power": args.laplacian_power,
        "min_illumination_fraction": args.min_illumination_fraction,
        "smooth_radius_x": args.smooth_radius_x,
        "smooth_radius_z": args.smooth_radius_z,
        "smooth_passes": args.smooth_passes,
        "direct_subtract_enabled": not args.no_direct_subtract,
        "direct_mute_enabled": not args.no_direct_mute,
        "metrics": metrics,
        "outputs": output_arrays | {"compare_figure": "scheme2_compare.png", "report": "scheme2_report.md"},
    }
    (args.output_dir / "scheme2_parameters.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    write_report(
        args.output_dir / "scheme2_report.md",
        args=args,
        shots=shots,
        padded_shots=padded_shots,
        arrays=output_arrays,
        metrics=metrics,
    )
    print(f"Saved Scheme 2 comparison outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
