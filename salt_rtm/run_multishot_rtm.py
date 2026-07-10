from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from salt_rtm.acoustic_rtm import (
    RTMConfig,
    crop_padded_model,
    crop_padded_record,
    multishot_reverse_time_migrate,
    multishot_reverse_time_migrate_parallel,
    mute_direct_arrivals,
    pad_rtm_config,
    pad_velocity_model,
    preprocess_migration_section,
    preprocess_stacked_record,
    read_binary_model,
    shot_positions_from_spacing,
    smooth_velocity_model,
)
from salt_rtm.plot_paper_style import (
    save_migration_figure,
    save_record_and_migration_figure,
    save_record_figure,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "data" / "seg676x230.bin"
DEFAULT_OUTPUT = ROOT / "outputs" / "seg_salt_multishot_rtm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run true prestack multi-shot acoustic RTM with zero-lag cross-correlation imaging."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--nx", type=int, default=676)
    parser.add_argument("--nz", type=int, default=230)
    parser.add_argument("--dx", type=float, default=10.0)
    parser.add_argument("--dz", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--nt", type=int, default=4001)
    parser.add_argument("--f0", type=float, default=20.0)
    parser.add_argument("--source-z", type=int, default=4)
    parser.add_argument("--receiver-z", type=int, default=4)
    parser.add_argument("--shot-spacing", type=float, default=30.0)
    parser.add_argument("--shot-margin-cells", type=int, default=4)
    parser.add_argument("--max-shots", type=int, default=0, help="0 means use all generated shots.")
    parser.add_argument("--fd-order", type=int, default=8)
    parser.add_argument("--absorb-cells", type=int, default=40)
    parser.add_argument("--absorb-strength", type=float, default=3.0)
    parser.add_argument("--laplacian-power", type=int, default=1)
    parser.add_argument("--min-illumination-fraction", type=float, default=0.01)
    parser.add_argument("--smooth-radius-x", type=int, default=10)
    parser.add_argument("--smooth-radius-z", type=int, default=10)
    parser.add_argument("--smooth-passes", type=int, default=2)
    parser.add_argument("--pad-x", type=int, default=0, help="Edge padding cells added to both left and right sides.")
    parser.add_argument("--pad-top", type=int, default=0, help="Edge padding cells added above the model.")
    parser.add_argument("--pad-bottom", type=int, default=0, help="Edge padding cells added below the model.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel shot workers. 1 keeps the serial implementation.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for per-shot accumulated checkpoint files. Parallel mode saves after completed shots.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume a parallel RTM run from --checkpoint-dir.")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1,
        help="Save checkpoint every N completed shots in parallel mode.",
    )
    parser.add_argument("--no-direct-subtract", action="store_true")
    parser.add_argument("--direct-velocity", type=float, default=2000.0)
    parser.add_argument("--direct-mute-padding", type=float, default=0.04)
    parser.add_argument("--direct-mute-taper", type=float, default=0.02)
    parser.add_argument("--no-direct-mute", action="store_true")
    parser.add_argument("--record-mute-time", type=float, default=0.0)
    parser.add_argument("--record-time-power", type=float, default=0.2)
    parser.add_argument("--migration-depth-power", type=float, default=0.15)
    parser.add_argument("--migration-clip-percentile", type=float, default=99.5)
    parser.add_argument("--migration-trace-balance", type=float, default=0.25)
    parser.add_argument("--migration-output-clip", type=float, default=0.80)
    return parser.parse_args()


def limited_shots(shots: list[int], max_shots: int) -> list[int]:
    if max_shots <= 0 or max_shots >= len(shots):
        return shots
    indices = np.linspace(0, len(shots) - 1, max_shots).round().astype(int)
    return [shots[i] for i in sorted(set(indices.tolist()))]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
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
    padded_cfg = pad_rtm_config(
        cfg,
        pad_x=args.pad_x,
        pad_top=args.pad_top,
        pad_bottom=args.pad_bottom,
    )
    rtm_velocity = pad_velocity_model(
        velocity,
        pad_x=args.pad_x,
        pad_top=args.pad_top,
        pad_bottom=args.pad_bottom,
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

    migration_velocity = smooth_velocity_model(
        rtm_velocity,
        radius_z=args.smooth_radius_z,
        radius_x=args.smooth_radius_x,
        passes=args.smooth_passes,
    )
    np.save(
        args.output_dir / "migration_velocity_smooth_padded.npy",
        migration_velocity.astype(np.float32),
    )
    np.save(
        args.output_dir / "migration_velocity_smooth.npy",
        crop_padded_model(
            migration_velocity,
            original_shape=original_shape,
            pad_x=args.pad_x,
            pad_top=args.pad_top,
        ).astype(np.float32),
    )

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

    print(f"Running true multi-shot RTM for {len(shots)} shots with {args.workers} worker(s)...", flush=True)
    if args.workers == 1:
        if args.checkpoint_dir is not None or args.resume:
            print("Checkpoint/resume is only used in parallel mode; serial run will ignore it.", flush=True)
        result = multishot_reverse_time_migrate(
            rtm_velocity,
            padded_cfg,
            shot_positions=padded_shots,
            wavefield_path=args.output_dir / "source_wavefield_reused_float32.dat",
            laplacian_power=args.laplacian_power,
            migration_velocity=migration_velocity,
            subtract_direct_wave=not args.no_direct_subtract,
            min_illumination_fraction=args.min_illumination_fraction,
            record_provider=receiver_record,
        )
    else:
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
            work_dir=args.output_dir / "parallel_wavefields",
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
    stacked_record = crop_padded_record(
        result.stacked_record,
        original_nx=args.nx,
        pad_x=args.pad_x,
    ).astype(np.float32)
    image = crop_padded_model(
        result.image,
        original_shape=original_shape,
        pad_x=args.pad_x,
        pad_top=args.pad_top,
    ).astype(np.float32)
    illumination = crop_padded_model(
        result.illumination,
        original_shape=original_shape,
        pad_x=args.pad_x,
        pad_top=args.pad_top,
    ).astype(np.float32)
    normalized_image = crop_padded_model(
        result.normalized_image,
        original_shape=original_shape,
        pad_x=args.pad_x,
        pad_top=args.pad_top,
    ).astype(np.float32)
    receiver_illumination = crop_padded_model(
        result.receiver_illumination,
        original_shape=original_shape,
        pad_x=args.pad_x,
        pad_top=args.pad_top,
    ).astype(np.float32)
    source_receiver_normalized = crop_padded_model(
        result.source_receiver_normalized_image,
        original_shape=original_shape,
        pad_x=args.pad_x,
        pad_top=args.pad_top,
    ).astype(np.float32)
    laplacian_image = crop_padded_model(
        result.laplacian_image,
        original_shape=original_shape,
        pad_x=args.pad_x,
        pad_top=args.pad_top,
    ).astype(np.float32)
    laplacian_normalized = crop_padded_model(
        result.laplacian_normalized_image,
        original_shape=original_shape,
        pad_x=args.pad_x,
        pad_top=args.pad_top,
    ).astype(np.float32)
    filtered_image = crop_padded_model(
        result.filtered_image,
        original_shape=original_shape,
        pad_x=args.pad_x,
        pad_top=args.pad_top,
    ).astype(np.float32)

    display_record = preprocess_stacked_record(
        stacked_record,
        dt=args.dt,
        mute_time=args.record_mute_time,
        time_power=args.record_time_power,
    )
    display_migration = preprocess_migration_section(
        filtered_image,
        depth_power=args.migration_depth_power,
        clip_percentile=args.migration_clip_percentile,
        trace_balance=args.migration_trace_balance,
        output_clip=args.migration_output_clip,
    )

    np.save(args.output_dir / "multishot_stacked_record_raw.npy", stacked_record)
    np.save(args.output_dir / "multishot_stacked_record.npy", display_record)
    np.save(args.output_dir / "multishot_rtm_image_raw.npy", image)
    np.save(args.output_dir / "multishot_rtm_illumination.npy", illumination)
    np.save(args.output_dir / "multishot_rtm_receiver_illumination.npy", receiver_illumination)
    np.save(args.output_dir / "multishot_rtm_source_normalized.npy", normalized_image)
    np.save(args.output_dir / "multishot_rtm_source_receiver_normalized.npy", source_receiver_normalized)
    np.save(args.output_dir / "multishot_rtm_laplacian_image.npy", laplacian_image)
    np.save(args.output_dir / "multishot_rtm_laplacian_source_normalized.npy", laplacian_normalized)
    np.save(args.output_dir / "multishot_rtm_laplacian_filtered.npy", filtered_image)
    np.save(args.output_dir / "multishot_rtm_display.npy", display_migration)
    display_record.astype(np.float32).tofile(args.output_dir / "multishot_stacked_record.bin")
    display_migration.astype(np.float32).tofile(args.output_dir / "multishot_rtm_display.bin")

    save_record_figure(
        args.output_dir / "multishot_stacked_record.png",
        display_record,
        dx=args.dx,
        dt=args.dt,
        title="True multi-shot stacked record",
    )
    save_migration_figure(
        args.output_dir / "multishot_rtm_migration_section.png",
        display_migration,
        dx=args.dx,
        dz=args.dz,
        title="True multi-shot RTM migration section",
    )
    save_record_and_migration_figure(
        args.output_dir / "multishot_record_and_rtm.png",
        display_record,
        display_migration,
        dx=args.dx,
        dz=args.dz,
        dt=args.dt,
    )

    params = {
        "model": str(args.model),
        "output_dir": str(args.output_dir),
        "config": cfg.__dict__,
        "padded_config": padded_cfg.__dict__,
        "padding": {
            "pad_x": args.pad_x,
            "pad_top": args.pad_top,
            "pad_bottom": args.pad_bottom,
            "crop_window": {
                "z_start": args.pad_top,
                "z_stop": args.pad_top + args.nz,
                "x_start": args.pad_x,
                "x_stop": args.pad_x + args.nx,
            },
        },
        "shot_spacing_m": args.shot_spacing,
        "shot_margin_cells": args.shot_margin_cells,
        "all_shot_count": len(all_shots),
        "used_shot_count": result.shot_count,
        "shot_positions": shots,
        "padded_shot_positions": padded_shots,
        "workers": args.workers,
        "checkpoint_dir": None if args.checkpoint_dir is None else str(args.checkpoint_dir),
        "resume": args.resume,
        "checkpoint_interval": args.checkpoint_interval,
        "laplacian_power": args.laplacian_power,
        "min_illumination_fraction": args.min_illumination_fraction,
        "smooth_radius_x": args.smooth_radius_x,
        "smooth_radius_z": args.smooth_radius_z,
        "smooth_passes": args.smooth_passes,
        "direct_subtract_enabled": not args.no_direct_subtract,
        "direct_mute_enabled": not args.no_direct_mute,
        "direct_velocity": args.direct_velocity,
        "direct_mute_padding": args.direct_mute_padding,
        "direct_mute_taper": args.direct_mute_taper,
        "record_mute_time": args.record_mute_time,
        "record_time_power": args.record_time_power,
        "migration_depth_power": args.migration_depth_power,
        "migration_clip_percentile": args.migration_clip_percentile,
        "migration_trace_balance": args.migration_trace_balance,
        "migration_output_clip": args.migration_output_clip,
        "migration_method": "prestack_multishot_zero_lag_cross_correlation",
        "outputs": {
            "stacked_record": "multishot_stacked_record.npy",
            "rtm_raw": "multishot_rtm_image_raw.npy",
            "rtm_receiver_illumination": "multishot_rtm_receiver_illumination.npy",
            "rtm_source_normalized": "multishot_rtm_source_normalized.npy",
            "rtm_source_receiver_normalized": "multishot_rtm_source_receiver_normalized.npy",
            "rtm_laplacian_image": "multishot_rtm_laplacian_image.npy",
            "rtm_laplacian_source_normalized": "multishot_rtm_laplacian_source_normalized.npy",
            "rtm_laplacian_filtered": "multishot_rtm_laplacian_filtered.npy",
            "rtm_display": "multishot_rtm_display.npy",
            "migration_velocity": "migration_velocity_smooth.npy",
            "migration_velocity_padded": "migration_velocity_smooth_padded.npy",
            "combined_figure": "multishot_record_and_rtm.png",
        },
    }
    (args.output_dir / "run_parameters.json").write_text(
        json.dumps(params, indent=2), encoding="utf-8"
    )
    print(f"Saved true multi-shot RTM outputs to {args.output_dir}")


if __name__ == "__main__":
    main()

