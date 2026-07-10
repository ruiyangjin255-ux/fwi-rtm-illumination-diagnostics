from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from admit_fwi.acoustic_rtm import (
    RTMConfig,
    preprocess_migration_section,
    preprocess_stacked_record,
    read_binary_model,
    reverse_time_boundary_migrate,
    shot_positions_from_spacing,
    stack_surface_records,
)
from admit_fwi.plot_paper_style import (
    save_migration_figure,
    save_record_and_migration_figure,
    save_record_figure,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "fd2d_pml" / "vel" / "seg676x230.bin"
DEFAULT_OUTPUT = ROOT / "admit_fwi" / "outputs" / "seg_salt_paper_style"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Wang-style stacked seismic record and boundary migration section."
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
    parser.add_argument("--source-z", type=int, default=1)
    parser.add_argument("--receiver-z", type=int, default=1)
    parser.add_argument("--shot-spacing", type=float, default=30.0)
    parser.add_argument("--shot-margin-cells", type=int, default=4)
    parser.add_argument("--max-shots", type=int, default=0, help="0 means use all generated shots.")
    parser.add_argument("--fd-order", type=int, default=8)
    parser.add_argument("--absorb-cells", type=int, default=40)
    parser.add_argument("--absorb-strength", type=float, default=3.0)
    parser.add_argument("--laplacian-power", type=int, default=1)
    parser.add_argument("--mute-time", type=float, default=0.18)
    parser.add_argument("--time-power", type=float, default=1.0)
    parser.add_argument("--migration-depth-power", type=float, default=0.15)
    parser.add_argument("--migration-clip-percentile", type=float, default=99.5)
    parser.add_argument("--migration-trace-balance", type=float, default=0.25)
    parser.add_argument("--migration-output-clip", type=float, default=0.80)
    parser.add_argument(
        "--stack-mode",
        choices=["mean", "signed_rms", "zero_offset", "normal_incidence"],
        default="normal_incidence",
    )
    parser.add_argument("--reuse-stacked", action="store_true")
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
    all_shots = shot_positions_from_spacing(
        nx=args.nx,
        dx=args.dx,
        spacing_m=args.shot_spacing,
        margin_cells=args.shot_margin_cells,
    )
    shots = limited_shots(all_shots, args.max_shots)
    if not shots:
        raise ValueError("no shots selected")

    raw_stack_path = args.output_dir / "paper_style_stacked_record_raw.npy"
    if args.reuse_stacked and raw_stack_path.exists():
        print(f"Reusing stacked record from {raw_stack_path}...")
        stacked_raw = np.load(raw_stack_path)
        shot_count = len(shots)
    else:
        if args.stack_mode == "normal_incidence":
            print("Synthesizing normal-incidence stacked seismic record...")
        else:
            print(f"Forward modeling and stacking {len(shots)} shots...")
        stacked_raw, shot_count = stack_surface_records(velocity, cfg, shots, stack_mode=args.stack_mode)
        np.save(raw_stack_path, stacked_raw)
    stacked_record = preprocess_stacked_record(
        stacked_raw,
        dt=args.dt,
        mute_time=args.mute_time,
        time_power=args.time_power,
    )
    print("Reverse-time migrating stacked record as surface boundary...")
    migration, migration_filtered = reverse_time_boundary_migrate(
        velocity,
        stacked_record,
        replace(cfg, source_x=shots[len(shots) // 2]),
        laplacian_power=args.laplacian_power,
    )
    display_migration = preprocess_migration_section(
        migration,
        depth_power=args.migration_depth_power,
        clip_percentile=args.migration_clip_percentile,
        trace_balance=args.migration_trace_balance,
        output_clip=args.migration_output_clip,
    )

    np.save(args.output_dir / "paper_style_stacked_record.npy", stacked_record)
    np.save(args.output_dir / "paper_style_migration_section.npy", display_migration)
    np.save(args.output_dir / "paper_style_migration_unfiltered.npy", migration)
    stacked_record.astype(np.float32).tofile(args.output_dir / "paper_style_stacked_record.bin")
    display_migration.astype(np.float32).tofile(args.output_dir / "paper_style_migration_section.bin")

    save_record_figure(
        args.output_dir / "paper_style_stacked_record.png",
        stacked_record,
        dx=args.dx,
        dt=args.dt,
        title="Stacked seismic record",
    )
    save_migration_figure(
        args.output_dir / "paper_style_migration_section.png",
        display_migration,
        dx=args.dx,
        dz=args.dz,
        title="Migration section",
    )
    save_record_and_migration_figure(
        args.output_dir / "paper_style_record_and_migration.png",
        stacked_record,
        display_migration,
        dx=args.dx,
        dz=args.dz,
        dt=args.dt,
    )

    params = {
        "model": str(args.model),
        "output_dir": str(args.output_dir),
        "config": cfg.__dict__,
        "shot_spacing_m": args.shot_spacing,
        "shot_margin_cells": args.shot_margin_cells,
        "all_shot_count": len(all_shots),
        "used_shot_count": shot_count,
        "shot_positions": shots,
        "mute_time": args.mute_time,
        "time_power": args.time_power,
        "migration_depth_power": args.migration_depth_power,
        "migration_clip_percentile": args.migration_clip_percentile,
        "migration_trace_balance": args.migration_trace_balance,
        "migration_output_clip": args.migration_output_clip,
        "stack_mode": args.stack_mode,
        "outputs": {
            "stacked_record": "paper_style_stacked_record.npy",
            "migration_section": "paper_style_migration_section.npy",
            "combined_figure": "paper_style_record_and_migration.png",
        },
    }
    (args.output_dir / "run_parameters.json").write_text(
        json.dumps(params, indent=2), encoding="utf-8"
    )
    print(f"Saved paper-style outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
