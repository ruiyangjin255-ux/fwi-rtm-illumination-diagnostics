from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from admit_fwi.acoustic_rtm import (
    RTMConfig,
    forward_model,
    read_binary_model,
    read_shot_record,
    reverse_time_boundary_migrate,
    reverse_time_migrate,
    save_boundary_migration_outputs,
    save_rtm_outputs,
)
from admit_fwi.plot_rtm_result import save_rtm_figure


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "fd2d_pml" / "vel" / "seg676x230.bin"
DEFAULT_RECORD = ROOT / "acoustic_results_multitime" / "seg" / "seg_record.bin"
DEFAULT_OUTPUT = ROOT / "admit_fwi" / "outputs" / "seg_salt_rtm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run acoustic RTM for the existing SEG/salt velocity model."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--nx", type=int, default=676)
    parser.add_argument("--nz", type=int, default=230)
    parser.add_argument("--dx", type=float, default=10.0)
    parser.add_argument("--dz", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--nt", type=int, default=4001)
    parser.add_argument("--record-nt", type=int, default=4001)
    parser.add_argument("--f0", type=float, default=20.0)
    parser.add_argument("--source-x", type=int, default=340)
    parser.add_argument("--source-z", type=int, default=1)
    parser.add_argument("--receiver-z", type=int, default=1)
    parser.add_argument("--fd-order", type=int, default=8)
    parser.add_argument("--absorb-cells", type=int, default=40)
    parser.add_argument("--absorb-strength", type=float, default=3.0)
    parser.add_argument("--laplacian-power", type=int, default=2)
    parser.add_argument(
        "--synthetic-record",
        action="store_true",
        help="Use the Python forward-modeled record instead of the existing C record.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = RTMConfig(
        nx=args.nx,
        nz=args.nz,
        dx=args.dx,
        dz=args.dz,
        dt=args.dt,
        nt=args.nt,
        f0=args.f0,
        source_x=args.source_x,
        source_z=args.source_z,
        receiver_z=args.receiver_z,
        absorb_cells=args.absorb_cells,
        absorb_strength=args.absorb_strength,
        fd_order=args.fd_order,
    )

    velocity = read_binary_model(args.model, nx=args.nx, nz=args.nz)
    source_wavefield_path = args.output_dir / "source_wavefield_float32.dat"

    print("Forward propagating source wavefield...")
    synthetic_record = forward_model(velocity, cfg, wavefield_path=source_wavefield_path)

    if args.synthetic_record:
        record = synthetic_record
        record_source = "python_forward_model"
    else:
        if args.nt > args.record_nt:
            raise ValueError("--nt cannot exceed --record-nt when reading an existing record")
        record = read_shot_record(args.record, nx=args.nx, nt=args.record_nt)[: args.nt, :]
        record_source = str(args.record)

    print("Reverse-time migrating receiver wavefield...")
    result = reverse_time_migrate(
        velocity,
        record,
        cfg,
        source_wavefield_path=source_wavefield_path,
        laplacian_power=args.laplacian_power,
    )
    save_rtm_outputs(args.output_dir, result, record=record)

    print("Reverse-time migrating with surface record as boundary...")
    boundary_image, boundary_filtered = reverse_time_boundary_migrate(
        velocity,
        record,
        cfg,
        laplacian_power=args.laplacian_power,
    )
    save_boundary_migration_outputs(args.output_dir, boundary_image, boundary_filtered)

    np.save(args.output_dir / "velocity.npy", velocity.astype(np.float32))
    save_rtm_figure(
        output_path=args.output_dir / "seg_salt_rtm_panel.png",
        velocity=velocity,
        record=record,
        raw_image=result.image,
        illumination=result.illumination,
        normalized_image=result.normalized_image,
        filtered_image=result.filtered_image,
        boundary_image=boundary_image,
        boundary_filtered_image=boundary_filtered,
        dx=args.dx,
        dz=args.dz,
        dt=args.dt,
    )

    params = {
        "model": str(args.model),
        "record_source": record_source,
        "output_dir": str(args.output_dir),
        "config": cfg.__dict__,
        "laplacian_power": args.laplacian_power,
        "outputs": {
            "source_wavefield": str(source_wavefield_path),
            "raw_image": "rtm_image_raw.npy",
            "source_normalized_image": "rtm_image_source_normalized.npy",
            "laplacian_filtered_image": "rtm_image_laplacian_filtered.npy",
            "wang_boundary_migration_image": "wang_boundary_migration_image.npy",
            "wang_boundary_migration_laplacian_filtered": "wang_boundary_migration_laplacian_filtered.npy",
            "figure": "seg_salt_rtm_panel.png",
        },
    }
    (args.output_dir / "run_parameters.json").write_text(
        json.dumps(params, indent=2), encoding="utf-8"
    )
    print(f"Saved RTM outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
