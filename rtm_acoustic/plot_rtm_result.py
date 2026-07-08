from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "rtm_acoustic" / "outputs" / "seg_salt_rtm"


def percentile_clip(image: np.ndarray, percentile: float = 99.0) -> float:
    value = float(np.percentile(np.abs(image), percentile))
    return value if value > 0.0 else 1.0


def save_rtm_figure(
    output_path: str | Path,
    velocity: np.ndarray,
    record: np.ndarray,
    raw_image: np.ndarray,
    illumination: np.ndarray,
    normalized_image: np.ndarray,
    filtered_image: np.ndarray,
    boundary_image: np.ndarray,
    boundary_filtered_image: np.ndarray,
    dx: float,
    dz: float,
    dt: float,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_km = np.arange(velocity.shape[1]) * dx / 1000.0
    z_km = np.arange(velocity.shape[0]) * dz / 1000.0
    t_s = np.arange(record.shape[0]) * dt
    extent_model = [x_km[0], x_km[-1], z_km[-1], z_km[0]]
    extent_record = [x_km[0], x_km[-1], t_s[-1], t_s[0]]

    fig, axes = plt.subplots(3, 3, figsize=(13.0, 10.2), constrained_layout=True)
    ax = axes[0, 0]
    im = ax.imshow(velocity / 1000.0, cmap="viridis", aspect="auto", extent=extent_model)
    ax.set_title("SEG/salt velocity")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Depth (km)")
    fig.colorbar(im, ax=ax, label="km/s")

    ax = axes[0, 1]
    clip = percentile_clip(record, 99.0)
    im = ax.imshow(record, cmap="gray", vmin=-clip, vmax=clip, aspect="auto", extent=extent_record)
    ax.set_title("Input shot record")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Time (s)")
    fig.colorbar(im, ax=ax)

    ax = axes[0, 2]
    clip = percentile_clip(raw_image, 99.0)
    im = ax.imshow(raw_image, cmap="seismic", vmin=-clip, vmax=clip, aspect="auto", extent=extent_model)
    ax.set_title("Zero-lag cross-correlation")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Depth (km)")
    fig.colorbar(im, ax=ax)

    ax = axes[1, 0]
    illum = np.log10(np.maximum(illumination, 1.0e-20))
    im = ax.imshow(illum, cmap="magma", aspect="auto", extent=extent_model)
    ax.set_title("Source illumination")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Depth (km)")
    fig.colorbar(im, ax=ax)

    ax = axes[1, 1]
    clip = percentile_clip(normalized_image, 99.0)
    im = ax.imshow(
        normalized_image,
        cmap="seismic",
        vmin=-clip,
        vmax=clip,
        aspect="auto",
        extent=extent_model,
    )
    ax.set_title("Source-normalized image")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Depth (km)")
    fig.colorbar(im, ax=ax)

    ax = axes[1, 2]
    clip = percentile_clip(filtered_image, 99.0)
    im = ax.imshow(
        filtered_image,
        cmap="seismic",
        vmin=-clip,
        vmax=clip,
        aspect="auto",
        extent=extent_model,
    )
    ax.set_title("High-order Laplacian filtered")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Depth (km)")
    fig.colorbar(im, ax=ax)

    ax = axes[2, 0]
    clip = percentile_clip(boundary_image, 99.0)
    im = ax.imshow(
        boundary_image,
        cmap="seismic",
        vmin=-clip,
        vmax=clip,
        aspect="auto",
        extent=extent_model,
    )
    ax.set_title("Wang-style boundary migration")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Depth (km)")
    fig.colorbar(im, ax=ax)

    ax = axes[2, 1]
    clip = percentile_clip(boundary_filtered_image, 99.0)
    im = ax.imshow(
        boundary_filtered_image,
        cmap="seismic",
        vmin=-clip,
        vmax=clip,
        aspect="auto",
        extent=extent_model,
    )
    ax.set_title("Boundary migration filtered")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Depth (km)")
    fig.colorbar(im, ax=ax)

    ax = axes[2, 2]
    clip = percentile_clip(boundary_filtered_image, 99.0)
    wiggle_image = boundary_filtered_image / clip
    ax.imshow(
        wiggle_image,
        cmap="gray",
        vmin=-1.0,
        vmax=1.0,
        aspect="auto",
        extent=extent_model,
    )
    ax.set_title("Boundary migration section")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Depth (km)")

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot acoustic RTM outputs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dx", type=float, default=10.0)
    parser.add_argument("--dz", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_rtm_figure(
        output_path=args.output_dir / "seg_salt_rtm_panel.png",
        velocity=np.load(args.output_dir / "velocity.npy"),
        record=np.load(args.output_dir / "input_record.npy"),
        raw_image=np.load(args.output_dir / "rtm_image_raw.npy"),
        illumination=np.load(args.output_dir / "rtm_source_illumination.npy"),
        normalized_image=np.load(args.output_dir / "rtm_image_source_normalized.npy"),
        filtered_image=np.load(args.output_dir / "rtm_image_laplacian_filtered.npy"),
        boundary_image=np.load(args.output_dir / "wang_boundary_migration_image.npy"),
        boundary_filtered_image=np.load(args.output_dir / "wang_boundary_migration_laplacian_filtered.npy"),
        dx=args.dx,
        dz=args.dz,
        dt=args.dt,
    )
    print(f"Saved {args.output_dir / 'seg_salt_rtm_panel.png'}")


if __name__ == "__main__":
    main()
