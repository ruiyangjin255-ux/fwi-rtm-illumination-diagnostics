from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "rtm_acoustic" / "outputs" / "seg_salt_paper_style"


def _clip(data: np.ndarray, percentile: float = 99.0) -> float:
    value = float(np.percentile(np.abs(data), percentile))
    return value if value > 0.0 else 1.0


def _active_vertical_extent(
    section: np.ndarray,
    *,
    sample_interval: float,
    threshold_ratio: float = 0.01,
    padding_ratio: float = 0.05,
) -> tuple[float, float]:
    rms = np.sqrt(np.mean(np.asarray(section, dtype=np.float64) ** 2, axis=1))
    if rms.size == 0:
        return 0.0, 0.0
    threshold = max(float(np.max(rms)) * threshold_ratio, 1.0e-12)
    active = np.flatnonzero(rms > threshold)
    if active.size == 0:
        return 0.0, float((section.shape[0] - 1) * sample_interval)
    padding = max(2, int(round(active.size * padding_ratio)))
    last = min(section.shape[0] - 1, int(active[-1]) + padding)
    return 0.0, float(last * sample_interval)


def _suppress_weak_amplitudes(section: np.ndarray, *, percentile: float = 55.0) -> np.ndarray:
    if percentile <= 0.0:
        return np.asarray(section, dtype=np.float32).copy()
    if percentile >= 100.0:
        raise ValueError("percentile must be less than 100")
    data = np.asarray(section, dtype=np.float32)
    abs_data = np.abs(data)
    nonzero = abs_data[abs_data > 0.0]
    if nonzero.size == 0:
        return data.copy()
    threshold = np.float32(np.percentile(nonzero, percentile))
    original_clip = np.float32(_clip(data, 99.5))
    enhanced = np.sign(data) * np.maximum(abs_data - threshold, np.float32(0.0))
    enhanced_clip = np.float32(_clip(enhanced, 99.5))
    if enhanced_clip > 0.0:
        enhanced *= original_clip / enhanced_clip
    np.clip(enhanced, -original_clip, original_clip, out=enhanced)
    return enhanced.astype(np.float32, copy=False)


def _plot_wiggle_section(
    ax,
    section: np.ndarray,
    *,
    dx: float,
    dt_or_dz: float,
    vertical_label: str,
    scale: float = 0.75,
    fill_positive: bool = True,
    max_traces: int = 240,
    trace_normalize: bool = True,
    variable_density: bool = True,
    density_alpha: float = 0.72,
    density_percentile: float = 98.5,
    line_width: float = 0.32,
    fill_alpha: float = 0.55,
    auto_vertical_extent: bool = False,
) -> None:
    n_vertical, nx = section.shape
    stride = max(1, int(np.ceil(nx / max_traces)))
    x_positions = np.arange(nx) * dx
    vertical = np.arange(n_vertical) * dt_or_dz
    if variable_density:
        clip = _clip(section, density_percentile)
        ax.imshow(
            section,
            cmap="gray",
            vmin=-clip,
            vmax=clip,
            aspect="auto",
            extent=[x_positions[0], x_positions[-1], vertical[-1], vertical[0]],
            interpolation="nearest",
            alpha=density_alpha,
        )
    trace_scale = _clip(section[:, ::stride], 99.0)
    global_gain = dx * stride * scale / trace_scale

    for ix in range(0, nx, stride):
        trace = section[:, ix].astype(np.float64, copy=True)
        if trace_normalize:
            local = float(np.percentile(np.abs(trace), 99.0))
            if local > 0.0:
                trace /= local
            trace *= dx * stride * scale
        else:
            trace *= global_gain
        base = x_positions[ix]
        x_trace = base + trace
        ax.plot(x_trace, vertical, color="black", linewidth=line_width)
        if fill_positive:
            ax.fill_betweenx(
                vertical,
                base,
                x_trace,
                where=x_trace >= base,
                color="black",
                alpha=fill_alpha,
                linewidth=0.0,
            )

    ax.set_xlim(x_positions[0], x_positions[-1])
    if auto_vertical_extent:
        top, bottom = _active_vertical_extent(section, sample_interval=dt_or_dz)
        ax.set_ylim(bottom, top)
    else:
        ax.set_ylim(vertical[-1], vertical[0])
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel(vertical_label)
    ax.grid(True, color="0.55", linewidth=0.35)


def save_record_figure(
    output_path: str | Path,
    record: np.ndarray,
    *,
    dx: float,
    dt: float,
    title: str = "Stacked seismic record",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.8, 5.8), constrained_layout=True)
    _plot_wiggle_section(
        ax,
        record,
        dx=dx,
        dt_or_dz=dt,
        vertical_label="Time (s)",
        auto_vertical_extent=True,
    )
    ax.set_title(title)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_migration_figure(
    output_path: str | Path,
    migration: np.ndarray,
    *,
    dx: float,
    dz: float,
    title: str = "Migration section",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    migration = _suppress_weak_amplitudes(migration, percentile=0.0)
    fig, ax = plt.subplots(figsize=(10.8, 5.8), constrained_layout=True)
    _plot_wiggle_section(
        ax,
        migration,
        dx=dx,
        dt_or_dz=dz,
        vertical_label="Depth (m)",
        scale=0.55,
        fill_positive=True,
        max_traces=180,
        trace_normalize=False,
        density_alpha=1.0,
        density_percentile=99.0,
        line_width=0.20,
        fill_alpha=0.14,
    )
    ax.set_title(title)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_record_and_migration_figure(
    output_path: str | Path,
    record: np.ndarray,
    migration: np.ndarray,
    *,
    dx: float,
    dz: float,
    dt: float,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    migration = _suppress_weak_amplitudes(migration, percentile=0.0)
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 10.4), constrained_layout=True)
    _plot_wiggle_section(
        axes[0],
        record,
        dx=dx,
        dt_or_dz=dt,
        vertical_label="Time (s)",
        auto_vertical_extent=True,
    )
    axes[0].set_title("Stacked seismic record")
    _plot_wiggle_section(
        axes[1],
        migration,
        dx=dx,
        dt_or_dz=dz,
        vertical_label="Depth (m)",
        scale=0.55,
        max_traces=180,
        trace_normalize=False,
        density_alpha=1.0,
        density_percentile=99.0,
        line_width=0.20,
        fill_alpha=0.14,
    )
    axes[1].set_title("Migration section")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot paper-style stacked record and migration.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dx", type=float, default=10.0)
    parser.add_argument("--dz", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = np.load(args.output_dir / "paper_style_stacked_record.npy")
    migration = np.load(args.output_dir / "paper_style_migration_section.npy")
    save_record_figure(args.output_dir / "paper_style_stacked_record.png", record, dx=args.dx, dt=args.dt)
    save_migration_figure(args.output_dir / "paper_style_migration_section.png", migration, dx=args.dx, dz=args.dz)
    save_record_and_migration_figure(
        args.output_dir / "paper_style_record_and_migration.png",
        record,
        migration,
        dx=args.dx,
        dz=args.dz,
        dt=args.dt,
    )
    print(f"Saved paper-style figures to {args.output_dir}")


if __name__ == "__main__":
    main()
