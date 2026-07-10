from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class RTMConfig:
    nx: int
    nz: int
    dx: float
    dz: float
    dt: float
    nt: int
    f0: float
    source_x: int
    source_z: int
    receiver_z: int
    absorb_cells: int = 40
    absorb_strength: float = 3.0
    fd_order: int = 8
    source_delay: Optional[float] = None
    absorb_top: bool = False
    max_cfl: float = 0.68


@dataclass(frozen=True)
class RTMResult:
    image: Array
    illumination: Array
    receiver_illumination: Array
    normalized_image: Array
    source_receiver_normalized_image: Array
    laplacian_image: Array
    laplacian_normalized_image: Array
    filtered_image: Array


@dataclass(frozen=True)
class MultishotRTMResult:
    image: Array
    illumination: Array
    receiver_illumination: Array
    normalized_image: Array
    source_receiver_normalized_image: Array
    laplacian_image: Array
    laplacian_normalized_image: Array
    filtered_image: Array
    stacked_record: Array
    shot_count: int


@dataclass(frozen=True)
class ShotRTMPartial:
    source_x: int
    image: Array
    illumination: Array
    receiver_illumination: Array
    stacked_record: Array


def _atomic_save_npy(path: Path, array: Array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("wb") as handle:
        np.save(handle, np.asarray(array, dtype=np.float32))
    tmp_path.replace(path)


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _checkpoint_run_signature(
    config: RTMConfig,
    shot_positions: list[int],
    *,
    laplacian_power: int,
    subtract_direct_wave: bool,
    min_illumination_fraction: float,
    direct_mute_params: Optional[dict[str, float]],
) -> dict:
    return {
        "config": {
            "nx": config.nx,
            "nz": config.nz,
            "dx": config.dx,
            "dz": config.dz,
            "dt": config.dt,
            "nt": config.nt,
            "f0": config.f0,
            "source_z": config.source_z,
            "receiver_z": config.receiver_z,
            "absorb_cells": config.absorb_cells,
            "absorb_strength": config.absorb_strength,
            "fd_order": config.fd_order,
            "source_delay": config.source_delay,
            "absorb_top": config.absorb_top,
            "max_cfl": config.max_cfl,
        },
        "shot_positions": [int(source_x) for source_x in shot_positions],
        "laplacian_power": int(laplacian_power),
        "subtract_direct_wave": bool(subtract_direct_wave),
        "min_illumination_fraction": float(min_illumination_fraction),
        "direct_mute_params": None
        if direct_mute_params is None
        else {key: float(value) for key, value in sorted(direct_mute_params.items())},
    }


def _save_multishot_checkpoint(
    checkpoint_dir: Path,
    *,
    signature: dict,
    completed_shots: list[int],
    image_sum: Array,
    illumination_sum: Array,
    receiver_illumination_sum: Array,
    stacked_record_sum: Array,
) -> None:
    _atomic_save_npy(checkpoint_dir / "image_sum.npy", image_sum)
    _atomic_save_npy(checkpoint_dir / "illumination_sum.npy", illumination_sum)
    _atomic_save_npy(checkpoint_dir / "receiver_illumination_sum.npy", receiver_illumination_sum)
    _atomic_save_npy(checkpoint_dir / "stacked_record_sum.npy", stacked_record_sum)
    manifest = {
        "version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "completed_count": len(completed_shots),
        "completed_shots": [int(source_x) for source_x in completed_shots],
        "signature": signature,
        "arrays": {
            "image_sum": "image_sum.npy",
            "illumination_sum": "illumination_sum.npy",
            "receiver_illumination_sum": "receiver_illumination_sum.npy",
            "stacked_record_sum": "stacked_record_sum.npy",
        },
    }
    _atomic_write_json(checkpoint_dir / "checkpoint_manifest.json", manifest)


def _load_multishot_checkpoint(checkpoint_dir: Path, signature: dict) -> tuple[list[int], Array, Array, Array, Array]:
    manifest_path = checkpoint_dir / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("signature") != signature:
        raise ValueError("checkpoint does not match current RTM run")
    completed_shots = [int(source_x) for source_x in manifest.get("completed_shots", [])]
    image_sum = np.load(checkpoint_dir / "image_sum.npy").astype(np.float32, copy=False)
    illumination_sum = np.load(checkpoint_dir / "illumination_sum.npy").astype(np.float32, copy=False)
    receiver_illumination_sum = np.load(checkpoint_dir / "receiver_illumination_sum.npy").astype(np.float32, copy=False)
    stacked_record_sum = np.load(checkpoint_dir / "stacked_record_sum.npy").astype(np.float32, copy=False)
    return completed_shots, image_sum, illumination_sum, receiver_illumination_sum, stacked_record_sum


def read_binary_model(path: str | Path, nx: int, nz: int, dtype=np.float32) -> Array:
    """Read C-written x-major model data and return a (nz, nx) array."""
    path = Path(path)
    data = np.fromfile(path, dtype=dtype)
    expected = nx * nz
    if data.size != expected:
        raise ValueError(f"{path} has {data.size} samples, expected {expected}")
    return data.reshape(nx, nz).T.astype(np.float32, copy=False)


def write_binary_model(path: str | Path, model: Array) -> None:
    """Write a (nz, nx) model in the x-major layout used by the C solver."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(model, dtype=np.float32).T.tofile(path)


def read_shot_record(path: str | Path, nx: int, nt: int, dtype=np.float32) -> Array:
    """Read C-written gather data and return a (nt, nx) record."""
    path = Path(path)
    data = np.fromfile(path, dtype=dtype)
    expected = nx * nt
    if data.size != expected:
        raise ValueError(f"{path} has {data.size} samples, expected {expected}")
    return data.reshape(nx, nt).T.astype(np.float32, copy=False)


def write_shot_record(path: str | Path, record: Array) -> None:
    """Write a (nt, nx) record in the x-major C solver layout."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(record, dtype=np.float32).T.tofile(path)


def pad_velocity_model(
    velocity: Array,
    pad_x: int = 0,
    pad_top: int = 0,
    pad_bottom: int = 0,
) -> Array:
    """Extend a velocity model with edge-value padding for absorbing boundaries."""
    if pad_x < 0 or pad_top < 0 or pad_bottom < 0:
        raise ValueError("padding values must be non-negative")
    velocity = np.asarray(velocity, dtype=np.float32)
    if velocity.ndim != 2:
        raise ValueError("velocity must be a 2-D array")
    return np.pad(
        velocity,
        pad_width=((int(pad_top), int(pad_bottom)), (int(pad_x), int(pad_x))),
        mode="edge",
    ).astype(np.float32, copy=False)


def crop_padded_model(
    model: Array,
    original_shape: tuple[int, int],
    pad_x: int = 0,
    pad_top: int = 0,
) -> Array:
    """Crop a padded (nz, nx) model or image back to the original physical window."""
    if pad_x < 0 or pad_top < 0:
        raise ValueError("padding values must be non-negative")
    original_nz, original_nx = original_shape
    z0 = int(pad_top)
    x0 = int(pad_x)
    return np.asarray(model)[z0 : z0 + int(original_nz), x0 : x0 + int(original_nx)]


def crop_padded_record(record: Array, original_nx: int, pad_x: int = 0) -> Array:
    """Crop a padded surface record back to the original receiver aperture."""
    if pad_x < 0:
        raise ValueError("pad_x must be non-negative")
    x0 = int(pad_x)
    return np.asarray(record)[:, x0 : x0 + int(original_nx)]


def pad_rtm_config(
    config: RTMConfig,
    pad_x: int = 0,
    pad_top: int = 0,
    pad_bottom: int = 0,
) -> RTMConfig:
    """Return an RTMConfig whose grid and source/receiver coordinates include padding."""
    if pad_x < 0 or pad_top < 0 or pad_bottom < 0:
        raise ValueError("padding values must be non-negative")
    return replace(
        config,
        nx=config.nx + 2 * int(pad_x),
        nz=config.nz + int(pad_top) + int(pad_bottom),
        source_x=config.source_x + int(pad_x),
        source_z=config.source_z + int(pad_top),
        receiver_z=config.receiver_z + int(pad_top),
    )


def ricker_wavelet(nt: int, dt: float, f0: float, delay: Optional[float] = None) -> Array:
    delay = 1.0 / f0 if delay is None else delay
    t = np.arange(nt, dtype=np.float64) * dt
    arg = np.pi * f0 * (t - delay)
    return ((1.0 - 2.0 * arg**2) * np.exp(-(arg**2))).astype(np.float32)


def _centered_ricker(f0: float, dt: float, half_length: int) -> Array:
    samples = np.arange(-half_length, half_length + 1, dtype=np.float64)
    arg = np.pi * f0 * samples * dt
    return ((1.0 - 2.0 * arg**2) * np.exp(-(arg**2))).astype(np.float32)


def synthesize_normal_incidence_stack(
    velocity: Array,
    dz: float,
    dt: float,
    nt: int,
    f0: float,
) -> Array:
    """Build a zero-offset stacked section from velocity reflectivity.

    This produces the paper-style stacked seismic section used as a clean
    boundary condition for Wang-style reverse-time migration. Reflectors are
    mapped to two-way vertical time and convolved trace-by-trace with a Ricker
    wavelet.
    """
    velocity = np.asarray(velocity, dtype=np.float32)
    nz, nx = velocity.shape
    record = np.zeros((nt, nx), dtype=np.float32)
    v_upper = velocity[:-1, :]
    v_lower = velocity[1:, :]
    reflectivity = (v_lower - v_upper) / np.maximum(v_lower + v_upper, 1.0e-6)
    lateral = np.zeros_like(reflectivity)
    lateral[:, 1:-1] = 0.25 * (velocity[:-1, 2:] - velocity[:-1, :-2]) / np.maximum(velocity[:-1, 1:-1], 1.0)
    reflectivity += lateral
    one_way = np.cumsum(dz / np.maximum(velocity[:-1, :], 1.0), axis=0)
    two_way_samples = np.rint(2.0 * one_way / dt).astype(np.int64)
    for ix in range(nx):
        valid = (two_way_samples[:, ix] >= 0) & (two_way_samples[:, ix] < nt)
        np.add.at(record[:, ix], two_way_samples[valid, ix], reflectivity[valid, ix])
    half_length = max(8, int(round(2.0 / (f0 * dt))))
    half_length = min(half_length, max(1, (nt - 1) // 2))
    wavelet = _centered_ricker(f0, dt, half_length)
    for ix in range(nx):
        record[:, ix] = np.convolve(record[:, ix], wavelet, mode="same")
    return record.astype(np.float32, copy=False)


def finite_difference_second_coefficients(order: int) -> Array:
    """Central finite-difference coefficients for a second derivative.

    The returned vector is [c0, c1, ... cr], where r = order / 2 and
    d2f/dx2 ~= (c0*f_i + sum_m c_m*(f_{i+m}+f_{i-m})) / dx**2.
    """
    if order < 2 or order % 2:
        raise ValueError("order must be an even integer >= 2")
    radius = order // 2
    matrix = np.zeros((radius + 1, radius + 1), dtype=np.float64)
    rhs = np.zeros(radius + 1, dtype=np.float64)
    for k in range(radius + 1):
        matrix[k, 0] = 1.0 if k == 0 else 0.0
        for m in range(1, radius + 1):
            matrix[k, m] = 2.0 * (m ** (2 * k))
        rhs[k] = 2.0 if k == 1 else 0.0
    return np.linalg.solve(matrix, rhs)


def laplacian(field: Array, dx: float, dz: float, order: int = 8) -> Array:
    coeff = finite_difference_second_coefficients(order)
    radius = len(coeff) - 1
    out = np.zeros_like(field, dtype=np.float32)
    if field.shape[0] <= 2 * radius or field.shape[1] <= 2 * radius:
        raise ValueError("field is too small for requested finite-difference order")

    rows = slice(radius, -radius)
    cols = slice(radius, -radius)
    core = field[rows, cols]
    out[rows, cols] = (coeff[0] / dx**2 + coeff[0] / dz**2) * core
    for m, c in enumerate(coeff[1:], start=1):
        out[rows, cols] += c * (
            field[rows, radius + m : field.shape[1] - radius + m]
            + field[rows, radius - m : field.shape[1] - radius - m]
        ) / dx**2
        out[rows, cols] += c * (
            field[radius + m : field.shape[0] - radius + m, cols]
            + field[radius - m : field.shape[0] - radius - m, cols]
        ) / dz**2
    return out


def make_absorbing_mask(config: RTMConfig) -> Array:
    mask = np.ones((config.nz, config.nx), dtype=np.float32)
    cells = max(0, int(config.absorb_cells))
    if cells == 0:
        return mask

    for i in range(cells):
        distance = (cells - i) / cells
        damp = np.exp(-config.absorb_strength * distance * distance).astype(np.float32)
        mask[:, i] *= damp
        mask[:, -(i + 1)] *= damp
        mask[-(i + 1), :] *= damp
        if config.absorb_top:
            mask[i, :] *= damp
    return mask


def validate_config(velocity: Array, config: RTMConfig) -> None:
    if velocity.shape != (config.nz, config.nx):
        raise ValueError(f"velocity shape {velocity.shape} != {(config.nz, config.nx)}")
    for name, value, limit in [
        ("source_x", config.source_x, config.nx),
        ("source_z", config.source_z, config.nz),
        ("receiver_z", config.receiver_z, config.nz),
    ]:
        if value < 0 or value >= limit:
            raise ValueError(f"{name}={value} is outside [0, {limit})")
    cfl = float(np.max(velocity) * config.dt * np.sqrt(1.0 / config.dx**2 + 1.0 / config.dz**2))
    if cfl >= config.max_cfl:
        raise ValueError(f"CFL={cfl:.3f} exceeds configured limit {config.max_cfl:.3f}")


def _allocate_wavefield(path: Optional[str | Path], shape: Tuple[int, int, int]) -> Array:
    if path is None:
        return np.zeros(shape, dtype=np.float32)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.memmap(path, mode="w+", dtype=np.float32, shape=shape)


def open_wavefield(path: str | Path, config: RTMConfig, mode: str = "r") -> Array:
    return np.memmap(path, mode=mode, dtype=np.float32, shape=(config.nt, config.nz, config.nx))


def _step_wavefield(prev: Array, curr: Array, velocity2_dt2: Array, mask: Array, config: RTMConfig) -> Array:
    nxt = 2.0 * curr - prev + velocity2_dt2 * laplacian(curr, config.dx, config.dz, config.fd_order)
    nxt *= mask
    return nxt.astype(np.float32, copy=False)


def forward_model(velocity: Array, config: RTMConfig, wavefield_path: Optional[str | Path] = None) -> Array:
    """Forward propagate the source wavefield and return the surface record."""
    validate_config(velocity, config)
    mask = make_absorbing_mask(config)
    velocity2_dt2 = (velocity.astype(np.float32) ** 2) * np.float32(config.dt * config.dt)
    wavelet = ricker_wavelet(config.nt, config.dt, config.f0, config.source_delay)
    wavefield = _allocate_wavefield(wavefield_path, (config.nt, config.nz, config.nx)) if wavefield_path else None
    record = np.zeros((config.nt, config.nx), dtype=np.float32)
    prev = np.zeros((config.nz, config.nx), dtype=np.float32)
    curr = np.zeros_like(prev)

    for it in range(config.nt):
        curr[config.source_z, config.source_x] += wavelet[it]
        curr *= mask
        if wavefield is not None:
            wavefield[it, :, :] = curr
        record[it, :] = curr[config.receiver_z, :]
        nxt = _step_wavefield(prev, curr, velocity2_dt2, mask, config)
        prev, curr = curr, nxt

    if isinstance(wavefield, np.memmap):
        wavefield.flush()
    return record


def shot_positions_from_spacing(
    nx: int,
    dx: float,
    spacing_m: float = 30.0,
    margin_cells: int = 4,
) -> list[int]:
    """Return integer surface shot positions for a requested physical spacing."""
    if spacing_m <= 0.0:
        raise ValueError("spacing_m must be positive")
    step = max(1, int(round(spacing_m / dx)))
    start = max(0, int(margin_cells))
    stop = max(start + 1, nx - int(margin_cells))
    positions = list(range(start, stop, step))
    if positions and positions[-1] != stop - 1 and (stop - 1 - positions[-1]) >= step // 2:
        positions.append(stop - 1)
    return positions


def stack_surface_records(
    velocity: Array,
    config: RTMConfig,
    shot_positions: list[int],
    stack_mode: str = "mean",
) -> tuple[Array, int]:
    """Forward model multiple surface shots and average their records.

    The resulting section follows the paper-style "stacked seismic record"
    convention used before boundary-condition reverse-time migration. It is a
    pragmatic shot-stack over a fixed receiver spread, not a CMP processor.
    """
    if not shot_positions:
        raise ValueError("shot_positions must not be empty")
    if stack_mode not in {"mean", "signed_rms", "zero_offset", "normal_incidence"}:
        raise ValueError("stack_mode must be 'mean', 'signed_rms', 'zero_offset', or 'normal_incidence'")
    if stack_mode == "normal_incidence":
        return (
            synthesize_normal_incidence_stack(
                velocity,
                dz=config.dz,
                dt=config.dt,
                nt=config.nt,
                f0=config.f0,
            ),
            len(shot_positions),
        )
    if stack_mode == "zero_offset":
        shot_positions_array = np.asarray(shot_positions, dtype=np.int64)
        traces = np.zeros((config.nt, len(shot_positions)), dtype=np.float32)
        for idx, source_x in enumerate(shot_positions):
            shot_cfg = replace(config, source_x=int(source_x))
            record = forward_model(velocity, shot_cfg)
            traces[:, idx] = record[:, int(source_x)]
        x_all = np.arange(config.nx, dtype=np.float32)
        stacked = np.zeros((config.nt, config.nx), dtype=np.float32)
        for it in range(config.nt):
            stacked[it, :] = np.interp(x_all, shot_positions_array, traces[it, :]).astype(np.float32)
        return stacked, len(shot_positions)

    summed = np.zeros((config.nt, config.nx), dtype=np.float32)
    summed_squares = np.zeros_like(summed) if stack_mode == "signed_rms" else None
    for source_x in shot_positions:
        shot_cfg = replace(config, source_x=int(source_x))
        record = forward_model(velocity, shot_cfg)
        summed += record
        if summed_squares is not None:
            summed_squares += record * record
    mean = summed / np.float32(len(shot_positions))
    if stack_mode == "mean":
        stacked = mean
    else:
        rms = np.sqrt(summed_squares / np.float32(len(shot_positions)))
        polarity = np.sign(mean)
        polarity[polarity == 0.0] = 1.0
        stacked = rms * polarity
    return stacked, len(shot_positions)


def preprocess_stacked_record(
    record: Array,
    dt: float,
    mute_time: float = 0.18,
    time_power: float = 1.0,
    eps: float = 1.0e-12,
) -> Array:
    """Prepare a stacked seismic section for paper-style display and migration.

    The operation removes trace means, mutes the strongest early direct-arrival
    zone, applies a simple time gain, and normalizes each trace. This matches
    the visual intent of stacked sections in older RTM papers, where coherent
    reflection events are emphasized before using the record as a boundary.
    """
    out = np.asarray(record, dtype=np.float32).copy()
    out -= np.mean(out, axis=0, keepdims=True)
    mute_samples = min(out.shape[0], max(0, int(round(mute_time / dt))))
    if mute_samples:
        out[:mute_samples, :] = 0.0
    if time_power != 0.0:
        gain = (np.arange(out.shape[0], dtype=np.float32) * np.float32(dt)) ** np.float32(time_power)
        out *= gain[:, None]
    scale = np.percentile(np.abs(out), 99.0, axis=0)
    scale = np.maximum(scale, np.float32(eps))
    out /= scale[None, :]
    np.clip(out, -1.0, 1.0, out=out)
    return np.nan_to_num(out).astype(np.float32, copy=False)


def mute_direct_arrivals(
    record: Array,
    config: RTMConfig,
    source_x: Optional[int] = None,
    direct_velocity: float = 2000.0,
    padding_time: float = 0.04,
    taper_time: float = 0.02,
) -> Array:
    """Mute direct arrivals before RTM imaging.

    The mute follows the first-break time from the shot to each surface
    receiver plus a small padding window. A cosine taper restores amplitudes
    after the muted zone to avoid a hard edge in the receiver wavefield.
    """
    if direct_velocity <= 0.0:
        raise ValueError("direct_velocity must be positive")
    out = np.asarray(record, dtype=np.float32).copy()
    if out.shape != (config.nt, config.nx):
        raise ValueError(f"record shape {out.shape} != {(config.nt, config.nx)}")
    sx = config.source_x if source_x is None else int(source_x)
    taper_samples = max(0, int(round(taper_time / config.dt)))
    x_offsets = (np.arange(config.nx, dtype=np.float32) - np.float32(sx)) * np.float32(config.dx)
    z_offset = np.float32((config.receiver_z - config.source_z) * config.dz)
    first_break = np.sqrt(x_offsets * x_offsets + z_offset * z_offset) / np.float32(direct_velocity)
    mute_samples = np.rint((first_break + np.float32(padding_time)) / np.float32(config.dt)).astype(np.int64)

    for ix, mute_until in enumerate(mute_samples):
        mute_until = max(0, min(config.nt, int(mute_until)))
        if mute_until:
            out[:mute_until, ix] = 0.0
        if taper_samples and mute_until < config.nt:
            taper_end = min(config.nt, mute_until + taper_samples)
            ramp = np.linspace(0.0, 1.0, taper_end - mute_until, dtype=np.float32)
            ramp = 0.5 - 0.5 * np.cos(np.pi * ramp)
            out[mute_until:taper_end, ix] *= ramp
    return out.astype(np.float32, copy=False)


def preprocess_migration_section(
    section: Array,
    depth_power: float = 0.15,
    clip_percentile: float = 99.5,
    trace_balance: float = 0.25,
    output_clip: float = 0.80,
    eps: float = 1.0e-12,
) -> Array:
    """Balance a migrated section for paper-style variable-density display."""
    if not 0.0 <= trace_balance <= 1.0:
        raise ValueError("trace_balance must be in [0, 1]")
    if clip_percentile <= 0.0 or clip_percentile >= 100.0:
        raise ValueError("clip_percentile must be between 0 and 100")
    if output_clip <= 0.0:
        raise ValueError("output_clip must be positive")
    out = np.asarray(section, dtype=np.float32).copy()
    out -= np.mean(out, axis=0, keepdims=True)
    if depth_power != 0.0:
        depth = np.arange(out.shape[0], dtype=np.float32)
        depth = np.maximum(depth, 1.0) ** np.float32(depth_power)
        depth /= np.max(depth)
        out *= depth[:, None]
    abs_out = np.abs(out)
    global_scale = np.float32(np.percentile(abs_out, clip_percentile))
    trace_scale = np.percentile(abs_out, clip_percentile, axis=0).astype(np.float32)
    global_scale = np.maximum(global_scale, np.float32(eps))
    trace_scale = np.maximum(trace_scale, np.float32(eps))
    scale = (np.float32(1.0 - trace_balance) * global_scale) + (
        np.float32(trace_balance) * trace_scale
    )
    out /= scale[None, :]
    np.clip(out, -output_clip, output_clip, out=out)
    return np.nan_to_num(out).astype(np.float32, copy=False)


def source_normalized_image(
    image: Array,
    illumination: Array,
    eps: float = 1.0e-12,
    min_illumination_fraction: float = 0.0,
) -> Array:
    if min_illumination_fraction < 0.0:
        raise ValueError("min_illumination_fraction must be non-negative")
    scale = np.max(np.abs(illumination))
    floor = eps * scale if scale > 0.0 else eps
    denominator = illumination + floor
    normalized = np.zeros_like(np.asarray(image, dtype=np.float32), dtype=np.float32)
    np.divide(image, denominator, out=normalized, where=denominator != 0.0)
    if min_illumination_fraction > 0.0 and scale > 0.0:
        normalized = np.asarray(normalized).copy()
        normalized[illumination < scale * min_illumination_fraction] = 0.0
    return normalized


def source_receiver_normalized_image(
    image: Array,
    source_illumination: Array,
    receiver_illumination: Array,
    eps: float = 1.0e-12,
    min_illumination_fraction: float = 0.0,
) -> Array:
    if min_illumination_fraction < 0.0:
        raise ValueError("min_illumination_fraction must be non-negative")
    src = np.asarray(source_illumination, dtype=np.float32)
    rec = np.asarray(receiver_illumination, dtype=np.float32)
    scale = np.sqrt(np.maximum(src * rec, 0.0)).astype(np.float32)
    max_scale = np.max(scale)
    floor = eps * max_scale if max_scale > 0.0 else eps
    denominator = scale + np.float32(floor)
    normalized = np.zeros_like(np.asarray(image, dtype=np.float32), dtype=np.float32)
    np.divide(image, denominator, out=normalized, where=denominator != 0.0)
    if min_illumination_fraction > 0.0 and max_scale > 0.0:
        normalized = np.asarray(normalized).copy()
        normalized[scale < max_scale * min_illumination_fraction] = 0.0
    return normalized


def high_order_laplacian_filter(image: Array, dx: float, dz: float, power: int = 2) -> Array:
    """Apply repeated Laplacian filtering to suppress RTM low-frequency noise."""
    if power < 1:
        return np.asarray(image, dtype=np.float32).copy()
    filtered = np.asarray(image, dtype=np.float32).copy()
    for _ in range(power):
        filtered = laplacian(filtered, dx, dz, order=2)
    filtered -= np.mean(filtered)
    return np.nan_to_num(filtered).astype(np.float32, copy=False)


def smooth_velocity_model(
    velocity: Array,
    radius_z: int = 8,
    radius_x: int = 8,
    passes: int = 2,
) -> Array:
    """Smooth velocity with repeated separable box filters.

    The smoothed model is intended as a migration/background velocity for
    direct-wave modeling and RTM propagation, while the original model can be
    used to synthesize full records containing reflections.
    """
    if radius_z < 0 or radius_x < 0:
        raise ValueError("smoothing radii must be non-negative")
    if passes < 0:
        raise ValueError("passes must be non-negative")
    out = np.asarray(velocity, dtype=np.float32).copy()
    if passes == 0 or (radius_z == 0 and radius_x == 0):
        return out

    def smooth_axis(data: Array, radius: int, axis: int) -> Array:
        if radius == 0:
            return data
        pad_width = [(0, 0), (0, 0)]
        pad_width[axis] = (radius, radius)
        padded = np.pad(data, pad_width, mode="edge")
        kernel_size = 2 * radius + 1
        cumsum = np.cumsum(padded, axis=axis, dtype=np.float64)
        cumsum = np.concatenate(
            [
                np.zeros_like(np.take(cumsum, [0], axis=axis)),
                cumsum,
            ],
            axis=axis,
        )
        head = [slice(None), slice(None)]
        tail = [slice(None), slice(None)]
        head[axis] = slice(kernel_size, None)
        tail[axis] = slice(None, -kernel_size)
        return ((cumsum[tuple(head)] - cumsum[tuple(tail)]) / kernel_size).astype(np.float32)

    for _ in range(passes):
        out = smooth_axis(out, radius_z, axis=0)
        out = smooth_axis(out, radius_x, axis=1)
    return out.astype(np.float32, copy=False)


def make_reflection_record(
    full_velocity: Array,
    smooth_velocity: Array,
    config: RTMConfig,
    full_wavefield_path: Optional[str | Path] = None,
    direct_wavefield_path: Optional[str | Path] = None,
) -> tuple[Array, Array]:
    """Return full and reflection-only records by subtracting a smooth model record."""
    full_record = forward_model(full_velocity, config, wavefield_path=full_wavefield_path)
    direct_record = forward_model(smooth_velocity, config, wavefield_path=direct_wavefield_path)
    reflection_record = full_record - direct_record
    return full_record.astype(np.float32, copy=False), reflection_record.astype(np.float32, copy=False)


def reverse_time_migrate(
    velocity: Array,
    record: Array,
    config: RTMConfig,
    source_wavefield: Optional[Array] = None,
    source_wavefield_path: Optional[str | Path] = None,
    laplacian_power: int = 2,
) -> RTMResult:
    """Back propagate receiver data and apply zero-lag imaging conditions."""
    validate_config(velocity, config)
    if record.shape != (config.nt, config.nx):
        raise ValueError(f"record shape {record.shape} != {(config.nt, config.nx)}")
    if source_wavefield is None:
        if source_wavefield_path is None:
            raise ValueError("source_wavefield or source_wavefield_path is required")
        source_wavefield = open_wavefield(source_wavefield_path, config, mode="r")
    if source_wavefield.shape != (config.nt, config.nz, config.nx):
        raise ValueError(f"source wavefield shape {source_wavefield.shape} is incompatible with config")

    mask = make_absorbing_mask(config)
    velocity2_dt2 = (velocity.astype(np.float32) ** 2) * np.float32(config.dt * config.dt)
    prev = np.zeros((config.nz, config.nx), dtype=np.float32)
    curr = np.zeros_like(prev)
    image = np.zeros_like(prev)
    illumination = np.zeros_like(prev)
    receiver_illumination = np.zeros_like(prev)

    for it in range(config.nt - 1, -1, -1):
        curr[config.receiver_z, :] += record[it, :]
        curr *= mask
        source = np.asarray(source_wavefield[it, :, :], dtype=np.float32)
        image += source * curr
        illumination += source * source
        receiver_illumination += curr * curr
        nxt = _step_wavefield(prev, curr, velocity2_dt2, mask, config)
        prev, curr = curr, nxt

    normalized = source_normalized_image(image, illumination)
    source_receiver_normalized = source_receiver_normalized_image(
        image,
        illumination,
        receiver_illumination,
    )
    laplacian_image = high_order_laplacian_filter(image, config.dx, config.dz, power=1)
    laplacian_normalized = source_normalized_image(laplacian_image, illumination)
    filtered = high_order_laplacian_filter(normalized, config.dx, config.dz, power=laplacian_power)
    return RTMResult(
        image=image.astype(np.float32, copy=False),
        illumination=illumination.astype(np.float32, copy=False),
        receiver_illumination=receiver_illumination.astype(np.float32, copy=False),
        normalized_image=normalized.astype(np.float32, copy=False),
        source_receiver_normalized_image=source_receiver_normalized.astype(np.float32, copy=False),
        laplacian_image=laplacian_image.astype(np.float32, copy=False),
        laplacian_normalized_image=laplacian_normalized.astype(np.float32, copy=False),
        filtered_image=filtered,
    )


def _run_multishot_single_shot(
    velocity: Array,
    migration_velocity: Array,
    config: RTMConfig,
    source_x: int,
    wavefield_path: str,
    laplacian_power: int,
    subtract_direct_wave: bool,
    direct_wavefield_path: Optional[str] = None,
    direct_mute_params: Optional[dict[str, float]] = None,
) -> ShotRTMPartial:
    wavefield_file = Path(wavefield_path)
    shot_cfg = replace(config, source_x=int(source_x))
    try:
        full_record = forward_model(velocity, shot_cfg)
        migration_record = full_record
        if subtract_direct_wave:
            if direct_wavefield_path is None:
                direct_record = forward_model(migration_velocity, shot_cfg)
            else:
                direct_record = forward_model(migration_velocity, shot_cfg, wavefield_path=direct_wavefield_path)
            migration_record = full_record - direct_record
        forward_model(migration_velocity, shot_cfg, wavefield_path=wavefield_file)
        receiver_record = migration_record
        if direct_mute_params is not None:
            receiver_record = mute_direct_arrivals(
                receiver_record,
                shot_cfg,
                source_x=int(source_x),
                direct_velocity=float(direct_mute_params["direct_velocity"]),
                padding_time=float(direct_mute_params["padding_time"]),
                taper_time=float(direct_mute_params["taper_time"]),
            )
        shot_result = reverse_time_migrate(
            migration_velocity,
            receiver_record,
            shot_cfg,
            source_wavefield_path=wavefield_file,
            laplacian_power=laplacian_power,
        )
        return ShotRTMPartial(
            source_x=int(source_x),
            image=shot_result.image.astype(np.float32, copy=False),
            illumination=shot_result.illumination.astype(np.float32, copy=False),
            receiver_illumination=shot_result.receiver_illumination.astype(np.float32, copy=False),
            stacked_record=receiver_record.astype(np.float32, copy=False),
        )
    finally:
        wavefield_file.unlink(missing_ok=True)


def multishot_reverse_time_migrate_parallel(
    velocity: Array,
    config: RTMConfig,
    shot_positions: list[int],
    work_dir: str | Path,
    workers: int = 2,
    laplacian_power: int = 2,
    migration_velocity: Optional[Array] = None,
    subtract_direct_wave: bool = False,
    min_illumination_fraction: float = 0.01,
    direct_mute_params: Optional[dict[str, float]] = None,
    checkpoint_dir: Optional[str | Path] = None,
    resume: bool = False,
    checkpoint_interval: int = 1,
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
) -> MultishotRTMResult:
    """Run multi-shot RTM with independent shot workers and stack the partial images."""
    validate_config(velocity, config)
    migration_velocity = np.asarray(
        velocity if migration_velocity is None else migration_velocity,
        dtype=np.float32,
    )
    validate_config(migration_velocity, config)
    if not shot_positions:
        raise ValueError("shot_positions must not be empty")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be >= 1")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    image_sum = np.zeros((config.nz, config.nx), dtype=np.float32)
    illumination_sum = np.zeros_like(image_sum)
    receiver_illumination_sum = np.zeros_like(image_sum)
    stacked_record = np.zeros((config.nt, config.nx), dtype=np.float32)
    total = len(shot_positions)
    shot_positions = [int(source_x) for source_x in shot_positions]
    checkpoint_path = None if checkpoint_dir is None else Path(checkpoint_dir)
    signature = _checkpoint_run_signature(
        config,
        shot_positions,
        laplacian_power=laplacian_power,
        subtract_direct_wave=subtract_direct_wave,
        min_illumination_fraction=min_illumination_fraction,
        direct_mute_params=direct_mute_params,
    )
    completed_shots: list[int] = []
    if resume and checkpoint_path is not None and (checkpoint_path / "checkpoint_manifest.json").exists():
        (
            completed_shots,
            image_sum,
            illumination_sum,
            receiver_illumination_sum,
            stacked_record,
        ) = _load_multishot_checkpoint(checkpoint_path, signature)
        if image_sum.shape != (config.nz, config.nx):
            raise ValueError("checkpoint does not match current RTM run")
        if illumination_sum.shape != (config.nz, config.nx):
            raise ValueError("checkpoint does not match current RTM run")
        if receiver_illumination_sum.shape != (config.nz, config.nx):
            raise ValueError("checkpoint does not match current RTM run")
        if stacked_record.shape != (config.nt, config.nx):
            raise ValueError("checkpoint does not match current RTM run")
    elif resume and checkpoint_path is not None:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
    elif checkpoint_path is not None:
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    completed_set = set(completed_shots)
    unknown_completed = completed_set.difference(shot_positions)
    if unknown_completed:
        raise ValueError("checkpoint does not match current RTM run")
    pending_shots = [source_x for source_x in shot_positions if source_x not in completed_set]

    def accumulate_partial(partial: ShotRTMPartial) -> int:
        nonlocal image_sum, illumination_sum, receiver_illumination_sum, stacked_record
        if partial.source_x in completed_set:
            return len(completed_shots)
        image_sum += partial.image
        illumination_sum += partial.illumination
        receiver_illumination_sum += partial.receiver_illumination
        stacked_record += partial.stacked_record
        completed_shots.append(int(partial.source_x))
        completed_set.add(int(partial.source_x))
        completed_count = len(completed_shots)
        if checkpoint_path is not None and (
            completed_count % checkpoint_interval == 0 or completed_count == total
        ):
            _save_multishot_checkpoint(
                checkpoint_path,
                signature=signature,
                completed_shots=completed_shots,
                image_sum=image_sum,
                illumination_sum=illumination_sum,
                receiver_illumination_sum=receiver_illumination_sum,
                stacked_record_sum=stacked_record,
            )
        return completed_count

    if workers == 1:
        for source_x in pending_shots:
            partial = _run_multishot_single_shot(
                velocity,
                migration_velocity,
                config,
                int(source_x),
                str(work_dir / f"source_wavefield_shot_{int(source_x):05d}.dat"),
                laplacian_power,
                subtract_direct_wave,
                None,
                direct_mute_params,
            )
            completed = accumulate_partial(partial)
            if progress_callback is not None:
                progress_callback(completed, total, partial.source_x)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _run_multishot_single_shot,
                    velocity,
                    migration_velocity,
                    config,
                    int(source_x),
                    str(work_dir / f"source_wavefield_shot_{int(source_x):05d}.dat"),
                    laplacian_power,
                    subtract_direct_wave,
                    None,
                    direct_mute_params,
                ): int(source_x)
                for source_x in pending_shots
            }
            for future in as_completed(futures):
                partial = future.result()
                completed = accumulate_partial(partial)
                if progress_callback is not None:
                    progress_callback(completed, total, partial.source_x)

    shot_count = len(shot_positions)
    stacked_record /= np.float32(shot_count)
    normalized = source_normalized_image(
        image_sum,
        illumination_sum,
        min_illumination_fraction=min_illumination_fraction,
    )
    source_receiver_normalized = source_receiver_normalized_image(
        image_sum,
        illumination_sum,
        receiver_illumination_sum,
        min_illumination_fraction=min_illumination_fraction,
    )
    laplacian_image = high_order_laplacian_filter(image_sum, config.dx, config.dz, power=1)
    laplacian_normalized = source_normalized_image(
        laplacian_image,
        illumination_sum,
        min_illumination_fraction=min_illumination_fraction,
    )
    filtered = high_order_laplacian_filter(normalized, config.dx, config.dz, power=laplacian_power)
    return MultishotRTMResult(
        image=image_sum.astype(np.float32, copy=False),
        illumination=illumination_sum.astype(np.float32, copy=False),
        receiver_illumination=receiver_illumination_sum.astype(np.float32, copy=False),
        normalized_image=normalized.astype(np.float32, copy=False),
        source_receiver_normalized_image=source_receiver_normalized.astype(np.float32, copy=False),
        laplacian_image=laplacian_image.astype(np.float32, copy=False),
        laplacian_normalized_image=laplacian_normalized.astype(np.float32, copy=False),
        filtered_image=filtered,
        stacked_record=stacked_record.astype(np.float32, copy=False),
        shot_count=shot_count,
    )


def multishot_reverse_time_migrate(
    velocity: Array,
    config: RTMConfig,
    shot_positions: list[int],
    wavefield_path: str | Path,
    laplacian_power: int = 2,
    record_provider: Optional[Callable[[int, Array], Array]] = None,
    migration_velocity: Optional[Array] = None,
    subtract_direct_wave: bool = False,
    direct_wavefield_path: Optional[str | Path] = None,
    min_illumination_fraction: float = 0.01,
) -> MultishotRTMResult:
    """Run prestack multi-shot RTM with a zero-lag cross-correlation image.

    Each shot is forward propagated to save its source wavefield. The matching
    receiver record is then reverse propagated and cross-correlated with that
    source wavefield. Shot images and illumination are accumulated before the
    final source-normalized and Laplacian-filtered images are formed.

    If ``record_provider`` is omitted, the synthetic forward-modeled record for
    each shot is used as receiver data. A provider can replace it with observed
    or preprocessed shot gathers while preserving the same source wavefield.
    """
    validate_config(velocity, config)
    migration_velocity = np.asarray(
        velocity if migration_velocity is None else migration_velocity,
        dtype=np.float32,
    )
    validate_config(migration_velocity, config)
    if not shot_positions:
        raise ValueError("shot_positions must not be empty")
    wavefield_path = Path(wavefield_path)
    wavefield_path.parent.mkdir(parents=True, exist_ok=True)
    direct_wavefield_path = Path(direct_wavefield_path) if direct_wavefield_path is not None else None

    image_sum = np.zeros((config.nz, config.nx), dtype=np.float32)
    illumination_sum = np.zeros_like(image_sum)
    receiver_illumination_sum = np.zeros_like(image_sum)
    stacked_record = np.zeros((config.nt, config.nx), dtype=np.float32)

    for source_x in shot_positions:
        shot_cfg = replace(config, source_x=int(source_x))
        full_record = forward_model(velocity, shot_cfg)
        migration_record = full_record
        if subtract_direct_wave:
            if direct_wavefield_path is None:
                direct_record = forward_model(migration_velocity, shot_cfg)
            else:
                direct_record = forward_model(
                    migration_velocity,
                    shot_cfg,
                    wavefield_path=direct_wavefield_path,
                )
            migration_record = full_record - direct_record
        source_record = forward_model(migration_velocity, shot_cfg, wavefield_path=wavefield_path)
        receiver_record = (
            np.asarray(record_provider(int(source_x), migration_record), dtype=np.float32)
            if record_provider is not None
            else migration_record
        )
        if receiver_record.shape != (config.nt, config.nx):
            raise ValueError(
                f"record for shot {source_x} has shape {receiver_record.shape}, "
                f"expected {(config.nt, config.nx)}"
            )
        shot_result = reverse_time_migrate(
            migration_velocity,
            receiver_record,
            shot_cfg,
            source_wavefield_path=wavefield_path,
            laplacian_power=laplacian_power,
        )
        image_sum += shot_result.image
        illumination_sum += shot_result.illumination
        receiver_illumination_sum += shot_result.receiver_illumination
        stacked_record += receiver_record

    shot_count = len(shot_positions)
    stacked_record /= np.float32(shot_count)
    normalized = source_normalized_image(
        image_sum,
        illumination_sum,
        min_illumination_fraction=min_illumination_fraction,
    )
    source_receiver_normalized = source_receiver_normalized_image(
        image_sum,
        illumination_sum,
        receiver_illumination_sum,
        min_illumination_fraction=min_illumination_fraction,
    )
    laplacian_image = high_order_laplacian_filter(image_sum, config.dx, config.dz, power=1)
    laplacian_normalized = source_normalized_image(
        laplacian_image,
        illumination_sum,
        min_illumination_fraction=min_illumination_fraction,
    )
    filtered = high_order_laplacian_filter(normalized, config.dx, config.dz, power=laplacian_power)
    return MultishotRTMResult(
        image=image_sum.astype(np.float32, copy=False),
        illumination=illumination_sum.astype(np.float32, copy=False),
        receiver_illumination=receiver_illumination_sum.astype(np.float32, copy=False),
        normalized_image=normalized.astype(np.float32, copy=False),
        source_receiver_normalized_image=source_receiver_normalized.astype(np.float32, copy=False),
        laplacian_image=laplacian_image.astype(np.float32, copy=False),
        laplacian_normalized_image=laplacian_normalized.astype(np.float32, copy=False),
        filtered_image=filtered,
        stacked_record=stacked_record.astype(np.float32, copy=False),
        shot_count=shot_count,
    )


def reverse_time_boundary_migrate(
    velocity: Array,
    record: Array,
    config: RTMConfig,
    laplacian_power: int = 2,
) -> tuple[Array, Array]:
    """Migrate by using the surface record as a reverse-time boundary.

    This matches the older acoustic RTM description used by Wang Chunyan:
    the recorded surface data are imposed at the surface for every reverse
    time step, and the back-propagated wavefield collapses toward subsurface
    reflectors. The wavefield at zero imaging time is returned as the migrated
    section, together with a Laplacian-filtered version for low-frequency
    noise suppression.
    """
    validate_config(velocity, config)
    if record.shape != (config.nt, config.nx):
        raise ValueError(f"record shape {record.shape} != {(config.nt, config.nx)}")

    mask = make_absorbing_mask(config)
    velocity2_dt2 = (velocity.astype(np.float32) ** 2) * np.float32(config.dt * config.dt)
    prev = np.zeros((config.nz, config.nx), dtype=np.float32)
    curr = np.zeros_like(prev)
    image_at_zero = None

    for it in range(config.nt - 1, -1, -1):
        # Boundary condition form: measured surface traces replace the
        # pressure values on the receiver line at each reverse-time step.
        curr[config.receiver_z, :] = record[it, :]
        curr *= mask
        if it == 0:
            image_at_zero = curr.copy()
        nxt = _step_wavefield(prev, curr, velocity2_dt2, mask, config)
        prev, curr = curr, nxt

    if image_at_zero is None:
        image_at_zero = curr.copy()
    filtered = high_order_laplacian_filter(image_at_zero, config.dx, config.dz, power=laplacian_power)
    return image_at_zero.astype(np.float32, copy=False), filtered


def save_rtm_outputs(output_dir: str | Path, result: RTMResult, record: Optional[Array] = None) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "rtm_image_raw.npy", result.image)
    np.save(output_dir / "rtm_source_illumination.npy", result.illumination)
    np.save(output_dir / "rtm_image_source_normalized.npy", result.normalized_image)
    np.save(output_dir / "rtm_image_laplacian_filtered.npy", result.filtered_image)
    result.image.astype(np.float32).tofile(output_dir / "rtm_image_raw.bin")
    result.normalized_image.astype(np.float32).tofile(output_dir / "rtm_image_source_normalized.bin")
    result.filtered_image.astype(np.float32).tofile(output_dir / "rtm_image_laplacian_filtered.bin")
    if record is not None:
        np.save(output_dir / "input_record.npy", np.asarray(record, dtype=np.float32))


def save_boundary_migration_outputs(
    output_dir: str | Path,
    boundary_image: Array,
    filtered_boundary_image: Array,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    boundary_image = np.asarray(boundary_image, dtype=np.float32)
    filtered_boundary_image = np.asarray(filtered_boundary_image, dtype=np.float32)
    np.save(output_dir / "wang_boundary_migration_image.npy", boundary_image)
    np.save(output_dir / "wang_boundary_migration_laplacian_filtered.npy", filtered_boundary_image)
    boundary_image.tofile(output_dir / "wang_boundary_migration_image.bin")
    filtered_boundary_image.tofile(output_dir / "wang_boundary_migration_laplacian_filtered.bin")
