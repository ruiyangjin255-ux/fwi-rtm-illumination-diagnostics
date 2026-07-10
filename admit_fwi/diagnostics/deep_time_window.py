from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DeepTimePlan:
    v_ref: float
    target_depth_m: float
    max_path_time: float
    mean_path_time: float
    margin_seconds: float
    required_time: float
    nt_required: int
    nt_recommended: int
    current_time: float
    current_satisfies: bool
    depth_rows: list[dict[str, float]]


def round_up(value: int, stride: int) -> int:
    if stride <= 1:
        return int(value)
    return int(math.ceil(value / stride) * stride)


def compute_required_record_time(
    model: np.ndarray,
    dx: float,
    dz: float,
    source_positions: list[int],
    receiver_positions: list[int],
    dt: float,
    wavelet_peak_time: float,
    f0: float,
    pml_thickness: int,
    target_depth_fraction: float = 0.95,
    safety_factor: float = 1.15,
    margin_seconds: float | None = None,
    v_ref_percentile: float = 10.0,
    current_nt: int | None = None,
    nt_floor: int = 5000,
    stride: int = 100,
) -> DeepTimePlan:
    velocity = np.asarray(model, dtype=np.float64)
    if velocity.ndim != 2:
        raise ValueError("model must be 2-D")
    if dt <= 0.0 or dx <= 0.0 or dz <= 0.0 or f0 <= 0.0:
        raise ValueError("dx, dz, dt and f0 must be positive")
    if not source_positions or not receiver_positions:
        raise ValueError("source and receiver positions must not be empty")
    nz, nx = velocity.shape
    effective_nz = nz - int(pml_thickness)
    if effective_nz <= 1:
        raise ValueError("PML thickness leaves no effective physical depth")
    physical = velocity[:effective_nz, :]
    if not np.isfinite(physical).all():
        raise ValueError("model contains NaN or Inf")
    v_ref = float(np.percentile(physical, float(v_ref_percentile)))
    if v_ref <= 0.0:
        raise ValueError("v_ref must be positive")
    margin = float(max(0.5, 4.0 / f0) if margin_seconds is None else margin_seconds)
    target_depth_m = float((effective_nz - 1) * dz * float(target_depth_fraction))
    target_depth_indices = np.linspace(max(1, int(0.25 * effective_nz)), int(target_depth_fraction * (effective_nz - 1)), 5).round().astype(int)
    target_x_indices = np.linspace(0, nx - 1, min(9, nx)).round().astype(int)
    src_x = np.asarray(source_positions, dtype=np.float64) * float(dx)
    rec_x = np.asarray(receiver_positions, dtype=np.float64) * float(dx)
    path_times: list[float] = []
    depth_rows: list[dict[str, float]] = []
    for iz in target_depth_indices:
        z_m = float(iz) * float(dz)
        layer_times: list[float] = []
        for ix in target_x_indices:
            x_m = float(ix) * float(dx)
            source_dist = np.min(np.sqrt((src_x - x_m) ** 2 + z_m**2))
            receiver_dist = np.min(np.sqrt((rec_x - x_m) ** 2 + z_m**2))
            t = float((source_dist + receiver_dist) / v_ref)
            path_times.append(t)
            layer_times.append(t)
        depth_rows.append(
            {
                "depth_m": z_m,
                "min_path_time_s": float(np.min(layer_times)),
                "mean_path_time_s": float(np.mean(layer_times)),
                "max_path_time_s": float(np.max(layer_times)),
            }
        )
    max_path = float(np.max(path_times))
    mean_path = float(np.mean(path_times))
    required_time = float(wavelet_peak_time + max_path * float(safety_factor) + margin)
    nt_required = round_up(int(math.ceil(required_time / dt)) + 1, stride)
    nt_recommended = round_up(max(int(nt_floor), nt_required), stride)
    current_time = float("nan") if current_nt is None else float(current_nt) * float(dt)
    return DeepTimePlan(
        v_ref=v_ref,
        target_depth_m=target_depth_m,
        max_path_time=max_path,
        mean_path_time=mean_path,
        margin_seconds=margin,
        required_time=required_time,
        nt_required=nt_required,
        nt_recommended=nt_recommended,
        current_time=current_time,
        current_satisfies=False if current_nt is None else bool(current_time >= required_time),
        depth_rows=depth_rows,
    )
