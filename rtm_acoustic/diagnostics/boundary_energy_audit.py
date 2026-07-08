from __future__ import annotations

import numpy as np


def boundary_mask(shape: tuple[int, int], cells: int) -> np.ndarray:
    nz, nx = shape
    c = max(1, int(cells))
    mask = np.zeros(shape, dtype=bool)
    mask[:, :c] = True
    mask[:, nx - c :] = True
    mask[nz - c :, :] = True
    return mask


def boundary_energy_ratio(boundary_energy: np.ndarray, physical_energy: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    b = np.asarray(boundary_energy, dtype=np.float64)
    p = np.asarray(physical_energy, dtype=np.float64)
    if b.shape != p.shape:
        raise ValueError("boundary and physical energy arrays must have the same shape")
    ratio = b / (p + float(eps))
    if not np.isfinite(ratio).all():
        raise ValueError("boundary energy ratio contains NaN or Inf")
    return ratio


def classify_boundary_energy(times: np.ndarray, ratio: np.ndarray, deep_peak_time: float, risk_threshold: float = 0.08, warning_threshold: float = 0.04) -> dict[str, float | str | bool]:
    t = np.asarray(times, dtype=np.float64)
    r = np.asarray(ratio, dtype=np.float64)
    if t.ndim != 1 or r.ndim != 1 or t.size != r.size or t.size == 0:
        raise ValueError("times and ratio must be non-empty 1-D arrays of equal length")
    if not np.isfinite(r).all():
        raise ValueError("boundary ratio contains NaN or Inf")
    near = np.abs(t - float(deep_peak_time)) <= max(0.1, 0.05 * max(float(t[-1]), 1.0))
    near_max = float(np.max(r[near])) if np.any(near) else float(np.max(r))
    tail_start = int(max(0, np.floor(0.9 * (t.size - 1))))
    tail_max = float(np.max(r[tail_start:]))
    pml_risk = bool(near_max >= risk_threshold)
    late_warning = bool(tail_max >= warning_threshold)
    if pml_risk:
        status = "PML_REFLECTION_RISK"
    elif late_warning:
        status = "LATE_BOUNDARY_REFLECTION_WARNING"
    else:
        status = "PML_OK"
    return {
        "status": status,
        "max_boundary_ratio": float(np.max(r)),
        "near_deep_peak_boundary_ratio": near_max,
        "final_window_boundary_ratio": tail_max,
        "pml_reflection_risk": pml_risk,
        "late_boundary_reflection_warning": late_warning,
    }
