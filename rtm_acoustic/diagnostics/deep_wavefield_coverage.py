from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RoiMasks:
    shallow: np.ndarray
    middle: np.ndarray
    deep: np.ndarray
    subsalt_proxy: np.ndarray
    physical: np.ndarray


def build_depth_roi_masks(shape: tuple[int, int], absorb_cells: int) -> RoiMasks:
    nz, nx = shape
    effective_nz = max(1, nz - int(absorb_cells))
    physical = np.zeros(shape, dtype=bool)
    physical[:effective_nz, :] = True
    q1 = max(1, effective_nz // 4)
    q2 = max(q1 + 1, effective_nz // 2)
    q3 = max(q2 + 1, (3 * effective_nz) // 4)
    shallow = np.zeros(shape, dtype=bool)
    middle = np.zeros(shape, dtype=bool)
    deep = np.zeros(shape, dtype=bool)
    shallow[:q1, :] = True
    middle[q1:q3, :] = True
    deep[q3:effective_nz, :] = True
    subsalt_proxy = deep.copy()
    return RoiMasks(shallow=shallow, middle=middle, deep=deep, subsalt_proxy=subsalt_proxy, physical=physical)


def energy(field: np.ndarray, mask: np.ndarray) -> float:
    values = np.asarray(field, dtype=np.float64)[mask]
    return float(np.sum(values * values))


def summarize_deep_energy(times: np.ndarray, deep_energy: np.ndarray, eps: float = 1.0e-12) -> dict[str, float | str | bool]:
    t = np.asarray(times, dtype=np.float64)
    e = np.asarray(deep_energy, dtype=np.float64)
    if t.ndim != 1 or e.ndim != 1 or t.size != e.size:
        raise ValueError("times and deep_energy must be 1-D arrays of equal length")
    if t.size == 0 or not np.isfinite(t).all() or not np.isfinite(e).all():
        raise ValueError("energy curve contains NaN/Inf or is empty")
    peak_idx = int(np.argmax(e))
    peak_time = float(t[peak_idx])
    max_energy = float(np.max(e))
    threshold = max(max_energy * 1.0e-3, eps)
    reached = bool(max_energy > eps)
    arrival_candidates = np.flatnonzero(e >= threshold)
    first_arrival = float(t[int(arrival_candidates[0])]) if arrival_candidates.size else None
    final_window_start = int(max(0, np.floor(0.9 * (t.size - 1))))
    final_slope = float(e[-1] - e[max(0, t.size - 5)])
    time_truncation_risk = bool(peak_idx >= final_window_start)
    time_truncation_confirmed = bool(reached and final_slope > max(max_energy * 1.0e-3, eps))
    if not reached:
        status = "DEEP_WAVEFIELD_NOT_REACHED"
    elif time_truncation_confirmed:
        status = "TIME_TRUNCATION_CONFIRMED"
    elif time_truncation_risk:
        status = "TIME_TRUNCATION_RISK"
    else:
        status = "DEEP_COVERAGE_OK"
    return {
        "status": status,
        "deep_energy_first_arrival_time": first_arrival,
        "deep_energy_peak_time": peak_time,
        "deep_energy_peak": max_energy,
        "late_time_deep_decay": float(max_energy - e[-1]),
        "time_truncation_risk": time_truncation_risk,
        "time_truncation_confirmed": time_truncation_confirmed,
    }
