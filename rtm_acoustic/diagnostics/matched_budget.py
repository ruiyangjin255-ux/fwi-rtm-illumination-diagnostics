from __future__ import annotations

import numpy as np


class BudgetMatchError(RuntimeError):
    """Raised when alpha_max prevents matching the requested update budget."""


def update_l2(alpha: np.ndarray, delta_model: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(alpha, dtype=float) * np.asarray(delta_model, dtype=float)))


def scale_to_budget(
    gate: np.ndarray,
    delta_model: np.ndarray,
    *,
    target_update_l2: float,
    alpha_max: float,
    rtol: float = 1.0e-4,
) -> np.ndarray:
    if target_update_l2 < 0.0:
        raise ValueError("target_update_l2 must be non-negative")
    if alpha_max <= 0.0:
        raise ValueError("alpha_max must be positive")
    gate_arr = np.nan_to_num(np.asarray(gate, dtype=float), copy=False)
    gate_arr = np.clip(gate_arr, 0.0, None)
    if not np.any(gate_arr > 0.0):
        raise BudgetMatchError("BUDGET_MATCH_FAILED: gate support is empty")
    max_norm = update_l2(alpha_max * gate_arr / float(np.max(gate_arr)), delta_model)
    if target_update_l2 > max_norm * (1.0 + rtol):
        raise BudgetMatchError(
            f"BUDGET_MATCH_FAILED: target_update_l2={target_update_l2:.6g} exceeds max feasible {max_norm:.6g}"
        )
    if target_update_l2 == 0.0:
        return np.zeros_like(gate_arr, dtype=np.float32)
    lo = 0.0
    hi = alpha_max / float(np.max(gate_arr))
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        current = update_l2(mid * gate_arr, delta_model)
        if current < target_update_l2:
            lo = mid
        else:
            hi = mid
    alpha = hi * gate_arr
    alpha = np.minimum(alpha, alpha_max)
    matched = update_l2(alpha, delta_model)
    if abs(matched - target_update_l2) > max(1.0e-8, abs(target_update_l2) * rtol):
        raise BudgetMatchError(
            f"BUDGET_MATCH_FAILED: matched={matched:.6g}, target={target_update_l2:.6g}, rtol={rtol}"
        )
    return alpha.astype(np.float32)

