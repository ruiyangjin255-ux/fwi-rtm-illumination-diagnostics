from __future__ import annotations

import numpy as np
import pytest

from rtm_acoustic.diagnostics.gate_ablation import build_matched_gate_suite, depth_matched_random_support
from rtm_acoustic.diagnostics.matched_budget import BudgetMatchError, scale_to_budget, update_l2


def test_scale_to_budget_matches_target_l2() -> None:
    gate = np.arange(1, 26, dtype=np.float32).reshape(5, 5)
    delta = np.ones((5, 5), dtype=np.float32)
    alpha = scale_to_budget(gate, delta, target_update_l2=0.75, alpha_max=0.5)
    assert update_l2(alpha, delta) == pytest.approx(0.75, rel=1.0e-4)
    assert float(np.max(alpha)) <= 0.5 + 1.0e-6


def test_budget_match_fails_when_alpha_max_too_small() -> None:
    with pytest.raises(BudgetMatchError):
        scale_to_budget(np.ones((4, 4)), np.ones((4, 4)), target_update_l2=100.0, alpha_max=0.1)


def test_random_gate_is_reproducible_for_fixed_seed() -> None:
    reference = np.zeros((5, 8), dtype=bool)
    reference[:, :3] = True
    a = depth_matched_random_support(reference, seed=3)
    b = depth_matched_random_support(reference, seed=3)
    c = depth_matched_random_support(reference, seed=4)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_matched_gate_suite_has_equal_update_budget() -> None:
    shape = (8, 10)
    delta = np.ones(shape, dtype=np.float32)
    illumination = np.tile(np.linspace(0.0, 1.0, shape[1], dtype=np.float32), (shape[0], 1))
    consensus = np.flipud(illumination)
    reliability = np.sqrt(np.clip(illumination * consensus, 0.0, 1.0))
    gates = build_matched_gate_suite(
        delta_model=delta,
        illumination=illumination,
        consensus=consensus,
        reliability=reliability,
        target_update_l2=0.5,
    )
    budgets = [update_l2(gate, delta) for gate in gates.values()]
    assert len(gates) >= 8
    for budget in budgets:
        assert budget == pytest.approx(0.5, rel=1.0e-4)

