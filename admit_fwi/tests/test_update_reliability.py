from __future__ import annotations

import numpy as np

from admit_fwi.diagnostics.update_reliability import (
    build_soft_gate,
    compute_descent_alignment,
    compute_gradient_consensus,
    compute_illumination_score,
    compute_reliability_score,
)


def test_identical_group_gradients_have_consensus_one() -> None:
    base = np.ones((4, 5), dtype=np.float32)
    groups = np.stack([base, base, base], axis=0)
    consensus = compute_gradient_consensus(groups)
    assert np.allclose(consensus, 1.0)


def test_opposite_group_gradients_have_low_consensus() -> None:
    base = np.ones((4, 5), dtype=np.float32)
    groups = np.stack([base, -base], axis=0)
    consensus = compute_gradient_consensus(groups)
    assert float(np.max(consensus)) < 1.0e-5


def test_descent_alignment_prefers_negative_gradient_direction() -> None:
    groups = np.stack([np.ones((4, 5), dtype=np.float32), np.ones((4, 5), dtype=np.float32)], axis=0)
    aggregate = np.sum(groups, axis=0)
    good_delta = -np.ones((4, 5), dtype=np.float32)
    bad_delta = np.ones((4, 5), dtype=np.float32)
    good = compute_descent_alignment(good_delta, aggregate, groups)
    bad = compute_descent_alignment(bad_delta, aggregate, groups)
    assert float(np.mean(good)) > float(np.mean(bad))


def test_reliability_score_and_gate_respect_alpha_max() -> None:
    score = np.linspace(0.0, 1.0, 25, dtype=np.float32).reshape(5, 5)
    delta = np.ones((5, 5), dtype=np.float32)
    alpha = build_soft_gate(
        score,
        coverage=0.4,
        sigma_x=1,
        sigma_z=1,
        alpha_max=0.3,
        target_update_l2=0.5,
        delta_model=delta,
    )
    assert float(np.max(alpha)) <= 0.3 + 1.0e-6
    assert alpha.shape == delta.shape


def test_illumination_and_reliability_are_bounded() -> None:
    proxy = np.arange(25, dtype=np.float32).reshape(5, 5)
    illumination = compute_illumination_score(proxy)
    reliability = compute_reliability_score(illumination, np.ones_like(proxy), np.ones_like(proxy))
    assert float(np.min(illumination)) >= 0.0
    assert float(np.max(illumination)) <= 1.0
    assert float(np.min(reliability)) >= 0.0
    assert float(np.max(reliability)) <= 1.0 + 1.0e-6

