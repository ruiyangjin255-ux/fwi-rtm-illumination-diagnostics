from __future__ import annotations

import numpy as np
import pytest

from admit_fwi.diagnostics.boundary_energy_audit import boundary_energy_ratio, boundary_mask, classify_boundary_energy


def test_boundary_mask_marks_sides_and_bottom() -> None:
    mask = boundary_mask((10, 12), cells=2)
    assert mask[:, :2].all()
    assert mask[:, -2:].all()
    assert mask[-2:, :].all()
    assert not mask[2, 5]


def test_boundary_ratio_is_finite() -> None:
    ratio = boundary_energy_ratio(np.array([1.0, 2.0]), np.array([10.0, 20.0]))
    assert np.isfinite(ratio).all()
    assert ratio[0] == pytest.approx(0.1)


def test_boundary_classifier_flags_risk_near_deep_peak() -> None:
    times = np.linspace(0.0, 5.0, 101)
    ratio = np.zeros_like(times)
    ratio[50] = 0.2
    summary = classify_boundary_energy(times, ratio, deep_peak_time=2.5)
    assert summary["status"] == "PML_REFLECTION_RISK"
