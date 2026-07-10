from __future__ import annotations

import numpy as np

from admit_fwi.diagnostics.deep_wavefield_coverage import build_depth_roi_masks, summarize_deep_energy


def test_depth_roi_masks_are_finite_and_disjoint_enough() -> None:
    masks = build_depth_roi_masks((100, 20), absorb_cells=20)
    assert masks.deep.any()
    assert masks.physical.sum() == 80 * 20
    assert not masks.deep[90:, :].any()


def test_deep_energy_rising_at_end_confirms_truncation() -> None:
    times = np.linspace(0.0, 5.0, 101)
    energy = np.linspace(0.0, 1.0, 101)
    summary = summarize_deep_energy(times, energy)
    assert summary["time_truncation_risk"] is True
    assert summary["time_truncation_confirmed"] is True
    assert summary["status"] == "TIME_TRUNCATION_CONFIRMED"


def test_deep_energy_peak_before_tail_is_ok() -> None:
    times = np.linspace(0.0, 5.0, 101)
    energy = np.exp(-((times - 2.0) ** 2))
    summary = summarize_deep_energy(times, energy)
    assert summary["status"] == "DEEP_COVERAGE_OK"
