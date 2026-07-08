from __future__ import annotations

import numpy as np
import pytest

from rtm_acoustic.diagnostics.time_sampling_audit import record_time, strict_disjoint, wavelet_summary


def test_record_time_uses_nt_times_dt() -> None:
    assert record_time(5000, 0.001) == pytest.approx(5.0)


def test_wavelet_summary_reports_peak_and_period() -> None:
    summary = wavelet_summary(15.0)
    assert summary["peak_time"] == pytest.approx(1.0 / 15.0)
    assert summary["dominant_period"] == pytest.approx(1.0 / 15.0)


def test_audit_shots_are_disjoint() -> None:
    assert strict_disjoint([1, 2, 3], [4, 5])
    assert not strict_disjoint([1, 2, 3], [3, 4])
