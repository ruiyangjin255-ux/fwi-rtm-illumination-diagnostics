from __future__ import annotations

import math

import numpy as np
import pytest

from rtm_acoustic.diagnostics.heldout_audit import (
    audit_record_pair,
    nrms_residual,
    paired_bootstrap,
    trace_correlation,
)


def test_nrms_uses_symmetric_denominator() -> None:
    observed = np.ones((4, 3), dtype=np.float32)
    predicted = np.zeros((4, 3), dtype=np.float32)
    assert nrms_residual(predicted, observed) == pytest.approx(2.0)


def test_trace_correlation_is_receiver_mean() -> None:
    t = np.arange(8, dtype=np.float32)
    observed = np.stack([t, -t], axis=1)
    predicted = observed.copy()
    assert trace_correlation(predicted, observed) == pytest.approx(1.0)


def test_record_pair_reports_finite_phase_and_envelope_metrics() -> None:
    t = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    observed = np.sin(2.0 * np.pi * 5.0 * t)[:, None].repeat(3, axis=1)
    predicted = np.sin(2.0 * np.pi * 5.0 * t + 0.1)[:, None].repeat(3, axis=1)
    metrics = audit_record_pair(predicted, observed)
    assert set(metrics) == {"normalized_l2_residual", "nrms_residual", "trace_correlation", "envelope_error", "phase_error"}
    assert all(math.isfinite(value) for value in metrics.values())


def test_nan_records_fail_fast() -> None:
    observed = np.ones((4, 3), dtype=np.float32)
    predicted = observed.copy()
    predicted[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        audit_record_pair(predicted, observed)


def test_paired_bootstrap_tracks_error_and_correlation_direction() -> None:
    rows = []
    for shot in range(4):
        rows.append(
            {
                "method": "ecg",
                "shot_index": shot,
                "normalized_l2_residual": 0.5,
                "nrms_residual": 0.5,
                "envelope_error": 0.5,
                "phase_error": 0.5,
                "trace_correlation": 0.9,
            }
        )
        rows.append(
            {
                "method": "global",
                "shot_index": shot,
                "normalized_l2_residual": 1.0,
                "nrms_residual": 1.0,
                "envelope_error": 1.0,
                "phase_error": 1.0,
                "trace_correlation": 0.2,
            }
        )
    result = paired_bootstrap(rows, reference_method="ecg", comparator_methods=["global"], samples=100, seed=1)
    assert result["global"]["normalized_l2_residual"]["probability_ecg_better"] == pytest.approx(1.0)
    assert result["global"]["trace_correlation"]["probability_ecg_better"] == pytest.approx(1.0)
