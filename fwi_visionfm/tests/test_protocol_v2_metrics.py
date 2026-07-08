from __future__ import annotations

import numpy as np
import pytest


def test_protocol_v2_metrics_run_on_toy_tensor():
    from fwi_visionfm.evaluation.metrics import compute_velocity_metrics

    target = np.linspace(0.0, 1.0, 2 * 1 * 6 * 7, dtype=np.float32).reshape(2, 1, 6, 7)
    prediction = target + 0.01
    metrics = compute_velocity_metrics(prediction, target, data_range=1.0)
    for key in ("mae", "rmse", "ssim", "psnr", "gradient_error", "gradient_mae", "edge_mae"):
        assert key in metrics
    assert metrics["mae"] > 0.0
    assert metrics["rmse"] > 0.0
    assert metrics["ssim_available"] is True


def test_protocol_v2_metrics_raise_clear_error_for_shape_mismatch():
    from fwi_visionfm.evaluation.metrics import compute_velocity_metrics

    with pytest.raises(ValueError, match="prediction and target shapes must match"):
        compute_velocity_metrics(np.zeros((1, 1, 6, 7), dtype=np.float32), np.zeros((1, 1, 5, 7), dtype=np.float32))

