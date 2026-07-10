from __future__ import annotations

import numpy as np
import pytest

from admit_fwi.evaluate_fwi_model_quality import evaluate_model_quality


def test_evaluate_model_quality_reports_model_and_edge_improvement() -> None:
    true_model = np.full((8, 10), 2000.0, dtype=np.float32)
    true_model[:, 5:] = 3200.0
    initial_model = np.full_like(true_model, 2000.0)
    inverted_model = initial_model.copy()
    inverted_model[:, 5:] = 2800.0

    metrics = evaluate_model_quality(
        true_model=true_model,
        initial_model=initial_model,
        inverted_model=inverted_model,
        update=inverted_model - initial_model,
        edge_percentile=80.0,
    )

    assert metrics["verdict"] in {"improved", "numerical_improvement_without_gradient_improvement"}
    assert metrics["mae_improvement_fraction"] > 0.0
    assert metrics["rmse_improvement_fraction"] > 0.0
    assert metrics["edge_fraction"] > 0.0
    assert metrics["update_l1_edge_fraction"] > 0.0
    assert metrics["update_true_error_correlation"] > 0.0


def test_evaluate_model_quality_rejects_mismatched_shapes() -> None:
    model = np.ones((4, 5), dtype=np.float32)

    with pytest.raises(ValueError, match="same shape"):
        evaluate_model_quality(
            true_model=model,
            initial_model=model,
            inverted_model=np.ones((5, 4), dtype=np.float32),
        )
