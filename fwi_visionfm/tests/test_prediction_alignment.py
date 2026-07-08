from __future__ import annotations

from pathlib import Path

import numpy as np


def _write_prediction(
    path: Path,
    *,
    sample_id: list[str],
    prediction: np.ndarray,
    target: np.ndarray,
    metric_space: str = "physical_velocity",
) -> None:
    np.savez(
        path,
        prediction=prediction.astype(np.float32),
        target=target.astype(np.float32),
        sample_id=np.asarray(sample_id),
        metric_space=np.asarray(metric_space),
        velocity_pred_physical=prediction.astype(np.float32),
        velocity_true_physical=target.astype(np.float32),
    )


def test_align_predictions_by_sample_id_keeps_common_sorted_ids(tmp_path: Path):
    from fwi_visionfm.evaluation.prediction_alignment import align_predictions_by_sample_id, load_prediction_npz

    pred_a = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    pred_b = pred_a[[2, 1, 0]] + 1.0
    target = np.ones((3, 2, 2), dtype=np.float32)
    _write_prediction(tmp_path / "a.npz", sample_id=["b", "a", "c"], prediction=pred_a, target=target)
    _write_prediction(tmp_path / "b.npz", sample_id=["c", "a", "b"], prediction=pred_b, target=target[[2, 1, 0]])

    aligned = align_predictions_by_sample_id(load_prediction_npz(tmp_path / "a.npz"), load_prediction_npz(tmp_path / "b.npz"))
    assert aligned["status"] == "ALIGNED"
    assert aligned["sample_id"] == ["a", "b", "c"]
    assert aligned["prediction_a"].shape == (3, 2, 2)
    assert aligned["prediction_b"].shape == (3, 2, 2)


def test_validate_prediction_targets_reports_mismatch(tmp_path: Path):
    from fwi_visionfm.evaluation.prediction_alignment import load_prediction_npz, validate_prediction_targets

    target_a = np.zeros((2, 2, 2), dtype=np.float32)
    target_b = target_a.copy()
    target_b[1, 0, 0] = 1.0
    prediction = np.zeros_like(target_a)
    _write_prediction(tmp_path / "a.npz", sample_id=["id0", "id1"], prediction=prediction, target=target_a)
    _write_prediction(tmp_path / "b.npz", sample_id=["id0", "id1"], prediction=prediction, target=target_b)

    status = validate_prediction_targets(load_prediction_npz(tmp_path / "a.npz"), load_prediction_npz(tmp_path / "b.npz"))
    assert status["status"] == "TARGET_MISMATCH"
    assert status["max_target_diff"] > 0.0

