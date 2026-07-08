from pathlib import Path
import json

import numpy as np

from scripts.run_protocol_v11_visionfm_crossfamily import (
    REQUIRED_PREDICTION_FIELDS,
    REQUIRED_SUCCESS_FILES,
    _is_complete_success,
    assert_target_test_isolation,
    write_prediction_npz,
)


def test_target_test_is_not_used_for_training_or_selection() -> None:
    manifest = {
        "train_samples": [{"path": "source_train_1.npz"}],
        "val_samples": [{"path": "source_val_1.npz"}],
        "in_family_test_samples": [{"path": "source_test_1.npz"}],
        "cross_family_test_samples": [{"path": "target_test_1.npz"}],
    }
    assert_target_test_isolation(manifest)


def test_prediction_npz_contains_physical_contract(tmp_path: Path) -> None:
    path = tmp_path / "predictions_cross_family_test.npz"
    prediction = np.zeros((2, 1, 70, 70), dtype=np.float32)
    target = np.ones_like(prediction)
    write_prediction_npz(
        path,
        prediction=prediction,
        target=target,
        sample_ids=["a", "b"],
        metadata={"model_id": "M1", "bridge_name": "raw_envelope_spectrum3", "source_family": "flatvel_a", "target_family": "curvevel_a", "seed": 0},
    )
    with np.load(path) as payload:
        assert REQUIRED_PREDICTION_FIELDS.issubset(payload.files)
        assert payload["velocity_pred_physical"].shape == (2, 1, 70, 70)


def test_resume_rejects_vision_run_without_registered_common_decoder(tmp_path: Path) -> None:
    for name in REQUIRED_SUCCESS_FILES:
        (tmp_path / name).write_text("", encoding="utf-8")
    (tmp_path / "config.json").write_text(json.dumps({"status": "SUCCESS"}), encoding="utf-8")
    (tmp_path / "model_card.json").write_text(
        json.dumps({"kind": "vision", "total_parameters": 21_628_800}), encoding="utf-8"
    )
    assert not _is_complete_success(tmp_path)

    (tmp_path / "model_card.json").write_text(
        json.dumps({"kind": "vision", "total_parameters": 23_630_881}), encoding="utf-8"
    )
    assert _is_complete_success(tmp_path)
