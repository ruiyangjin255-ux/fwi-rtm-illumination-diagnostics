from __future__ import annotations

import numpy as np
from pathlib import Path


def _write_npz(path: Path, prediction: np.ndarray, target: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, prediction=prediction.astype("float32"), target=target.astype("float32"))


def test_average_fusion_selects_best_alpha_on_fake_predictions(tmp_path: Path):
    from fwi_visionfm.scripts.fuse_protocol_v4_predictions import run_pair_fusion

    target = np.ones((2, 8, 8), dtype="float32")
    pred_a_val = target.copy()
    pred_b_val = np.zeros_like(target)
    pred_a_test = target * 0.9
    pred_b_test = target * 0.2
    a = tmp_path / "a"
    b = tmp_path / "b"
    _write_npz(a / "predictions_in_family_test.npz", pred_a_val, target)
    _write_npz(b / "predictions_in_family_test.npz", pred_b_val, target)
    _write_npz(a / "predictions_cross_family_test.npz", pred_a_test, target)
    _write_npz(b / "predictions_cross_family_test.npz", pred_b_test, target)

    result = run_pair_fusion(
        output_dir=tmp_path / "fusion",
        run_a=a,
        run_b=b,
        method="average_fusion",
        optimize_on="val",
        reference_only=False,
    )

    assert result["best_param"] == 1.0
    assert (tmp_path / "fusion" / "fused_predictions_cross_family_test.npz").exists()
    assert (tmp_path / "fusion" / "fused_metrics_cross_family_test.json").exists()


def test_low_high_and_edge_aware_fusion_shape():
    from fwi_visionfm.scripts.fuse_protocol_v4_predictions import edge_aware_fusion, low_high_fusion

    pred_a = np.ones((3, 12, 10), dtype="float32")
    pred_b = np.random.RandomState(0).randn(3, 12, 10).astype("float32")

    assert low_high_fusion(pred_a, pred_b, beta=0.5).shape == pred_a.shape
    assert edge_aware_fusion(pred_a, pred_b, edge_scale=1.0).shape == pred_a.shape


def test_pair_fusion_rejects_mismatched_targets(tmp_path: Path):
    from fwi_visionfm.scripts.fuse_protocol_v4_predictions import run_pair_fusion

    pred = np.zeros((1, 4, 4), dtype="float32")
    a = tmp_path / "a"
    b = tmp_path / "b"
    _write_npz(a / "predictions_in_family_test.npz", pred, np.zeros_like(pred))
    _write_npz(b / "predictions_in_family_test.npz", pred, np.ones_like(pred))
    _write_npz(a / "predictions_cross_family_test.npz", pred, np.zeros_like(pred))
    _write_npz(b / "predictions_cross_family_test.npz", pred, np.ones_like(pred))

    try:
        run_pair_fusion(output_dir=tmp_path / "fusion", run_a=a, run_b=b, method="average_fusion", optimize_on="val", reference_only=False)
    except ValueError as exc:
        assert "targets differ" in str(exc)
    else:
        raise AssertionError("expected mismatched targets to be rejected")
