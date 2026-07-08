import pytest

from scripts.run_protocol_v12_spectrogram_dinov2_confirmation import assert_v12_target_isolation


def test_target_test_is_evaluation_only() -> None:
    manifest = {
        "source_family": "flatvel_a",
        "target_family": "curvevel_a",
        "train_samples": [{"sample_id": "s1", "path": "s1.npz"}],
        "val_samples": [{"sample_id": "s2", "path": "s2.npz"}],
        "in_family_test_samples": [{"sample_id": "s3", "path": "s3.npz"}],
        "cross_family_test_samples": [{"sample_id": "t1", "path": "t1.npz"}],
    }
    assert_v12_target_isolation(manifest)
    manifest["train_samples"].append({"sample_id": "t1", "path": "t1.npz"})
    with pytest.raises(ValueError, match="target test leakage"):
        assert_v12_target_isolation(manifest)

