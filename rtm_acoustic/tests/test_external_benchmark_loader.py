from __future__ import annotations

import numpy as np
import pytest

from rtm_acoustic.models.external_benchmark_loader import prepare_external_crop


def test_missing_external_model_is_not_faked(tmp_path):
    manifest = prepare_external_crop(name="marmousi", source_path=tmp_path / "missing.npy", output_root=tmp_path)
    assert manifest["status"] == "MISSING_EXTERNAL_MODEL"
    assert not (tmp_path / "marmousi_crop" / "true_velocity.npy").exists()


def test_external_npy_crop_writes_distinct_initial(tmp_path):
    src = tmp_path / "marmousi.npy"
    model = np.arange(40 * 50, dtype=np.float32).reshape(40, 50) + 1500
    np.save(src, model)
    manifest = prepare_external_crop(name="marmousi", source_path=src, output_root=tmp_path, crop=(0, 30, 0, 20), downsample=2)
    assert manifest["status"] == "READY"
    true = np.load(tmp_path / "marmousi_crop" / "true_velocity.npy")
    initial = np.load(tmp_path / "marmousi_crop" / "initial_velocity.npy")
    assert true.shape == initial.shape == (10, 15)
    assert not np.allclose(true, initial)


def test_unsupported_external_format_fails(tmp_path):
    src = tmp_path / "sigsbee2a.segy"
    src.write_bytes(b"not parsed")
    with pytest.raises(ValueError, match="UNSUPPORTED_EXTERNAL_MODEL_FORMAT"):
        prepare_external_crop(name="sigsbee2a", source_path=src, output_root=tmp_path)
