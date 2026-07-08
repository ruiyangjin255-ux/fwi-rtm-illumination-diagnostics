from __future__ import annotations

import numpy as np

from rtm_acoustic.models.synthetic_model_bank import build_model, save_synthetic_model


def test_simple_models_have_distinct_true_and_initial(tmp_path):
    for name in ["simple_layered", "simple_fault"]:
        true, initial = build_model(name, nx=48, nz=24)
        assert true.shape == initial.shape == (24, 48)
        assert not np.allclose(true, initial)
        assert 1400.0 <= float(true.min()) <= float(true.max()) <= 5000.0
        manifest = save_synthetic_model(name, tmp_path, nx=48, nz=24)
        assert manifest["model_source"] == "SYNTHETIC_DIAGNOSTIC_MODEL"
        assert (tmp_path / name / "true_velocity.npy").exists()
