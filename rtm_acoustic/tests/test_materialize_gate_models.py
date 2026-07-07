from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rtm_acoustic.scripts.materialize_gate_models import GATE_TO_MODEL, materialize_models


def test_materialize_models_uses_frozen_initial_delta_and_gates(tmp_path: Path) -> None:
    result_dir = tmp_path / "result"
    fwi_dir = tmp_path / "fwi"
    (result_dir / "diagnostics").mkdir(parents=True)
    (result_dir / "gates").mkdir(parents=True)
    fwi_dir.mkdir(parents=True)
    initial = np.full((4, 5), 2000.0, dtype=np.float32)
    delta = np.arange(20, dtype=np.float32).reshape(4, 5)
    np.save(fwi_dir / "full_salt_initial_model.npy", initial)
    np.save(fwi_dir / "full_salt_inverted_model.npy", initial + delta)
    np.save(result_dir / "diagnostics" / "delta_model.npy", delta)
    for index, gate_name in enumerate(GATE_TO_MODEL):
        gate = np.full((4, 5), 0.01 * (index + 1), dtype=np.float32)
        np.save(result_dir / "gates" / f"{gate_name}.npy", gate)

    manifest = materialize_models(result_dir=result_dir, fwi_dir=fwi_dir)

    assert manifest["status"] == "READY"
    assert manifest["true_model_used"] is False
    assert len(manifest["models"]) == len(GATE_TO_MODEL)
    model_path = result_dir / "models" / "ecg_reliability_gate_model.npy"
    assert model_path.exists()
    ecg_index = list(GATE_TO_MODEL).index("ecg_reliability_gate")
    expected = initial + np.float32(0.01 * (ecg_index + 1)) * delta
    np.testing.assert_allclose(np.load(model_path), expected)
    payload = json.loads((result_dir / "models" / "gate_model_manifest.json").read_text(encoding="utf-8"))
    assert payload["formula"] == "model = initial_model + alpha_gate * frozen_delta_model"
