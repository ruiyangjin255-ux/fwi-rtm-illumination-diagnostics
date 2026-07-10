from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from admit_fwi.diagnostics.admit_input_audit import audit_inputs
from admit_fwi.scripts.run_holdout_gate_audit import MODEL_FILES


def test_input_audit_finds_seg_salt_and_missing_external(tmp_path: Path):
    root = tmp_path
    fwi = root / "outputs" / "FWI" / "full_salt_fwi_cg_audit0_train_ecg_v1"
    models = root / "outputs" / "salt_reliability_gate_audit0_v1" / "models"
    audit = root / "outputs" / "salt_reliability_gate_audit0_v1" / "audit"
    rtm = root / "outputs" / "RTM" / "audit0_gate_rtm_v1"
    deep = root / "outputs" / "deep_time_preflight_v1"
    fwi.mkdir(parents=True)
    models.mkdir(parents=True)
    audit.mkdir(parents=True)
    rtm.mkdir(parents=True)
    (deep / "wavefield_smoke").mkdir(parents=True)
    (deep / "boundary_energy").mkdir(parents=True)
    arr = np.ones((3, 4), dtype=np.float32)
    np.save(fwi / "full_salt_true_model.npy", arr)
    np.save(fwi / "full_salt_initial_model.npy", arr * 0.9)
    (fwi / "full_salt_fwi_summary.json").write_text(json.dumps({"audit_split": {"train_shots": [1], "audit_shots": [2]}}), encoding="utf-8")
    for filename in MODEL_FILES.values():
        np.save(models / filename, arr)
    for path in [
        audit / "audit_method_summary.csv",
        audit / "audit_method_summary.md",
        audit / "heldout_audit_manifest.json",
        rtm / "gate_rtm_method_summary.csv",
        rtm / "gate_rtm_method_summary.md",
        rtm / "gate_rtm_manifest.json",
        deep / "deep_time_plan.json",
        deep / "time_sampling_audit.json",
        deep / "wavefield_smoke" / "deep_energy_summary.json",
        deep / "boundary_energy" / "boundary_energy_summary.json",
    ]:
        path.write_text("{}", encoding="utf-8")
    report = audit_inputs(root, tmp_path / "out", search_dirs=[tmp_path / "no_models"])
    assert report["can_enter_p1_seg_salt"] is True
    assert report["seg_salt"]["current_model_shape"] == [3, 4]
    assert "marmousi" in report["missing_external_models"]
