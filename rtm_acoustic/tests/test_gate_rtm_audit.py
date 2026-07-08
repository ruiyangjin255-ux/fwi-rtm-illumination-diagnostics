from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import rtm_acoustic.scripts.run_gate_rtm_audit as gate_rtm
from rtm_acoustic.scripts.run_gate_rtm_audit import run_gate_rtm_audit
from rtm_acoustic.scripts.run_holdout_gate_audit import MODEL_FILES


def test_run_gate_rtm_audit_writes_summary_for_all_models(tmp_path: Path, monkeypatch) -> None:
    nz, nx = 24, 28
    fwi_dir = tmp_path / "fwi"
    model_dir = tmp_path / "models"
    output_dir = tmp_path / "rtm"
    fwi_dir.mkdir()
    model_dir.mkdir()
    true_model = np.full((nz, nx), 2000.0, dtype=np.float32)
    true_model[12:, :] = 2600.0
    np.save(fwi_dir / "full_salt_true_model.npy", true_model)
    summary = {
        "config": {
            "nx": nx,
            "nz": nz,
            "dx": 10.0,
            "dz": 10.0,
            "dt": 0.001,
            "nt": 35,
            "f0": 18.0,
            "source_z": 4,
            "receiver_z": 4,
            "absorb_cells": 4,
            "fd_order": 4,
        },
        "audit_split": {"audit_shots": [5, 15, 20]},
    }
    (fwi_dir / "full_salt_fwi_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    for index, filename in enumerate(MODEL_FILES.values()):
        model = true_model + np.float32(index)
        np.save(model_dir / filename, model)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("output_dir: unused\n", encoding="utf-8")

    def fake_rtm(velocity, cfg, shot_positions, **kwargs):
        model = kwargs.get("migration_velocity", velocity)
        scale = float(np.mean(model) - np.mean(velocity)) * 1.0e-4
        image = np.full((cfg.nz, cfg.nx), scale, dtype=np.float32)
        return SimpleNamespace(
            stacked_record=np.zeros((cfg.nt, cfg.nx), dtype=np.float32),
            image=image,
            normalized_image=image + 1.0,
            source_receiver_normalized_image=image + 2.0,
            filtered_image=image + 3.0,
            illumination=np.ones_like(image),
            receiver_illumination=np.ones_like(image),
        )

    monkeypatch.setattr(gate_rtm, "multishot_reverse_time_migrate_parallel", fake_rtm)
    monkeypatch.setattr(gate_rtm, "save_migration_figure", lambda *args, **kwargs: Path(args[0]).write_text("fig", encoding="utf-8"))
    monkeypatch.setattr(gate_rtm, "save_record_and_migration_figure", lambda *args, **kwargs: Path(args[0]).write_text("fig", encoding="utf-8"))

    manifest = run_gate_rtm_audit(
        config_path=config_path,
        fwi_dir=fwi_dir,
        model_dir=model_dir,
        output_dir=output_dir,
        nt=35,
        f0=18.0,
        max_shots=2,
        workers=1,
        smoke=True,
    )

    assert manifest["status"] == "READY"
    assert manifest["shot_count"] == 2
    assert (output_dir / "gate_rtm_method_summary.csv").exists()
    assert (output_dir / "gate_rtm_manifest.json").exists()
    assert (output_dir / "ecg" / "rtm_laplacian_filtered_physical.npy").exists()
