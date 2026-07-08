from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import rtm_acoustic.diagnostics.rtm_split_runner as runner
from rtm_acoustic.diagnostics.rtm_split_runner import P0C_AUDIT_RTM_SHOTS, run_split_subsets, selected_subsets


def test_subset_a_b_are_disjoint_and_cover_p0c_shots() -> None:
    subsets = selected_subsets(smoke=False)
    a = set(subsets["subset_A"])
    b = set(subsets["subset_B"])
    assert a.isdisjoint(b)
    assert sorted(a | b) == sorted(P0C_AUDIT_RTM_SHOTS)


def test_split_runner_writes_new_directory_and_no_git_manifest(tmp_path: Path, monkeypatch) -> None:
    fwi = tmp_path / "fwi"
    models = tmp_path / "models"
    out = tmp_path / "audit0_gate_rtm_split_v1"
    fwi.mkdir()
    models.mkdir()
    nz, nx = 8, 10
    true = np.full((nz, nx), 2000.0, dtype=np.float32)
    np.save(fwi / "full_salt_true_model.npy", true)
    (fwi / "full_salt_fwi_summary.json").write_text(
        json.dumps(
            {
                "config": {
                    "nx": nx,
                    "nz": nz,
                    "dx": 10.0,
                    "dz": 10.0,
                    "source_z": 2,
                    "receiver_z": 2,
                    "absorb_cells": 2,
                    "fd_order": 4,
                }
            }
        ),
        encoding="utf-8",
    )
    np.save(models / "initial_model.npy", true)

    def fake_rtm(velocity, cfg, shots, **kwargs):
        image = np.ones((cfg.nz, cfg.nx), dtype=np.float32) * float(sum(shots))
        return SimpleNamespace(
            image=image,
            filtered_image=image,
            normalized_image=image,
            source_receiver_normalized_image=image,
            stacked_record=np.zeros((cfg.nt, cfg.nx), dtype=np.float32),
            illumination=np.ones_like(image),
            receiver_illumination=np.ones_like(image),
        )

    monkeypatch.setattr(runner, "multishot_reverse_time_migrate_parallel", fake_rtm)
    monkeypatch.setattr(runner, "save_migration_figure", lambda *args, **kwargs: Path(args[0]).write_text("fig", encoding="utf-8"))
    monkeypatch.setattr(runner, "save_record_and_migration_figure", lambda *args, **kwargs: Path(args[0]).write_text("fig", encoding="utf-8"))
    monkeypatch.setattr(runner, "git_commit", lambda root: "NO_GIT_REPOSITORY")
    manifest = run_split_subsets(root=tmp_path, fwi_dir=fwi, model_dir=models, output_dir=out, methods=["initial"], smoke=True, workers=1, command="cmd")
    assert manifest["git_commit"] == "NO_GIT_REPOSITORY"
    assert (out / "subset_A" / "initial" / "rtm_metadata.json").exists()
    assert not (tmp_path / "audit0_gate_rtm_v1").exists()
    meta = json.loads((out / "subset_A" / "initial" / "rtm_metadata.json").read_text(encoding="utf-8"))
    assert meta["dt"] == 0.001
    assert meta["nt"] == 600
    assert meta["f0"] == 15.0
