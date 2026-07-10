from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


CONFIG = Path("admit_fwi/configs/salt_reliability_gate_v1.yaml")
OUTPUT = Path("admit_fwi/outputs/salt_reliability_gate_v1")


def test_replay_smoke_fails_fast_without_required_diagnostics(tmp_path: Path) -> None:
    output = Path("admit_fwi/outputs/reliability_missing_diagnostics_smoke")
    shutil.rmtree(output, ignore_errors=True)
    config = tmp_path / "missing_diagnostics.yaml"
    config.write_text(
        "\n".join(
            [
                "name: reliability_missing_diagnostics_smoke",
                "output_dir: admit_fwi/outputs/reliability_missing_diagnostics_smoke",
                "num_shot_groups: 2",
                "required_diagnostics:",
                "  - gradient_group_00.npy",
                "  - gradient_group_01.npy",
                "  - source_adjoint_energy_proxy.npy",
                "  - delta_model.npy",
                "  - step_length.json",
            ]
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "admit_fwi/scripts/replay_fwi_diagnostics.py",
            "--config",
            str(config),
            "--smoke",
        ],
        cwd=Path.cwd(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode != 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "BLOCKED_MISSING_FWI_DIAGNOSTICS"
    assert manifest["missing"]


def test_holdout_smoke_writes_split_manifest() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "admit_fwi/scripts/run_holdout_gate_audit.py",
            "--config",
            str(CONFIG),
            "--all-audit-folds",
            "--smoke",
        ],
        cwd=Path.cwd(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads((OUTPUT / "audit" / "heldout_audit_manifest.json").read_text(encoding="utf-8"))
    assert payload["status"] == "SPLIT_PREPARED"
    assert len(payload["folds"]) == 4


def test_replay_and_ablation_generate_real_gate_outputs(tmp_path: Path) -> None:
    output = Path("admit_fwi/outputs/reliability_script_smoke")
    shutil.rmtree(output, ignore_errors=True)
    diagnostics = output / "diagnostics"
    diagnostics.mkdir(parents=True)
    rng = np.random.default_rng(7)
    shape = (8, 9)
    delta = rng.normal(size=shape).astype(np.float32)
    np.save(diagnostics / "gradient_group_00.npy", -delta + 0.05 * rng.normal(size=shape).astype(np.float32))
    np.save(diagnostics / "gradient_group_01.npy", -delta + 0.05 * rng.normal(size=shape).astype(np.float32))
    np.save(diagnostics / "source_adjoint_energy_proxy.npy", np.abs(rng.normal(size=shape)).astype(np.float32) + 1.0)
    np.save(diagnostics / "delta_model.npy", delta)
    (diagnostics / "step_length.json").write_text('{"step_scale": 1.0}', encoding="utf-8")
    config = tmp_path / "reliability_smoke.yaml"
    config.write_text(
        "\n".join(
            [
                "name: reliability_script_smoke",
                "output_dir: admit_fwi/outputs/reliability_script_smoke",
                "num_shot_groups: 2",
                "shot_group_mode: interleaved",
                "coverage_candidates: [0.5]",
                "alpha_max_candidates: [0.3]",
                "sigma_x_candidates: [1]",
                "sigma_z_candidates: [1]",
                "eps: 1.0e-8",
                "target_update_fraction: 0.02",
                "required_diagnostics:",
                "  - gradient_group_00.npy",
                "  - gradient_group_01.npy",
                "  - source_adjoint_energy_proxy.npy",
                "  - delta_model.npy",
                "  - step_length.json",
            ]
        ),
        encoding="utf-8",
    )
    replay = subprocess.run(
        [sys.executable, "admit_fwi/scripts/replay_fwi_diagnostics.py", "--config", str(config), "--smoke"],
        cwd=Path.cwd(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert replay.returncode == 0, replay.stderr
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "READY"
    assert (output / "gates" / "ecg_reliability_gate.npy").exists()
    ablation = subprocess.run(
        [sys.executable, "admit_fwi/scripts/run_reliability_gate_ablation.py", "--config", str(config), "--smoke"],
        cwd=Path.cwd(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert ablation.returncode == 0, ablation.stderr
    gate_manifest = json.loads((output / "gates" / "gate_ablation_manifest.json").read_text(encoding="utf-8"))
    assert gate_manifest["status"] == "READY"
    assert gate_manifest["gate_count"] >= 6
