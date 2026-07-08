from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rtm_acoustic.compare_full_salt_fwi_optimizers import (
    compare_optimizer_runs,
    write_optimizer_comparison,
)


def _write_history(path: Path, optimizer: str, misfits: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "iteration,completed_shots,mean_misfit,step_scale,optimizer,cg_beta,max_abs_update,model_min,model_max"
    ]
    for iteration, misfit in enumerate(misfits):
        lines.append(f"{iteration},224,{misfit},4.0,{optimizer},0.0,100.0,1450.0,4410.0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_run_dir(base: Path, name: str, optimizer: str, misfits: list[float]) -> Path:
    run_dir = base / name
    _write_history(run_dir / "fwi_iteration_history.csv", optimizer, misfits)
    np.save(run_dir / "full_salt_initial_model.npy", np.zeros((3, 4), dtype=np.float32))
    np.save(run_dir / "full_salt_inverted_model.npy", np.ones((3, 4), dtype=np.float32))
    np.save(run_dir / "full_salt_model_update.npy", np.ones((3, 4), dtype=np.float32))
    checkpoint_dir = run_dir / "checkpoint"
    checkpoint_dir.mkdir()
    for iteration in range(len(misfits)):
        (checkpoint_dir / f"iteration_{iteration:03d}.json").write_text(
            json.dumps({"iteration": iteration, "completed_shots": [4, 7, 10]}),
            encoding="utf-8",
        )
    return run_dir


def test_compare_optimizer_runs_marks_partial_when_pcg_has_fewer_iterations(tmp_path: Path) -> None:
    cg_dir = _make_run_dir(tmp_path, "cg", "cg", [0.9, 0.7])
    pcg_dir = _make_run_dir(tmp_path, "pcg", "p-cg", [0.9])

    comparison = compare_optimizer_runs(cg_dir=cg_dir, pcg_dir=pcg_dir)

    assert comparison["status"] == "partial"
    assert comparison["cg"]["iterations_completed"] == 2
    assert comparison["pcg"]["iterations_completed"] == 1
    assert comparison["pcg"]["missing_iterations_vs_cg"] == 1
    assert comparison["cg"]["misfit_reduction_fraction"] > 0.0


def test_write_optimizer_comparison_uses_separate_report_directory(tmp_path: Path) -> None:
    cg_dir = _make_run_dir(tmp_path, "cg", "cg", [0.9, 0.7])
    pcg_dir = _make_run_dir(tmp_path, "pcg", "p-cg", [0.9, 0.65])
    comparison = compare_optimizer_runs(cg_dir=cg_dir, pcg_dir=pcg_dir)
    report_dir = tmp_path / "report"

    written = write_optimizer_comparison(comparison, output_dir=report_dir)

    assert written["json"] == report_dir / "full_salt_fwi_optimizer_compare.json"
    assert written["markdown"] == report_dir / "full_salt_fwi_optimizer_compare.md"
    assert written["json"].exists()
    assert written["markdown"].exists()
    assert not (cg_dir / "full_salt_fwi_optimizer_compare.json").exists()
    assert not (pcg_dir / "full_salt_fwi_optimizer_compare.json").exists()
    assert "CG/P-CG 优化器对照报告" in written["markdown"].read_text(encoding="utf-8")
