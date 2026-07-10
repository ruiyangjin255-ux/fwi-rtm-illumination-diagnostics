from __future__ import annotations

import numpy as np

from admit_fwi.acoustic_rtm import write_binary_model
from admit_fwi.run_full_salt_fwi import (
    FullSaltFWIConfig,
    _compute_cg_direction,
    limited_shots,
    run_full_salt_fwi,
    select_fwi_shots,
)


def test_limited_shots_returns_uniform_subset() -> None:
    shots = [4, 10, 16, 22, 28]

    assert limited_shots(shots, max_shots=3) == [4, 16, 28]
    assert limited_shots(shots, max_shots=0) == shots


def test_full_salt_fwi_writes_checkpoint_and_summary(tmp_path) -> None:
    model = np.full((24, 36), 2100.0, dtype=np.float32)
    model[12:, 14:24] = 2600.0
    model_path = tmp_path / "model.bin"
    write_binary_model(model_path, model)
    cfg = FullSaltFWIConfig(
        nx=36,
        nz=24,
        nt=60,
        f0=10.0,
        fd_order=4,
        absorb_cells=8,
        iterations=1,
        shot_spacing=160.0,
        max_shots=2,
        step_scale=1.0,
        max_update=10.0,
        smooth_radius_z=2,
        smooth_radius_x=3,
        smooth_passes=1,
    )

    summary = run_full_salt_fwi(
        model_path=model_path,
        output_dir=tmp_path / "out",
        config=cfg,
        write_figures=False,
    )

    assert summary["model_shape"] == [24, 36]
    assert summary["shot_count"] == len(select_fwi_shots(cfg))
    assert (tmp_path / "out" / "checkpoint" / "iteration_000.json").exists()
    assert (tmp_path / "out" / "full_salt_fwi_summary.json").exists()
    assert np.isfinite(np.load(tmp_path / "out" / "full_salt_inverted_model.npy")).all()
    assert "全范围盐丘模型 FWI" in (tmp_path / "out" / "full_salt_fwi_summary.md").read_text(encoding="utf-8")


def test_full_salt_fwi_supports_cg_optimizer(tmp_path) -> None:
    model = np.full((24, 36), 2100.0, dtype=np.float32)
    model[12:, 14:24] = 2600.0
    model_path = tmp_path / "model.bin"
    write_binary_model(model_path, model)
    cfg = FullSaltFWIConfig(
        nx=36,
        nz=24,
        nt=60,
        f0=10.0,
        fd_order=4,
        absorb_cells=8,
        iterations=2,
        shot_spacing=160.0,
        max_shots=2,
        step_scale=1.0,
        max_update=10.0,
        smooth_radius_z=2,
        smooth_radius_x=3,
        smooth_passes=1,
        optimizer="cg",
    )

    summary = run_full_salt_fwi(
        model_path=model_path,
        output_dir=tmp_path / "cg",
        config=cfg,
        write_figures=False,
    )

    assert summary["config"]["optimizer"] == "cg"
    assert (tmp_path / "cg" / "optimizer_state" / "previous_gradient.npy").exists()
    assert (tmp_path / "cg" / "optimizer_state" / "previous_direction.npy").exists()
    history = (tmp_path / "cg" / "fwi_iteration_history.csv").read_text(encoding="utf-8")
    assert "cg_beta" in history


def test_full_salt_fwi_supports_preconditioned_cg_optimizer(tmp_path) -> None:
    model = np.full((24, 36), 2100.0, dtype=np.float32)
    model[12:, 14:24] = 2600.0
    model_path = tmp_path / "model.bin"
    write_binary_model(model_path, model)
    cfg = FullSaltFWIConfig(
        nx=36,
        nz=24,
        nt=60,
        f0=10.0,
        fd_order=4,
        absorb_cells=8,
        iterations=1,
        shot_spacing=160.0,
        max_shots=2,
        step_scale=1.0,
        max_update=10.0,
        smooth_radius_z=2,
        smooth_radius_x=3,
        smooth_passes=1,
        optimizer="p-cg",
        preconditioner_epsilon=0.5,
    )

    summary = run_full_salt_fwi(
        model_path=model_path,
        output_dir=tmp_path / "pcg",
        config=cfg,
        write_figures=False,
    )

    assert summary["config"]["optimizer"] == "p-cg"
    assert summary["config"]["preconditioner_epsilon"] == 0.5
    assert np.isfinite(np.load(tmp_path / "pcg" / "full_salt_model_update.npy")).all()


def test_full_salt_fwi_outer_padding_keeps_physical_outputs_cropped(tmp_path) -> None:
    model = np.full((24, 36), 2100.0, dtype=np.float32)
    model[12:, 14:24] = 2600.0
    model_path = tmp_path / "model.bin"
    write_binary_model(model_path, model)
    cfg = FullSaltFWIConfig(
        nx=36,
        nz=24,
        nt=60,
        f0=10.0,
        fd_order=4,
        absorb_cells=8,
        iterations=1,
        shot_spacing=160.0,
        max_shots=2,
        step_scale=1.0,
        max_update=10.0,
        smooth_radius_z=2,
        smooth_radius_x=3,
        smooth_passes=1,
        pad_x=3,
        pad_top=2,
        pad_bottom=8,
    )

    summary = run_full_salt_fwi(
        model_path=model_path,
        output_dir=tmp_path / "padded",
        config=cfg,
        write_figures=False,
    )

    assert summary["model_shape"] == [24, 36]
    assert summary["padded_model_shape"] == [34, 42]
    assert summary["padding"]["pad_x"] == 3
    assert summary["padding"]["pad_top"] == 2
    assert summary["padding"]["pad_bottom"] == 8
    assert np.load(tmp_path / "padded" / "full_salt_inverted_model.npy").shape == (24, 36)
    assert np.load(tmp_path / "padded" / "full_salt_model_update.npy").shape == (24, 36)
    assert np.load(tmp_path / "padded" / "full_salt_inverted_model_padded.npy").shape == (34, 42)
    assert np.load(tmp_path / "padded" / "full_salt_model_update_padded.npy").shape == (34, 42)
    assert np.load(tmp_path / "padded" / "observations" / "shot_00004.npy").shape == (60, 42)


def test_compute_cg_direction_uses_float64_energy() -> None:
    current = np.full((2, 2), 1.0e20, dtype=np.float32)
    previous = np.full((2, 2), 5.0e19, dtype=np.float32)
    previous_direction = np.ones((2, 2), dtype=np.float32)

    direction, beta = _compute_cg_direction(current, previous, previous_direction)

    assert np.isfinite(beta)
    assert beta > 0.0
    assert np.isfinite(direction).all()
