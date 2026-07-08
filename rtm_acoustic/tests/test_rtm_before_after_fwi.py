from __future__ import annotations

from pathlib import Path

import numpy as np

from rtm_acoustic.run_rtm_before_after_fwi import run_before_after_rtm


def _write_binary_model(path: Path, model: np.ndarray) -> None:
    model.astype(np.float32).T.tofile(path)


def test_run_before_after_rtm_writes_summary_and_case_outputs(tmp_path: Path) -> None:
    nz, nx = 24, 28
    true_model = np.full((nz, nx), 2000.0, dtype=np.float32)
    true_model[12:, :] = 2600.0
    true_model[15:20, 10:18] = 3200.0
    initial_model = np.full_like(true_model, 2100.0)
    initial_model[13:, :] = 2450.0
    inverted_model = initial_model.copy()
    inverted_model[12:, :] = 2550.0
    inverted_model[15:20, 10:18] = 2850.0

    true_path = tmp_path / "true.bin"
    initial_path = tmp_path / "initial.npy"
    inverted_path = tmp_path / "inverted.npy"
    _write_binary_model(true_path, true_model)
    np.save(initial_path, initial_model)
    np.save(inverted_path, inverted_model)

    metrics = run_before_after_rtm(
        true_model_path=true_path,
        initial_model_path=initial_path,
        inverted_model_path=inverted_path,
        output_dir=tmp_path / "rtm_compare",
        nx=nx,
        nz=nz,
        nt=35,
        f0=18.0,
        source_z=4,
        receiver_z=4,
        absorb_cells=4,
        fd_order=4,
        shot_spacing=160.0,
        shot_margin_cells=5,
        max_shots=1,
        direct_mute=False,
    )

    output_dir = tmp_path / "rtm_compare"
    assert metrics["shot_count"] == 1
    assert metrics["verdict"] in {"after_fwi_closer_to_reference", "after_fwi_not_closer_to_reference"}
    assert (output_dir / "rtm_before_after_summary.json").exists()
    assert (output_dir / "rtm_before_after_summary.md").exists()
    for case in ("reference_true_velocity", "before_initial_velocity", "after_fwi_velocity"):
        assert (output_dir / case / "rtm_laplacian_filtered_physical.npy").exists()
        assert (output_dir / case / "rtm_display.png").exists()
