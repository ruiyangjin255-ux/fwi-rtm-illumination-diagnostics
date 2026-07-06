from __future__ import annotations

from pathlib import Path

import numpy as np

from rtm_acoustic.run_optimized_fwi_rtm_pipeline import run_pipeline


def test_run_pipeline_writes_quality_scale_and_report_without_rtm(tmp_path: Path) -> None:
    fwi_dir = tmp_path / "fwi"
    fwi_dir.mkdir()
    true_model = np.full((6, 8), 2000.0, dtype=np.float32)
    true_model[:, 4:] = 2600.0
    initial_model = np.full_like(true_model, 2000.0)
    inverted_model = initial_model.copy()
    inverted_model[:, 4:] = 2400.0
    np.save(fwi_dir / "full_salt_true_model.npy", true_model)
    np.save(fwi_dir / "full_salt_initial_model.npy", initial_model)
    np.save(fwi_dir / "full_salt_inverted_model.npy", inverted_model)

    report = run_pipeline(
        fwi_dir=fwi_dir,
        true_model=tmp_path / "true.bin",
        output_dir=tmp_path / "pipeline",
        rtm_dir=tmp_path / "missing_rtm",
        run_rtm=False,
        alphas=[0.0, 0.25, 0.5, 1.0],
        gradient_tolerance=1.0,
        nt=10,
        max_shots=1,
        pad_x=0,
        pad_bottom=0,
    )

    assert report["rtm_validation"]["available"] is False
    assert Path(report["selected_model"]).exists()
    assert Path(report["pipeline_report_paths"]["json"]).exists()
    assert Path(report["pipeline_report_paths"]["markdown"]).exists()
    assert Path(report["full_update_quality_paths"]["json"]).exists()
    assert Path(report["update_scale_paths"]["json"]).exists()
