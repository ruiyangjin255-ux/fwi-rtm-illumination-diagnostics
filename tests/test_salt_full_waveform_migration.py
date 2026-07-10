from __future__ import annotations

import numpy as np
import pytest

from salt_rtm.run_salt_full_waveform_migration import (
    FullWaveformMigrationConfig,
    choose_default_crop,
    default_shot_positions,
    parse_float_values,
    run_direct_mute_scan_demo,
    run_full_waveform_migration_demo,
)


def test_default_crop_and_shots_are_inside_model() -> None:
    model = np.ones((80, 120), dtype=np.float32)
    cfg = FullWaveformMigrationConfig(crop_nz=40, crop_nx=70)

    z0, x0, crop_nz, crop_nx = choose_default_crop(model, cfg.crop_nz, cfg.crop_nx)
    shots = default_shot_positions(crop_nx)

    assert 0 <= z0 <= model.shape[0] - crop_nz
    assert 0 <= x0 <= model.shape[1] - crop_nx
    assert shots == [8, 35, 61]


def test_small_demo_writes_expected_outputs(tmp_path) -> None:
    true_model = np.full((30, 48), 2000.0, dtype=np.float32)
    true_model[14:, 20:30] = 2600.0
    cfg = FullWaveformMigrationConfig(crop_nz=30, crop_nx=48, nt=80, f0=12.0, absorb_cells=8, fd_order=4)

    summary = run_full_waveform_migration_demo(
        true_model=true_model,
        output_dir=tmp_path,
        config=cfg,
        shot_positions=[10, 24, 37],
    )

    assert summary["shot_count"] == 3
    assert summary["full_waveform"]["image_abs_p99"] > 0.0
    assert summary["reflection_only"]["image_abs_p99"] >= 0.0
    assert np.isfinite(np.load(tmp_path / "full_waveform_image.npy")).all()
    assert np.isfinite(np.load(tmp_path / "reflection_only_image.npy")).all()
    assert (tmp_path / "summary.json").exists()


def test_small_demo_writes_chinese_report(tmp_path) -> None:
    true_model = np.full((24, 36), 2100.0, dtype=np.float32)
    true_model[12:, 15:23] = 2800.0
    cfg = FullWaveformMigrationConfig(crop_nz=24, crop_nx=36, nt=70, absorb_cells=8, fd_order=4)

    run_full_waveform_migration_demo(
        true_model=true_model,
        output_dir=tmp_path,
        config=cfg,
        shot_positions=[8, 18, 27],
    )

    report = (tmp_path / "full_waveform_migration_summary.md").read_text(encoding="utf-8")
    assert "全波形偏移" in report
    assert "反射波偏移" in report
    assert (tmp_path / "migration_compare.png").exists()
    assert (tmp_path / "stacked_record_compare.png").exists()


def test_parse_float_values_allows_zero_only_when_requested() -> None:
    assert parse_float_values("0,0.01,0.02", name="padding_time", allow_zero=True) == [0.0, 0.01, 0.02]
    with pytest.raises(ValueError, match="taper_time"):
        parse_float_values("0,0.02", name="taper_time")


def test_mute_scan_writes_metrics_and_report(tmp_path) -> None:
    true_model = np.full((24, 36), 2100.0, dtype=np.float32)
    true_model[12:, 14:24] = 2800.0
    cfg = FullWaveformMigrationConfig(crop_nz=24, crop_nx=36, nt=70, absorb_cells=8, fd_order=4)

    summary = run_direct_mute_scan_demo(
        true_model=true_model,
        output_dir=tmp_path,
        config=cfg,
        shot_positions=[8, 18, 27],
        padding_times=[0.0, 0.01],
        taper_times=[0.01],
    )

    assert summary["scan_count"] == 2
    assert summary["best_case"]["case_name"].startswith("mute_pad")
    assert (tmp_path / "mute_scan_metrics.csv").exists()
    assert (tmp_path / "mute_scan_summary.json").exists()
    assert (tmp_path / "mute_scan_compare.png").exists()
    report = (tmp_path / "mute_scan_best_summary.md").read_text(encoding="utf-8")
    assert "直达波静音" in report
    assert "照明归一化" in report
    assert "Laplacian" in report

