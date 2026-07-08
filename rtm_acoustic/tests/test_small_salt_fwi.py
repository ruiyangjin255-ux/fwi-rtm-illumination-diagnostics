import numpy as np
import pytest

from rtm_acoustic.acoustic_rtm import RTMConfig
from rtm_acoustic.run_small_salt_fwi import (
    FWIConfig,
    apply_illumination_preconditioner,
    configure_chinese_matplotlib,
    build_initial_model,
    choose_default_crop,
    clip_velocity_update,
    compute_source_illumination,
    compute_update_direction,
    compute_record_misfit,
    parse_epsilon_values,
    parse_float_values,
    run_adaptive_line_search_compare,
    run_fwi_adaptive_line_search_demo,
    run_illumination_2d_scan,
    run_illumination_scan,
    run_fwi_line_search_demo,
    run_fwi_compare,
    run_fwi_demo,
    run_line_search_compare,
)


def test_choose_default_crop_returns_requested_window():
    model = np.zeros((230, 676), dtype=np.float32)

    z0, x0, nz, nx = choose_default_crop(model, crop_nz=70, crop_nx=120)

    assert 0 <= z0 <= model.shape[0] - nz
    assert 0 <= x0 <= model.shape[1] - nx
    assert (nz, nx) == (70, 120)


def test_build_initial_model_smooths_true_model():
    true_model = np.full((40, 50), 2000.0, dtype=np.float32)
    true_model[20:, 20:35] = 4200.0

    initial = build_initial_model(true_model, radius_z=3, radius_x=4, passes=2)

    assert initial.shape == true_model.shape
    assert np.isfinite(initial).all()
    assert initial.max() < true_model.max()
    assert initial.min() >= true_model.min()


def test_compute_record_misfit_uses_half_l2_norm():
    residual = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
    predicted = residual.copy()
    observed = np.zeros_like(predicted)

    misfit = compute_record_misfit(predicted, observed)

    assert np.isclose(misfit, 0.5 * np.mean(residual * residual))


def test_clip_velocity_update_limits_step_amplitude():
    update = np.array([[-100.0, 0.0, 100.0]], dtype=np.float32)

    clipped = clip_velocity_update(update, max_update=20.0)

    assert clipped.min() >= -20.0
    assert clipped.max() <= 20.0
    assert clipped.dtype == np.float32


def test_parse_epsilon_values_returns_positive_floats():
    assert parse_epsilon_values("0.01,0.05,0.2") == [0.01, 0.05, 0.2]


def test_parse_epsilon_values_rejects_nonpositive_values():
    with pytest.raises(ValueError, match="epsilon"):
        parse_epsilon_values("0.01,0")


def test_parse_float_values_returns_positive_floats():
    assert parse_float_values("20,35,80", name="max_update") == [20.0, 35.0, 80.0]


def test_parse_float_values_rejects_nonpositive_values():
    with pytest.raises(ValueError, match="max_update"):
        parse_float_values("20,-1", name="max_update")


def test_parse_float_values_supports_step_scales():
    assert parse_float_values("0.25,0.5,1.0", name="step_scale") == [0.25, 0.5, 1.0]


def test_configure_chinese_matplotlib_sets_font_family():
    selected = configure_chinese_matplotlib()

    assert selected in {"Microsoft YaHei", "SimHei", "SimSun"}


def test_compute_update_direction_returns_finite_model_shaped_array(tmp_path):
    nz, nx = 26, 32
    velocity = np.full((nz, nx), 2000.0, dtype=np.float32)
    cfg = RTMConfig(
        nx=nx,
        nz=nz,
        dx=10.0,
        dz=10.0,
        dt=0.001,
        nt=35,
        f0=12.0,
        source_x=nx // 2,
        source_z=4,
        receiver_z=4,
        absorb_cells=6,
        fd_order=4,
    )
    residual = np.zeros((cfg.nt, cfg.nx), dtype=np.float32)
    residual[10:20, 8:24] = 1.0

    from rtm_acoustic.acoustic_rtm import forward_model

    source_path = tmp_path / "source.dat"
    forward_model(velocity, cfg, wavefield_path=source_path)
    update = compute_update_direction(
        velocity=velocity,
        config=cfg,
        residual=residual,
        source_wavefield_path=source_path,
    )

    assert update.shape == velocity.shape
    assert np.isfinite(update).all()


def test_compute_source_illumination_is_nonnegative(tmp_path):
    nz, nx = 20, 24
    velocity = np.full((nz, nx), 2000.0, dtype=np.float32)
    cfg = RTMConfig(
        nx=nx,
        nz=nz,
        dx=10.0,
        dz=10.0,
        dt=0.001,
        nt=30,
        f0=12.0,
        source_x=12,
        source_z=4,
        receiver_z=4,
        absorb_cells=5,
        fd_order=4,
    )
    source_path = tmp_path / "source.dat"

    from rtm_acoustic.acoustic_rtm import forward_model

    forward_model(velocity, cfg, wavefield_path=source_path)
    illum = compute_source_illumination(source_path, cfg)

    assert illum.shape == velocity.shape
    assert np.isfinite(illum).all()
    assert illum.min() >= 0.0
    assert illum.max() > 0.0


def test_apply_illumination_preconditioner_preserves_shape_and_finiteness():
    update = np.ones((4, 5), dtype=np.float32)
    illumination = np.zeros((4, 5), dtype=np.float32)
    illumination[:, 2:] = 10.0

    conditioned = apply_illumination_preconditioner(update, illumination, epsilon=0.1)

    assert conditioned.shape == update.shape
    assert np.isfinite(conditioned).all()
    assert conditioned.dtype == np.float32


def test_run_fwi_demo_writes_summary_and_reduces_misfit(tmp_path):
    true_model = np.full((34, 42), 2000.0, dtype=np.float32)
    true_model[18:, 14:30] = 2600.0
    cfg = FWIConfig(crop_nx=42, crop_nz=34, nt=45, iterations=2, absorb_cells=6, max_update=20.0)

    summary = run_fwi_demo(
        true_model=true_model,
        config=cfg,
        output_dir=tmp_path,
        shot_positions=[14, 28],
        write_figures=False,
    )

    assert (tmp_path / "summary.json").exists()
    assert len(summary["misfit_history"]) == cfg.iterations + 1
    assert summary["final_misfit"] <= summary["initial_misfit"]


def test_run_fwi_compare_writes_compare_summary(tmp_path):
    true_model = np.full((30, 36), 2000.0, dtype=np.float32)
    true_model[16:, 12:26] = 2600.0
    cfg = FWIConfig(crop_nx=36, crop_nz=30, nt=40, iterations=1, absorb_cells=6, max_update=15.0)

    summary = run_fwi_compare(
        true_model=true_model,
        config=cfg,
        output_dir=tmp_path,
        shot_positions=[12, 24],
        write_figures=False,
    )

    assert (tmp_path / "summary_compare.json").exists()
    assert "baseline" in summary
    assert "illumination_preconditioned" in summary
    assert (tmp_path / "baseline_inverted_model.npy").exists()
    assert (tmp_path / "preconditioned_inverted_model.npy").exists()


def test_run_illumination_scan_writes_scan_summary(tmp_path):
    true_model = np.full((28, 34), 2000.0, dtype=np.float32)
    true_model[15:, 10:24] = 2600.0
    cfg = FWIConfig(crop_nx=34, crop_nz=28, nt=38, iterations=1, absorb_cells=6, max_update=15.0)

    summary = run_illumination_scan(
        true_model=true_model,
        config=cfg,
        output_dir=tmp_path,
        epsilons=[0.02, 0.1],
        shot_positions=[11, 23],
        write_figures=False,
    )

    assert (tmp_path / "summary_scan.json").exists()
    assert (tmp_path / "scan_results.csv").exists()
    assert "baseline" in summary
    assert len(summary["preconditioned_runs"]) == 2
    assert "best_preconditioned" in summary


def test_run_illumination_2d_scan_writes_summary(tmp_path):
    true_model = np.full((26, 32), 2000.0, dtype=np.float32)
    true_model[14:, 10:22] = 2600.0
    cfg = FWIConfig(crop_nx=32, crop_nz=26, nt=36, iterations=1, absorb_cells=6, max_update=15.0)

    summary = run_illumination_2d_scan(
        true_model=true_model,
        config=cfg,
        output_dir=tmp_path,
        epsilons=[0.05, 0.2],
        max_updates=[10.0, 20.0],
        shot_positions=[10, 21],
        write_figures=False,
    )

    assert (tmp_path / "summary_2d_scan.json").exists()
    assert (tmp_path / "scan_2d_results.csv").exists()
    assert "baseline" in summary
    assert len(summary["preconditioned_runs"]) == 4
    assert "best_preconditioned" in summary


def test_run_fwi_line_search_demo_writes_selected_steps(tmp_path):
    true_model = np.full((26, 32), 2000.0, dtype=np.float32)
    true_model[14:, 10:22] = 2600.0
    cfg = FWIConfig(crop_nx=32, crop_nz=26, nt=36, iterations=1, absorb_cells=6, max_update=15.0)

    summary = run_fwi_line_search_demo(
        true_model=true_model,
        config=cfg,
        output_dir=tmp_path,
        step_scales=[0.5, 1.0],
        shot_positions=[10, 21],
        write_figures=False,
    )

    assert (tmp_path / "line_search_summary.json").exists()
    assert (tmp_path / "line_search_results.csv").exists()
    assert len(summary["selected_step_scales"]) == cfg.iterations
    assert summary["final_misfit"] <= summary["initial_misfit"]


def test_run_line_search_compare_writes_summary(tmp_path):
    true_model = np.full((26, 32), 2000.0, dtype=np.float32)
    true_model[14:, 10:22] = 2600.0
    cfg = FWIConfig(crop_nx=32, crop_nz=26, nt=36, iterations=1, absorb_cells=6, max_update=15.0)

    summary = run_line_search_compare(
        true_model=true_model,
        config=cfg,
        output_dir=tmp_path,
        step_scales=[0.5, 1.0],
        shot_positions=[10, 21],
        write_figures=False,
    )

    assert (tmp_path / "line_search_summary.json").exists()
    assert "baseline" in summary
    assert "illumination_preconditioned" in summary


def test_run_fwi_adaptive_line_search_demo_writes_tested_steps(tmp_path):
    true_model = np.full((26, 32), 2000.0, dtype=np.float32)
    true_model[14:, 10:22] = 2600.0
    cfg = FWIConfig(crop_nx=32, crop_nz=26, nt=36, iterations=1, absorb_cells=6, max_update=15.0)

    summary = run_fwi_adaptive_line_search_demo(
        true_model=true_model,
        config=cfg,
        output_dir=tmp_path,
        initial_step_scales=[0.5, 1.0],
        expanded_step_scales=[1.5, 2.0],
        shot_positions=[10, 21],
        write_figures=False,
    )

    assert (tmp_path / "adaptive_line_search_summary.json").exists()
    assert (tmp_path / "adaptive_line_search_results.csv").exists()
    assert len(summary["selected_step_scales"]) == cfg.iterations
    assert len(summary["tested_step_scales_by_iteration"]) == cfg.iterations
    assert summary["final_misfit"] <= summary["initial_misfit"]


def test_run_adaptive_line_search_compare_writes_summary(tmp_path):
    true_model = np.full((26, 32), 2000.0, dtype=np.float32)
    true_model[14:, 10:22] = 2600.0
    cfg = FWIConfig(crop_nx=32, crop_nz=26, nt=36, iterations=1, absorb_cells=6, max_update=15.0)

    summary = run_adaptive_line_search_compare(
        true_model=true_model,
        config=cfg,
        output_dir=tmp_path,
        initial_step_scales=[0.5, 1.0],
        expanded_step_scales=[1.5, 2.0],
        shot_positions=[10, 21],
        write_figures=False,
    )

    assert (tmp_path / "adaptive_line_search_summary.json").exists()
    assert "baseline" in summary
    assert "illumination_preconditioned" in summary
