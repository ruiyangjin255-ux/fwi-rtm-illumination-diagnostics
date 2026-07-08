from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest

from rtm_acoustic.acoustic_rtm import (
    RTMConfig,
    finite_difference_second_coefficients,
    forward_model,
    high_order_laplacian_filter,
    multishot_reverse_time_migrate,
    multishot_reverse_time_migrate_parallel,
    mute_direct_arrivals,
    make_reflection_record,
    crop_padded_model,
    crop_padded_record,
    pad_rtm_config,
    pad_velocity_model,
    preprocess_migration_section,
    preprocess_stacked_record,
    read_binary_model,
    reverse_time_boundary_migrate,
    reverse_time_migrate,
    shot_positions_from_spacing,
    smooth_velocity_model,
    source_receiver_normalized_image,
    source_normalized_image,
    stack_surface_records,
    synthesize_normal_incidence_stack,
)
from rtm_acoustic.plot_paper_style import (
    _active_vertical_extent,
    _suppress_weak_amplitudes,
    save_record_and_migration_figure,
)


def test_binary_model_reader_uses_c_x_major_layout(tmp_path: Path):
    path = tmp_path / "model.bin"
    x_major = np.arange(12, dtype=np.float32).reshape(4, 3)
    x_major.tofile(path)

    model = read_binary_model(path, nx=4, nz=3)

    assert model.shape == (3, 4)
    np.testing.assert_array_equal(model, x_major.T)


def test_high_order_second_derivative_coefficients_reproduce_quadratic():
    coeff = finite_difference_second_coefficients(order=4)
    offsets = np.arange(-2, 3, dtype=np.float64)
    values = offsets**2

    second = coeff[0] * values[2] + coeff[1] * (values[3] + values[1]) + coeff[2] * (
        values[4] + values[0]
    )

    assert np.isclose(second, 2.0)


def test_source_normalized_image_limits_source_illumination_bias():
    raw = np.array([[2.0, 8.0], [18.0, 32.0]], dtype=np.float64)
    illum = np.array([[1.0, 4.0], [9.0, 16.0]], dtype=np.float64)

    normalized = source_normalized_image(raw, illum, eps=0.0)

    np.testing.assert_allclose(normalized, [[2.0, 2.0], [2.0, 2.0]])


def test_source_normalized_image_masks_low_illumination():
    raw = np.array([[10.0, 10.0], [10.0, 10.0]], dtype=np.float32)
    illum = np.array([[100.0, 1.0], [0.0, 50.0]], dtype=np.float32)

    normalized = source_normalized_image(raw, illum, eps=0.0, min_illumination_fraction=0.1)

    assert normalized[0, 0] > 0.0
    assert normalized[1, 1] > 0.0
    assert normalized[0, 1] == 0.0
    assert normalized[1, 0] == 0.0


def test_source_receiver_normalized_image_uses_geometric_illumination():
    raw = np.array([[6.0, 12.0], [18.0, 24.0]], dtype=np.float64)
    source_illum = np.array([[4.0, 4.0], [9.0, 9.0]], dtype=np.float64)
    receiver_illum = np.array([[9.0, 36.0], [4.0, 16.0]], dtype=np.float64)

    normalized = source_receiver_normalized_image(raw, source_illum, receiver_illum, eps=0.0)

    np.testing.assert_allclose(normalized, [[1.0, 1.0], [3.0, 2.0]])


def test_laplacian_filter_preserves_shape_and_removes_constant_component():
    image = np.ones((9, 11), dtype=np.float64)
    image[4, 5] = 10.0

    filtered = high_order_laplacian_filter(image, dx=10.0, dz=10.0, power=2)

    assert filtered.shape == image.shape
    assert np.isfinite(filtered).all()
    assert abs(filtered.mean()) < abs(image.mean())


def test_smooth_velocity_model_reduces_sharp_contrasts():
    velocity = np.full((28, 32), 2000.0, dtype=np.float32)
    velocity[10:18, 12:20] = 4500.0

    smoothed = smooth_velocity_model(velocity, radius_z=3, radius_x=4, passes=2)

    assert smoothed.shape == velocity.shape
    assert np.isfinite(smoothed).all()
    assert smoothed.max() < velocity.max()
    assert smoothed.min() >= velocity.min()
    assert np.std(smoothed) < np.std(velocity)


def test_pad_velocity_model_extends_edges_and_crops_back():
    velocity = np.array(
        [
            [1500.0, 1600.0, 1700.0],
            [1800.0, 1900.0, 2000.0],
        ],
        dtype=np.float32,
    )

    padded = pad_velocity_model(velocity, pad_x=2, pad_top=1, pad_bottom=2)

    assert padded.shape == (5, 7)
    np.testing.assert_array_equal(padded[1:3, 2:5], velocity)
    np.testing.assert_array_equal(padded[:, 0], padded[:, 2])
    np.testing.assert_array_equal(padded[:, -1], padded[:, -3])
    np.testing.assert_array_equal(padded[0, :], padded[1, :])
    np.testing.assert_array_equal(padded[-1, :], padded[-3, :])
    np.testing.assert_array_equal(
        crop_padded_model(padded, original_shape=velocity.shape, pad_x=2, pad_top=1),
        velocity,
    )


def test_pad_rtm_config_and_record_crop_shift_to_physical_window():
    cfg = RTMConfig(
        nx=12,
        nz=8,
        dx=10.0,
        dz=10.0,
        dt=0.001,
        nt=5,
        f0=20.0,
        source_x=4,
        source_z=3,
        receiver_z=3,
        absorb_cells=2,
        fd_order=4,
    )
    padded_cfg = pad_rtm_config(cfg, pad_x=3, pad_top=1, pad_bottom=4)
    record = np.arange(cfg.nt * padded_cfg.nx, dtype=np.float32).reshape(cfg.nt, padded_cfg.nx)

    assert padded_cfg.nx == 18
    assert padded_cfg.nz == 13
    assert padded_cfg.source_x == 7
    assert padded_cfg.source_z == 4
    assert padded_cfg.receiver_z == 4
    np.testing.assert_array_equal(crop_padded_record(record, original_nx=cfg.nx, pad_x=3), record[:, 3:15])


def test_make_reflection_record_subtracts_smooth_model_direct_wave(tmp_path: Path):
    nz, nx = 28, 34
    velocity = np.full((nz, nx), 2000.0, dtype=np.float32)
    smooth_velocity = velocity.copy()
    cfg = RTMConfig(
        nx=nx,
        nz=nz,
        dx=10.0,
        dz=10.0,
        dt=0.001,
        nt=45,
        f0=18.0,
        source_x=nx // 2,
        source_z=4,
        receiver_z=4,
        absorb_cells=6,
        fd_order=4,
    )

    full_record, reflection_record = make_reflection_record(
        velocity,
        smooth_velocity,
        cfg,
        full_wavefield_path=tmp_path / "full.dat",
        direct_wavefield_path=tmp_path / "direct.dat",
    )

    assert full_record.shape == (cfg.nt, cfg.nx)
    assert reflection_record.shape == (cfg.nt, cfg.nx)
    assert np.max(np.abs(reflection_record)) < 1.0e-5


def test_small_rtm_produces_finite_nonzero_image(tmp_path: Path):
    nz, nx = 36, 48
    velocity = np.full((nz, nx), 1800.0, dtype=np.float32)
    velocity[20:, :] = 2400.0
    cfg = RTMConfig(
        nx=nx,
        nz=nz,
        dx=10.0,
        dz=10.0,
        dt=0.001,
        nt=90,
        f0=20.0,
        source_x=nx // 2,
        source_z=4,
        receiver_z=4,
        absorb_cells=8,
        fd_order=4,
    )
    wavefield_path = tmp_path / "source_wavefield.dat"

    record = forward_model(velocity, cfg, wavefield_path=wavefield_path)
    result = reverse_time_migrate(velocity, record, cfg, source_wavefield_path=wavefield_path)

    assert record.shape == (cfg.nt, cfg.nx)
    assert result.image.shape == (nz, nx)
    assert result.normalized_image.shape == (nz, nx)
    assert result.receiver_illumination.shape == (nz, nx)
    assert result.source_receiver_normalized_image.shape == (nz, nx)
    assert result.laplacian_image.shape == (nz, nx)
    assert result.laplacian_normalized_image.shape == (nz, nx)
    assert np.isfinite(result.normalized_image).all()
    assert np.isfinite(result.receiver_illumination).all()
    assert np.isfinite(result.source_receiver_normalized_image).all()
    assert np.isfinite(result.laplacian_image).all()
    assert np.isfinite(result.laplacian_normalized_image).all()
    assert np.max(np.abs(result.normalized_image)) > 0.0
    assert np.max(result.receiver_illumination) > 0.0


def test_multishot_rtm_stacks_cross_correlation_images(tmp_path: Path):
    nz, nx = 34, 44
    velocity = np.full((nz, nx), 1900.0, dtype=np.float32)
    velocity[18:, :] = 2500.0
    cfg = RTMConfig(
        nx=nx,
        nz=nz,
        dx=10.0,
        dz=10.0,
        dt=0.001,
        nt=70,
        f0=18.0,
        source_x=nx // 2,
        source_z=3,
        receiver_z=3,
        absorb_cells=8,
        fd_order=4,
    )

    result = multishot_reverse_time_migrate(
        velocity,
        cfg,
        shot_positions=[12, 22, 32],
        wavefield_path=tmp_path / "source_wavefield.dat",
        laplacian_power=1,
    )

    assert result.shot_count == 3
    assert result.stacked_record.shape == (cfg.nt, cfg.nx)
    assert result.image.shape == (nz, nx)
    assert result.illumination.shape == (nz, nx)
    assert result.receiver_illumination.shape == (nz, nx)
    assert result.normalized_image.shape == (nz, nx)
    assert result.source_receiver_normalized_image.shape == (nz, nx)
    assert result.laplacian_image.shape == (nz, nx)
    assert result.laplacian_normalized_image.shape == (nz, nx)
    assert result.filtered_image.shape == (nz, nx)
    assert np.isfinite(result.filtered_image).all()
    assert np.isfinite(result.source_receiver_normalized_image).all()
    assert np.isfinite(result.laplacian_normalized_image).all()
    assert np.max(np.abs(result.filtered_image)) > 0.0
    assert np.count_nonzero(result.stacked_record) > 0


def test_multishot_rtm_can_use_smooth_migration_velocity(tmp_path: Path):
    nz, nx = 34, 44
    full_velocity = np.full((nz, nx), 1900.0, dtype=np.float32)
    full_velocity[18:, :] = 2500.0
    migration_velocity = smooth_velocity_model(full_velocity, radius_z=2, radius_x=2, passes=1)
    cfg = RTMConfig(
        nx=nx,
        nz=nz,
        dx=10.0,
        dz=10.0,
        dt=0.001,
        nt=160,
        f0=18.0,
        source_x=nx // 2,
        source_z=4,
        receiver_z=4,
        absorb_cells=8,
        fd_order=4,
    )

    result = multishot_reverse_time_migrate(
        full_velocity,
        cfg,
        shot_positions=[14, 28],
        wavefield_path=tmp_path / "source_wavefield.dat",
        laplacian_power=1,
        migration_velocity=migration_velocity,
        subtract_direct_wave=True,
        direct_wavefield_path=tmp_path / "direct_wavefield.dat",
    )

    assert result.shot_count == 2
    assert result.stacked_record.shape == (cfg.nt, cfg.nx)
    assert np.isfinite(result.filtered_image).all()
    assert np.count_nonzero(result.stacked_record) > 0
    assert np.max(np.abs(result.filtered_image)) > 0.0


def test_parallel_multishot_rtm_matches_serial_result(tmp_path: Path):
    nz, nx = 26, 32
    full_velocity = np.full((nz, nx), 1900.0, dtype=np.float32)
    full_velocity[14:, :] = 2450.0
    migration_velocity = smooth_velocity_model(full_velocity, radius_z=1, radius_x=1, passes=1)
    cfg = RTMConfig(
        nx=nx,
        nz=nz,
        dx=10.0,
        dz=10.0,
        dt=0.001,
        nt=45,
        f0=18.0,
        source_x=nx // 2,
        source_z=4,
        receiver_z=4,
        absorb_cells=6,
        fd_order=4,
    )
    shots = [10, 20]

    serial = multishot_reverse_time_migrate(
        full_velocity,
        cfg,
        shot_positions=shots,
        wavefield_path=tmp_path / "serial_source.dat",
        laplacian_power=1,
        migration_velocity=migration_velocity,
        subtract_direct_wave=True,
        min_illumination_fraction=0.01,
    )
    parallel = multishot_reverse_time_migrate_parallel(
        full_velocity,
        cfg,
        shot_positions=shots,
        work_dir=tmp_path / "parallel",
        workers=2,
        laplacian_power=1,
        migration_velocity=migration_velocity,
        subtract_direct_wave=True,
        min_illumination_fraction=0.01,
    )

    assert parallel.shot_count == serial.shot_count
    np.testing.assert_allclose(parallel.stacked_record, serial.stacked_record, rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(parallel.image, serial.image, rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(parallel.illumination, serial.illumination, rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(parallel.receiver_illumination, serial.receiver_illumination, rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(parallel.normalized_image, serial.normalized_image, rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(
        parallel.source_receiver_normalized_image,
        serial.source_receiver_normalized_image,
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    np.testing.assert_allclose(parallel.laplacian_image, serial.laplacian_image, rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(
        parallel.laplacian_normalized_image,
        serial.laplacian_normalized_image,
        rtol=1.0e-5,
        atol=1.0e-5,
    )


def test_parallel_multishot_rtm_resumes_from_checkpoint(tmp_path: Path):
    nz, nx = 26, 32
    full_velocity = np.full((nz, nx), 1900.0, dtype=np.float32)
    full_velocity[14:, :] = 2450.0
    migration_velocity = smooth_velocity_model(full_velocity, radius_z=1, radius_x=1, passes=1)
    cfg = RTMConfig(
        nx=nx,
        nz=nz,
        dx=10.0,
        dz=10.0,
        dt=0.001,
        nt=45,
        f0=18.0,
        source_x=nx // 2,
        source_z=4,
        receiver_z=4,
        absorb_cells=6,
        fd_order=4,
    )
    shots = [10, 16, 22, 28]
    checkpoint_dir = tmp_path / "checkpoints"

    expected = multishot_reverse_time_migrate_parallel(
        full_velocity,
        cfg,
        shot_positions=shots,
        work_dir=tmp_path / "expected",
        workers=2,
        laplacian_power=1,
        migration_velocity=migration_velocity,
        subtract_direct_wave=True,
        min_illumination_fraction=0.01,
    )

    def stop_after_two(completed: int, total: int, source_x: int) -> None:
        if completed == 2:
            raise RuntimeError("stop after checkpoint")

    with pytest.raises(RuntimeError, match="stop after checkpoint"):
        multishot_reverse_time_migrate_parallel(
            full_velocity,
            cfg,
            shot_positions=shots,
            work_dir=tmp_path / "interrupted",
            workers=2,
            laplacian_power=1,
            migration_velocity=migration_velocity,
            subtract_direct_wave=True,
            min_illumination_fraction=0.01,
            checkpoint_dir=checkpoint_dir,
            resume=False,
            checkpoint_interval=1,
            progress_callback=stop_after_two,
        )

    resumed = multishot_reverse_time_migrate_parallel(
        full_velocity,
        cfg,
        shot_positions=shots,
        work_dir=tmp_path / "resumed",
        workers=2,
        laplacian_power=1,
        migration_velocity=migration_velocity,
        subtract_direct_wave=True,
        min_illumination_fraction=0.01,
        checkpoint_dir=checkpoint_dir,
        resume=True,
        checkpoint_interval=1,
    )

    assert resumed.shot_count == expected.shot_count
    np.testing.assert_allclose(resumed.image, expected.image, rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(resumed.illumination, expected.illumination, rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(
        resumed.receiver_illumination,
        expected.receiver_illumination,
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    np.testing.assert_allclose(resumed.stacked_record, expected.stacked_record, rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(
        resumed.source_receiver_normalized_image,
        expected.source_receiver_normalized_image,
        rtol=1.0e-5,
        atol=1.0e-5,
    )


def test_parallel_multishot_rtm_rejects_mismatched_checkpoint(tmp_path: Path):
    nz, nx = 24, 30
    velocity = np.full((nz, nx), 1900.0, dtype=np.float32)
    velocity[13:, :] = 2450.0
    migration_velocity = smooth_velocity_model(velocity, radius_z=1, radius_x=1, passes=1)
    cfg = RTMConfig(
        nx=nx,
        nz=nz,
        dx=10.0,
        dz=10.0,
        dt=0.001,
        nt=42,
        f0=18.0,
        source_x=nx // 2,
        source_z=4,
        receiver_z=4,
        absorb_cells=6,
        fd_order=4,
    )
    shots = [10, 20]
    checkpoint_dir = tmp_path / "checkpoints"

    multishot_reverse_time_migrate_parallel(
        velocity,
        cfg,
        shot_positions=shots,
        work_dir=tmp_path / "initial",
        workers=2,
        laplacian_power=1,
        migration_velocity=migration_velocity,
        subtract_direct_wave=True,
        min_illumination_fraction=0.01,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=1,
    )

    mismatched_cfg = replace(cfg, nt=cfg.nt + 1)
    with pytest.raises(ValueError, match="checkpoint does not match current RTM run"):
        multishot_reverse_time_migrate_parallel(
            velocity,
            mismatched_cfg,
            shot_positions=shots,
            work_dir=tmp_path / "resume",
            workers=2,
            laplacian_power=1,
            migration_velocity=np.pad(migration_velocity, ((0, 0), (0, 0))),
            subtract_direct_wave=True,
            min_illumination_fraction=0.01,
            checkpoint_dir=checkpoint_dir,
            resume=True,
            checkpoint_interval=1,
        )


def test_boundary_reverse_migration_repositions_surface_record(tmp_path: Path):
    nz, nx = 34, 44
    velocity = np.full((nz, nx), 2000.0, dtype=np.float32)
    velocity[18:, :] = 2600.0
    cfg = RTMConfig(
        nx=nx,
        nz=nz,
        dx=10.0,
        dz=10.0,
        dt=0.001,
        nt=80,
        f0=18.0,
        source_x=nx // 2,
        source_z=3,
        receiver_z=3,
        absorb_cells=8,
        fd_order=4,
    )

    record = forward_model(velocity, cfg, wavefield_path=tmp_path / "source.dat")
    image, filtered = reverse_time_boundary_migrate(velocity, record, cfg, laplacian_power=1)

    assert image.shape == (nz, nx)
    assert filtered.shape == (nz, nx)
    assert np.isfinite(image).all()
    assert np.isfinite(filtered).all()
    assert np.max(np.abs(image)) > 0.0


def test_stack_surface_records_averages_multiple_shots():
    nz, nx = 30, 36
    velocity = np.full((nz, nx), 1900.0, dtype=np.float32)
    velocity[16:, :] = 2400.0
    cfg = RTMConfig(
        nx=nx,
        nz=nz,
        dx=10.0,
        dz=10.0,
        dt=0.001,
        nt=55,
        f0=20.0,
        source_x=nx // 2,
        source_z=3,
        receiver_z=3,
        absorb_cells=6,
        fd_order=4,
    )
    shots = shot_positions_from_spacing(nx=nx, dx=10.0, spacing_m=90.0, margin_cells=6)

    stacked, count = stack_surface_records(velocity, cfg, shots)

    assert count == len(shots)
    assert stacked.shape == (cfg.nt, cfg.nx)
    assert np.isfinite(stacked).all()
    assert np.max(np.abs(stacked)) > 0.0


def test_zero_offset_stack_returns_full_interpolated_section():
    nz, nx = 28, 34
    velocity = np.full((nz, nx), 1900.0, dtype=np.float32)
    velocity[14:, :] = 2400.0
    cfg = RTMConfig(
        nx=nx,
        nz=nz,
        dx=10.0,
        dz=10.0,
        dt=0.001,
        nt=45,
        f0=20.0,
        source_x=nx // 2,
        source_z=3,
        receiver_z=3,
        absorb_cells=6,
        fd_order=4,
    )
    shots = [6, 16, 26]

    stacked, count = stack_surface_records(velocity, cfg, shots, stack_mode="zero_offset")

    assert count == len(shots)
    assert stacked.shape == (cfg.nt, cfg.nx)
    assert np.isfinite(stacked).all()
    assert np.max(np.abs(stacked[:, shots])) > 0.0


def test_signed_rms_stack_preserves_energy_when_phases_cancel():
    traces = [
        np.array([[1.0, -1.0], [2.0, -2.0]], dtype=np.float32),
        np.array([[-1.0, 1.0], [-2.0, 2.0]], dtype=np.float32),
    ]

    mean = sum(traces) / 2.0
    rms = np.sqrt(sum(t * t for t in traces) / 2.0)

    assert np.allclose(mean, 0.0)
    assert np.count_nonzero(rms) == 4


def test_preprocess_stacked_record_mutes_and_trace_normalizes():
    record = np.zeros((100, 4), dtype=np.float32)
    record[5, :] = 100.0
    record[40:, :] = np.linspace(0.1, 1.0, 60)[:, None]

    processed = preprocess_stacked_record(record, dt=0.001, mute_time=0.02, time_power=1.0)

    assert processed.shape == record.shape
    assert np.allclose(processed[:20], 0.0)
    assert np.isfinite(processed).all()
    assert np.max(np.abs(processed)) <= 1.0


def test_mute_direct_arrivals_removes_early_first_breaks():
    cfg = RTMConfig(
        nx=9,
        nz=12,
        dx=10.0,
        dz=10.0,
        dt=0.001,
        nt=80,
        f0=20.0,
        source_x=4,
        source_z=4,
        receiver_z=4,
        absorb_cells=2,
        fd_order=4,
    )
    record = np.ones((cfg.nt, cfg.nx), dtype=np.float32)

    muted = mute_direct_arrivals(
        record,
        cfg,
        source_x=4,
        direct_velocity=2000.0,
        padding_time=0.004,
        taper_time=0.0,
    )

    assert np.allclose(muted[:4, 4], 0.0)
    assert muted[20, 4] == 1.0
    assert muted[10, 0] == 0.0


def test_preprocess_migration_section_balances_traces():
    migration = np.zeros((30, 5), dtype=np.float32)
    migration[8:12, :] = np.array([1, 2, 3, 4, 5], dtype=np.float32)[None, :]
    migration[18:20, :] = -2.0

    processed = preprocess_migration_section(migration)

    assert processed.shape == migration.shape
    assert np.isfinite(processed).all()
    assert np.max(np.abs(processed)) <= 1.0
    assert np.count_nonzero(processed) > 0


def test_preprocess_migration_section_uses_gentle_display_gain():
    migration = np.zeros((80, 8), dtype=np.float32)
    migration[24:28, :] = np.linspace(1.0, 8.0, 8, dtype=np.float32)[None, :]
    migration[50:53, :] = -0.45
    migration[:, 0] += 0.02

    processed = preprocess_migration_section(
        migration,
        depth_power=0.0,
        trace_balance=0.25,
        output_clip=0.80,
    )

    saturated = np.mean(np.abs(processed) >= 0.799)
    assert processed.shape == migration.shape
    assert np.isfinite(processed).all()
    assert np.max(np.abs(processed)) <= 0.800001
    assert saturated < 0.08


def test_paper_style_figure_writes_record_and_migration(tmp_path: Path):
    nt, nx, nz = 60, 42, 28
    t = np.linspace(0.0, 1.0, nt)[:, None]
    x = np.linspace(-1.0, 1.0, nx)[None, :]
    record = np.sin(40.0 * (t - 0.3 * x * x)).astype(np.float32) * np.exp(-2.0 * t)
    migration = np.zeros((nz, nx), dtype=np.float32)
    migration[10:13, 8:34] = np.hanning(26)[None, :]
    migration[18:20, 14:28] = -1.0

    out = tmp_path / "paper_style.png"
    save_record_and_migration_figure(out, record, migration, dx=10.0, dz=10.0, dt=0.001)

    assert out.exists()
    assert out.stat().st_size > 10_000


def test_active_vertical_extent_trims_blank_record_tail():
    section = np.zeros((100, 12), dtype=np.float32)
    section[20:46, :] = 1.0
    section[80:, :] = 1.0e-8

    start, stop = _active_vertical_extent(section, sample_interval=0.001)

    assert start == 0.0
    assert 0.046 < stop < 0.060


def test_suppress_weak_amplitudes_keeps_strong_wiggle_events():
    section = np.array(
        [
            [0.01, -0.02, 0.03],
            [0.20, -0.40, 0.60],
            [-0.80, 0.90, -1.00],
        ],
        dtype=np.float32,
    )

    enhanced = _suppress_weak_amplitudes(section, percentile=50.0)

    assert enhanced.shape == section.shape
    assert np.count_nonzero(enhanced[0]) == 0
    assert np.sign(enhanced[2, 0]) == np.sign(section[2, 0])
    assert np.sign(enhanced[2, 1]) == np.sign(section[2, 1])
    assert np.max(np.abs(enhanced)) <= np.max(np.abs(section))


def test_normal_incidence_stack_maps_reflector_to_two_way_time():
    nz, nx, nt = 40, 12, 260
    velocity = np.full((nz, nx), 2000.0, dtype=np.float32)
    velocity[20:, :] = 3000.0

    record = synthesize_normal_incidence_stack(
        velocity,
        dz=10.0,
        dt=0.001,
        nt=nt,
        f0=25.0,
    )

    expected_sample = int(round(2.0 * 20 * 10.0 / 2000.0 / 0.001))
    rms = np.sqrt(np.mean(record.astype(np.float64) ** 2, axis=1))

    assert record.shape == (nt, nx)
    assert np.isfinite(record).all()
    assert abs(int(np.argmax(rms)) - expected_sample) <= 8
