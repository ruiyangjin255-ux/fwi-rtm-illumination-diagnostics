import numpy as np

from rtm_acoustic.optimize_existing_salt_result import (
    build_candidate_products,
    compute_diagnostics,
    depth_balanced_display,
    make_paper_ready_products,
    mask_low_illumination,
    robust_symmetric_display,
    soft_threshold_display,
)


def test_mask_low_illumination_zeros_underlit_samples():
    image = np.arange(6, dtype=np.float32).reshape(2, 3)
    illumination = np.array([[10.0, 1.0, 0.0], [9.0, 5.0, 2.0]], dtype=np.float32)

    masked = mask_low_illumination(image, illumination, fraction=0.3)

    np.testing.assert_array_equal(masked, [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])


def test_robust_symmetric_display_clips_and_normalizes():
    image = np.array([[-10.0, -1.0, 0.0], [1.0, 10.0, 1000.0]], dtype=np.float32)

    display = robust_symmetric_display(image, percentile=90.0, output_clip=0.8)

    assert display.shape == image.shape
    assert np.isfinite(display).all()
    assert np.max(display) <= 0.800001
    assert np.min(display) >= -0.800001


def test_depth_balanced_display_boosts_deeper_weak_event():
    image = np.zeros((12, 4), dtype=np.float32)
    image[2, :] = 4.0
    image[10, :] = 0.4

    display = depth_balanced_display(image, percentile=99.0, output_clip=1.0)

    assert display.shape == image.shape
    assert np.isfinite(display).all()
    assert np.max(np.abs(display)) <= 1.0
    assert abs(float(display[10, 1])) > abs(float(robust_symmetric_display(image)[10, 1]))


def test_compute_diagnostics_reports_illumination_and_energy_zones():
    image = np.ones((9, 6), dtype=np.float32)
    image[:, 3:] *= 2.0
    illumination = np.ones_like(image)
    illumination[:3, :2] = 0.0

    metrics = compute_diagnostics(
        raw=image,
        normalized=image,
        filtered=image * 0.5,
        illumination=illumination,
    )

    assert metrics["shape"] == [9, 6]
    assert metrics["low_illumination_fraction"] > 0.0
    assert metrics["lateral_energy"]["right"] > metrics["lateral_energy"]["left"]
    assert metrics["laplacian_energy_retention"] == 0.25
    assert set(metrics["depth_energy"]) == {"shallow", "middle", "deep"}


def test_build_candidate_products_returns_named_candidates_and_decision():
    velocity = np.ones((10, 8), dtype=np.float32) * 2000.0
    raw = np.random.default_rng(0).normal(size=(10, 8)).astype(np.float32)
    illumination = np.ones((10, 8), dtype=np.float32)
    illumination[:, :2] = 0.01
    normalized = raw.copy()
    filtered = raw * 0.1
    display = raw * 0.05

    products, metrics, decision = build_candidate_products(
        velocity=velocity,
        raw=raw,
        illumination=illumination,
        normalized=normalized,
        filtered=filtered,
        current_display=display,
    )

    assert {
        "current_display",
        "improved_display",
        "low_illumination_masked",
        "depth_balanced",
        "recommended",
    }.issubset(products)
    assert products["recommended"].shape == raw.shape
    assert metrics["low_illumination_fraction"] > 0.0
    assert decision in {"display_dominated", "imaging_condition_limited"}


def test_soft_threshold_display_suppresses_background_without_changing_shape():
    image = np.zeros((8, 8), dtype=np.float32)
    image[2, 2] = 0.05
    image[4, 4] = 2.0

    display = soft_threshold_display(image, threshold_percentile=50.0, clip_percentile=99.0)

    assert display.shape == image.shape
    assert np.isfinite(display).all()
    assert abs(float(display[2, 2])) < abs(float(display[4, 4]))
    assert np.max(np.abs(display)) <= 0.900001


def test_make_paper_ready_products_returns_conservative_and_enhanced_versions():
    filtered = np.zeros((12, 10), dtype=np.float32)
    filtered[4:6, 2:8] = 0.5
    filtered[8:9, 1:9] = -0.2
    current_display = filtered * 0.5

    products = make_paper_ready_products(filtered=filtered, current_display=current_display)

    assert {"paper_conservative", "paper_enhanced", "paper_recommended"}.issubset(products)
    assert products["paper_recommended"].shape == filtered.shape
    assert np.isfinite(products["paper_recommended"]).all()
    assert np.max(np.abs(products["paper_recommended"])) <= 0.900001
