from __future__ import annotations

from admit_fwi.build_target_zone_illumination_diagnostics import (
    build_target_zone_masks,
    compute_target_zone_metrics,
    load_diagnostic_arrays,
)


def test_target_zone_masks_and_metrics_are_nonempty() -> None:
    arrays = load_diagnostic_arrays()
    masks = build_target_zone_masks(arrays.true_velocity)

    assert set(masks) == {"salt_top", "salt_flanks", "subsalt_shadow"}
    assert all(mask.any() for mask in masks.values())

    rows = compute_target_zone_metrics(arrays)
    by_zone = {row["zone"]: row for row in rows}

    assert by_zone["salt_top"]["mean_source_receiver_illumination_norm"] > by_zone["subsalt_shadow"][
        "mean_source_receiver_illumination_norm"
    ]
    assert by_zone["salt_top"]["mean_full_update_abs_ms"] > by_zone["subsalt_shadow"]["mean_full_update_abs_ms"]
