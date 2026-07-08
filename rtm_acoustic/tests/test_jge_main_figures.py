from __future__ import annotations

from pathlib import Path


def test_jge_main_figures_exist_in_submission_formats() -> None:
    figure_dir = Path("rtm_acoustic/docs/jge_main_figures")
    stems = [
        "figure1_fwi_quality_gate",
        "figure2_rtm_before_after_validation",
        "figure3_imaging_condition_diagnostics",
        "figure4_spatial_update_gate",
        "figure5_target_zone_illumination_diagnostics",
    ]

    for stem in stems:
        for ext in ["png", "pdf", "svg"]:
            path = figure_dir / f"{stem}.{ext}"
            assert path.exists(), path
            assert path.stat().st_size > 10_000, path

    tiff_paths = [figure_dir / f"{stem}.tiff" for stem in stems]
    if all(path.exists() for path in tiff_paths):
        for path in tiff_paths:
            assert path.stat().st_size > 10_000, path


def test_jge_main_figure_captions_cover_all_main_figures() -> None:
    captions = Path("rtm_acoustic/docs/jge_main_figures/jge_main_figure_captions.md").read_text(encoding="utf-8")

    for number in range(1, 6):
        assert f"## Figure {number}." in captions
