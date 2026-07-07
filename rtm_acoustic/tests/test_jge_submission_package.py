from __future__ import annotations

import json
from pathlib import Path

from rtm_acoustic.build_jge_submission_package import build_package


def test_build_package_writes_manifest_index_and_checklist(tmp_path: Path) -> None:
    manifest = build_package(tmp_path / "package", figure_formats=["png"])

    manifest_path = Path(manifest["manifest_path"])
    file_index = Path(manifest["file_index_path"])
    checklist = Path(manifest["files"]["checklist"]["package_path"])

    assert manifest_path.exists()
    assert file_index.exists()
    assert checklist.exists()
    assert (tmp_path / "package" / "manuscript" / "sci_fwi_rtm_innovation_manuscript_draft.md").exists()
    assert (tmp_path / "package" / "figures" / "figure1_fwi_quality_gate.png").exists()
    assert (tmp_path / "package" / "figures" / "figure2_rtm_before_after_validation.png").exists()
    assert (tmp_path / "package" / "figures" / "figure3_imaging_condition_diagnostics.png").exists()
    assert (tmp_path / "package" / "figures" / "figure4_spatial_update_gate.png").exists()
    assert (tmp_path / "package" / "figures" / "figure5_target_zone_illumination_diagnostics.png").exists()
    assert (tmp_path / "package" / "figures" / "jge_figure_alt_text.md").exists()
    assert (tmp_path / "package" / "tables" / "fwi_update_scale_optimization.csv").exists()
    assert (tmp_path / "package" / "tables" / "method_synthesis_matrix.csv").exists()
    assert (tmp_path / "package" / "tables" / "jge_innovation_framework.csv").exists()
    assert (tmp_path / "package" / "tables" / "spatial_update_gate_candidates.csv").exists()
    assert (tmp_path / "package" / "tables" / "target_zone_illumination_metrics.csv").exists()
    assert (tmp_path / "package" / "reports" / "jge_innovation_framework.md").exists()

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["pipeline_metrics"]["selected_alpha"] == 0.1
    assert data["files"]["figures"]["figure5_target_zone_illumination_diagnostics.png"]["bytes"] > 0
    assert data["files"]["innovation_framework"]["bytes"] > 0
