from __future__ import annotations

from pathlib import Path

from rtm_acoustic.check_jge_submission_readiness import check_manuscript, write_report


def test_check_manuscript_reports_core_jge_limits(tmp_path: Path) -> None:
    result = check_manuscript(
        Path("rtm_acoustic/docs/jge_submission_package_mainfigures/manuscript/sci_fwi_rtm_innovation_manuscript_draft.md")
    )
    checks = {check["item"]: check for check in result["checks"]}

    assert checks["English abstract <=250 words"]["value"] > 0
    assert checks["References <=50"]["status"] == "pass"
    assert checks["Data/code availability statement present"]["status"] == "pass"
    assert "OUP Journal of Geophysics and Engineering" in result["guideline_source"]

    written = write_report(result, tmp_path / "readiness.md")
    assert written["markdown"].exists()
    assert written["json"].exists()
