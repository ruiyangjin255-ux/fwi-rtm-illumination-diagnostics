from __future__ import annotations

from pathlib import Path

from admit_fwi.check_reference_integrity import audit_references, write_audit


def test_audit_references_parses_numbering_and_metadata(tmp_path: Path) -> None:
    manuscript = Path("admit_fwi/docs/jge_submission_package/manuscript/sci_fwi_rtm_innovation_manuscript_draft.md")
    result = audit_references(manuscript)

    assert result["reference_count"] == 8
    assert result["numbering_ok"] is True
    assert any(row["doi"].startswith("10.1190/") for row in result["rows"])
    assert any("openreview.net" in row["url"] for row in result["rows"])

    written = write_audit(result, tmp_path / "reference_audit.md")
    assert written["markdown"].exists()
    assert written["json"].exists()
    assert written["csv"].exists()
