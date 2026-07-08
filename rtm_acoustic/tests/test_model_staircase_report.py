from __future__ import annotations

import json
from pathlib import Path

from rtm_acoustic.diagnostics.model_staircase_report import summarize_model_staircase


def test_model_staircase_report_classifies_simple_fault(tmp_path: Path):
    case = tmp_path / "simple_fault"
    case.mkdir()
    (case / "manifest.json").write_text(json.dumps({"model_name": "simple_fault", "status": "READY", "proxy_type": "SIMPLIFIED_DIAGNOSTIC_PROXY_NOT_FWI", "verdict": "selective_update_needed"}), encoding="utf-8")
    rows = summarize_model_staircase(tmp_path)
    assert rows[0]["main_failure_mode"] == "EDGE_LOCALIZATION_RISK"
    assert "full FWI/RTM benchmark" in rows[0]["forbidden_claim"]
