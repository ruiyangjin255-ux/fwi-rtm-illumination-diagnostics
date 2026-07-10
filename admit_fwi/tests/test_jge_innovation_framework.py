from __future__ import annotations

import json
from pathlib import Path

from admit_fwi.build_jge_innovation_framework import build


def test_build_jge_innovation_framework_maps_claims_to_outputs(tmp_path: Path) -> None:
    paths = build(tmp_path)

    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0

    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    claims = data["innovation_claims"]
    alt_text = data["figure_alt_text"]

    assert data["jge_paper_limits"]["abstract_words"] == 250
    assert len(claims) == 5
    assert len(alt_text) == 5
    assert {claim["claim_id"] for claim in claims} >= {
        "illumination_trust_spatial_update_gate",
        "quality_gated_fwi_rtm",
        "target_zone_illumination_diagnostics",
        "imaging_condition_separation",
    }
    for claim in claims:
        assert claim["implemented_program"]
        assert claim["primary_outputs"]
        assert claim["main_figure"]
        assert "not" in claim["claim_boundary"].lower() or "remain" in claim["claim_boundary"].lower()
