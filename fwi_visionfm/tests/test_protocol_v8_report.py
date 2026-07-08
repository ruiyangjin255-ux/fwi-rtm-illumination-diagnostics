from __future__ import annotations

import json
from pathlib import Path


def test_protocol_v8_report_and_continued_mae_plan_keep_claims_conservative(tmp_path: Path):
    from fwi_visionfm.scripts.report_protocol_v8_continued_mae_plan import write_continued_mae_plan
    from fwi_visionfm.scripts.report_protocol_v8_seismic_fm_probe import write_protocol_v8_seismic_fm_probe_report

    root = tmp_path / "protocol_v8_seismic_fm_probe"
    root.mkdir(parents=True, exist_ok=True)
    (root / "availability_report.json").write_text(
        json.dumps(
            {
                "overall_status": "UNAVAILABLE_REPO",
                "variants": [
                    {
                        "variant": "ncs_2p5d",
                        "status": "UNAVAILABLE_REPO",
                        "message": "repo missing",
                    }
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    feature_cache_dir = root / "feature_cache"
    feature_cache_dir.mkdir(parents=True, exist_ok=True)
    (feature_cache_dir / "metadata.json").write_text(
        json.dumps(
            {
                "backbone_name": "fallback_random",
                "bridge_name": "raw_envelope_spectrum3",
                "tokenizer_name": "fallback_tokenization",
                "status": "SKIPPED_NCS_UNAVAILABLE",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    decoder_dir = root / "decoder_probe"
    decoder_dir.mkdir(parents=True, exist_ok=True)
    (decoder_dir / "config.json").write_text(
        json.dumps(
            {
                "status": "DUMMY_CACHE_SMOKE",
                "metric_space": "physical_velocity",
                "is_real_ncs_feature": False,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    plan_path = write_continued_mae_plan(root)
    report_payload = write_protocol_v8_seismic_fm_probe_report(root)

    assert plan_path.exists()
    assert report_payload["report_path"].exists()
    assert report_payload["claims_path"].exists()

    report_text = report_payload["report_path"].read_text(encoding="utf-8")
    claims_text = report_payload["claims_path"].read_text(encoding="utf-8")
    plan_text = plan_path.read_text(encoding="utf-8")

    assert "not benchmark-level proof" in report_text
    assert "NCS improves FWI" not in report_text
    assert "## Can Claim" in claims_text
    assert "## Cannot Claim" in claims_text
    assert "continued MAE" in plan_text
