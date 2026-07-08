from __future__ import annotations

import json
from pathlib import Path


def test_protocol_v9_report_handles_no_real_feature_graceful_skip(tmp_path: Path):
    from fwi_visionfm.scripts.train_v9_real_frozen_feature_decoder_probe import train_v9_real_frozen_feature_decoder_probe
    from fwi_visionfm.scripts.report_protocol_v9_weight_setup_and_real_probe import write_protocol_v9_weight_setup_and_real_probe_report

    feature_cache_root = tmp_path / "feature_cache"
    feature_cache_root.mkdir(parents=True, exist_ok=True)
    cache_dir = feature_cache_root / "fallback_random"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "cache_config.json").write_text(
        json.dumps({"backbone_name": "fallback_random", "status": "FALLBACK_FEATURE_ONLY", "is_real_feature": False}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    probe_root = tmp_path / "probe"
    result = train_v9_real_frozen_feature_decoder_probe(feature_cache_root=feature_cache_root, output_dir=probe_root, epochs=2, seed=0, device="cpu")
    assert result["status"] == "SKIPPED_NO_REAL_FEATURE_CACHE"

    download_root = tmp_path / "download"
    download_root.mkdir(parents=True, exist_ok=True)
    (download_root / "download_status.json").write_text(json.dumps({"entries": []}, indent=2), encoding="utf-8")
    (download_root / "availability_report.json").write_text(json.dumps({"models": []}, indent=2), encoding="utf-8")
    payload = write_protocol_v9_weight_setup_and_real_probe_report(download_root=download_root, probe_root=probe_root, output_dir=probe_root)
    report_text = payload["report_path"].read_text(encoding="utf-8")
    claims_text = payload["claims_path"].read_text(encoding="utf-8")
    assert "SKIPPED_NO_REAL_FEATURE_CACHE" in report_text
    assert "not benchmark-level proof" in report_text
    assert "NCS improves FWI" not in report_text
    assert "fallback feature" in report_text.lower()
    assert "## Can Claim" in claims_text
    assert "## Cannot Claim" in claims_text
