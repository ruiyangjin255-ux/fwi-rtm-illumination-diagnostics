from __future__ import annotations

import json
from pathlib import Path


def _write_probe_run(path: Path, *, seed: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "decoder_name": "lightweight_feature_decoder",
                "loss_name": "default_l1",
                "epochs": 2,
                "status": "SUCCESS",
                "is_real_feature": True,
                "feature_cache_status": "AVAILABLE",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (path / "metrics_val.json").write_text(json.dumps({"mae": 420.0 + seed, "rmse": 520.0 + seed, "ssim": 0.8, "metric_space": "physical_velocity"}, indent=2), encoding="utf-8")
    (path / "metrics_cross_family_test.json").write_text(
        json.dumps(
            {"mae": 423.0 + seed, "rmse": 523.0 + seed, "ssim": 0.83, "gradient_error": 55.0 + seed, "edge_mae": 135.0 + seed, "metric_space": "physical_velocity"},
            indent=2,
        ),
        encoding="utf-8",
    )


def test_protocol_v9_ncs2d_seed_stability_report_contains_required_limitations(tmp_path: Path):
    from fwi_visionfm.scripts.report_protocol_v9_ncs2d_seed_stability import write_protocol_v9_ncs2d_seed_stability_report

    root = tmp_path / "protocol_v9_ncs2d_seed_stability"
    seed0_dir = tmp_path / "seed0"
    _write_probe_run(seed0_dir, seed=0)
    _write_probe_run(root / "decoder_probe" / "ncs_2d_seed1", seed=1)
    _write_probe_run(root / "decoder_probe" / "ncs_2d_seed2", seed=2)

    adapter_report = tmp_path / "adapter_repair.md"
    adapter_report.write_text(
        "previous ncs_2d IMPORT_ERROR has been repaired\nncs_2d now uses transformers-compatible real feature extraction\nncs_2p5d remains adapter pending\nnot benchmark-level proof\n",
        encoding="utf-8",
    )

    payload = write_protocol_v9_ncs2d_seed_stability_report(root=root, seed0_dir=seed0_dir, adapter_repair_report=adapter_report, output_dir=root)
    report_text = payload["report_path"].read_text(encoding="utf-8")
    claims_text = payload["claims_path"].read_text(encoding="utf-8")
    assert "ncs_2d" in report_text
    assert "is_real_feature=True" in report_text
    assert "previous ncs_2d IMPORT_ERROR has been repaired" in report_text
    assert "ncs_2p5d remains adapter pending" in report_text
    assert "not benchmark-level proof" in report_text
    assert "NCS improves FWI" not in report_text
    assert "## Can Claim" in claims_text
    assert "## Cannot Claim" in claims_text
