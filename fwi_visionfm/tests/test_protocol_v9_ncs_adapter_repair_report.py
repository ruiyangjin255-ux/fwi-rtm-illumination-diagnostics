from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _write_npz(path: Path, *, feature_dim: int = 32, count: int = 4, is_real_feature: bool = True) -> None:
    np.savez(
        path,
        features=np.ones((count, feature_dim), dtype=np.float32),
        target=np.ones((count, 8, 10), dtype=np.float32),
        sample_id=np.asarray([f"s{i}" for i in range(count)], dtype=object),
        backbone_name=np.asarray("ncs_2d"),
        bridge_name=np.asarray("raw_envelope_spectrum3"),
        tokenizer_name=np.asarray("vit_pixel_values"),
        feature_shape=np.asarray([feature_dim], dtype=np.int32),
        source_split=np.asarray("train"),
        target_split=np.asarray("curvevel_a_subset500"),
        status=np.asarray("AVAILABLE"),
        is_real_feature=np.asarray(bool(is_real_feature)),
        metadata_json=np.asarray(json.dumps({"sample_id_count": count, "feature_shape": [feature_dim], "is_real_feature": is_real_feature}, ensure_ascii=False)),
    )


def test_protocol_v9_ncs_adapter_repair_report_contains_required_limitations(tmp_path: Path):
    from fwi_visionfm.scripts.report_protocol_v9_ncs_adapter_repair import write_protocol_v9_ncs_adapter_repair_report

    root = tmp_path / "protocol_v9_ncs_adapter_repair"
    root.mkdir(parents=True, exist_ok=True)
    availability = {
        "models": [
            {"name": "ncs_2d", "status": "AVAILABLE", "message": "transformers adapter ready", "load_status": "LOAD_OK", "forward_status": "FORWARD_OK", "adapter_status": "TRANSFORMERS_OK"},
            {"name": "ncs_2p5d", "status": "WEIGHTS_PRESENT_ADAPTER_PENDING", "message": "builder pending", "load_status": "WEIGHTS_OK", "forward_status": "", "adapter_status": "PENDING"},
            {"name": "vit_mae_base", "status": "AVAILABLE", "message": "available"},
        ]
    }
    (root / "availability_report.json").write_text(json.dumps(availability, indent=2, ensure_ascii=False), encoding="utf-8")
    feature_dir = root / "feature_cache" / "ncs_2d"
    feature_dir.mkdir(parents=True, exist_ok=True)
    (feature_dir / "cache_config.json").write_text(json.dumps({"backbone_name": "ncs_2d", "status": "AVAILABLE", "is_real_feature": True, "feature_shape": [32], "sample_id_count": 100}, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_npz(feature_dir / "train_features.npz", count=100)
    _write_npz(feature_dir / "val_features.npz", count=50)
    _write_npz(feature_dir / "cross_family_test_features.npz", count=50)

    probe_dir = root / "decoder_probe" / "ncs_2d"
    probe_dir.mkdir(parents=True, exist_ok=True)
    (probe_dir / "config.json").write_text(json.dumps({"status": "SUCCESS"}, indent=2), encoding="utf-8")
    (probe_dir / "metrics_val.json").write_text(json.dumps({"mae": 1.0, "rmse": 2.0, "ssim": 0.5, "gradient_error": 3.0, "edge_mae": 4.0, "metric_space": "physical_velocity"}, indent=2), encoding="utf-8")
    (probe_dir / "metrics_cross_family_test.json").write_text(json.dumps({"mae": 1.5, "rmse": 2.5, "ssim": 0.4, "gradient_error": 3.5, "edge_mae": 4.5, "metric_space": "physical_velocity"}, indent=2), encoding="utf-8")
    previous_report = root / "previous_v9.md"
    previous_report.write_text("ncs_2d previously IMPORT_ERROR\nnot benchmark-level proof\n", encoding="utf-8")

    payload = write_protocol_v9_ncs_adapter_repair_report(root=root, previous_v9_report=previous_report, output_dir=root)

    report_text = payload["report_path"].read_text(encoding="utf-8")
    claims_text = payload["claims_path"].read_text(encoding="utf-8")
    assert "previously IMPORT_ERROR" in report_text
    assert "not benchmark-level proof" in report_text
    assert "NCS improves FWI" not in report_text
    assert "## Can Claim" in claims_text
    assert "## Cannot Claim" in claims_text
