from __future__ import annotations

import csv
from pathlib import Path


def test_protocol_v6_fake_summary_generates_report(tmp_path: Path):
    from fwi_visionfm.scripts.report_protocol_v6_geometry_aggregation import write_protocol_v6_geometry_aggregation_report

    summary_path = tmp_path / "protocol_v6_geometry_aggregation_summary.csv"
    rows = [
        {
            "run_id": "run_1",
            "bridge": "raw_envelope_spectrum3",
            "geometry_enabled": "False",
            "geometry_mode": "",
            "geometry_fusion": "",
            "projection_to_3ch": "False",
            "aggregator": "mean",
            "seed": "0",
            "train_size": "100",
            "val_size": "50",
            "test_size": "50",
            "epochs": "2",
            "val_MAE": "1.0",
            "val_RMSE": "2.0",
            "val_SSIM": "0.8",
            "cross_family_MAE": "1.1",
            "cross_family_RMSE": "2.1",
            "cross_family_SSIM": "0.7",
            "cross_family_gradient_error": "0.3",
            "cross_family_edge_MAE": "0.4",
            "attention_entropy": "",
            "status": "SUCCESS",
            "skip_reason": "",
        },
        {
            "run_id": "run_4",
            "bridge": "raw_envelope_spectrum3",
            "geometry_enabled": "True",
            "geometry_mode": "sinusoidal",
            "geometry_fusion": "concat",
            "projection_to_3ch": "True",
            "aggregator": "source_aware_attention",
            "seed": "0",
            "train_size": "100",
            "val_size": "50",
            "test_size": "50",
            "epochs": "2",
            "val_MAE": "1.2",
            "val_RMSE": "2.2",
            "val_SSIM": "0.75",
            "cross_family_MAE": "1.3",
            "cross_family_RMSE": "2.3",
            "cross_family_SSIM": "0.72",
            "cross_family_gradient_error": "0.31",
            "cross_family_edge_MAE": "0.41",
            "attention_entropy": "0.6",
            "status": "SUCCESS",
            "skip_reason": "",
        },
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    report = write_protocol_v6_geometry_aggregation_report(tmp_path)
    text = report.read_text(encoding="utf-8")
    assert "not benchmark-level proof" in text
    assert "geometry is fallback index-based unless real source/receiver metadata are provided" in text
    assert "geometry-aware bridge improves FWI generalization" not in text
