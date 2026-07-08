from __future__ import annotations

import csv
from pathlib import Path


def test_protocol_v8_continued_mae_tiny_report_handles_skipped_natural_init(tmp_path: Path):
    from fwi_visionfm.scripts.report_protocol_v8_continued_mae_tiny_probe import write_protocol_v8_continued_mae_tiny_report

    rows = [
        {
            "run_id": "A",
            "encoder_init": "random",
            "seismic_mae_pretraining": "false",
            "masking": "trace_dropout",
            "seed": "0",
            "train_size": "100",
            "val_size": "50",
            "test_size": "50",
            "pretrain_loss_final": "",
            "val_MAE": "100",
            "val_RMSE": "200",
            "val_SSIM": "0.70",
            "cross_family_MAE": "110",
            "cross_family_RMSE": "220",
            "cross_family_SSIM": "0.68",
            "cross_family_gradient_error": "30",
            "cross_family_edge_MAE": "40",
            "status": "SUCCESS",
            "skip_reason": "",
        },
        {
            "run_id": "B",
            "encoder_init": "random",
            "seismic_mae_pretraining": "true",
            "masking": "trace_dropout",
            "seed": "0",
            "train_size": "100",
            "val_size": "50",
            "test_size": "50",
            "pretrain_loss_final": "0.1",
            "val_MAE": "95",
            "val_RMSE": "190",
            "val_SSIM": "0.72",
            "cross_family_MAE": "105",
            "cross_family_RMSE": "210",
            "cross_family_SSIM": "0.69",
            "cross_family_gradient_error": "29",
            "cross_family_edge_MAE": "39",
            "status": "SUCCESS",
            "skip_reason": "",
        },
        {
            "run_id": "C",
            "encoder_init": "natural_image_mae",
            "seismic_mae_pretraining": "false",
            "masking": "trace_dropout",
            "seed": "0",
            "train_size": "100",
            "val_size": "50",
            "test_size": "50",
            "pretrain_loss_final": "",
            "val_MAE": "",
            "val_RMSE": "",
            "val_SSIM": "",
            "cross_family_MAE": "",
            "cross_family_RMSE": "",
            "cross_family_SSIM": "",
            "cross_family_gradient_error": "",
            "cross_family_edge_MAE": "",
            "status": "SKIPPED_NATURAL_IMAGE_INIT_UNAVAILABLE",
            "skip_reason": "weights unavailable",
        },
        {
            "run_id": "D",
            "encoder_init": "natural_image_mae",
            "seismic_mae_pretraining": "true",
            "masking": "trace_dropout",
            "seed": "0",
            "train_size": "100",
            "val_size": "50",
            "test_size": "50",
            "pretrain_loss_final": "",
            "val_MAE": "",
            "val_RMSE": "",
            "val_SSIM": "",
            "cross_family_MAE": "",
            "cross_family_RMSE": "",
            "cross_family_SSIM": "",
            "cross_family_gradient_error": "",
            "cross_family_edge_MAE": "",
            "status": "SKIPPED_NATURAL_IMAGE_INIT_UNAVAILABLE",
            "skip_reason": "weights unavailable",
        },
    ]
    summary_path = tmp_path / "protocol_v8_continued_mae_tiny_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = write_protocol_v8_continued_mae_tiny_report(tmp_path)
    report_text = payload["report_path"].read_text(encoding="utf-8")
    claims_text = payload["claims_path"].read_text(encoding="utf-8")

    assert "not benchmark-level proof" in report_text
    assert "natural-image continued pretraining improves FWI" not in report_text
    assert "SKIPPED_NATURAL_IMAGE_INIT_UNAVAILABLE" in report_text
    assert "## Can Claim" in claims_text
    assert "## Cannot Claim" in claims_text
