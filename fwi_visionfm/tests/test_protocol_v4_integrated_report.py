from __future__ import annotations

import csv
from pathlib import Path


FIELDS = [
    "model_name",
    "bridge",
    "decoder_name",
    "loss_name",
    "seed",
    "MAE",
    "RMSE",
    "SSIM",
    "gradient_error",
    "edge_MAE",
    "visual_score",
    "visual_rank",
    "numerical_rank",
    "structural_rank",
    "status",
    "is_probe",
    "is_structural_control",
]


def test_integrated_report_identifies_probe_and_limitation(tmp_path: Path):
    from fwi_visionfm.scripts.report_protocol_v4_integrated import write_integrated_report

    rows = [
        {
            "model_name": "cnn_baseline",
            "bridge": "raw_repeat3",
            "decoder_name": "unet_decoder",
            "loss_name": "default_l1",
            "seed": 0,
            "MAE": 10,
            "RMSE": 11,
            "SSIM": 0.5,
            "gradient_error": 2,
            "edge_MAE": 3,
            "visual_score": 0.8,
            "visual_rank": 1,
            "numerical_rank": 2,
            "structural_rank": 1,
            "status": "SUCCESS",
            "is_probe": False,
            "is_structural_control": True,
        },
        {
            "model_name": "dinov2_lora_smoke",
            "bridge": "raw_spectrogram",
            "decoder_name": "unet_decoder",
            "loss_name": "default_l1",
            "seed": 0,
            "MAE": 12,
            "RMSE": 13,
            "SSIM": 0.4,
            "gradient_error": 4,
            "edge_MAE": 5,
            "visual_score": 0.3,
            "visual_rank": 2,
            "numerical_rank": 2,
            "structural_rank": 2,
            "status": "SUCCESS",
            "is_probe": True,
            "is_structural_control": False,
        },
    ]
    with (tmp_path / "protocol_v4_integrated_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    path = write_integrated_report(tmp_path)
    text = path.read_text(encoding="utf-8")

    assert "not application-level" in text
    assert "DINOv2-LoRA" in text
    assert "probe" in text
    assert "raw_repeat3" in text
    assert "VisionFM improves FWI generalization" not in text
