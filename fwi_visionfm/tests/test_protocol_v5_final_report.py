from __future__ import annotations

import csv
from pathlib import Path


def test_protocol_v5_final_report_contains_seed_stability_and_benchmark_caveat(tmp_path: Path):
    from fwi_visionfm.scripts.report_protocol_v5_final_stage import write_protocol_v5_final_stage_report

    rows = [
        {
            "model_type": "pretrained_local_mae",
            "bridge": "raw_envelope_spectrum3",
            "mask_type": "trace_dropout",
            "decoder_name": "unet_decoder",
            "loss_name": "default_l1",
            "seed": "0",
            "pretrain_epochs": "5",
            "decoder_epochs": "3",
            "reconstruction_loss": "0.10",
            "MAE": "100",
            "RMSE": "200",
            "SSIM": "0.80",
            "gradient_error": "10",
            "edge_MAE": "20",
            "visual_score": "0.90",
            "status": "SUCCESS",
            "skip_reason": "",
        },
        {
            "model_type": "random_mae_encoder",
            "bridge": "raw_envelope_spectrum3",
            "mask_type": "trace_dropout",
            "decoder_name": "unet_decoder",
            "loss_name": "default_l1",
            "seed": "0",
            "pretrain_epochs": "0",
            "decoder_epochs": "3",
            "reconstruction_loss": "",
            "MAE": "110",
            "RMSE": "210",
            "SSIM": "0.75",
            "gradient_error": "11",
            "edge_MAE": "21",
            "visual_score": "0.70",
            "status": "SUCCESS",
            "skip_reason": "",
        },
        {
            "model_type": "pretrained_local_mae",
            "bridge": "raw_envelope_spectrum3",
            "mask_type": "trace_dropout",
            "decoder_name": "unet_decoder",
            "loss_name": "default_l1",
            "seed": "1",
            "pretrain_epochs": "5",
            "decoder_epochs": "3",
            "reconstruction_loss": "0.11",
            "MAE": "102",
            "RMSE": "202",
            "SSIM": "0.81",
            "gradient_error": "12",
            "edge_MAE": "22",
            "visual_score": "0.88",
            "status": "SUCCESS",
            "skip_reason": "",
        },
        {
            "model_type": "random_mae_encoder",
            "bridge": "raw_envelope_spectrum3",
            "mask_type": "trace_dropout",
            "decoder_name": "unet_decoder",
            "loss_name": "default_l1",
            "seed": "1",
            "pretrain_epochs": "0",
            "decoder_epochs": "3",
            "reconstruction_loss": "",
            "MAE": "108",
            "RMSE": "208",
            "SSIM": "0.78",
            "gradient_error": "10",
            "edge_MAE": "20",
            "visual_score": "0.74",
            "status": "SUCCESS",
            "skip_reason": "",
        },
    ]
    root = tmp_path / "root"
    root.mkdir()
    summary_path = root / "local_mae_ablation_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (root / "local_mae_ablation_report.md").write_text("# placeholder\n", encoding="utf-8")

    output_dir = tmp_path / "stage"
    report_path = write_protocol_v5_final_stage_report(root, output_dir)
    text = report_path.read_text(encoding="utf-8")
    assert "Trace-dropout Seed Stability" in text
    assert "not benchmark-level proof" in text
    assert "stable numerical benefit" in text

