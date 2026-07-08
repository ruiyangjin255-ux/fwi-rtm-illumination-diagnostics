from __future__ import annotations

import csv
from pathlib import Path


def test_protocol_v5_seed_stability_plot_writes_csv_and_png(tmp_path: Path):
    from fwi_visionfm.scripts.plot_protocol_v5_seed_stability import write_protocol_v5_seed_stability_artifacts

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
            "MAE": "101",
            "RMSE": "201",
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
            "MAE": "105",
            "RMSE": "205",
            "SSIM": "0.79",
            "gradient_error": "10",
            "edge_MAE": "20",
            "visual_score": "0.74",
            "status": "SUCCESS",
            "skip_reason": "",
        },
        {
            "model_type": "pretrained_local_mae",
            "bridge": "raw_envelope_spectrum3",
            "mask_type": "trace_dropout",
            "decoder_name": "unet_decoder",
            "loss_name": "default_l1",
            "seed": "2",
            "pretrain_epochs": "5",
            "decoder_epochs": "3",
            "reconstruction_loss": "0.12",
            "MAE": "104",
            "RMSE": "204",
            "SSIM": "0.82",
            "gradient_error": "11",
            "edge_MAE": "19",
            "visual_score": "0.86",
            "status": "SUCCESS",
            "skip_reason": "",
        },
        {
            "model_type": "random_mae_encoder",
            "bridge": "raw_envelope_spectrum3",
            "mask_type": "trace_dropout",
            "decoder_name": "unet_decoder",
            "loss_name": "default_l1",
            "seed": "2",
            "pretrain_epochs": "0",
            "decoder_epochs": "3",
            "reconstruction_loss": "",
            "MAE": "106",
            "RMSE": "206",
            "SSIM": "0.77",
            "gradient_error": "12",
            "edge_MAE": "21",
            "visual_score": "0.72",
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

    output_dir = tmp_path / "out"
    result = write_protocol_v5_seed_stability_artifacts(root, output_dir)
    csv_text = result["csv_path"].read_text(encoding="utf-8")
    assert "pretrained_win_MAE" in csv_text
    assert "win_count" in csv_text
    assert "MAE,3" in csv_text.replace("\r\n", "\n")
    assert result["comparison_png"].exists()
    assert result["win_count_png"].exists()
