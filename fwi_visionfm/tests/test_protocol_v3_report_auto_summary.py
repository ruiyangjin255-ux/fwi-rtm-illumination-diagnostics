from __future__ import annotations

import csv
from pathlib import Path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_protocol_v3_report_writes_auto_interpretation(tmp_path: Path):
    from fwi_visionfm.scripts.report_protocol_v3 import build_report

    _write_csv(
        tmp_path / "protocol_v3_summary.csv",
        [
            {
                "source_family": "flatvel",
                "target_family": "curvevel",
                "model_name": "dinov2_lora_smoke",
                "bridge": "raw_spectrogram",
                "decoder_name": "unet_decoder",
                "loss_name": "default_l1",
                "seed": 0,
                "metric_space": "physical_velocity",
                "cross_family_MAE": 12.0,
                "cross_family_RMSE": 22.0,
                "cross_family_SSIM": 0.8,
                "cross_family_gradient_error": 3.0,
                "cross_family_edge_MAE": 4.0,
                "status": "SUCCESS",
                "skip_reason": "",
                "runtime_seconds": 1.0,
            }
        ],
    )
    _write_csv(
        tmp_path / "protocol_v3_decoder_comparison.csv",
        [
            {
                "source_family": "flatvel",
                "target_family": "curvevel",
                "model_name": "vit_tiny_scratch",
                "bridge": "raw_spectrogram",
                "loss_name": "default_l1",
                "seed": 0,
                "simple_MAE": 10.0,
                "unet_MAE": 11.0,
                "delta_MAE": 1.0,
                "simple_RMSE": 20.0,
                "unet_RMSE": 21.0,
                "delta_RMSE": 1.0,
                "simple_SSIM": 0.8,
                "unet_SSIM": 0.82,
                "delta_SSIM": 0.02,
                "simple_gradient_error": 5.0,
                "unet_gradient_error": 3.0,
                "delta_gradient_error": -2.0,
                "simple_edge_MAE": 7.0,
                "unet_edge_MAE": 4.0,
                "delta_edge_MAE": -3.0,
                "numerical_winner": "simple_bounded_decoder",
                "structural_winner": "unet_decoder",
                "tradeoff_type": "simple numerical advantage vs unet structural advantage",
            }
        ],
    )
    _write_csv(
        tmp_path / "protocol_v3_loss_comparison.csv",
        [
            {
                "source_family": "flatvel",
                "target_family": "curvevel",
                "model_name": "vit_tiny_scratch",
                "bridge": "raw_spectrogram",
                "decoder_name": "unet_decoder",
                "seed": 0,
                "default_MAE": 10.0,
                "gradient_l1_MAE": 10.5,
                "structure_loss_MAE": 11.0,
                "default_gradient_error": 5.0,
                "gradient_l1_gradient_error": 4.0,
                "structure_loss_gradient_error": 3.0,
                "default_edge_MAE": 7.0,
                "gradient_l1_edge_MAE": 6.0,
                "structure_loss_edge_MAE": 5.0,
                "best_loss_by_MAE": "default_l1",
                "best_loss_by_gradient_error": "structure_loss",
                "best_loss_by_edge_MAE": "structure_loss",
                "loss_tradeoff_summary": "structure loss improves gradient metrics with numerical tradeoff",
            }
        ],
    )
    _write_csv(
        tmp_path / "protocol_v3_top_configs.csv",
        [
            {
                "rank_type": "pareto_candidates",
                "source_family": "flatvel",
                "target_family": "curvevel",
                "model_name": "vit_tiny_scratch",
                "bridge": "raw_spectrogram",
                "decoder_name": "unet_decoder",
                "loss_name": "structure_loss",
                "seed": 0,
                "MAE": 11.0,
                "RMSE": 21.0,
                "SSIM": 0.8,
                "gradient_error": 3.0,
                "edge_MAE": 5.0,
                "note": "pareto candidate",
            }
        ],
    )

    report = build_report(tmp_path)
    text = report.read_text(encoding="utf-8")
    assert "simple_bounded_decoder is generally better on MAE/RMSE" in text
    assert "unet_decoder improves gradient_error and edge_MAE" in text
    assert "Structure-aware losses reduce gradient_error slightly but do not yet improve MAE/RMSE" in text
    assert "not a benchmark" in text
