from __future__ import annotations

import csv
from pathlib import Path


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {field: "" for field in fieldnames}
            payload.update(row)
            writer.writerow(payload)


def test_report_writes_multiseed_stability_section(tmp_path: Path):
    from fwi_visionfm.scripts.report_protocol_v3 import build_report
    from fwi_visionfm.scripts.summarize_protocol_v3 import (
        DECODER_COMPARISON_FIELDS,
        FIELDNAMES,
        LOSS_COMPARISON_FIELDS,
        TOP_CONFIG_FIELDS,
    )

    summary_rows = [
        {
            "source_family": "flatvel_a_subset2k",
            "target_family": "curvevel_a_subset500",
            "model_name": "dinov2_lora_smoke",
            "bridge": "raw_spectrogram",
            "decoder_name": "unet_decoder",
            "loss_name": "default_l1",
            "seed": seed,
            "metric_space": "physical_velocity",
            "status": "SUCCESS",
        }
        for seed in (0, 1)
    ]
    _write_csv(tmp_path / "protocol_v3_summary.csv", FIELDNAMES, summary_rows)
    _write_csv(
        tmp_path / "protocol_v3_decoder_comparison.csv",
        DECODER_COMPARISON_FIELDS,
        [
            {
                "source_family": "flatvel_a_subset2k",
                "target_family": "curvevel_a_subset500",
                "model_name": "cnn_baseline",
                "bridge": "raw_repeat3",
                "loss_name": "default_l1",
                "seed": seed,
                "numerical_winner": "simple_bounded_decoder",
                "structural_winner": "unet_decoder",
                "tradeoff_type": "simple numerical advantage vs unet structural advantage",
            }
            for seed in (0, 1, 2)
        ],
    )
    _write_csv(
        tmp_path / "protocol_v3_loss_comparison.csv",
        LOSS_COMPARISON_FIELDS,
        [
            {
                "source_family": "flatvel_a_subset2k",
                "target_family": "curvevel_a_subset500",
                "model_name": "vit_tiny_scratch",
                "bridge": "raw_spectrogram",
                "decoder_name": "unet_decoder",
                "seed": seed,
                "loss_tradeoff_summary": "structure loss improves gradient metrics with numerical tradeoff",
            }
            for seed in (0, 1, 2)
        ],
    )
    _write_csv(tmp_path / "protocol_v3_top_configs.csv", TOP_CONFIG_FIELDS, [])

    report = build_report(tmp_path)
    text = report.read_text(encoding="utf-8")

    assert "## Multi-seed Stability" in text
    assert "selected single-pair multi-seed validation" in text
    assert "limited-seed probe" in text
    assert "single pair / single seed" not in text
    assert "simple decoder MAE/RMSE wins: 3/3 seeds" in text
    assert "U-Net decoder gradient_error/edge_MAE wins: 3/3 seeds" in text
    assert "structure_loss lowers gradient_error: 3/3 seeds" in text
    assert "DINOv2-LoRA probe seed status: success=2, skipped=0, failed=0" in text
    assert "VisionFM improves FWI generalization" not in text
