from __future__ import annotations

import csv
from pathlib import Path


SUMMARY_FIELDS = [
    "source_family",
    "target_family",
    "model_name",
    "bridge",
    "decoder_name",
    "loss_name",
    "seed",
    "metric_space",
    "cross_family_MAE",
    "cross_family_RMSE",
    "cross_family_SSIM",
    "cross_family_gradient_error",
    "cross_family_edge_MAE",
    "status",
    "skip_reason",
    "runtime_seconds",
]


def _write_summary(root: Path, rows: list[dict[str, object]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "protocol_v3_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            payload = {field: "" for field in SUMMARY_FIELDS}
            payload.update(row)
            writer.writerow(payload)


def test_protocol_v3_summary_writes_decoder_loss_and_top_config_tables(tmp_path: Path):
    from fwi_visionfm.scripts.summarize_protocol_v3 import write_summary

    base = {
        "source_family": "flatvel",
        "target_family": "curvevel",
        "model_name": "vit_tiny_scratch",
        "bridge": "raw_spectrogram",
        "seed": 0,
        "metric_space": "physical_velocity",
        "status": "SUCCESS",
        "skip_reason": "",
        "runtime_seconds": 1.0,
    }
    rows = [
        {**base, "decoder_name": "simple_bounded_decoder", "loss_name": "default_l1", "cross_family_MAE": 10.0, "cross_family_RMSE": 20.0, "cross_family_SSIM": 0.8, "cross_family_gradient_error": 5.0, "cross_family_edge_MAE": 7.0},
        {**base, "decoder_name": "unet_decoder", "loss_name": "default_l1", "cross_family_MAE": 11.0, "cross_family_RMSE": 21.0, "cross_family_SSIM": 0.82, "cross_family_gradient_error": 3.0, "cross_family_edge_MAE": 4.0},
        {**base, "decoder_name": "unet_decoder", "loss_name": "gradient_l1", "cross_family_MAE": 11.2, "cross_family_RMSE": 21.2, "cross_family_SSIM": 0.81, "cross_family_gradient_error": 2.8, "cross_family_edge_MAE": 3.8},
        {**base, "decoder_name": "unet_decoder", "loss_name": "structure_loss", "cross_family_MAE": 11.5, "cross_family_RMSE": 21.5, "cross_family_SSIM": 0.79, "cross_family_gradient_error": 2.6, "cross_family_edge_MAE": 3.5},
    ]
    _write_summary(tmp_path, rows)

    paths = write_summary(tmp_path)
    assert paths["decoder_comparison"].exists()
    assert paths["loss_comparison"].exists()
    assert paths["top_configs"].exists()

    with paths["decoder_comparison"].open("r", encoding="utf-8", newline="") as handle:
        decoder_rows = list(csv.DictReader(handle))
    assert decoder_rows[0]["numerical_winner"] == "simple_bounded_decoder"
    assert decoder_rows[0]["structural_winner"] == "unet_decoder"
    assert decoder_rows[0]["tradeoff_type"] == "simple numerical advantage vs unet structural advantage"

    with paths["loss_comparison"].open("r", encoding="utf-8", newline="") as handle:
        loss_rows = list(csv.DictReader(handle))
    assert loss_rows[0]["best_loss_by_gradient_error"] == "structure_loss"
    assert loss_rows[0]["best_loss_by_edge_MAE"] == "structure_loss"
    assert loss_rows[0]["loss_tradeoff_summary"] == "structure loss improves gradient metrics with numerical tradeoff"

    with paths["top_configs"].open("r", encoding="utf-8", newline="") as handle:
        top_rows = list(csv.DictReader(handle))
    assert any(row["rank_type"] == "top_3_by_MAE" for row in top_rows)
    assert any(row["rank_type"] == "pareto_candidates" for row in top_rows)
