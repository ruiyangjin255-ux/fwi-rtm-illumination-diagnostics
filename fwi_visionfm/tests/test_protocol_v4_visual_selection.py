from __future__ import annotations

import csv
import json
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


def test_protocol_v4_visual_selection_outputs_best_models_and_report(tmp_path: Path):
    from fwi_visionfm.scripts.select_best_visual_model import run_visual_selection

    base = {
        "source_family": "flatvel_a_subset2k",
        "target_family": "curvevel_a_subset500",
        "bridge": "raw_repeat3",
        "loss_name": "default_l1",
        "seed": 0,
        "metric_space": "physical_velocity",
        "status": "SUCCESS",
    }
    rows = [
        {
            **base,
            "model_name": "cnn_baseline",
            "decoder_name": "simple_bounded_decoder",
            "cross_family_MAE": 10,
            "cross_family_RMSE": 20,
            "cross_family_SSIM": 0.7,
            "cross_family_gradient_error": 8,
            "cross_family_edge_MAE": 9,
        },
        {
            **base,
            "model_name": "cnn_baseline",
            "decoder_name": "unet_decoder",
            "cross_family_MAE": 12,
            "cross_family_RMSE": 22,
            "cross_family_SSIM": 0.8,
            "cross_family_gradient_error": 4,
            "cross_family_edge_MAE": 5,
        },
        {
            **base,
            "model_name": "dinov2_lora_smoke",
            "bridge": "raw_spectrogram",
            "decoder_name": "unet_decoder",
            "cross_family_MAE": 13,
            "cross_family_RMSE": 23,
            "cross_family_SSIM": 0.9,
            "cross_family_gradient_error": 3,
            "cross_family_edge_MAE": 4,
        },
    ]
    _write_summary(tmp_path, rows)

    paths = run_visual_selection(tmp_path)

    assert paths["visual_summary"].exists()
    assert paths["best_models_csv"].exists()
    assert paths["best_models_json"].exists()
    assert paths["report"].exists()

    with paths["visual_summary"].open("r", encoding="utf-8", newline="") as handle:
        visual_rows = list(csv.DictReader(handle))
    assert "visual_score" in visual_rows[0]
    assert max(float(row["visual_score"]) for row in visual_rows) <= 1.0

    with paths["best_models_json"].open("r", encoding="utf-8") as handle:
        best = json.load(handle)
    assert best["best_by_MAE"]["decoder_name"] == "simple_bounded_decoder"
    assert best["best_by_gradient_error"]["decoder_name"] == "unet_decoder"
    assert "best_by_visual_score" in best
    assert isinstance(best["pareto_candidates"], list)

    report = paths["report"].read_text(encoding="utf-8")
    assert "Protocol V4 shifts the objective from metric-only comparison to visual-quality-driven model selection" in report
    assert "DINOv2-LoRA remains a limited-seed probe" in report
    assert "VisionFM improves FWI generalization" not in report
