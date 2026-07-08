from __future__ import annotations

import csv
import json
from pathlib import Path


def test_integrated_summary_writes_visual_and_control_ranks(tmp_path: Path):
    from fwi_visionfm.scripts.summarize_protocol_v4_integrated import write_integrated_summary

    for index, (bridge, mae, grad) in enumerate(
        [("raw_repeat3", 10.0, 2.0), ("raw_spectrogram", 8.0, 5.0), ("spectrogram_multiband", 9.0, 4.0)]
    ):
        run_dir = tmp_path / "runs" / bridge / f"seed_{index}"
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text(
            json.dumps(
                {
                    "protocol": "protocol_v4_integrated_bridge_visual_search",
                    "model_name": "cnn_baseline",
                    "bridge": bridge,
                    "decoder_name": "unet_decoder",
                    "loss_name": "default_l1",
                    "seed": index,
                    "status": "SUCCESS",
                    "is_probe": False,
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "metrics_cross_family_test.json").write_text(
            json.dumps({"mae": mae, "rmse": mae + 1.0, "ssim": 0.5, "gradient_error": grad, "edge_mae": grad + 1.0}),
            encoding="utf-8",
        )

    path = write_integrated_summary(tmp_path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert {row["bridge"] for row in rows} >= {"raw_repeat3", "raw_spectrogram"}
    assert rows[0]["visual_rank"] != ""
    assert rows[0]["numerical_rank"] != ""
    assert rows[0]["structural_rank"] != ""
    raw = next(row for row in rows if row["bridge"] == "raw_repeat3")
    assert raw["is_structural_control"] == "True"


def test_integrated_entries_keep_raw_repeat3_and_mark_dino_probe():
    from fwi_visionfm.scripts.run_protocol_v4_integrated_visual_search import integrated_entries

    entries = integrated_entries()
    assert any(entry["bridge"] == "raw_repeat3" for entry in entries)
    dino_entries = [entry for entry in entries if entry["model_name"] == "dinov2_lora_smoke"]
    assert dino_entries
    assert all(entry.get("is_probe") for entry in dino_entries)
