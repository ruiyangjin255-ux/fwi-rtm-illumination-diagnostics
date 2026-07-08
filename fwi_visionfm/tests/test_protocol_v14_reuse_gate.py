from __future__ import annotations

import json
from pathlib import Path


def test_protocol_v14_reuse_gate_marks_only_exact_b0_runs_reusable(tmp_path: Path):
    from scripts.verify_v12_v13_reuse_for_v14 import verify_v12_v13_reuse_for_v14

    v12_root = tmp_path / "v12"
    v13_root = tmp_path / "v13"
    for root, method in ((v12_root, "dinov2_frozen"), (v13_root, "ncs2d_frozen")):
        run_dir = root / "runs" / "flatvel_a_to_curvevel_a" / method / "seed_0"
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text(
            json.dumps(
                {
                    "status": "SUCCESS",
                    "run_id": f"flatvel_a_to_curvevel_a__{method}__seed0",
                    "method_key": method,
                    "transfer_id": "flatvel_a_to_curvevel_a",
                    "source_family": "flatvel_a",
                    "target_family": "curvevel_a",
                    "seed": 0,
                    "bridge": "raw_envelope_spectrum3",
                    "shot_count": 5,
                    "image_size": 224,
                    "decoder": "common_bounded_velocity_decoder",
                    "loss": "default_l1",
                    "epochs": 2,
                    "metric_space": "physical_velocity",
                    "manifest_combined_hash": "hash-a",
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "model_card.json").write_text(
            json.dumps({"decoder_fully_registered": True, "optimizer_parameters": 10, "trainable_parameters": 10, "is_real_feature": method == "ncs2d_frozen"}),
            encoding="utf-8",
        )
        (run_dir / "predictions_cross_family_test.npz").write_bytes(b"fake")
    config = tmp_path / "protocol_v14.yaml"
    config.write_text("epochs: 2\nshot_count: 5\nimage_size: 224\ndecoder: common_bounded_velocity_decoder\nloss: default_l1\nmetric_space: physical_velocity\n", encoding="utf-8")
    out = tmp_path / "reuse"
    payload = verify_v12_v13_reuse_for_v14(v12_root=v12_root, v13_root=v13_root, config_path=config, output_dir=out)
    assert payload["total_runs"] == 2
    assert payload["reusable_count"] == 2
    assert (out / "reusable_runs.csv").is_file()
