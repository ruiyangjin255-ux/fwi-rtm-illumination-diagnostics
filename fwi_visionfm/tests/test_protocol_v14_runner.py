from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def test_protocol_v14_grid_png_is_rendered_from_predictions(tmp_path: Path) -> None:
    from PIL import Image

    from scripts.run_protocol_v14_geometry_aware_trace_bridge import _write_grid_png

    prediction = np.linspace(1800.0, 4200.0, 3 * 70 * 70, dtype=np.float32).reshape(3, 1, 70, 70)
    target = np.linspace(2000.0, 4000.0, 3 * 70 * 70, dtype=np.float32).reshape(3, 1, 70, 70)
    npz_path = tmp_path / "predictions_cross_family_test.npz"
    np.savez_compressed(
        npz_path,
        velocity_pred_physical=prediction,
        velocity_true_physical=target,
        error_map_physical=prediction - target,
        sample_id=np.asarray(["a", "b", "c"], dtype=str),
        metric_space=np.asarray("physical_velocity"),
    )

    output_path = tmp_path / "prediction_grid.png"
    _write_grid_png(output_path, npz_path=npz_path, gradient=False)

    assert output_path.is_file()
    image = Image.open(output_path)
    assert image.size[0] > 32
    assert image.size[1] > 32


def test_protocol_v14_runner_reuses_b0_and_trains_non_b0(tmp_path: Path, monkeypatch) -> None:
    from scripts.run_protocol_v14_geometry_aware_trace_bridge import run_protocol_v14_geometry_aware_trace_bridge

    protocol_root = tmp_path / "protocol_v14"
    protocol_root.mkdir()
    manifests = protocol_root / "manifests"
    manifests.mkdir()
    manifest = {
        "source_family": "flatvel_a",
        "target_family": "curvevel_a",
        "seed": 0,
        "manifest_combined_hash": "hash-1",
        "train_samples": [],
        "val_samples": [],
        "in_family_test_samples": [],
        "cross_family_test_samples": [],
        "stats_path": str(tmp_path / "stats.json"),
    }
    (tmp_path / "stats.json").write_text(json.dumps({"velocity": {"min": 1500.0, "max": 4500.0}}), encoding="utf-8")
    (manifests / "flatvel_a_to_curvevel_a_seed0_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    matrix_path = protocol_root / "protocol_v14_run_matrix.csv"
    with matrix_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "transfer_id",
                "source_family",
                "target_family",
                "seed",
                "method_id",
                "method_key",
                "method_name",
                "bridge_id",
                "bridge_name",
                "geometry_mode",
                "trace_context_radius",
                "use_shot_global_context",
                "use_multiscale_context",
                "shot_count",
                "image_size",
                "decoder",
                "loss",
                "epochs",
                "metric_space",
                "geometry_provenance",
                "status",
                "skip_reason",
                "reused_from",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "run_id": "flatvel_a_to_curvevel_a__dinov2_frozen__B0__seed0",
                "transfer_id": "flatvel_a_to_curvevel_a",
                "source_family": "flatvel_a",
                "target_family": "curvevel_a",
                "seed": 0,
                "method_id": "M3",
                "method_key": "dinov2_frozen",
                "method_name": "DINOv2 frozen",
                "bridge_id": "B0",
                "bridge_name": "raw_envelope_spectrum3",
                "geometry_mode": "none",
                "trace_context_radius": 0,
                "use_shot_global_context": False,
                "use_multiscale_context": False,
                "shot_count": 5,
                "image_size": 224,
                "decoder": "common_bounded_velocity_decoder",
                "loss": "default_l1",
                "epochs": 2,
                "metric_space": "physical_velocity",
                "geometry_provenance": "CANONICAL_RECONSTRUCTED",
                "status": "REUSE_GATE_PASSED",
                "skip_reason": "",
                "reused_from": str(tmp_path / "reused" / "source_run"),
            }
        )
        writer.writerow(
            {
                "run_id": "flatvel_a_to_curvevel_a__dinov2_frozen__B1__seed0",
                "transfer_id": "flatvel_a_to_curvevel_a",
                "source_family": "flatvel_a",
                "target_family": "curvevel_a",
                "seed": 0,
                "method_id": "M3",
                "method_key": "dinov2_frozen",
                "method_name": "DINOv2 frozen",
                "bridge_id": "B1",
                "bridge_name": "geometry_aware_trace_bridge_geometry",
                "geometry_mode": "trace_geometry_only",
                "trace_context_radius": 0,
                "use_shot_global_context": False,
                "use_multiscale_context": False,
                "shot_count": 5,
                "image_size": 224,
                "decoder": "common_bounded_velocity_decoder",
                "loss": "default_l1",
                "epochs": 2,
                "metric_space": "physical_velocity",
                "geometry_provenance": "CANONICAL_RECONSTRUCTED",
                "status": "PENDING",
                "skip_reason": "",
                "reused_from": "",
            }
        )
    reused = tmp_path / "reused" / "source_run"
    reused.mkdir(parents=True)
    for name in [
        "config.json",
        "config_hash.txt",
        "model_card.json",
        "train_history.csv",
        "metrics_val.json",
        "metrics_in_family_test.json",
        "metrics_cross_family_test.json",
        "predictions_in_family_test.npz",
        "predictions_cross_family_test.npz",
        "prediction_grid.png",
        "gradient_grid.png",
        "run_log.txt",
    ]:
        path = reused / name
        if name.endswith(".json"):
            path.write_text("{}", encoding="utf-8")
        elif name.endswith(".csv"):
            path.write_text("epoch,train_l1\n1,1.0\n", encoding="utf-8")
        elif name.endswith(".npz"):
            np.savez(path, prediction=np.zeros((1, 1, 70, 70), dtype=np.float32), target=np.zeros((1, 1, 70, 70), dtype=np.float32))
        else:
            path.write_text("ok\n", encoding="utf-8")
    config_path = tmp_path / "protocol_v14.yaml"
    config_path.write_text(
        "\n".join(
            [
                "protocol: protocol_v14_geometry_aware_trace_bridge",
                f"manifest_root: {manifests.as_posix()}",
                "seeds: [0]",
                "shot_count: 5",
                "image_size: 224",
                "epochs: 2",
                "batch_size: 2",
                "learning_rate: 0.001",
                "velocity_shape: [70, 70]",
                "decoder: common_bounded_velocity_decoder",
                "decoder_base_channels: 16",
                "loss: default_l1",
                "metric_space: physical_velocity",
                "backbones:",
                "  dinov2_frozen:",
                "    method_name: DINOv2 frozen",
                "    backbone_type: dummy",
                "    model_name: vit_tiny_patch16_224",
            ]
        ),
        encoding="utf-8",
    )

    call_count = {"value": 0}

    def fake_train_run(**kwargs):
        call_count["value"] += 1
        run_dir = kwargs["run_dir"]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps({"status": "SUCCESS"}), encoding="utf-8")
        (run_dir / "config_hash.txt").write_text("hash\n", encoding="utf-8")
        (run_dir / "model_card.json").write_text(json.dumps({"decoder_fully_registered": True, "optimizer_parameters": 1, "trainable_parameters": 1}), encoding="utf-8")
        (run_dir / "geometry_metadata.json").write_text(json.dumps({"geometry_provenance": "CANONICAL_RECONSTRUCTED"}), encoding="utf-8")
        (run_dir / "train_history.csv").write_text("epoch,train_l1\n1,1.0\n", encoding="utf-8")
        for filename in ["metrics_val.json", "metrics_in_family_test.json", "metrics_cross_family_test.json"]:
            (run_dir / filename).write_text(json.dumps({"metric_space": "physical_velocity"}), encoding="utf-8")
        for filename in ["predictions_in_family_test.npz", "predictions_cross_family_test.npz"]:
            np.savez(run_dir / filename, velocity_pred_physical=np.zeros((1, 1, 70, 70), dtype=np.float32), velocity_true_physical=np.zeros((1, 1, 70, 70), dtype=np.float32), error_map_physical=np.zeros((1, 1, 70, 70), dtype=np.float32), sample_id=np.array(["a"]), model_id=np.array("M3"), bridge_name=np.array("B1"), geometry_mode=np.array("trace_geometry_only"), geometry_provenance=np.array("CANONICAL_RECONSTRUCTED"), trace_context_radius=np.array(0), use_shot_global_context=np.array(False), use_multiscale_context=np.array(False), source_family=np.array("flatvel_a"), target_family=np.array("curvevel_a"), seed=np.array(0), metric_space=np.array("physical_velocity"), is_real_feature=np.array(True))
        (run_dir / "prediction_grid.png").write_bytes(b"png")
        (run_dir / "gradient_grid.png").write_bytes(b"png")
        (run_dir / "run_log.txt").write_text("ok\n", encoding="utf-8")
        return {"status": "SUCCESS"}

    monkeypatch.setattr("scripts.run_protocol_v14_geometry_aware_trace_bridge._train_single_v14_run", fake_train_run)
    result = run_protocol_v14_geometry_aware_trace_bridge(config_path=config_path, output_dir=protocol_root, stage="screening", seeds=[0], device="cpu", resume=True)
    assert result["success"] == 2
    assert call_count["value"] == 1
    assert (protocol_root / "runs" / "flatvel_a_to_curvevel_a" / "dinov2_frozen" / "seed_0" / "B0" / "config.json").is_file()
    assert (protocol_root / "runs" / "flatvel_a_to_curvevel_a" / "dinov2_frozen" / "seed_0" / "B1" / "geometry_metadata.json").is_file()
    second = run_protocol_v14_geometry_aware_trace_bridge(config_path=config_path, output_dir=protocol_root, stage="screening", seeds=[0], device="cpu", resume=True)
    assert second["success"] == 2
    assert call_count["value"] == 1


def test_protocol_v14_trace_feature_extraction_is_chunked(monkeypatch) -> None:
    import torch

    from scripts.run_protocol_v14_geometry_aware_trace_bridge import _extract_trace_features_for_method

    call_sizes: list[int] = []

    class FakeBridge:
        def __call__(self, x):
            batch = x.shape[0]
            return torch.ones((batch, 3, 16, 16), dtype=torch.float32)

    class FakeBackbone:
        def to(self, device):
            return self

        def __call__(self, images):
            call_sizes.append(int(images.shape[0]))
            return torch.ones((images.shape[0], 4, 8), dtype=torch.float32)

    monkeypatch.setattr("scripts.run_protocol_v14_geometry_aware_trace_bridge.SeismicToVisionBridge", lambda **kwargs: FakeBridge())
    monkeypatch.setattr("scripts.run_protocol_v14_geometry_aware_trace_bridge.build_vision_backbone", lambda **kwargs: FakeBackbone())
    records = np.zeros((100, 5, 70, 20), dtype=np.float32)
    config = {"image_size": 16, "backbones": {"dinov2_frozen": {"backbone_type": "dummy", "model_name": "dummy"}}}
    features = _extract_trace_features_for_method(
        method_key="dinov2_frozen",
        bridge_name="raw_envelope_spectrum3",
        records=records,
        config=config,
        device="cpu",
    )
    assert features.shape == (100, 5, 70, 8)
    assert len(call_sizes) > 1


def test_protocol_v14_cache_key_is_shared_across_b1_b2_b3() -> None:
    from scripts.run_protocol_v14_geometry_aware_trace_bridge import _trace_cache_dir

    b1 = _trace_cache_dir(root=Path("out"), transfer_id="flatvel_a_to_curvevel_a", method_key="dinov2_frozen", seed=0)
    b2 = _trace_cache_dir(root=Path("out"), transfer_id="flatvel_a_to_curvevel_a", method_key="dinov2_frozen", seed=0)
    assert b1 == b2
    assert "B1" not in str(b1)
