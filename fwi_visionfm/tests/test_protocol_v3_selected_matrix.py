from __future__ import annotations

import json
from pathlib import Path


def _write_manifest(root: Path, source: str, target: str, seed: int) -> None:
    manifest_dir = root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / f"{source}_to_{target}_seed{seed}_manifest.json").write_text(
        json.dumps(
            {
                "source_family": source,
                "target_family": target,
                "seed": seed,
                "train_samples": [{"path": "sample_0.npz"}],
                "val_samples": [{"path": "sample_0.npz"}],
                "in_family_test_samples": [{"path": "sample_0.npz"}],
                "cross_family_test_samples": [{"path": "sample_0.npz"}],
                "stats_path": str(root / "stats.json"),
            }
        ),
        encoding="utf-8",
    )


def test_selected_matrix_expands_only_requested_configs(monkeypatch, tmp_path: Path):
    from fwi_visionfm.scripts import run_protocol_v3_selected_matrix as selected

    source = "flatvel_a_subset2k"
    target = "curvevel_a_subset500"
    calls: list[tuple[str, str, str, str, int]] = []

    def fake_build_splits(*, output_root, seeds, **_kwargs):
        for seed in seeds:
            _write_manifest(Path(output_root), source, target, int(seed))

    def fake_run_single(*, model_name, bridge, decoder_name, loss_name, manifest, **_kwargs):
        calls.append((model_name, bridge, decoder_name, loss_name, int(manifest["seed"])))
        return {
            "status": "SUCCESS",
            "metric_space": "physical_velocity",
            "is_probe": model_name.startswith("dinov2"),
            "actual_epochs": 1 if model_name.startswith("dinov2") else 3,
        }

    monkeypatch.setattr(selected, "build_protocol_v2_splits", fake_build_splits)
    monkeypatch.setattr(selected, "_run_single", fake_run_single)

    summary = selected.run_protocol_v3_selected_matrix(
        data_root=tmp_path / "data",
        output_root=tmp_path / "out",
        source=source,
        target=target,
        seeds=[0, 1, 2],
        train_size=300,
        val_size=100,
        test_size=100,
        epochs=3,
        device="cpu",
    )

    assert summary["run_count"] == 24
    assert len(calls) == 24
    assert len({call[:4] for call in calls}) == 8
    assert ("vit_tiny_scratch", "raw_spectrogram", "unet_decoder", "structure_loss", 2) in calls
    assert ("dinov2_lora_smoke", "raw_spectrogram", "unet_decoder", "default_l1", 1) in calls
    assert (tmp_path / "out" / "matrix_run_summary.json").exists()
