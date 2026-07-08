from __future__ import annotations

import json
from pathlib import Path

import torch


def test_local_mae_feature_cache_contains_sample_id_and_feature_shape(tmp_path: Path):
    from fwi_visionfm.scripts.cache_local_mae_features import write_local_mae_feature_cache

    sample_root = Path("D:/ryjin/fwi_visionfm/data/flatvel_a_subset2k")
    manifest_rows = []
    for index, path in enumerate(sorted(sample_root.glob("sample_*.npz"))[:6]):
        manifest_rows.append({"path": str(path), "data_file": str(path), "model_file": str(path), "local_index": 0, "global_index": index})

    manifest = {
        "source_family": "flatvel_a_subset2k",
        "target_family": "curvevel_a_subset500",
        "seed": 0,
        "train_samples": manifest_rows[:2],
        "val_samples": manifest_rows[2:4],
        "in_family_test_samples": manifest_rows[4:5],
        "cross_family_test_samples": manifest_rows[5:6],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    ckpt_path = tmp_path / "best_mae_encoder.pt"
    torch.save({"model_state": {}, "bridge": "raw_envelope_spectrum3"}, ckpt_path)

    output_dir = tmp_path / "cache"
    result = write_local_mae_feature_cache(
        manifest_path=manifest_path,
        checkpoint_path=ckpt_path,
        output_root=output_dir,
        bridge="raw_envelope_spectrum3",
        device="cpu",
    )
    assert result["status"] == "SUCCESS"
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["encoder_type"] == "local_seismic_mae"
    assert metadata["metric_space"] == "physical_velocity"
    payload = torch.load(output_dir / "val_features.pt", map_location="cpu")
    assert "sample_ids" in payload
    assert "features" in payload

