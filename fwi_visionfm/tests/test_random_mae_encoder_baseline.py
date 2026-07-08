from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def test_random_same_architecture_encoder_cache_writes_features(tmp_path: Path):
    from fwi_visionfm.scripts.cache_random_mae_encoder_features import write_random_mae_feature_cache

    sample_root = Path("D:/ryjin/fwi_visionfm/data/flatvel_a_subset2k")
    rows = []
    for index, path in enumerate(sorted(sample_root.glob("sample_*.npz"))[:6]):
        rows.append({"path": str(path), "data_file": str(path), "model_file": str(path), "local_index": 0, "global_index": index})
    manifest = {
        "source_family": "flatvel_a_subset2k",
        "target_family": "curvevel_a_subset500",
        "seed": 0,
        "train_samples": rows[:2],
        "val_samples": rows[2:4],
        "in_family_test_samples": rows[4:5],
        "cross_family_test_samples": rows[5:6],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "cache"
    result = write_random_mae_feature_cache(manifest_path=manifest_path, output_root=out, bridge="raw_envelope_spectrum3", device="cpu")
    assert result["status"] == "SUCCESS"
    payload = torch.load(out / "val_features.pt", map_location="cpu")
    assert "features" in payload
    assert "sample_ids" in payload

