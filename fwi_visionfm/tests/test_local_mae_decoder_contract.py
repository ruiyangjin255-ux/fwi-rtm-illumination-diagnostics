from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def _write_cache(path: Path, name: str, n: int) -> None:
    rng = np.random.default_rng(abs(hash(name)) % 10000)
    torch.save(
        {
            "features": torch.as_tensor(rng.normal(size=(n, 64)).astype(np.float32)),
            "target": torch.as_tensor((1500.0 + 3000.0 * rng.random(size=(n, 8, 8))).astype(np.float32)),
            "records_preview": torch.as_tensor(rng.normal(size=(n, 5, 18, 14)).astype(np.float32)),
            "sample_ids": [f"{name}_{i}" for i in range(n)],
            "target_shape": [8, 8],
        },
        path,
    )


def test_local_mae_decoder_outputs_predictions_val_npz(tmp_path: Path):
    from fwi_visionfm.scripts.train_local_mae_feature_decoder import train_local_mae_decoder_from_cache

    cache_root = tmp_path / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    _write_cache(cache_root / "train_features.pt", "train", 4)
    _write_cache(cache_root / "val_features.pt", "val", 2)
    _write_cache(cache_root / "in_family_test_features.pt", "in", 2)
    _write_cache(cache_root / "cross_family_test_features.pt", "cross", 2)
    (cache_root / "metadata.json").write_text(
        json.dumps({"bridge": "raw_envelope_spectrum3", "encoder_type": "local_seismic_mae", "metric_space": "physical_velocity", "is_pretrained": True}),
        encoding="utf-8",
    )

    run_dir = tmp_path / "run"
    result = train_local_mae_decoder_from_cache(
        cache_root=cache_root,
        output_dir=run_dir,
        decoder_name="unet_decoder",
        loss_name="default_l1",
        epochs=1,
        batch_size=2,
        device="cpu",
    )
    assert result["status"] == "SUCCESS"
    with np.load(run_dir / "predictions_val.npz", allow_pickle=True) as payload:
        assert "sample_id" in payload.files
        assert "metric_space" in payload.files
        assert "velocity_pred_physical" in payload.files

