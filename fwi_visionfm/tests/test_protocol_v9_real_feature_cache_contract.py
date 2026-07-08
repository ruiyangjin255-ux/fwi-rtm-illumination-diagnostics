from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _write_sample(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    records = rng.normal(size=(5, 18, 14)).astype(np.float32)
    velocity = (1500.0 + 3000.0 * rng.random(size=(8, 10))).astype(np.float32)
    source_positions = np.linspace(0.12, 0.88, 5, dtype=np.float32)
    np.savez(path, records=records, velocity=velocity, source_positions=source_positions)


def test_protocol_v9_real_and_fallback_cache_contract(tmp_path: Path):
    from fwi_visionfm.scripts.cache_v9_real_frozen_features import write_feature_npz

    rows = []
    for idx in range(2):
        sample = tmp_path / f"sample_{idx}.npz"
        _write_sample(sample, idx)
        rows.append({"path": str(sample), "data_file": str(sample), "local_index": 0})

    out_real = tmp_path / "real.npz"
    meta_real = write_feature_npz(
        rows=rows,
        output_path=out_real,
        backbone_name="ncs_2d",
        bridge_name="raw_envelope_spectrum3",
        tokenizer_name="vit_pixel_values",
        feature_builder=lambda _: np.ones((32,), dtype=np.float32),
        source_split="train",
        target_split="curvevel_a_subset500",
        status="AVAILABLE",
        is_real_feature=True,
    )
    with np.load(out_real, allow_pickle=True) as payload:
        assert payload["is_real_feature"].item() == True
        assert "sample_id" in payload
    assert meta_real["is_real_feature"] is True

    out_fallback = tmp_path / "fallback.npz"
    meta_fallback = write_feature_npz(
        rows=rows,
        output_path=out_fallback,
        backbone_name="fallback_random",
        bridge_name="raw_envelope_spectrum3",
        tokenizer_name="fallback_tokenizer",
        feature_builder=lambda _: np.zeros((8,), dtype=np.float32),
        source_split="train",
        target_split="curvevel_a_subset500",
        status="FALLBACK_FEATURE_ONLY",
        is_real_feature=False,
    )
    with np.load(out_fallback, allow_pickle=True) as payload:
        assert payload["is_real_feature"].item() == False
    assert meta_fallback["status"] == "FALLBACK_FEATURE_ONLY"

