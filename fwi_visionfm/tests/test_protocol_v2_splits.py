from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _write_family(root: Path, family: str, samples: int = 12) -> None:
    data_dir = root / family / "data"
    model_dir = root / family / "model"
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    offset = float(len(family))
    records = (np.arange(samples * 5 * 10 * 7, dtype=np.float32).reshape(samples, 5, 10, 7) + offset)
    velocity = (np.arange(samples * 1 * 6 * 7, dtype=np.float32).reshape(samples, 1, 6, 7) + offset)
    np.save(data_dir / "data1.npy", records)
    np.save(model_dir / "model1.npy", velocity)


def _sample_ids(samples: list[dict]) -> set[tuple[str, int]]:
    return {(row["data_file"], int(row["local_index"])) for row in samples}


def test_protocol_v2_split_builder_writes_manifest_stats_and_nonoverlapping_source_splits(tmp_path: Path):
    from fwi_visionfm.scripts.build_protocol_v2_splits import build_protocol_v2_splits

    data_root = tmp_path / "OpenFWI"
    _write_family(data_root, "FlatVel_A")
    _write_family(data_root, "CurveVel_A")
    _write_family(data_root, "FlatFault_A")

    summary = build_protocol_v2_splits(
        data_root=data_root,
        output_root=tmp_path / "outputs",
        train_size=4,
        val_size=2,
        test_size=2,
        seeds=[0],
    )

    assert summary["fault_family"] == "FlatFault_A"
    assert summary["manifest_count"] == 3
    manifest_path = Path(summary["manifests"][0])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert {
        "train_samples",
        "val_samples",
        "in_family_test_samples",
        "cross_family_test_samples",
        "source_family",
        "target_family",
        "seed",
        "data_shape",
        "velocity_shape",
        "stats_path",
        "split_files",
    }.issubset(manifest)
    assert len(manifest["train_samples"]) == 4
    assert len(manifest["val_samples"]) == 2
    assert len(manifest["in_family_test_samples"]) == 2
    assert len(manifest["cross_family_test_samples"]) == 2
    assert manifest["data_shape"] == [5, 10, 7]
    assert manifest["velocity_shape"] == [1, 6, 7]
    train_ids = _sample_ids(manifest["train_samples"])
    val_ids = _sample_ids(manifest["val_samples"])
    in_ids = _sample_ids(manifest["in_family_test_samples"])
    cross_ids = _sample_ids(manifest["cross_family_test_samples"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(in_ids)
    assert val_ids.isdisjoint(in_ids)
    if manifest["source_family"] == manifest["target_family"]:
        assert train_ids.isdisjoint(cross_ids)
    stats = json.loads(Path(manifest["stats_path"]).read_text(encoding="utf-8"))
    assert stats["stats_split"] == "source_train_only"
    assert stats["sample_count"] == 4
    assert all(manifest["source_family"] in path for path in stats["source_files"])
    assert all(Path(path).exists() for path in manifest["split_files"].values())

