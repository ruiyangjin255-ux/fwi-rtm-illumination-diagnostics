import numpy as np
import pytest

from fwi_visionfm.pasd.protocol import DatasetRef, ProtocolManifest, build_protocol, load_protocol, load_protocol_bundles


def test_protocol_keeps_source_splits_disjoint_and_target_isolated(tmp_path):
    src_records = np.zeros((10, 3, 16, 8), dtype=np.float32)
    src_models = np.zeros((10, 12, 12), dtype=np.float32)
    tgt_records = np.ones((6, 3, 16, 8), dtype=np.float32)
    tgt_models = np.ones((6, 12, 12), dtype=np.float32)
    src_r = tmp_path / "src_records.npy"
    src_m = tmp_path / "src_models.npy"
    tgt_r = tmp_path / "tgt_records.npy"
    tgt_m = tmp_path / "tgt_models.npy"
    np.save(src_r, src_records)
    np.save(src_m, src_models)
    np.save(tgt_r, tgt_records)
    np.save(tgt_m, tgt_models)
    manifest = build_protocol(
        DatasetRef(str(src_r), str(src_m), "Flat"),
        DatasetRef(str(tgt_r), str(tgt_m), "Curve"),
        train_size=5,
        val_size=2,
        in_family_test_size=3,
        cross_family_test_size=3,
        seed=5,
    )
    path = manifest.save(tmp_path / "protocol.json")
    loaded = load_protocol(path)
    source, target = load_protocol_bundles(loaded)
    assert source.family == "Flat"
    assert target is not None and target.family == "Curve"
    assert len(loaded.train_indices) == 5
    assert len(loaded.cross_family_test_indices) == 3


def test_protocol_rejects_source_split_leakage():
    with pytest.raises(ValueError, match="Split leakage"):
        ProtocolManifest(
            version="x",
            source=DatasetRef("records.npy", "models.npy", "Flat"),
            target=None,
            train_indices=(0, 1),
            val_indices=(1, 2),
            in_family_test_indices=(3,),
        ).validate_structure()
