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


def _write_manifest(tmp_path: Path) -> Path:
    sample_root = tmp_path / "samples"
    sample_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(6):
        path = sample_root / f"sample_{index}.npz"
        _write_sample(path, index)
        rows.append({"path": str(path), "data_file": str(path), "model_file": str(path), "local_index": 0, "global_index": index})
    stats_path = tmp_path / "train_stats.json"
    stats_path.write_text(json.dumps({"velocity": {"min": 1500.0, "max": 4500.0}}), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_family": "flatvel_a_subset2k",
                "target_family": "curvevel_a_subset500",
                "seed": 0,
                "stats_path": str(stats_path),
                "train_samples": rows[:2],
                "val_samples": rows[2:4],
                "in_family_test_samples": rows[4:5],
                "cross_family_test_samples": rows[5:6],
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_protocol_v8_feature_cache_contract_supports_fallback_and_fake_available_ncs(tmp_path: Path):
    from fwi_visionfm.scripts.cache_seismic_fm_features import cache_seismic_fm_features

    manifest_path = _write_manifest(tmp_path)

    fake_repo = tmp_path / "fake_ncs_repo"
    fake_repo.mkdir(parents=True, exist_ok=True)
    (fake_repo / "ncs.py").write_text(
        "\n".join(
            [
                "import numpy as np",
                "",
                "class FakeNCSModel:",
                "    def encode(self, tokens):",
                "        array = np.asarray(tokens, dtype=np.float32)",
                "        if array.ndim == 2:",
                "            return array.mean(axis=0)",
                "        if array.ndim == 3:",
                "            return array.mean(axis=(0, 1))",
                "        return array.reshape(-1)",
                "",
                "def load_model(variant='ncs_2d', device='cpu'):",
                "    return FakeNCSModel()",
            ]
        ),
        encoding="utf-8",
    )
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    (weights_dir / "fake_weights.pt").write_bytes(b"fake")

    output_dir = tmp_path / "cache"
    result = cache_seismic_fm_features(
        manifest_path=manifest_path,
        output_dir=output_dir,
        backbone_name="ncs_2d",
        bridge_name="raw_envelope_spectrum3",
        repo_path=fake_repo,
        weights_path=weights_dir,
        allow_fallback=False,
        device="cpu",
    )

    assert result["status"] in {"SUCCESS", "FALLBACK_FEATURE_ONLY", "SKIPPED_NCS_UNAVAILABLE"}
    assert (output_dir / "val_features.npz").exists()

    with np.load(output_dir / "val_features.npz", allow_pickle=True) as payload:
        assert "features" in payload
        assert "sample_id" in payload
        metadata = json.loads(str(payload["metadata_json"].item()))
        assert "status" in metadata

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] in {"SUCCESS", "FALLBACK_FEATURE_ONLY", "SKIPPED_NCS_UNAVAILABLE"}
    assert metadata["backbone_name"] == "ncs_2d"

