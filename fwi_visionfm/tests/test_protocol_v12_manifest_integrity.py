import csv
from pathlib import Path

from scripts.build_protocol_v12_manifests import compute_manifest_hashes, validate_locked_splits


def _write(path: Path, ids: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "path"])
        writer.writeheader()
        for sample_id in ids:
            writer.writerow({"sample_id": sample_id, "path": f"{sample_id}.npz"})


def test_manifest_hash_is_stable_and_splits_do_not_overlap(tmp_path: Path) -> None:
    _write(tmp_path / "flatvel_a_train200.csv", ["a", "b"])
    _write(tmp_path / "flatvel_a_val50.csv", ["c"])
    _write(tmp_path / "flatvel_a_test50.csv", ["d"])
    validate_locked_splits(tmp_path, ["flatvel_a"])
    first = compute_manifest_hashes(tmp_path)
    second = compute_manifest_hashes(tmp_path)
    assert first == second
    assert set(first) == {"flatvel_a_train200.csv", "flatvel_a_val50.csv", "flatvel_a_test50.csv"}

