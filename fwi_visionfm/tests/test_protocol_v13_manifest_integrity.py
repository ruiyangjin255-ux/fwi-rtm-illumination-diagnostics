from scripts.verify_v12_reuse_for_v13 import verify_manifest_hash


def test_manifest_hash_must_match_locked_value(tmp_path) -> None:
    path = tmp_path / "split.csv"; path.write_text("sample_id,path\na,a.npz\n", encoding="utf-8")
    current = verify_manifest_hash({"split.csv": path}, expected=None)
    assert verify_manifest_hash({"split.csv": path}, expected=current) == current

