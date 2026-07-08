import numpy as np

from fwi_visionfm.pasd.bootstrap import aligned_metric_difference


def test_phase3_dual_target_archives_require_identical_sample_ids(tmp_path):
    base = tmp_path / "base.npz"
    candidate = tmp_path / "candidate.npz"
    target = np.zeros((3, 4, 4), dtype=np.float32)
    np.savez_compressed(base, sample_id=np.array([1, 2, 3]), prediction=target, target=target)
    np.savez_compressed(candidate, sample_id=np.array([1, 2, 4]), prediction=target, target=target)
    try:
        aligned_metric_difference(base, candidate, "mae")
    except ValueError as exc:
        assert "identical sample_id" in str(exc)
    else:
        raise AssertionError("Expected incomplete archive alignment to fail")
