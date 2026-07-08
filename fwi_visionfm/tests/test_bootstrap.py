import numpy as np

from fwi_visionfm.pasd.bootstrap import paired_bootstrap


def test_paired_bootstrap_requires_and_uses_aligned_sample_ids(tmp_path):
    target = np.ones((4, 8, 8), dtype=np.float32)
    baseline = target - 0.5
    candidate = target - 0.1
    base_path = tmp_path / "baseline.npz"
    cand_path = tmp_path / "candidate.npz"
    np.savez_compressed(base_path, sample_id=np.array([4, 2, 1, 3]), prediction=baseline, target=target)
    np.savez_compressed(cand_path, sample_id=np.array([1, 2, 3, 4]), prediction=candidate[[2, 1, 3, 0]], target=target[[2, 1, 3, 0]])
    result = paired_bootstrap(base_path, cand_path, metric="mae", n_resamples=100, seed=0)
    assert result["n_samples"] == 4
    assert result["candidate_minus_baseline_mean"] < 0.0
    assert result["improvement_probability"] == 1.0
