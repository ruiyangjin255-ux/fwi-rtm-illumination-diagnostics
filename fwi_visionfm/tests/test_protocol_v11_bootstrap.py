import numpy as np

from scripts.bootstrap_protocol_v11_comparisons import paired_bootstrap_mae_delta


def test_bootstrap_uses_paired_sample_ids_and_reports_negative_delta() -> None:
    target = np.zeros((4, 1, 2, 2), dtype=np.float32)
    candidate = np.full_like(target, 1.0)
    baseline = np.full_like(target, 2.0)
    result = paired_bootstrap_mae_delta(
        candidate_prediction=candidate,
        candidate_target=target,
        candidate_ids=["d", "b", "a", "c"],
        baseline_prediction=baseline[[2, 1, 3, 0]],
        baseline_target=target,
        baseline_ids=["a", "b", "c", "d"],
        n_bootstrap=200,
        seed=7,
    )
    assert result["aligned_sample_count"] == 4
    assert result["mean_difference"] < 0
    assert result["ci_high"] < 0
    assert result["win_probability"] == 1.0

