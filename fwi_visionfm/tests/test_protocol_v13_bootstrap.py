import numpy as np

from scripts.bootstrap_protocol_v13_pretraining_source import paired_bootstrap_metric_deltas


def test_bootstrap_aligns_target_sample_ids() -> None:
    target = np.stack([np.arange(9).reshape(1, 3, 3), np.arange(9).reshape(1, 3, 3) * 2]).astype(np.float32)
    candidate = target * .9; baseline = target * .3
    result = paired_bootstrap_metric_deltas(candidate_prediction=candidate, candidate_target=target, candidate_ids=["a", "b"], baseline_prediction=baseline[::-1], baseline_target=target[::-1], baseline_ids=["b", "a"], n_bootstrap=100, seed=0)
    assert result["paired"] is True and result["aligned_sample_count"] == 2
    assert result["mae_mean_difference"] < 0

