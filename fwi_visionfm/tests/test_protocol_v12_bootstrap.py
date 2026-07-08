import numpy as np

from scripts.bootstrap_protocol_v12_comparisons import paired_bootstrap_metric_deltas


def test_multimetric_bootstrap_aligns_sample_ids_before_pairing() -> None:
    base = np.arange(9, dtype=np.float32).reshape(1, 3, 3)
    target = np.stack([base, base * 2]).astype(np.float32)
    candidate = (target * 0.9).astype(np.float32)
    baseline = (target * 0.3).astype(np.float32)
    result = paired_bootstrap_metric_deltas(
        candidate_prediction=candidate,
        candidate_target=target,
        candidate_ids=["a", "b"],
        baseline_prediction=baseline[::-1],
        baseline_target=target[::-1],
        baseline_ids=["b", "a"],
        n_bootstrap=100,
        seed=0,
    )
    assert result["aligned_sample_count"] == 2
    assert result["paired"] is True
    for metric in ("mae", "rmse", "gradient_error", "edge_mae"):
        assert result[f"{metric}_mean_difference"] < 0
