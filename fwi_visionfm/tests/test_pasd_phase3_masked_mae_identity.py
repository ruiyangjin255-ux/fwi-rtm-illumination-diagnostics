import numpy as np

from fwi_visionfm.pasd.corrected_metrics import corrected_sample_metrics


def test_phase3r_masked_mae_identity_holds():
    target = np.zeros((5, 5), dtype=np.float32)
    target[:, 3:] = 3.0
    pred = target + 0.5
    metrics, coverage = corrected_sample_metrics(pred, target, tau_source=0.1, tau_pred=0.1)
    reconstructed = coverage["edge_coverage"] * metrics["source_threshold_edge_MAE"] + (1 - coverage["edge_coverage"]) * metrics["nonedge_MAE"]
    assert abs(metrics["MAE"] - reconstructed) <= 1e-6
