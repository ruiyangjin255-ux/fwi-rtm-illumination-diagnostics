import numpy as np

from fwi_visionfm.pasd.corrected_metrics import corrected_sample_metrics


def test_phase3r_edge_mask_is_strict_greater_than_tau():
    target = np.zeros((4, 4), dtype=np.float32)
    target[:, 2:] = 1.0
    pred = target.copy()
    _, coverage_equal_tau = corrected_sample_metrics(pred, target, tau_source=1.0, tau_pred=1.0)
    _, coverage_below_tau = corrected_sample_metrics(pred, target, tau_source=0.999, tau_pred=0.999)
    assert coverage_equal_tau["edge_pixels"] == 0
    assert coverage_below_tau["edge_pixels"] > 0
