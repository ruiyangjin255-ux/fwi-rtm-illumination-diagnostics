import numpy as np

from fwi_visionfm.pasd.corrected_metrics import corrected_sample_metrics


def test_phase3r_corrected_metrics_use_physical_arrays():
    target = np.zeros((6, 6), dtype=np.float32)
    pred = target + 2.0
    metrics, coverage = corrected_sample_metrics(pred, target, tau_source=0.1, tau_pred=0.1)
    assert metrics["MAE"] == 2.0
    assert metrics["RMSE"] == 2.0
    assert coverage["mask_condition"] == "strict_gt"
