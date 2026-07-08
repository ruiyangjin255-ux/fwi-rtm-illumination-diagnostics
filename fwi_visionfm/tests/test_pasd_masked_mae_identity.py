import numpy as np

from fwi_visionfm.pasd.diagnostics import masked_mae_identity


def test_masked_mae_identity_is_exact_for_binary_partition():
    target = np.zeros((4, 4), dtype=np.float32)
    prediction = np.ones((4, 4), dtype=np.float32)
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2] = True
    out = masked_mae_identity(prediction, target, mask)
    assert out["edge_coverage"] == 0.5
    assert out["weighted_identity_error"] <= 1e-7
