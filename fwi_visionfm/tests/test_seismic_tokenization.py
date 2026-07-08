from __future__ import annotations

import numpy as np


def test_seismic_tokenizers_return_expected_shapes():
    from fwi_visionfm.models.tokenizers.seismic_tokenization import (
        dummy_feature,
        tokenize_2d,
        tokenize_2p5d,
    )

    records = np.random.RandomState(0).randn(2, 16, 12).astype("float32")
    tokens_2d = tokenize_2d(records, patch_size=4)
    tokens_25d = tokenize_2p5d(records, patch_size=4)
    feature = dummy_feature(records, feature_dim=8)

    assert tokens_2d.ndim == 2
    assert tokens_25d.ndim == 3
    assert feature.shape == (8,)
    assert np.isfinite(tokens_2d).all()
    assert np.isfinite(tokens_25d).all()
    assert np.isfinite(feature).all()
