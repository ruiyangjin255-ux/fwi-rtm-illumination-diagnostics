from pathlib import Path

import numpy as np
import pytest

from scripts.run_protocol_v13_ncs2d_frozen import validate_real_feature_cache


def test_real_ncs_cache_is_required(tmp_path: Path) -> None:
    real = tmp_path / "real.npz"; np.savez(real, features=np.ones((2, 4)), target=np.ones((2, 1, 70, 70)), sample_id=np.array(["a", "b"]), is_real_feature=np.asarray(True))
    metadata = validate_real_feature_cache(real)
    assert metadata["is_real_feature"] is True
    fallback = tmp_path / "fallback.npz"; np.savez(fallback, features=np.ones((2, 4)), target=np.ones((2, 1, 70, 70)), sample_id=np.array(["a", "b"]), is_real_feature=np.asarray(False))
    with pytest.raises(ValueError, match="real NCS2D"):
        validate_real_feature_cache(fallback)

