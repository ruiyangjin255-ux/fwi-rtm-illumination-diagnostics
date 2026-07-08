from __future__ import annotations

import numpy as np


def raw_bridge(records: np.ndarray, source_positions: np.ndarray | None = None) -> np.ndarray:
    if records.ndim != 4:
        raise ValueError(f"raw_bridge expects (batch, shots, receivers, time), got {records.shape}")
    scale = np.maximum(np.max(np.abs(records), axis=(-1, -2), keepdims=True), 1.0e-6)
    return (records / scale)[:, :, None, :, :].astype(np.float32)
