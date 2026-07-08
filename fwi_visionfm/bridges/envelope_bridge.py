from __future__ import annotations

import numpy as np


def envelope_bridge(records: np.ndarray, source_positions: np.ndarray | None = None) -> np.ndarray:
    if records.ndim != 4:
        raise ValueError(f"envelope_bridge expects (batch, shots, receivers, time), got {records.shape}")
    envelope = np.abs(records).astype(np.float32)
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    padded = np.pad(envelope, ((0, 0), (0, 0), (0, 0), (1, 1)), mode="edge")
    smoothed = (
        kernel[0] * padded[..., :-2]
        + kernel[1] * padded[..., 1:-1]
        + kernel[2] * padded[..., 2:]
    )
    return smoothed[:, :, None, :, :]
