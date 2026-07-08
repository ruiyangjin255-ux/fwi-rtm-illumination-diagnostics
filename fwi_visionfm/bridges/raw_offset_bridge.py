from __future__ import annotations

import numpy as np

from fwi_visionfm.bridges.raw_bridge import raw_bridge


def raw_offset_bridge(records: np.ndarray, source_positions: np.ndarray) -> np.ndarray:
    if records.ndim != 4:
        raise ValueError(f"raw_offset_bridge expects (batch, shots, receivers, time), got {records.shape}")
    batch, shots, receivers, time = records.shape
    if source_positions.shape != (batch, shots):
        raise ValueError(f"source_positions must have shape {(batch, shots)}, got {source_positions.shape}")
    raw = raw_bridge(records)[:, :, 0]
    receiver_positions = np.linspace(0.0, 1.0, receivers, dtype=np.float32)[None, None, :, None]
    offsets = receiver_positions - source_positions.astype(np.float32)[:, :, None, None]
    offsets = np.broadcast_to(offsets, (batch, shots, receivers, time))
    return np.stack([raw, offsets], axis=2).astype(np.float32)
