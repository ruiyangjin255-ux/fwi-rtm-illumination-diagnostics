from __future__ import annotations

import numpy as np

from fwi_visionfm.config import BridgeConfig

Array = np.ndarray


def _normalize(channel: Array, mode: str) -> Array:
    data = np.asarray(channel, dtype=np.float32)
    if mode == "none":
        return data
    if mode == "zscore":
        std = float(np.std(data))
        return ((data - float(np.mean(data))) / (std + 1.0e-6)).astype(np.float32)
    if mode == "maxabs":
        scale = float(np.max(np.abs(data)))
        return (data / (scale + 1.0e-6)).astype(np.float32)
    raise ValueError(f"unsupported normalization mode: {mode}")


def _envelope_channel(shot: Array) -> Array:
    return np.sqrt(np.asarray(shot, dtype=np.float32) ** 2 + 1.0e-8).astype(np.float32)


def _spectrum_channel(shot: Array) -> Array:
    spectrum = np.abs(np.fft.rfft(shot, axis=1)).astype(np.float32)
    if spectrum.shape[1] == 1:
        resized = np.repeat(spectrum, shot.shape[1], axis=1)
    else:
        source_x = np.linspace(0.0, 1.0, spectrum.shape[1], dtype=np.float32)
        target_x = np.linspace(0.0, 1.0, shot.shape[1], dtype=np.float32)
        resized = np.vstack([np.interp(target_x, source_x, row) for row in spectrum]).astype(np.float32)
    return resized


def _offset_channel(shot: Array, source_position: float | None, receiver_positions: Array | None) -> Array:
    if source_position is None:
        source_position = 0.5
    if receiver_positions is None:
        receiver_positions = np.linspace(0.0, 1.0, shot.shape[0], dtype=np.float32)
    receivers = np.asarray(receiver_positions, dtype=np.float32)
    if receivers.shape != (shot.shape[0],):
        raise ValueError(f"receiver_positions must have shape ({shot.shape[0]},), got {receivers.shape}")
    offsets = receivers - float(source_position)
    return np.repeat(offsets[:, None], shot.shape[1], axis=1).astype(np.float32)


def build_shot_image(
    shot: Array,
    cfg: BridgeConfig,
    *,
    source_position: float | None = None,
    receiver_positions: Array | None = None,
) -> Array:
    shot = np.asarray(shot, dtype=np.float32)
    if shot.ndim != 2:
        raise ValueError(f"shot gather must be 2-D (receivers, time), got {shot.shape}")
    channels: list[Array] = []
    for name in cfg.channels:
        if name == "raw":
            channel = shot
        elif name == "envelope":
            channel = _envelope_channel(shot)
        elif name == "spectrum":
            channel = _spectrum_channel(shot)
        elif name == "offset":
            channel = _offset_channel(shot, source_position, receiver_positions)
        else:
            raise ValueError(f"unsupported bridge channel: {name}")
        if name == "offset":
            channels.append(channel.T.astype(np.float32))
        else:
            channels.append(_normalize(channel, cfg.normalize).T)
    return np.stack(channels, axis=0).astype(np.float32)


def bridge_multishot_record(
    records: Array,
    cfg: BridgeConfig,
    *,
    source_positions: Array | None = None,
    receiver_positions: Array | None = None,
) -> Array:
    records = np.asarray(records, dtype=np.float32)
    if records.ndim != 3:
        raise ValueError(f"records must be 3-D (shots, receivers, time), got {records.shape}")
    if source_positions is not None:
        source_positions = np.asarray(source_positions, dtype=np.float32)
        if source_positions.shape != (records.shape[0],):
            raise ValueError(f"source_positions must have shape ({records.shape[0]},), got {source_positions.shape}")
    return np.stack(
        [
            build_shot_image(
                shot,
                cfg,
                source_position=None if source_positions is None else float(source_positions[index]),
                receiver_positions=receiver_positions,
            )
            for index, shot in enumerate(records)
        ],
        axis=0,
    )
