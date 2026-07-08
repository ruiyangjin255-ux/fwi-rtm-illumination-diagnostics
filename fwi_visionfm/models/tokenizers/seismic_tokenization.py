from __future__ import annotations

from typing import Any

import numpy as np


def _to_records(records: Any) -> np.ndarray:
    array = np.asarray(records, dtype=np.float32)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"records must have shape (shots,time,receivers) or (time,receivers), got {array.shape}")
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize(array: np.ndarray) -> tuple[np.ndarray, str]:
    mean = float(array.mean())
    std = float(array.std())
    if std <= 1.0e-6:
        return array - mean, "mean_center"
    return (array - mean) / std, "zscore"


def _patch_tokens(image: np.ndarray, patch_size: int = 8) -> np.ndarray:
    patch = int(max(1, patch_size))
    h = image.shape[-2] // patch * patch
    w = image.shape[-1] // patch * patch
    cropped = image[..., :h, :w]
    if h == 0 or w == 0:
        flat = image.reshape(1, -1)
        stats = np.stack([flat.mean(axis=1), flat.std(axis=1), np.sqrt(np.mean(flat * flat, axis=1))], axis=1)
        return np.concatenate([stats, flat], axis=1).astype(np.float32)
    patches = cropped.reshape(h // patch, patch, w // patch, patch).transpose(0, 2, 1, 3)
    flat = patches.reshape(-1, patch * patch)
    means = flat.mean(axis=1, keepdims=True)
    stds = flat.std(axis=1, keepdims=True)
    energy = np.sqrt(np.mean(flat * flat, axis=1, keepdims=True))
    return np.concatenate([means, stds, energy, flat], axis=1).astype(np.float32)


def _pack(tokens: np.ndarray, *, tokenization_name: str, variant: str, bridge_name: str, view_mode: str, normalization: str, input_shape: tuple[int, ...]) -> dict[str, Any]:
    return {
        "tokens": np.asarray(tokens, dtype=np.float32),
        "metadata": {
            "tokenization_name": tokenization_name,
            "variant": variant,
            "input_shape": list(input_shape),
            "output_shape": list(np.asarray(tokens).shape),
            "bridge_name": bridge_name,
            "view_mode": view_mode,
            "normalization": normalization,
            "status": "SUCCESS",
        },
    }


def ncs_2d_tokenization(records: Any, bridge_name: str, *, patch_size: int = 8) -> dict[str, Any]:
    array = _to_records(records)
    normalized, normalization = _normalize(array.mean(axis=0))
    tokens = _patch_tokens(normalized, patch_size=patch_size)
    return _pack(
        tokens,
        tokenization_name="ncs_2d_tokenization",
        variant="ncs_2d",
        bridge_name=bridge_name,
        view_mode="shot_stack_mean",
        normalization=normalization,
        input_shape=tuple(array.shape),
    )


def ncs_2p5d_tokenization(records: Any, bridge_name: str, *, view_mode: str = "shot_view", patch_size: int = 8) -> dict[str, Any]:
    array = _to_records(records)
    if view_mode == "shot_view":
        views = array
    elif view_mode == "attribute_view":
        shot_mean = array.mean(axis=0)
        shot_std = array.std(axis=0)
        shot_energy = np.sqrt(np.mean(array * array, axis=0))
        views = np.stack([shot_mean, shot_std, shot_energy], axis=0)
    elif view_mode == "offset_view":
        views = np.stack([array[0], array[len(array) // 2], array[-1]], axis=0)
    else:
        raise ValueError(f"unsupported ncs_2p5d view_mode: {view_mode}")
    normalized, normalization = _normalize(views)
    tokens = np.stack([_patch_tokens(view, patch_size=patch_size) for view in normalized], axis=0).astype(np.float32)
    return _pack(
        tokens,
        tokenization_name="ncs_2p5d_tokenization",
        variant="ncs_2p5d",
        bridge_name=bridge_name,
        view_mode=view_mode,
        normalization=normalization,
        input_shape=tuple(array.shape),
    )


def fallback_tokenization(records: Any, variant: str, *, bridge_name: str = "fallback", feature_dim: int = 128) -> dict[str, Any]:
    array = _to_records(records)
    stats = np.array(
        [
            float(array.mean()),
            float(array.std()),
            float(np.min(array)),
            float(np.max(array)),
            float(np.mean(np.abs(array))),
            float(np.sqrt(np.mean(array * array))),
        ],
        dtype=np.float32,
    )
    repeats = int(np.ceil(int(feature_dim) / stats.shape[0]))
    tokens = np.tile(stats, repeats)[: int(feature_dim)].astype(np.float32)
    return _pack(
        tokens,
        tokenization_name="fallback_tokenization",
        variant=variant,
        bridge_name=bridge_name,
        view_mode="fallback_dummy",
        normalization="summary_stats",
        input_shape=tuple(array.shape),
    )


def tokenize_2d(records: Any, patch_size: int = 16) -> np.ndarray:
    return np.asarray(ncs_2d_tokenization(records, bridge_name="legacy", patch_size=patch_size)["tokens"], dtype=np.float32)


def tokenize_2p5d(records: Any, patch_size: int = 16) -> np.ndarray:
    return np.asarray(ncs_2p5d_tokenization(records, bridge_name="legacy", patch_size=patch_size)["tokens"], dtype=np.float32)


def dummy_feature(records: Any, feature_dim: int = 128) -> np.ndarray:
    return np.asarray(fallback_tokenization(records, variant="dummy", feature_dim=feature_dim)["tokens"], dtype=np.float32)
