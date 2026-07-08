"""Data, scaling, and strict split helpers for PASD-FWI experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class VelocityScaler:
    """Train-only min-max scaler for target velocity fields."""

    minimum: float
    maximum: float

    @classmethod
    def fit(cls, values: np.ndarray) -> "VelocityScaler":
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        if maximum <= minimum:
            maximum = minimum + 1.0
        return cls(minimum=minimum, maximum=maximum)

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.minimum) / (self.maximum - self.minimum)).astype(np.float32)

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        return (values * (self.maximum - self.minimum) + self.minimum).astype(np.float32)

    def as_dict(self) -> dict[str, float]:
        return {"minimum": self.minimum, "maximum": self.maximum}


@dataclass(frozen=True)
class ArrayBundle:
    """Array collection retaining family metadata and stable sample identifiers."""

    records: np.ndarray
    velocities: np.ndarray
    sample_ids: np.ndarray
    family: str = "unknown"
    source_positions: np.ndarray | None = None
    receiver_positions: np.ndarray | None = None

    def __post_init__(self) -> None:
        validate_arrays(self.records, self.velocities)
        _validate_geometry_arrays(self.records, self.source_positions, self.receiver_positions)
        if len(self.sample_ids) != len(self.records):
            raise ValueError("sample_ids must be aligned with the first records dimension.")
        if np.unique(self.sample_ids).size != self.sample_ids.size:
            raise ValueError("sample_ids must be unique within a data bundle.")


class OpenFWINpyDataset(Dataset):
    """Dataset for records [N,S,T,R] and velocities [N,H,W] / [N,1,H,W]."""

    def __init__(
        self,
        records: np.ndarray,
        velocities: np.ndarray,
        indices: Iterable[int],
        scaler: VelocityScaler,
        sample_ids: np.ndarray | None = None,
        source_positions: np.ndarray | None = None,
        receiver_positions: np.ndarray | None = None,
    ) -> None:
        self.records = np.asarray(records, dtype=np.float32)
        self.velocities = canonical_velocities(velocities)
        validate_arrays(self.records, self.velocities)
        self.indices = np.asarray(list(indices), dtype=np.int64)
        if self.indices.size == 0:
            raise ValueError("A PASD dataset split cannot be empty.")
        if self.indices.min(initial=0) < 0 or self.indices.max(initial=0) >= len(self.records):
            raise IndexError("Split indices are outside the available data range.")
        self.scaler = scaler
        self.sample_ids = np.arange(len(self.records), dtype=np.int64) if sample_ids is None else np.asarray(sample_ids, dtype=np.int64)
        self.source_positions = None if source_positions is None else np.asarray(source_positions, dtype=np.float32)
        self.receiver_positions = None if receiver_positions is None else np.asarray(receiver_positions, dtype=np.float32)
        _validate_geometry_arrays(self.records, self.source_positions, self.receiver_positions)
        if len(self.sample_ids) != len(self.records):
            raise ValueError("sample_ids must have one entry per sample.")

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> dict[str, Tensor]:
        index = int(self.indices[item])
        item = {
            "records": torch.from_numpy(np.array(self.records[index], dtype=np.float32, copy=True)),
            "velocity": torch.from_numpy(self.scaler.normalize(self.velocities[index])).unsqueeze(0),
            "sample_id": torch.tensor(int(self.sample_ids[index]), dtype=torch.long),
            "array_index": torch.tensor(index, dtype=torch.long),
        }
        if hasattr(self, "source_positions") and self.source_positions is not None:
            source = self.source_positions if self.source_positions.ndim == 1 else self.source_positions[index]
            item["source_positions"] = torch.from_numpy(np.array(source, dtype=np.float32, copy=True))
        if hasattr(self, "receiver_positions") and self.receiver_positions is not None:
            receiver = self.receiver_positions
            if receiver.ndim == 2 and receiver.shape[0] == len(self.records):
                receiver = receiver[index]
            elif receiver.ndim == 3:
                receiver = receiver[index]
            item["receiver_positions"] = torch.from_numpy(np.array(receiver, dtype=np.float32, copy=True))
        return item


def _load_geometry_array(path: str | Path | None, n: int) -> np.ndarray | None:
    if path is None:
        return None
    values = np.load(path, mmap_mode="r")
    values = np.asarray(values, dtype=np.float32)
    if values.ndim >= 2 and values.shape[0] >= n:
        values = values[:n]
    return values


def _validate_geometry_arrays(records: np.ndarray, source_positions: np.ndarray | None, receiver_positions: np.ndarray | None) -> None:
    n, shots, _, receivers = np.asarray(records).shape
    if source_positions is not None:
        source = np.asarray(source_positions)
        if source.shape not in {(shots,), (n, shots)}:
            raise ValueError(f"source_positions must have shape [S] or [N,S], got {source.shape}.")
    if receiver_positions is not None:
        receiver = np.asarray(receiver_positions)
        valid = {(receivers,), (n, receivers), (n, shots, receivers)}
        if receiver.shape not in valid:
            raise ValueError(f"receiver_positions must have shape [R], [N,R], or [N,S,R], got {receiver.shape}.")


def canonical_velocities(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 4 and values.shape[1] == 1:
        values = values[:, 0]
    return values


def validate_arrays(records: np.ndarray, velocities: np.ndarray) -> None:
    records = np.asarray(records)
    velocities = canonical_velocities(velocities)
    if records.ndim != 4:
        raise ValueError(f"records must be [N,S,T,R], got {records.shape}.")
    if velocities.ndim != 3:
        raise ValueError(f"velocities must be [N,H,W] or [N,1,H,W], got {velocities.shape}.")
    if records.shape[0] != velocities.shape[0]:
        raise ValueError("records and velocities must have the same number of samples.")
    if min(records.shape[1:]) <= 0 or min(velocities.shape[1:]) <= 0:
        raise ValueError("records and velocities must have non-empty physical dimensions.")


def load_sample_ids(path: str | Path | None, count: int) -> np.ndarray:
    """Load stable IDs from .npy/.txt/.csv or generate 0..N-1 when unavailable."""

    if path is None:
        return np.arange(count, dtype=np.int64)
    path = Path(path)
    if path.suffix.lower() == ".npy":
        values = np.load(path)
    else:
        values = np.loadtxt(path, delimiter="," if path.suffix.lower() == ".csv" else None, dtype=np.int64)
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    if len(values) != count:
        raise ValueError(f"sample ID file contains {len(values)} IDs but data contain {count} samples.")
    return values


def load_arrays(
    records_path: str | Path,
    models_path: str | Path,
    max_samples: int | None = None,
    sample_ids_path: str | Path | None = None,
    family: str = "unknown",
    source_positions_path: str | Path | None = None,
    receiver_positions_path: str | Path | None = None,
) -> ArrayBundle:
    """Load only the requested leading block to keep CPU smoke runs memory-bounded."""

    records_mm = np.load(records_path, mmap_mode="r")
    models_mm = np.load(models_path, mmap_mode="r")
    n = min(len(records_mm), len(models_mm))
    if max_samples is not None:
        n = min(n, int(max_samples))
    records = np.asarray(records_mm[:n], dtype=np.float32)
    velocities = canonical_velocities(np.asarray(models_mm[:n], dtype=np.float32))
    sample_ids = load_sample_ids(sample_ids_path, len(records_mm))[:n]
    source_positions = _load_geometry_array(source_positions_path, n)
    receiver_positions = _load_geometry_array(receiver_positions_path, n)
    return ArrayBundle(records=records, velocities=velocities, sample_ids=sample_ids, family=family, source_positions=source_positions, receiver_positions=receiver_positions)


def fixed_split_indices(n: int, train_size: int, val_size: int, test_size: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create one deterministic source-family split without target-family involvement."""

    if min(train_size, val_size, test_size) <= 0:
        raise ValueError("train_size, val_size, and test_size must all be positive.")
    if train_size + val_size + test_size > n:
        raise ValueError("Requested split exceeds available sample count.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    train = order[:train_size]
    val = order[train_size : train_size + val_size]
    test = order[train_size + val_size : train_size + val_size + test_size]
    assert_disjoint_splits({"train": train, "val": val, "in_family_test": test})
    return train, val, test


def deterministic_subset_indices(n: int, size: int, seed: int = 0) -> np.ndarray:
    """Select a fixed target-family evaluation subset; no target samples enter training/scaler fitting."""

    if size <= 0 or size > n:
        raise ValueError(f"Requested target subset size {size} is not within [1, {n}].")
    return np.random.default_rng(seed).permutation(n)[:size]


def assert_disjoint_splits(splits: Mapping[str, Sequence[int] | np.ndarray]) -> None:
    """Reject source split leakage before any model training begins."""

    keys = list(splits)
    for i, left_name in enumerate(keys):
        left = set(np.asarray(splits[left_name], dtype=np.int64).tolist())
        if not left:
            raise ValueError(f"Split '{left_name}' is empty.")
        for right_name in keys[i + 1 :]:
            right = set(np.asarray(splits[right_name], dtype=np.int64).tolist())
            overlap = left.intersection(right)
            if overlap:
                preview = sorted(overlap)[:5]
                raise ValueError(f"Split leakage between '{left_name}' and '{right_name}': {preview}.")


def synthetic_openfwi_like(
    n: int = 24,
    shots: int = 5,
    time: int = 128,
    receivers: int = 48,
    model_size: int = 32,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a structured smoke dataset only; it is not a geophysical benchmark."""

    rng = np.random.default_rng(seed)
    xx, zz = np.meshgrid(np.linspace(-1, 1, model_size), np.linspace(0, 1, model_size))
    models: list[np.ndarray] = []
    records: list[np.ndarray] = []
    t = np.linspace(0, 1, time, dtype=np.float32)
    r = np.linspace(-1, 1, receivers, dtype=np.float32)
    for _ in range(n):
        background = 0.25 + 0.5 * zz
        center_x, center_z = rng.uniform(-0.5, 0.5), rng.uniform(0.35, 0.8)
        anomaly = 0.18 * np.exp(-((xx - center_x) ** 2 / 0.08 + (zz - center_z) ** 2 / 0.04))
        fault = 0.06 * (xx > (0.4 * zz + rng.uniform(-0.25, 0.25))).astype(np.float32)
        velocity = np.clip(background + anomaly + fault, 0.0, 1.0).astype(np.float32)
        models.append(velocity)
        gathers: list[np.ndarray] = []
        for shot in range(shots):
            shot_pos = -1.0 + 2.0 * shot / max(shots - 1, 1)
            curvature = 0.12 + 0.08 * (shot_pos - center_x) ** 2
            travel = 0.25 + center_z * 0.45 + curvature * (r - shot_pos) ** 2
            wave = np.exp(-((t[:, None] - travel[None, :]) ** 2) / 0.0025)
            wave *= 1.0 + 0.3 * np.sin(16 * np.pi * t[:, None])
            wave += 0.05 * rng.normal(size=(time, receivers))
            gathers.append(wave.astype(np.float32))
        records.append(np.stack(gathers, axis=0))
    return np.stack(records), np.stack(models)
