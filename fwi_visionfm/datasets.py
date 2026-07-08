from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fwi_visionfm.config import DataConfig

Array = np.ndarray


@dataclass(frozen=True)
class FWISample:
    records: Array
    velocity: Array
    source_positions: Array


@dataclass(frozen=True)
class FWIBatch:
    records: Array
    velocity: Array
    source_positions: Array
    paths: list[Path]


class NPZSampleDataset:
    def __init__(self, paths: list[str | Path]) -> None:
        self.paths = [Path(path) for path in paths]
        if not self.paths:
            raise ValueError("NPZSampleDataset requires at least one sample path")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> FWISample:
        return load_npz_sample(self.paths[index])

    def iter_batches(self, batch_size: int) -> list[FWIBatch]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        batches: list[FWIBatch] = []
        for start in range(0, len(self.paths), batch_size):
            batch_paths = self.paths[start : start + batch_size]
            samples = [load_npz_sample(path) for path in batch_paths]
            batches.append(
                FWIBatch(
                    records=np.stack([sample.records for sample in samples], axis=0).astype(np.float32),
                    velocity=np.stack([sample.velocity for sample in samples], axis=0).astype(np.float32),
                    source_positions=np.stack([sample.source_positions for sample in samples], axis=0).astype(np.float32),
                    paths=batch_paths,
                )
            )
        return batches


def _make_velocity(depth: int, width: int, rng: np.random.Generator) -> Array:
    z = np.linspace(0.0, 1.0, depth, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    background = 1700.0 + 1200.0 * z
    salt_body = np.exp(-((x - 0.25) ** 2 / 0.12 + (z - 0.58) ** 2 / 0.045))
    fault_step = (x + 0.25 * z > 0.15).astype(np.float32) * 120.0
    perturbation = rng.normal(0.0, 20.0, size=(depth, width)).astype(np.float32)
    return (background + 900.0 * salt_body + fault_step + perturbation).astype(np.float32)


def _make_record(velocity: Array, cfg: DataConfig, source_x: float, rng: np.random.Generator) -> Array:
    receivers = np.linspace(0.0, 1.0, cfg.num_receivers, dtype=np.float32)
    time = np.linspace(0.0, 1.0, cfg.num_time_samples, dtype=np.float32)
    mean_velocity = float(np.mean(velocity))
    velocity_contrast = float(np.std(velocity) / max(mean_velocity, 1.0))
    offset = receivers[None, :] - source_x
    t = time[:, None]
    event_1 = np.sin(2.0 * np.pi * (12.0 * t - 1.8 * offset * offset))
    event_2 = np.sin(2.0 * np.pi * (22.0 * t + 0.7 * offset))
    envelope = np.exp(-3.0 * (t - 0.25 - 0.35 * np.abs(offset)) ** 2)
    record = (event_1 * envelope + 0.35 * event_2 * np.exp(-2.0 * t)) * (1.0 + velocity_contrast)
    if cfg.noise_std > 0.0:
        record = record + rng.normal(0.0, cfg.noise_std, size=record.shape)
    return record.T.astype(np.float32)


def make_synthetic_sample(cfg: DataConfig, seed: int = 0) -> FWISample:
    rng = np.random.default_rng(seed)
    velocity = _make_velocity(cfg.velocity_depth, cfg.velocity_width, rng)
    source_positions = np.linspace(0.12, 0.88, cfg.num_shots, dtype=np.float32)
    records = np.stack([_make_record(velocity, cfg, float(src), rng) for src in source_positions], axis=0)
    return FWISample(records=records.astype(np.float32), velocity=velocity, source_positions=source_positions)


def _default_source_positions(num_shots: int) -> Array:
    if num_shots <= 1:
        return np.array([0.5], dtype=np.float32)
    return np.linspace(0.12, 0.88, num_shots, dtype=np.float32)


def load_npz_sample(path: str | Path) -> FWISample:
    """Load an OpenFWI-style local sample from a compact npz file.

    The first scaffold format expects `records` shaped `(shots, receivers, time)`
    and `velocity` shaped `(depth, width)`. `source_positions` is optional and
    normalized to `[0, 1]` when absent.
    """
    with np.load(Path(path)) as data:
        if "records" not in data or "velocity" not in data:
            raise ValueError("npz sample must contain 'records' and 'velocity' arrays")
        records = np.asarray(data["records"], dtype=np.float32)
        velocity = np.asarray(data["velocity"], dtype=np.float32)
        if records.ndim != 3:
            raise ValueError(f"records must have shape (shots, receivers, time), got {records.shape}")
        if velocity.ndim != 2:
            raise ValueError(f"velocity must have shape (depth, width), got {velocity.shape}")
        if "source_positions" in data:
            source_positions = np.asarray(data["source_positions"], dtype=np.float32)
        else:
            source_positions = _default_source_positions(records.shape[0])
    if source_positions.shape != (records.shape[0],):
        raise ValueError(
            f"source_positions must have shape ({records.shape[0]},), got {source_positions.shape}"
        )
    return FWISample(records=records, velocity=velocity, source_positions=source_positions)


def discover_npz_samples(root: str | Path) -> list[Path]:
    return sorted(Path(root).glob("*.npz"))


def split_sample_paths(
    paths: list[str | Path],
    *,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    seed: int = 0,
) -> dict[str, list[Path]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be less than 1")
    normalized = [Path(path) for path in paths]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(normalized))
    shuffled = [normalized[int(index)] for index in order]
    train_count = int(len(shuffled) * train_fraction)
    val_count = int(len(shuffled) * val_fraction)
    train = shuffled[:train_count]
    val = shuffled[train_count : train_count + val_count]
    test = shuffled[train_count + val_count :]
    return {"train": train, "val": val, "test": test}
