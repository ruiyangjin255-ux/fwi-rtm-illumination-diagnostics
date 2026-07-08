from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

Array = np.ndarray


def _load_array_file(path: str | Path, key: str | None = None) -> Array:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=np.float32)
    if path.suffix.lower() == ".npz":
        with np.load(path) as bundle:
            if key is None:
                if len(bundle.files) != 1:
                    raise ValueError(f"{path} contains multiple arrays; pass an explicit key")
                key = bundle.files[0]
            if key not in bundle.files:
                raise ValueError(f"{path} does not contain key '{key}'")
            return np.asarray(bundle[key], dtype=np.float32)
    raise ValueError(f"unsupported array file extension for {path}")


def _canonical_records(records: Array) -> Array:
    records = np.asarray(records, dtype=np.float32)
    if records.ndim == 4:
        return records
    if records.ndim == 3:
        return records[:, None, :, :]
    raise ValueError(
        "records must have shape (samples, shots, receivers, time) "
        f"or (samples, receivers, time), got {records.shape}"
    )


def _reorder_openfwi_records(records: Array, records_layout: str) -> Array:
    records = _canonical_records(records)
    if records_layout == "samples_shots_receivers_time":
        return records
    if records_layout == "samples_shots_time_receivers":
        return np.transpose(records, (0, 1, 3, 2)).astype(np.float32, copy=False)
    raise ValueError(
        "records_layout must be one of: "
        "samples_shots_receivers_time, samples_shots_time_receivers"
    )


def _canonical_velocity(velocity: Array) -> Array:
    velocity = np.asarray(velocity, dtype=np.float32)
    if velocity.ndim == 3:
        return velocity
    if velocity.ndim == 4 and velocity.shape[1] == 1:
        return velocity[:, 0, :, :]
    if velocity.ndim == 4 and velocity.shape[-1] == 1:
        return velocity[:, :, :, 0]
    raise ValueError(
        "velocity must have shape (samples, depth, width), "
        "(samples, 1, depth, width), or (samples, depth, width, 1), "
        f"got {velocity.shape}"
    )


def _validate_arrays(records: Array, velocity: Array, source_positions: Array | None) -> tuple[Array, Array, Array | None]:
    records = np.asarray(records, dtype=np.float32)
    velocity = np.asarray(velocity, dtype=np.float32)
    if records.ndim != 4:
        raise ValueError(f"records must have shape (samples, shots, receivers, time), got {records.shape}")
    if velocity.ndim != 3:
        raise ValueError(f"velocity must have shape (samples, depth, width), got {velocity.shape}")
    if records.shape[0] != velocity.shape[0]:
        raise ValueError("records and velocity must have the same sample count")
    if source_positions is None:
        return records, velocity, None
    source_positions = np.asarray(source_positions, dtype=np.float32)
    expected = (records.shape[0], records.shape[1])
    if source_positions.shape != expected:
        raise ValueError(f"source_positions must have shape {expected}, got {source_positions.shape}")
    return records, velocity, source_positions


def _select_subset(
    records: Array,
    velocity: Array,
    source_positions: Array | None,
    *,
    max_samples: int | None,
    sample_stride: int,
) -> tuple[Array, Array, Array | None]:
    if sample_stride <= 0:
        raise ValueError("sample_stride must be positive")
    indices = np.arange(records.shape[0])[::sample_stride]
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive when provided")
        indices = indices[:max_samples]
    records = records[indices]
    velocity = velocity[indices]
    if source_positions is not None:
        source_positions = source_positions[indices]
    return records, velocity, source_positions


def _build_openfwi_manifest(
    *,
    dataset_name: str,
    family: str,
    split_name: str,
    subset_name: str,
    source_records_path: str | Path | None,
    source_velocity_path: str | Path | None,
    records_original_shape: tuple[int, ...],
    velocity_original_shape: tuple[int, ...],
    records_layout: str,
    records: Array,
    velocity: Array,
    max_samples: int | None,
    sample_stride: int,
    source_positions: Array | None,
) -> dict[str, Any]:
    return {
        "dataset_name": dataset_name,
        "family": family,
        "split_name": split_name,
        "subset_name": subset_name,
        "source_records_path": str(source_records_path) if source_records_path is not None else "",
        "source_velocity_path": str(source_velocity_path) if source_velocity_path is not None else "",
        "records_original_shape": [int(v) for v in records_original_shape],
        "velocity_original_shape": [int(v) for v in velocity_original_shape],
        "records_layout": records_layout,
        "output_records_shape": [int(v) for v in records.shape[1:]],
        "output_velocity_shape": [int(v) for v in velocity.shape[1:]],
        "max_samples": None if max_samples is None else int(max_samples),
        "sample_stride": int(sample_stride),
        "sample_count": int(records.shape[0]),
        "has_source_positions": source_positions is not None,
    }


def convert_array_dataset_to_npz(
    records: Array,
    velocity: Array,
    output_dir: str | Path,
    *,
    dataset_name: str = "array_dataset",
    source_positions: Array | None = None,
    sample_metadata: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Convert batched FWI arrays into the local one-sample-per-file npz format."""
    records, velocity, source_positions = _validate_arrays(records, velocity, source_positions)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    existing_count = len(list(output_path.glob("sample_*.npz")))
    for index in range(records.shape[0]):
        global_index = existing_count + index
        sample_name = f"sample_{global_index:06d}.npz"
        sample_path = output_path / sample_name
        payload: dict[str, Array] = {
            "records": records[index].astype(np.float32, copy=False),
            "velocity": velocity[index].astype(np.float32, copy=False),
        }
        if source_positions is not None:
            payload["source_positions"] = source_positions[index].astype(np.float32, copy=False)
        np.savez(sample_path, **payload)
        sample_entry: dict[str, Any] = {"index": global_index, "path": sample_name}
        if sample_metadata is not None:
            sample_entry.update(sample_metadata[index])
        samples.append(sample_entry)
    manifest: dict[str, Any] = {
        "dataset_name": dataset_name,
        "sample_count": int(records.shape[0]),
        "record_shape": [int(value) for value in records.shape[1:]],
        "velocity_shape": [int(value) for value in velocity.shape[1:]],
        "has_source_positions": source_positions is not None,
        "samples": samples,
    }
    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def convert_openfwi_file_groups_to_npz(
    *,
    records_paths: list[str | Path],
    velocity_paths: list[str | Path],
    output_dir: str | Path,
    dataset_name: str = "openfwi",
    family: str = "",
    split_name: str = "",
    subset_name: str = "",
    records_layout: str = "samples_shots_time_receivers",
    max_samples: int | None = None,
    sample_stride: int = 1,
    dry_run: bool = False,
) -> dict[str, Any]:
    if len(records_paths) != len(velocity_paths):
        raise ValueError("records_paths and velocity_paths must have the same length")
    if not records_paths:
        raise ValueError("at least one records/velocity file pair is required")

    output_path = Path(output_dir)
    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)

    combined_samples: list[dict[str, Any]] = []
    source_records_paths = [str(Path(path)) for path in records_paths]
    source_velocity_paths = [str(Path(path)) for path in velocity_paths]
    records_original_shapes: list[list[int]] = []
    velocity_original_shapes: list[list[int]] = []
    output_records_shape: list[int] | None = None
    output_velocity_shape: list[int] | None = None
    total_written = 0
    remaining_samples = max_samples

    for pair_index, (records_path, velocity_path) in enumerate(zip(records_paths, velocity_paths)):
        if remaining_samples is not None and remaining_samples <= 0:
            break
        records = _load_array_file(records_path)
        velocity = _load_array_file(velocity_path)
        source_positions = None
        records_original_shapes.append([int(v) for v in np.asarray(records).shape])
        velocity_original_shapes.append([int(v) for v in np.asarray(velocity).shape])

        canonical_records = _reorder_openfwi_records(records, records_layout)
        canonical_velocity = _canonical_velocity(velocity)
        canonical_records, canonical_velocity, source_positions = _validate_arrays(canonical_records, canonical_velocity, source_positions)
        canonical_records, canonical_velocity, source_positions = _select_subset(
            canonical_records,
            canonical_velocity,
            source_positions,
            max_samples=remaining_samples,
            sample_stride=sample_stride,
        )
        output_records_shape = [int(v) for v in canonical_records.shape[1:]]
        output_velocity_shape = [int(v) for v in canonical_velocity.shape[1:]]
        sample_count = int(canonical_records.shape[0])
        if dry_run:
            total_written += sample_count
            if remaining_samples is not None:
                remaining_samples -= sample_count
            continue
        sample_metadata = [
            {
                "family": family,
                "split_name": split_name,
                "sample_index": total_written + local_index,
                "source_file_index": pair_index,
                "local_index": local_index,
                "global_index": total_written + local_index,
            }
            for local_index in range(sample_count)
        ]
        partial_manifest = convert_array_dataset_to_npz(
            canonical_records,
            canonical_velocity,
            output_path,
            dataset_name=dataset_name,
            source_positions=source_positions,
            sample_metadata=sample_metadata,
        )
        combined_samples.extend(partial_manifest["samples"])
        total_written += sample_count
        if remaining_samples is not None:
            remaining_samples -= sample_count

    manifest = {
        "dataset_name": dataset_name,
        "family": family,
        "split_name": split_name,
        "subset_name": subset_name,
        "records_layout": records_layout,
        "sample_count": int(total_written),
        "output_records_shape": output_records_shape or [],
        "output_velocity_shape": output_velocity_shape or [],
        "source_records_paths": source_records_paths,
        "source_velocity_paths": source_velocity_paths,
        "records_original_shapes": records_original_shapes,
        "velocity_original_shapes": velocity_original_shapes,
        "max_samples": None if max_samples is None else int(max_samples),
        "sample_stride": int(sample_stride),
        "samples": combined_samples,
        "sample_files": [sample["path"] for sample in combined_samples],
    }
    if dry_run:
        return manifest
    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def convert_openfwi_arrays_to_npz(
    records: Array,
    velocity: Array,
    output_dir: str | Path,
    *,
    dataset_name: str = "openfwi",
    family: str = "",
    split_name: str = "",
    subset_name: str = "",
    records_layout: str = "samples_shots_time_receivers",
    source_positions: Array | None = None,
    max_samples: int | None = None,
    sample_stride: int = 1,
    dry_run: bool = False,
    source_records_path: str | Path | None = None,
    source_velocity_path: str | Path | None = None,
) -> dict[str, Any]:
    records_original_shape = tuple(np.asarray(records).shape)
    velocity_original_shape = tuple(np.asarray(velocity).shape)
    canonical_records = _reorder_openfwi_records(records, records_layout)
    canonical_velocity = _canonical_velocity(velocity)
    canonical_records, canonical_velocity, source_positions = _validate_arrays(canonical_records, canonical_velocity, source_positions)
    canonical_records, canonical_velocity, source_positions = _select_subset(
        canonical_records,
        canonical_velocity,
        source_positions,
        max_samples=max_samples,
        sample_stride=sample_stride,
    )
    manifest = _build_openfwi_manifest(
        dataset_name=dataset_name,
        family=family,
        split_name=split_name,
        subset_name=subset_name,
        source_records_path=source_records_path,
        source_velocity_path=source_velocity_path,
        records_original_shape=records_original_shape,
        velocity_original_shape=velocity_original_shape,
        records_layout=records_layout,
        records=canonical_records,
        velocity=canonical_velocity,
        max_samples=max_samples,
        sample_stride=sample_stride,
        source_positions=source_positions,
    )
    if dry_run:
        return manifest
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    full_manifest = convert_array_dataset_to_npz(
        canonical_records,
        canonical_velocity,
        output_path,
        dataset_name=dataset_name,
        source_positions=source_positions,
        sample_metadata=[
            {
                "family": family,
                "split_name": split_name,
                "sample_index": int(index),
            }
            for index in range(canonical_records.shape[0])
        ],
    )
    full_manifest.update(manifest)
    (output_path / "manifest.json").write_text(json.dumps(full_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return full_manifest


def convert_openfwi_files_to_npz(
    *,
    records_path: str | Path,
    velocity_path: str | Path,
    output_dir: str | Path,
    records_key: str | None = None,
    velocity_key: str | None = None,
    dataset_name: str = "openfwi",
    family: str = "",
    split_name: str = "",
    subset_name: str = "",
    source_positions_path: str | Path | None = None,
    source_positions_key: str | None = None,
    records_layout: str = "samples_shots_receivers_time",
    max_samples: int | None = None,
    sample_stride: int = 1,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Convert OpenFWI-style array files into the local npz sample format.

    This accepts common `.npy` files directly and `.npz` bundles with optional
    explicit keys. Records may be either 4-D `(N, S, R, T)` or 3-D `(N, R, T)`.
    Velocity may be 3-D `(N, Z, X)` or 4-D with one singleton channel.
    """
    records = _load_array_file(records_path, records_key)
    velocity = _load_array_file(velocity_path, velocity_key)
    source_positions = None
    if source_positions_path is not None:
        source_positions = _load_array_file(source_positions_path, source_positions_key)
    return convert_openfwi_arrays_to_npz(
        records,
        velocity,
        output_dir,
        dataset_name=dataset_name,
        family=family,
        split_name=split_name,
        subset_name=subset_name,
        records_layout=records_layout,
        source_positions=source_positions,
        max_samples=max_samples,
        sample_stride=sample_stride,
        dry_run=dry_run,
        source_records_path=records_path,
        source_velocity_path=velocity_path,
    )
