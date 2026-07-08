"""Fixed-manifest PASD protocol for source-family training and isolated target-family evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .data import ArrayBundle, assert_disjoint_splits, count_samples, deterministic_subset_indices, fixed_split_indices, inspect_array_source, load_arrays


@dataclass(frozen=True)
class DatasetRef:
    records: str
    models: str
    family: str
    sample_ids: str | None = None
    source_positions: str | None = None
    receiver_positions: str | None = None


@dataclass(frozen=True)
class ProtocolManifest:
    """Serializable protocol with source-only train/validation scaling and an isolated target test set."""

    version: str
    source: DatasetRef
    target: DatasetRef | None
    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]
    in_family_test_indices: tuple[int, ...]
    cross_family_test_indices: tuple[int, ...] = ()
    seed: int = 0
    notes: str = ""
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], base_dir: str | Path | None = None) -> "ProtocolManifest":
        base_dir = Path(base_dir) if base_dir is not None else None

        def resolve_ref(value: Mapping[str, Any]) -> DatasetRef:
            def resolve_path(raw: str | None) -> str | None:
                if raw is None:
                    return None
                candidate = Path(raw)
                if base_dir is not None and not candidate.is_absolute():
                    candidate = base_dir / candidate
                return str(candidate)

            return DatasetRef(
                records=str(resolve_path(str(value["records"]))),
                models=str(resolve_path(str(value["models"]))),
                family=str(value.get("family", "unknown")),
                sample_ids=resolve_path(value.get("sample_ids")),
                source_positions=resolve_path(value.get("source_positions")),
                receiver_positions=resolve_path(value.get("receiver_positions")),
            )

        split = payload.get("split", payload)
        source = resolve_ref(payload["source"])
        target_payload = payload.get("target")
        target = resolve_ref(target_payload) if target_payload else None
        manifest = cls(
            version=str(payload.get("version", "pasd_protocol_v1")),
            source=source,
            target=target,
            train_indices=tuple(int(x) for x in split["train"]),
            val_indices=tuple(int(x) for x in split["val"]),
            in_family_test_indices=tuple(int(x) for x in split["in_family_test"]),
            cross_family_test_indices=tuple(int(x) for x in split.get("cross_family_test", [])),
            seed=int(payload.get("seed", 0)),
            notes=str(payload.get("notes", "")),
            metadata=dict(payload.get("metadata", {})),
        )
        manifest.validate_structure()
        return manifest

    def validate_structure(self) -> None:
        assert_disjoint_splits(
            {
                "train": self.train_indices,
                "val": self.val_indices,
                "in_family_test": self.in_family_test_indices,
            }
        )
        if self.target is None and self.cross_family_test_indices:
            raise ValueError("cross_family_test indices cannot be supplied without a target data reference.")
        if self.target is not None and not self.cross_family_test_indices:
            raise ValueError("Target dataset is defined but cross_family_test indices are empty.")

    def to_dict(self, relative_to: str | Path | None = None) -> dict[str, Any]:
        root = Path(relative_to) if relative_to is not None else None

        def serialise_ref(ref: DatasetRef | None) -> dict[str, Any] | None:
            if ref is None:
                return None

            def maybe_relative(value: str | None) -> str | None:
                if value is None or root is None:
                    return value
                try:
                    return str(Path(value).resolve().relative_to(root.resolve()))
                except ValueError:
                    return value

            return {
                "records": maybe_relative(ref.records),
                "models": maybe_relative(ref.models),
                "family": ref.family,
                "sample_ids": maybe_relative(ref.sample_ids),
                "source_positions": maybe_relative(ref.source_positions),
                "receiver_positions": maybe_relative(ref.receiver_positions),
            }

        payload = {
            "version": self.version,
            "source": serialise_ref(self.source),
            "target": serialise_ref(self.target),
            "split": {
                "train": list(self.train_indices),
                "val": list(self.val_indices),
                "in_family_test": list(self.in_family_test_indices),
                "cross_family_test": list(self.cross_family_test_indices),
            },
            "seed": self.seed,
            "notes": self.notes,
            "metadata": self.metadata or {},
        }
        return payload

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(relative_to=path.parent), handle, indent=2, ensure_ascii=False)
        return path


def load_protocol(path: str | Path) -> ProtocolManifest:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return ProtocolManifest.from_dict(payload, base_dir=path.parent)


def build_protocol(
    source: DatasetRef,
    target: DatasetRef | None,
    train_size: int,
    val_size: int,
    in_family_test_size: int,
    cross_family_test_size: int | None,
    seed: int,
    max_source_samples: int | None = None,
    max_target_samples: int | None = None,
    notes: str = "",
) -> ProtocolManifest:
    """Create a reproducible protocol by reading only shapes; target data never define source scaling/splits."""

    source_n = count_samples(source.records, source.models)
    if max_source_samples is not None:
        source_n = min(source_n, int(max_source_samples))
    train, val, in_test = fixed_split_indices(source_n, train_size, val_size, in_family_test_size, seed=seed)

    cross: tuple[int, ...] = ()
    if target is not None:
        target_n = count_samples(target.records, target.models)
        if max_target_samples is not None:
            target_n = min(target_n, int(max_target_samples))
        size = int(cross_family_test_size or in_family_test_size)
        cross = tuple(int(x) for x in deterministic_subset_indices(target_n, size=size, seed=seed))

    manifest = ProtocolManifest(
        version="pasd_protocol_v1",
        source=source,
        target=target,
        train_indices=tuple(int(x) for x in train),
        val_indices=tuple(int(x) for x in val),
        in_family_test_indices=tuple(int(x) for x in in_test),
        cross_family_test_indices=cross,
        seed=seed,
        notes=notes,
        metadata={
            "source": inspect_array_source(source.records, source.models),
            "target": inspect_array_source(target.records, target.models) if target is not None else None,
            "scaler_fit_split": "source.train",
            "target_role": "cross_family_test_only" if target is not None else None,
            "geometry_mode": "deterministic_fallback",
            "geometry_fallback": "source is normalized shot index; receivers are normalized grid; mean offset is averaged absolute receiver-source offset.",
        },
    )
    manifest.validate_structure()
    return manifest


def load_protocol_bundles(manifest: ProtocolManifest) -> tuple[ArrayBundle, ArrayBundle | None]:
    """Load the exact protocol arrays; the caller is responsible for source-only scaler fitting."""

    source_max = max(manifest.train_indices + manifest.val_indices + manifest.in_family_test_indices) + 1
    source_bundle = load_arrays(
        manifest.source.records,
        manifest.source.models,
        max_samples=source_max,
        sample_ids_path=manifest.source.sample_ids,
        family=manifest.source.family,
        source_positions_path=manifest.source.source_positions,
        receiver_positions_path=manifest.source.receiver_positions,
    )
    _validate_indices(manifest.train_indices, len(source_bundle.records), "train")
    _validate_indices(manifest.val_indices, len(source_bundle.records), "val")
    _validate_indices(manifest.in_family_test_indices, len(source_bundle.records), "in_family_test")

    target_bundle: ArrayBundle | None = None
    if manifest.target is not None:
        target_max = max(manifest.cross_family_test_indices) + 1
        target_bundle = load_arrays(
            manifest.target.records,
            manifest.target.models,
            max_samples=target_max,
            sample_ids_path=manifest.target.sample_ids,
            family=manifest.target.family,
            source_positions_path=manifest.target.source_positions,
            receiver_positions_path=manifest.target.receiver_positions,
        )
        _validate_indices(manifest.cross_family_test_indices, len(target_bundle.records), "cross_family_test")
    return source_bundle, target_bundle


def _validate_indices(indices: tuple[int, ...], n: int, name: str) -> None:
    if not indices:
        raise ValueError(f"Protocol split '{name}' is empty.")
    if min(indices) < 0 or max(indices) >= n:
        raise IndexError(f"Protocol split '{name}' has an index outside [0, {n - 1}].")
