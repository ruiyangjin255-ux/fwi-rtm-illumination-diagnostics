from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


_DATA_PATTERN = re.compile(r"^data(?P<index>\d+)\.npy$", re.IGNORECASE)
_MODEL_PATTERN = re.compile(r"^model(?P<index>\d+)\.npy$", re.IGNORECASE)


@dataclass(frozen=True)
class _SampleRef:
    family: str
    data_file: Path
    model_file: Path
    local_index: int
    global_index: int


def _normalize_array(array: np.ndarray, mode: str, stats: dict[str, float] | None, eps: float = 1.0e-6) -> np.ndarray:
    if mode == "none":
        return array.astype(np.float32, copy=False)
    if stats is None:
        return array.astype(np.float32, copy=False)
    if mode == "zscore":
        mean = float(stats.get("mean", stats.get("input_mean", stats.get("target_mean", 0.0))))
        std = float(stats.get("std", stats.get("input_std", stats.get("target_std", 1.0))))
        return ((array - mean) / max(std, eps)).astype(np.float32, copy=False)
    if mode == "minmax":
        min_value = float(stats.get("min", stats.get("input_min", stats.get("target_min", 0.0))))
        max_value = float(stats.get("max", stats.get("input_max", stats.get("target_max", 1.0))))
        return ((array - min_value) / max(max_value - min_value, eps)).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported normalization mode: {mode}")


def _infer_family_root(path: Path) -> Path:
    parent = path.parent
    if parent.name.lower() in {"data", "model"} and parent.parent != parent:
        return parent.parent
    return parent


def _infer_family_name(path: Path) -> str:
    return _infer_family_root(path).name


class OpenFWINpyDataset(Dataset):
    def __init__(
        self,
        root: str,
        split_file: str | None = None,
        family: str | None = None,
        max_samples: int | None = None,
        input_norm: str = "zscore",
        target_norm: str = "minmax",
        stats_file: str | None = None,
        fit_stats: bool = False,
        mmap: bool = True,
        dtype: str = "float32",
    ) -> None:
        self.root = Path(root)
        self.family_override = family
        self.max_samples = max_samples
        self.input_norm = input_norm
        self.target_norm = target_norm
        self.stats_file = Path(stats_file) if stats_file else None
        self.fit_stats = fit_stats
        self.mmap_mode = "r" if mmap else None
        self.dtype = np.dtype(dtype)

        if not self.root.exists():
            raise FileNotFoundError(f"OpenFWI root does not exist: {self.root}")

        self._array_cache: dict[Path, np.ndarray] = {}
        self._sample_refs, self.split_name = self._build_sample_refs(split_file)
        if not self._sample_refs:
            raise ValueError("OpenFWINpyDataset found no valid samples")
        if self.max_samples is not None:
            if self.max_samples <= 0:
                raise ValueError("max_samples must be positive")
            self._sample_refs = self._sample_refs[: self.max_samples]

        if self.fit_stats and self.split_name not in (None, "", "train"):
            raise ValueError("fit_stats is only allowed for train splits")

        self.input_stats: dict[str, float] | None = None
        self.target_stats: dict[str, float] | None = None
        if self.stats_file and self.stats_file.exists():
            loaded = json.loads(self.stats_file.read_text(encoding="utf-8"))
            if "seismic" in loaded and "velocity" in loaded:
                self.input_stats = {key: float(value) for key, value in loaded["seismic"].items() if isinstance(value, (int, float))}
                self.target_stats = {key: float(value) for key, value in loaded["velocity"].items() if isinstance(value, (int, float))}
            else:
                self.input_stats = {
                    "mean": float(loaded["input_mean"]),
                    "std": float(loaded["input_std"]),
                }
                self.target_stats = {
                    "min": float(loaded["target_min"]),
                    "max": float(loaded["target_max"]),
                }
        elif self.fit_stats:
            self.input_stats, self.target_stats = self._fit_stats()
            if self.stats_file is None:
                raise ValueError("stats_file must be provided when fit_stats=True")
            self.stats_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "seismic": self.input_stats,
                "velocity": self.target_stats,
                "input_mean": float(self.input_stats.get("mean", 0.0)),
                "input_std": float(self.input_stats.get("std", 1.0)),
                "target_min": float(self.target_stats.get("min", 0.0)),
                "target_max": float(self.target_stats.get("max", 1.0)),
                "sample_count": len(self._sample_refs),
                "split_name": self.split_name or "train",
            }
            self.stats_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def __len__(self) -> int:
        return len(self._sample_refs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ref = self._sample_refs[idx]
        seismic = self._load_sample_array(ref.data_file, ref.local_index, kind="records")
        velocity = self._load_sample_array(ref.model_file, ref.local_index, kind="velocity")

        seismic = _normalize_array(seismic, self.input_norm, self.input_stats)
        velocity = _normalize_array(velocity, self.target_norm, self.target_stats)

        return {
            "seismic": torch.from_numpy(np.array(seismic, dtype=self.dtype, copy=True)),
            "velocity": torch.from_numpy(np.array(velocity, dtype=self.dtype, copy=True)),
            "meta": {
                "family": ref.family,
                "data_file": str(ref.data_file),
                "model_file": str(ref.model_file),
                "local_index": ref.local_index,
                "global_index": ref.global_index,
            },
        }

    def _build_sample_refs(self, split_file: str | None) -> tuple[list[_SampleRef], str | None]:
        if split_file:
            return self._load_split_refs(Path(split_file))
        return self._discover_refs(), None

    def _load_split_refs(self, split_path: Path) -> tuple[list[_SampleRef], str | None]:
        if split_path.suffix.lower() == ".csv":
            return self._load_split_refs_from_csv(split_path)
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            entries = payload
            split_name = None
        elif isinstance(payload, dict):
            if "samples" in payload:
                entries = payload["samples"]
                split_name = payload.get("split_name")
            else:
                for candidate in ("train", "val", "test"):
                    if candidate in payload and isinstance(payload[candidate], list):
                        entries = payload[candidate]
                        split_name = candidate
                        break
                else:
                    raise ValueError(f"Unsupported split file format: {split_path}")
        else:
            raise ValueError(f"Unsupported split file payload: {split_path}")

        refs: list[_SampleRef] = []
        for item in entries:
            if not isinstance(item, dict):
                raise ValueError("split file entries must be objects")
            data_file = Path(item["data_file"])
            model_file = Path(item["model_file"])
            local_index = int(item["local_index"])
            global_index = int(item.get("global_index", len(refs)))
            family = str(item.get("family") or self.family_override or _infer_family_name(data_file))
            self._validate_pair_shapes(data_file, model_file)
            refs.append(
                _SampleRef(
                    family=family,
                    data_file=data_file,
                    model_file=model_file,
                    local_index=local_index,
                    global_index=global_index,
                )
            )
        return refs, split_name

    def _load_split_refs_from_csv(self, split_path: Path) -> tuple[list[_SampleRef], str | None]:
        refs: list[_SampleRef] = []
        with split_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"data_file", "model_file", "local_index", "global_index", "family"}
            if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
                raise ValueError(f"split csv missing required columns: {sorted(required)}")
            for row in reader:
                data_file = Path(row["data_file"])
                model_file = Path(row["model_file"])
                local_index = int(row["local_index"])
                global_index = int(row["global_index"])
                family = str(row.get("family") or self.family_override or _infer_family_name(data_file))
                self._validate_pair_shapes(data_file, model_file)
                refs.append(
                    _SampleRef(
                        family=family,
                        data_file=data_file,
                        model_file=model_file,
                        local_index=local_index,
                        global_index=global_index,
                    )
                )
        return refs, split_path.stem

    def _discover_refs(self) -> list[_SampleRef]:
        pairs = self._discover_pairs()
        refs: list[_SampleRef] = []
        global_index = 0
        for data_file, model_file in pairs:
            records_array = self._load_array(data_file)
            velocity_array = self._load_array(model_file)
            self._validate_shapes(records_array, velocity_array, data_file, model_file)
            family = self.family_override or _infer_family_name(data_file)
            sample_count = int(records_array.shape[0])
            if sample_count != int(velocity_array.shape[0]):
                raise ValueError(
                    f"data/model sample count mismatch for {data_file} and {model_file}: "
                    f"{records_array.shape[0]} vs {velocity_array.shape[0]}"
                )
            for local_index in range(sample_count):
                refs.append(
                    _SampleRef(
                        family=family,
                        data_file=data_file,
                        model_file=model_file,
                        local_index=local_index,
                        global_index=global_index,
                    )
                )
                global_index += 1
        return refs

    def _discover_pairs(self) -> list[tuple[Path, Path]]:
        data_files: dict[tuple[Path, str], Path] = {}
        model_files: dict[tuple[Path, str], Path] = {}
        for path in self.root.rglob("*.npy"):
            data_match = _DATA_PATTERN.match(path.name)
            if data_match:
                data_files[(_infer_family_root(path), data_match.group("index"))] = path
                continue
            model_match = _MODEL_PATTERN.match(path.name)
            if model_match:
                model_files[(_infer_family_root(path), model_match.group("index"))] = path

        data_keys = set(data_files.keys())
        model_keys = set(model_files.keys())
        if data_keys != model_keys:
            missing_models = sorted(str(data_files[key]) for key in data_keys - model_keys)
            missing_data = sorted(str(model_files[key]) for key in model_keys - data_keys)
            raise ValueError(
                "OpenFWI data/model pairing mismatch. "
                f"Missing model files for: {missing_models or '[]'}. "
                f"Missing data files for: {missing_data or '[]'}."
            )
        if not data_keys:
            raise ValueError(f"No OpenFWI data/model .npy pairs found under {self.root}")
        return [(data_files[key], model_files[key]) for key in sorted(data_keys, key=lambda item: (str(item[0]), int(item[1])))]

    def _fit_stats(self) -> tuple[dict[str, float], dict[str, float]]:
        input_sum = 0.0
        input_sq_sum = 0.0
        input_count = 0
        target_min = float("inf")
        target_max = float("-inf")

        for ref in self._sample_refs:
            seismic = self._load_sample_array(ref.data_file, ref.local_index, kind="records")
            velocity = self._load_sample_array(ref.model_file, ref.local_index, kind="velocity")
            input_sum += float(seismic.sum(dtype=np.float64))
            input_sq_sum += float(np.square(seismic, dtype=np.float64).sum(dtype=np.float64))
            input_count += int(seismic.size)
            target_min = min(target_min, float(velocity.min()))
            target_max = max(target_max, float(velocity.max()))

        input_mean = input_sum / max(input_count, 1)
        input_var = max(input_sq_sum / max(input_count, 1) - input_mean * input_mean, 0.0)
        input_std = float(np.sqrt(input_var))
        return (
            {"mean": float(input_mean), "std": input_std},
            {"min": target_min, "max": target_max},
        )

    def _validate_pair_shapes(self, data_file: Path, model_file: Path) -> None:
        self._validate_shapes(self._load_array(data_file), self._load_array(model_file), data_file, model_file)

    def _validate_shapes(self, records_array: np.ndarray, velocity_array: np.ndarray, data_file: Path, model_file: Path) -> None:
        if records_array.ndim != 4:
            raise ValueError(f"OpenFWI records file must have shape [N,S,T,R] or [N,S,R,T], got {records_array.shape} at {data_file}")
        if velocity_array.ndim not in (3, 4):
            raise ValueError(f"OpenFWI model file must have shape [N,H,W] or [N,1,H,W], got {velocity_array.shape} at {model_file}")
        if velocity_array.ndim == 4 and velocity_array.shape[1] != 1 and velocity_array.shape[-1] != 1:
            raise ValueError(f"OpenFWI model file must use a singleton channel dimension, got {velocity_array.shape} at {model_file}")

    def _load_array(self, path: Path) -> np.ndarray:
        cached = self._array_cache.get(path)
        if cached is None:
            cached = np.load(path, mmap_mode=self.mmap_mode)
            self._array_cache[path] = cached
        return cached

    def _load_sample_array(self, path: Path, local_index: int, *, kind: str) -> np.ndarray:
        array = self._load_array(path)
        if local_index < 0 or local_index >= int(array.shape[0]):
            raise IndexError(f"local_index {local_index} is out of range for {path}")
        sample = np.asarray(array[local_index], dtype=np.float32)
        if kind == "records":
            if sample.ndim != 3:
                raise ValueError(f"OpenFWI records sample must be 3D, got {sample.shape} from {path}")
            return sample
        if kind == "velocity":
            if sample.ndim == 2:
                sample = sample[None, ...]
            elif sample.ndim == 3 and sample.shape[0] == 1:
                pass
            elif sample.ndim == 3 and sample.shape[-1] == 1:
                sample = np.transpose(sample, (2, 0, 1))
            else:
                raise ValueError(f"OpenFWI velocity sample must be [H,W], [1,H,W], or [H,W,1], got {sample.shape} from {path}")
            return sample.astype(np.float32, copy=False)
        raise ValueError(f"Unsupported sample kind: {kind}")
