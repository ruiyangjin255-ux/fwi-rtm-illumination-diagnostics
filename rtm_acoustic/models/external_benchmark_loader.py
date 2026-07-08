from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rtm_acoustic.diagnostics.admit_common import file_hash, git_commit, now_utc, text_hash, write_json
from rtm_acoustic.models.synthetic_model_bank import _smooth


def load_velocity(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path).astype(np.float32)
    if suffix == ".npz":
        data = np.load(path)
        key = "velocity" if "velocity" in data else list(data.keys())[0]
        return data[key].astype(np.float32)
    raise ValueError(f"UNSUPPORTED_EXTERNAL_MODEL_FORMAT: {path}")


def prepare_external_crop(
    *,
    name: str,
    source_path: Path,
    output_root: Path,
    crop: tuple[int, int, int, int] | None = None,
    downsample: int = 1,
) -> dict[str, Any]:
    out = output_root / f"{name}_crop"
    out.mkdir(parents=True, exist_ok=True)
    if not source_path.exists():
        manifest = {
            "status": "MISSING_EXTERNAL_MODEL",
            "timestamp_utc": now_utc(),
            "git_commit": git_commit(),
            "command": "prepare_external_benchmark_crops.py",
            "config_hash": text_hash(f"{name}:{source_path}"),
            "input_hash": "MISSING",
            "model_source": name,
            "source_path": str(source_path),
        }
        write_json(out / "manifest.json", manifest)
        return manifest
    true = load_velocity(source_path)
    if true.ndim != 2:
        raise ValueError(f"{name} must be 2D after loading, got {true.shape}")
    if crop:
        x0, x1, z0, z1 = crop
        true = true[z0:z1, x0:x1]
    if downsample > 1:
        true = true[::downsample, ::downsample]
    if true.size == 0:
        raise ValueError(f"{name} crop is empty")
    initial = _smooth(true, 8.0)
    if np.allclose(true, initial):
        raise ValueError("initial model must differ from true model")
    np.save(out / "true_velocity.npy", true.astype(np.float32))
    np.save(out / "initial_velocity.npy", initial.astype(np.float32))
    config = {
        "model_name": f"{name}_crop",
        "model_source": name,
        "source_path": str(source_path),
        "shape": list(true.shape),
        "downsample": downsample,
    }
    (out / "model_config.yaml").write_text("\n".join(f"{k}: {v}" for k, v in config.items()) + "\n", encoding="utf-8")
    manifest = {
        "status": "READY",
        "timestamp_utc": now_utc(),
        "git_commit": git_commit(),
        "command": "prepare_external_benchmark_crops.py",
        "config_hash": text_hash(str(config)),
        "input_hash": file_hash(source_path),
        "model_source": name,
        "source_path": str(source_path),
        "true_hash": file_hash(out / "true_velocity.npy"),
        "initial_hash": file_hash(out / "initial_velocity.npy"),
    }
    write_json(out / "manifest.json", manifest)
    return manifest
