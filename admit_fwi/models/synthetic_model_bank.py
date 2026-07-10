from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from admit_fwi.diagnostics.admit_common import file_hash, git_commit, now_utc, text_hash, write_json

try:
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover
    gaussian_filter = None


def _smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
    if gaussian_filter is not None:
        return gaussian_filter(arr, sigma=sigma).astype(np.float32)
    out = arr.astype(np.float32).copy()
    for _ in range(max(1, int(sigma))):
        out[1:-1, 1:-1] = 0.2 * (out[1:-1, 1:-1] + out[:-2, 1:-1] + out[2:, 1:-1] + out[1:-1, :-2] + out[1:-1, 2:])
    return out


def simple_layered(nx: int = 240, nz: int = 120) -> np.ndarray:
    model = np.empty((nz, nx), dtype=np.float32)
    model[: nz // 4] = 1800.0
    model[nz // 4 : nz // 2] = 2300.0
    model[nz // 2 : 3 * nz // 4] = 2850.0
    model[3 * nz // 4 :] = 3400.0
    return model


def simple_fault(nx: int = 240, nz: int = 120) -> np.ndarray:
    model = simple_layered(nx, nz)
    for x in range(nx):
        throw = int(0.16 * nz) if x > nx // 2 else 0
        model[:, x] = np.roll(model[:, x], throw)
        if throw:
            model[:throw, x] = model[throw, x]
    return _smooth(model, 1.0)


def build_model(name: str, nx: int = 240, nz: int = 120) -> tuple[np.ndarray, np.ndarray]:
    if name == "simple_layered":
        true = simple_layered(nx, nz)
    elif name == "simple_fault":
        true = simple_fault(nx, nz)
    else:
        raise ValueError(f"unknown synthetic model: {name}")
    initial = _smooth(true, 8.0)
    if np.allclose(true, initial):
        raise ValueError("initial model must differ from true model")
    return true.astype(np.float32), initial.astype(np.float32)


def save_synthetic_model(name: str, output_root: Path, *, nx: int = 240, nz: int = 120, dx: float = 20.0, dz: float = 20.0, dt: float = 0.001, f0: float = 10.0) -> dict[str, Any]:
    true, initial = build_model(name, nx, nz)
    out = output_root / name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "true_velocity.npy", true)
    np.save(out / "initial_velocity.npy", initial)
    config = {
        "model_name": name,
        "model_source": "SYNTHETIC_DIAGNOSTIC_MODEL",
        "nx": nx,
        "nz": nz,
        "dx": dx,
        "dz": dz,
        "dt": dt,
        "f0": f0,
        "nt": 800,
        "source_z": 4,
        "receiver_z": 4,
        "shots": 16,
    }
    (out / "model_config.yaml").write_text("\n".join(f"{k}: {v}" for k, v in config.items()) + "\n", encoding="utf-8")
    manifest = {
        "status": "READY",
        "timestamp_utc": now_utc(),
        "git_commit": git_commit(),
        "command": f"generate_admit_simple_models.py --models {name}",
        "config_hash": text_hash(str(config)),
        "input_hash": "SYNTHETIC_DIAGNOSTIC_MODEL",
        "model_source": "SYNTHETIC_DIAGNOSTIC_MODEL",
        "true_hash": file_hash(out / "true_velocity.npy"),
        "initial_hash": file_hash(out / "initial_velocity.npy"),
        "velocity_min": float(np.min(true)),
        "velocity_max": float(np.max(true)),
    }
    write_json(out / "manifest.json", manifest)
    return manifest
