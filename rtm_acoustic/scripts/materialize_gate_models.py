from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from rtm_acoustic.scripts._common import read_simple_yaml, write_json


GATE_TO_MODEL = {
    "global_matched": "global_matched_model.npy",
    "illumination_only_matched": "illumination_only_matched_model.npy",
    "gradient_consensus_only_matched": "gradient_consensus_only_matched_model.npy",
    "depth_matched": "depth_matched_model.npy",
    "inverse_illumination_negative_control": "inverse_illumination_negative_control_model.npy",
    "ecg_reliability_gate": "ecg_reliability_gate_model.npy",
    "random_matched_seed_0": "random_matched_seed_0_model.npy",
    "random_matched_seed_1": "random_matched_seed_1_model.npy",
    "random_matched_seed_2": "random_matched_seed_2_model.npy",
    "random_matched_seed_3": "random_matched_seed_3_model.npy",
    "random_matched_seed_4": "random_matched_seed_4_model.npy",
}


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def array_hash(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()[:16]


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return result.stdout.strip()
    except Exception:
        return "UNKNOWN"


def _save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("wb") as handle:
        np.save(handle, np.asarray(array, dtype=np.float32))
    tmp.replace(path)


def _require_finite(name: str, array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")


def materialize_models(
    *,
    result_dir: Path,
    fwi_dir: Path,
    smoke: bool = False,
) -> dict[str, Any]:
    initial_path = fwi_dir / "full_salt_initial_model.npy"
    delta_path = result_dir / "diagnostics" / "delta_model.npy"
    if not initial_path.exists():
        raise FileNotFoundError(f"missing initial model: {initial_path}")
    if not delta_path.exists():
        raise FileNotFoundError(f"missing delta model: {delta_path}")
    initial = np.load(initial_path).astype(np.float32, copy=False)
    delta = np.load(delta_path).astype(np.float32, copy=False)
    _require_finite("initial", initial)
    _require_finite("delta", delta)
    if initial.shape != delta.shape:
        raise ValueError(f"initial and delta shapes differ: {initial.shape} vs {delta.shape}")

    models_dir = result_dir / "models"
    gates_dir = result_dir / "gates"
    records = []
    for gate_name, model_filename in GATE_TO_MODEL.items():
        gate_path = gates_dir / f"{gate_name}.npy"
        if not gate_path.exists():
            raise FileNotFoundError(f"missing gate array for {gate_name}: {gate_path}")
        gate = np.load(gate_path).astype(np.float32, copy=False)
        _require_finite(gate_name, gate)
        if gate.shape != initial.shape:
            raise ValueError(f"{gate_name} shape differs from initial: {gate.shape} vs {initial.shape}")
        model = initial + gate * delta
        _require_finite(model_filename, model)
        model_path = models_dir / model_filename
        _save_array(model_path, model)
        records.append(
            {
                "name": gate_name,
                "gate": str(gate_path),
                "model": str(model_path),
                "gate_hash": file_hash(gate_path),
                "model_hash": file_hash(model_path),
                "model_array_hash": array_hash(model),
                "model_min": float(np.min(model)),
                "model_max": float(np.max(model)),
                "model_mean": float(np.mean(model)),
            }
        )

    full_model_path = fwi_dir / "full_salt_inverted_model.npy"
    if full_model_path.exists():
        _save_array(models_dir / "full_fwi_model.npy", np.load(full_model_path).astype(np.float32, copy=False))
    _save_array(models_dir / "initial_model.npy", initial)

    manifest = {
        "status": "READY",
        "script": "materialize_gate_models.py",
        "smoke": bool(smoke),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "formula": "model = initial_model + alpha_gate * frozen_delta_model",
        "true_model_used": False,
        "result_dir": str(result_dir),
        "fwi_dir": str(fwi_dir),
        "initial_model": str(initial_path),
        "delta_model": str(delta_path),
        "initial_hash": file_hash(initial_path),
        "delta_hash": file_hash(delta_path),
        "models": records,
    }
    write_json(models_dir / "gate_model_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize frozen gate velocity models from frozen alpha gates.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fwi-dir", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = read_simple_yaml(args.config)
    result_dir = ROOT / config.get("output_dir", "rtm_acoustic/outputs/salt_reliability_gate_audit0_v1")
    if args.smoke:
        result_dir = ROOT / "rtm_acoustic" / "outputs" / "smoke_reliability_gate_audit0"
    fwi_dir = args.fwi_dir
    if fwi_dir is None:
        fwi_dir = ROOT / "rtm_acoustic" / "outputs" / "FWI" / "full_salt_fwi_cg_audit0_train_ecg_v1"
    manifest = materialize_models(result_dir=result_dir, fwi_dir=fwi_dir, smoke=args.smoke)
    print(json.dumps({"status": manifest["status"], "models": len(manifest["models"]), "manifest": str(result_dir / "models" / "gate_model_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
