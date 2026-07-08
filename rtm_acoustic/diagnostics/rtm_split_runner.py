from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from rtm_acoustic.acoustic_rtm import (
    RTMConfig,
    crop_padded_model,
    crop_padded_record,
    multishot_reverse_time_migrate_parallel,
    preprocess_migration_section,
    preprocess_stacked_record,
)
from rtm_acoustic.plot_paper_style import save_migration_figure, save_record_and_migration_figure
from rtm_acoustic.scripts.run_gate_rtm_audit import MODEL_FILES, file_hash

P0C_AUDIT_RTM_SHOTS = [4, 64, 124, 184, 244, 304, 364, 424, 484, 544, 604, 664]
SUBSETS = {
    "subset_A": P0C_AUDIT_RTM_SHOTS[0::2],
    "subset_B": P0C_AUDIT_RTM_SHOTS[1::2],
}


def git_commit(root: Path) -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if result.returncode != 0:
            return "NO_GIT_REPOSITORY"
        return result.stdout.strip() or "NO_GIT_REPOSITORY"
    except Exception:
        return "NO_GIT_REPOSITORY"


def selected_subsets(smoke: bool) -> dict[str, list[int]]:
    if not smoke:
        return {name: list(shots) for name, shots in SUBSETS.items()}
    return {name: [shots[0]] for name, shots in SUBSETS.items()}


def rtm_config_from_fwi_summary(summary: dict[str, Any], *, nt: int = 600, f0: float = 15.0) -> RTMConfig:
    cfg = summary["config"]
    return RTMConfig(
        nx=int(cfg["nx"]),
        nz=int(cfg["nz"]),
        dx=float(cfg["dx"]),
        dz=float(cfg["dz"]),
        dt=0.001,
        nt=int(nt),
        f0=float(f0),
        source_x=int(cfg["nx"]) // 2,
        source_z=int(cfg["source_z"]),
        receiver_z=int(cfg["receiver_z"]),
        absorb_cells=int(cfg["absorb_cells"]),
        fd_order=int(cfg["fd_order"]),
    )


def _case_complete(case_dir: Path, shots: list[int], cfg: RTMConfig) -> bool:
    meta_path = case_dir / "rtm_metadata.json"
    required = [
        case_dir / "rtm_raw.npy",
        case_dir / "rtm_laplacian_filtered_physical.npy",
        case_dir / "rtm_source_normalized_physical.npy",
        meta_path,
    ]
    if not all(path.exists() for path in required):
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta.get("shot_indices") == shots and int(meta.get("nt", -1)) == cfg.nt and float(meta.get("f0", -1.0)) == cfg.f0


def save_split_case(case_dir: Path, result: Any, *, cfg: RTMConfig, original_shape: tuple[int, int], method: str, subset_name: str, shots: list[int], model_path: Path, command: str, root: Path) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "rtm_raw": result.image.astype(np.float32),
        "rtm_laplacian_filtered": result.filtered_image.astype(np.float32),
        "rtm_source_normalized": result.normalized_image.astype(np.float32),
        "rtm_source_receiver_normalized": result.source_receiver_normalized_image.astype(np.float32),
        "stacked_record_raw": result.stacked_record.astype(np.float32),
        "illumination": result.illumination.astype(np.float32),
        "receiver_illumination": result.receiver_illumination.astype(np.float32),
    }
    for name, arr in arrays.items():
        np.save(case_dir / f"{name}.npy", arr)
    cropped_filtered = crop_padded_model(arrays["rtm_laplacian_filtered"], original_shape, 0, 0)
    cropped_norm = crop_padded_model(arrays["rtm_source_normalized"], original_shape, 0, 0)
    cropped_record = crop_padded_record(arrays["stacked_record_raw"], original_nx=original_shape[1], pad_x=0)
    np.save(case_dir / "rtm_laplacian_filtered_physical.npy", cropped_filtered.astype(np.float32))
    np.save(case_dir / "rtm_source_normalized_physical.npy", cropped_norm.astype(np.float32))
    display_record = preprocess_stacked_record(cropped_record, dt=cfg.dt, mute_time=0.0, time_power=0.2)
    display_migration = preprocess_migration_section(cropped_filtered, depth_power=0.15, clip_percentile=99.5, trace_balance=0.25, output_clip=0.80)
    save_migration_figure(case_dir / "rtm_display.png", display_migration, dx=cfg.dx, dz=cfg.dz, title=f"{method} {subset_name}")
    save_record_and_migration_figure(case_dir / "record_and_rtm.png", display_record, display_migration, dx=cfg.dx, dz=cfg.dz, dt=cfg.dt)
    metadata = {
        "method": method,
        "velocity_model_path": str(model_path),
        "subset_name": subset_name,
        "shot_indices": shots,
        "dt": cfg.dt,
        "nt": cfg.nt,
        "T_record": cfg.dt * cfg.nt,
        "f0": cfg.f0,
        "rtm_config": asdict(cfg),
        "rtm_config_hash": file_hash(model_path) + f":nt{cfg.nt}:f0{cfg.f0}",
        "input_model_hash": file_hash(model_path),
        "command_line": command,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(root),
    }
    (case_dir / "rtm_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def run_split_subsets(
    *,
    root: Path,
    fwi_dir: Path,
    model_dir: Path,
    output_dir: Path,
    methods: list[str],
    smoke: bool,
    workers: int,
    command: str | None = None,
) -> dict[str, Any]:
    if output_dir.name == "audit0_gate_rtm_v1":
        raise ValueError("refusing to write into existing stacked RTM directory")
    summary = json.loads((fwi_dir / "full_salt_fwi_summary.json").read_text(encoding="utf-8"))
    true_velocity = np.load(fwi_dir / "full_salt_true_model.npy").astype(np.float32)
    cfg = rtm_config_from_fwi_summary(summary, nt=600, f0=15.0)
    subsets = selected_subsets(smoke)
    records = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for subset_name, shots in subsets.items():
        for method in methods:
            if method not in MODEL_FILES:
                raise ValueError(f"unknown method: {method}")
            model_path = model_dir / MODEL_FILES[method]
            if not model_path.exists():
                raise FileNotFoundError(f"missing model for {method}: {model_path}")
            model = np.load(model_path).astype(np.float32)
            if model.shape != true_velocity.shape:
                raise ValueError(f"{method} shape {model.shape} != {true_velocity.shape}")
            case_dir = output_dir / subset_name / method
            status = "READY_CACHED"
            if not _case_complete(case_dir, shots, cfg):
                print(f"RTM split {subset_name} {method} shots={shots}", flush=True)
                status = "READY"
                result = multishot_reverse_time_migrate_parallel(
                    true_velocity,
                    cfg,
                    shots,
                    work_dir=case_dir / "work",
                    workers=max(1, int(workers)),
                    laplacian_power=1,
                    migration_velocity=model,
                    direct_mute_params={"direct_velocity": 2000.0, "padding_time": 0.03, "taper_time": 0.02},
                )
                save_split_case(
                    case_dir,
                    result,
                    cfg=cfg,
                    original_shape=true_velocity.shape,
                    method=method,
                    subset_name=subset_name,
                    shots=shots,
                    model_path=model_path,
                    command=command or " ".join(sys.argv),
                    root=root,
                )
            records.append({"subset": subset_name, "method": method, "status": status, "shot_indices": shots, "path": str(case_dir)})
    manifest = {
        "status": "READY",
        "script": "run_rtm_split_subsets_audit0.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": command or " ".join(sys.argv),
        "git_commit": git_commit(root),
        "smoke": bool(smoke),
        "rtm_config": asdict(cfg),
        "subsets": subsets,
        "methods": methods,
        "records": records,
        "output_dir": str(output_dir),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
