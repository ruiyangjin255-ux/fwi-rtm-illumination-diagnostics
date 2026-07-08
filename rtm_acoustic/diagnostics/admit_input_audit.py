from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rtm_acoustic.diagnostics.admit_common import file_hash, git_commit, now_utc, text_hash
from rtm_acoustic.scripts.run_holdout_gate_audit import MODEL_FILES


DEFAULT_SEARCH_DIRS = [
    Path(r"D:\ryjin\rtm_acoustic\data"),
    Path(r"D:\data"),
    Path(r"D:\data\geophysics"),
    Path(r"D:\data\marmousi"),
    Path(r"D:\data\sigsbee"),
    Path(r"D:\data\bp2004"),
    Path(r"D:\data\seg_models"),
    Path(r"D:\ryjin\datasets"),
    Path(r"D:\Workspace"),
]
MODEL_EXTENSIONS = {".npy", ".npz", ".mat", ".h5", ".hdf5", ".segy", ".sgy", ".bin", ".rsf"}
BENCHMARK_KEYS = {
    "marmousi": ["marmousi"],
    "marmousi2": ["marmousi2", "marmousi_2"],
    "sigsbee2a": ["sigsbee2a", "sigsbee_2a", "sigsbee"],
    "bp2004": ["bp2004", "bp_2004", "bpvelocity", "bp_velocity"],
    "overthrust": ["overthrust"],
}


def _exists(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "hash": file_hash(path) if path.is_file() else ""}


def _shape(path: Path) -> list[int] | None:
    if not path.exists() or path.suffix.lower() != ".npy":
        return None
    return list(np.load(path, mmap_mode="r").shape)


def discover_external_models(search_dirs: list[Path] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base in search_dirs or DEFAULT_SEARCH_DIRS:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in MODEL_EXTENSIONS:
                continue
            lower = path.name.lower()
            matched = [name for name, needles in BENCHMARK_KEYS.items() if any(key in lower for key in needles)]
            if not matched:
                continue
            rows.append(
                {
                    "benchmark": ";".join(matched),
                    "path": str(path),
                    "extension": path.suffix.lower(),
                    "bytes": path.stat().st_size,
                    "status": "AVAILABLE_CANDIDATE",
                }
            )
    return rows


def audit_inputs(root: Path, output_dir: Path, search_dirs: list[Path] | None = None) -> dict[str, Any]:
    fwi_dir = root / "outputs" / "FWI" / "full_salt_fwi_cg_audit0_train_ecg_v1"
    gate_root = root / "outputs" / "salt_reliability_gate_audit0_v1"
    model_dir = gate_root / "models"
    audit_dir = gate_root / "audit"
    rtm_dir = root / "outputs" / "RTM" / "audit0_gate_rtm_v1"
    deep_dir = root / "outputs" / "deep_time_preflight_v1"

    true_model_path = fwi_dir / "full_salt_true_model.npy"
    initial_model_path = fwi_dir / "full_salt_initial_model.npy"
    summary_path = fwi_dir / "full_salt_fwi_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    split = summary.get("audit_split", {})

    gate_rows = []
    for method, filename in MODEL_FILES.items():
        path = model_dir / filename
        gate_rows.append(
            {
                "method": method,
                "path": str(path),
                "exists": path.exists(),
                "shape": _shape(path),
                "hash": file_hash(path) if path.exists() else "MISSING",
            }
        )

    discovered = discover_external_models(search_dirs)
    found_text = " ".join(row["benchmark"] for row in discovered).lower()
    missing_external = [name for name in BENCHMARK_KEYS if name not in found_text]

    p0_b_files = {
        "audit_method_summary_csv": audit_dir / "audit_method_summary.csv",
        "audit_method_summary_md": audit_dir / "audit_method_summary.md",
        "audit_manifest": audit_dir / "heldout_audit_manifest.json",
    }
    p0_c_files = {
        "gate_rtm_summary_csv": rtm_dir / "gate_rtm_method_summary.csv",
        "gate_rtm_summary_md": rtm_dir / "gate_rtm_method_summary.md",
        "gate_rtm_manifest": rtm_dir / "gate_rtm_manifest.json",
    }
    deep_files = {
        "deep_time_plan": deep_dir / "deep_time_plan.json",
        "time_sampling_audit": deep_dir / "time_sampling_audit.json",
        "deep_energy_summary": deep_dir / "wavefield_smoke" / "deep_energy_summary.json",
        "boundary_energy_summary": deep_dir / "boundary_energy" / "boundary_energy_summary.json",
    }

    p1_ready = (
        summary_path.exists()
        and all(row["exists"] for row in gate_rows)
        and all(path.exists() for path in p0_b_files.values())
        and all(path.exists() for path in p0_c_files.values())
        and all(path.exists() for path in deep_files.values())
    )
    p2_ready = true_model_path.exists() and initial_model_path.exists()

    audit = {
        "status": "READY",
        "timestamp_utc": now_utc(),
        "git_commit": git_commit(),
        "command": "audit_admit_fwi_inputs.py",
        "config_hash": text_hash("default_admit_input_audit_v1"),
        "input_hash": file_hash(summary_path),
        "seg_salt": {
            "true_model": _exists(true_model_path),
            "initial_model": _exists(initial_model_path),
            "summary": _exists(summary_path),
            "current_model_shape": _shape(true_model_path),
            "train_shots": split.get("train_shots"),
            "audit_shots": split.get("audit_shots"),
        },
        "gate_models": gate_rows,
        "p0_b_files": {key: _exists(path) for key, path in p0_b_files.items()},
        "p0_c_files": {key: _exists(path) for key, path in p0_c_files.items()},
        "deep_time_preflight_files": {key: _exists(path) for key, path in deep_files.items()},
        "has_rtm_split_consistency_module": (root / "diagnostics" / "rtm_split_consistency.py").exists(),
        "has_salt_region_code": (root / "build_target_zone_illumination_diagnostics.py").exists(),
        "discovered_external_models": discovered,
        "missing_external_models": missing_external,
        "can_enter_p1_seg_salt": p1_ready,
        "can_enter_p2_model_staircase": p2_ready,
    }
    return audit
