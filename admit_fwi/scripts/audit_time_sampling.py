from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from admit_fwi.diagnostics.time_sampling_audit import array_stats, record_time, strict_disjoint, wavelet_summary
from admit_fwi.scripts._common import read_simple_yaml, write_json


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def git_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return result.stdout.strip() or "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit short/deep-time sampling and existing FWI/RTM paths.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = read_simple_yaml(args.config)
    out_dir = ROOT / config.get("output_dir", "admit_fwi/outputs/deep_time_preflight_v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    fwi_dir = ROOT / config.get("fwi_dir", "admit_fwi/outputs/FWI/full_salt_fwi_cg_audit0_train_ecg_v1")
    short_rtm_dir = ROOT / config.get("short_rtm_dir", "admit_fwi/outputs/RTM/audit0_gate_rtm_v1")
    gate_dir = ROOT / config.get("gate_dir", "admit_fwi/outputs/salt_reliability_gate_audit0_v1")
    fwi_summary_path = fwi_dir / "full_salt_fwi_summary.json"
    short_manifest_path = short_rtm_dir / "gate_rtm_manifest.json"
    if not fwi_summary_path.exists():
        raise FileNotFoundError(fwi_summary_path)
    if not short_manifest_path.exists():
        raise FileNotFoundError(short_manifest_path)
    fwi_summary = json.loads(fwi_summary_path.read_text(encoding="utf-8"))
    rtm_manifest = json.loads(short_manifest_path.read_text(encoding="utf-8"))
    fwi_cfg = fwi_summary["config"]
    rtm_cfg = rtm_manifest["rtm_config"]
    train_shots = [int(v) for v in fwi_summary["shot_positions"]]
    audit_shots = [int(v) for v in fwi_summary["audit_split"]["audit_shots"]]
    observed_dir = fwi_dir / "observations"
    observed_samples = sorted(observed_dir.glob("shot_*.npy"))[:3]
    payload: dict[str, Any] = {
        "status": "READY",
        "script": "audit_time_sampling.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "git_commit": git_commit(),
        "config_path": str(args.config),
        "config_hash": file_hash(args.config),
        "paths": {"fwi_dir": str(fwi_dir), "short_rtm_dir": str(short_rtm_dir), "gate_dir": str(gate_dir)},
        "fwi": {"dt": fwi_cfg["dt"], "nt": fwi_cfg["nt"], "t_record": record_time(int(fwi_cfg["nt"]), float(fwi_cfg["dt"]))},
        "audit0_fwi": {"dt": fwi_cfg["dt"], "nt": fwi_cfg["nt"], "t_record": record_time(int(fwi_cfg["nt"]), float(fwi_cfg["dt"]))},
        "p0c_rtm": {"dt": rtm_cfg["dt"], "nt": rtm_cfg["nt"], "t_record": record_time(int(rtm_cfg["nt"]), float(rtm_cfg["dt"])), "shallow_time_window": int(rtm_cfg["nt"]) == 600},
        "wavelet": wavelet_summary(float(rtm_cfg["f0"]), rtm_cfg.get("source_delay")),
        "model": {
            "true": array_stats(fwi_dir / "full_salt_true_model.npy"),
            "initial": array_stats(fwi_dir / "full_salt_initial_model.npy"),
            "inverted": array_stats(fwi_dir / "full_salt_inverted_model.npy"),
            "dx": fwi_cfg["dx"],
            "dz": fwi_cfg["dz"],
            "effective_depth_m": (int(fwi_cfg["nz"]) - int(fwi_cfg["absorb_cells"])) * float(fwi_cfg["dz"]),
            "pml_thickness_cells": fwi_cfg["absorb_cells"],
        },
        "geometry": {
            "source_positions_train": train_shots,
            "audit_shots": audit_shots,
            "receiver_positions": list(range(int(fwi_cfg["nx"]))),
            "train_audit_disjoint": strict_disjoint(train_shots, audit_shots),
            "rtm_audit_shots": rtm_manifest["audit_shots"],
        },
        "observed_data": [array_stats(path) for path in observed_samples],
        "snapshot_time_basis": "index unless file name explicitly contains physical seconds; deep-time smoke will use physical seconds",
        "pml": {"absorb_cells": fwi_cfg["absorb_cells"], "absorb_strength": rtm_cfg.get("absorb_strength", 3.0), "absorb_top": rtm_cfg.get("absorb_top", False)},
    }
    write_json(out_dir / "time_sampling_audit.json", payload)
    lines = [
        "# Time Sampling Audit",
        "",
        f"- FWI: dt={payload['fwi']['dt']}, nt={payload['fwi']['nt']}, T={payload['fwi']['t_record']} s",
        f"- P0-C RTM: dt={payload['p0c_rtm']['dt']}, nt={payload['p0c_rtm']['nt']}, T={payload['p0c_rtm']['t_record']} s",
        f"- SHALLOW_TIME_WINDOW_FOR_RTM = {str(payload['p0c_rtm']['shallow_time_window']).lower()}",
        f"- Audit train/audit disjoint = {payload['geometry']['train_audit_disjoint']}",
        f"- Effective physical depth excluding bottom absorbing cells = {payload['model']['effective_depth_m']} m",
        f"- Wavelet f0={payload['wavelet']['f0']} Hz, peak_time={payload['wavelet']['peak_time']} s, dominant_period={payload['wavelet']['dominant_period']} s",
    ]
    (out_dir / "time_sampling_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_dir / "time_sampling_audit.md")


if __name__ == "__main__":
    main()
