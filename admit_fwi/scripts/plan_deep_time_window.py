from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from admit_fwi.diagnostics.deep_time_window import compute_required_record_time
from admit_fwi.scripts._common import read_simple_yaml, write_json


def yaml_none(value):
    return None if isinstance(value, str) and value.lower() in {"null", "none"} else value


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
    parser = argparse.ArgumentParser(description="Plan a CFL-stable deep-time record window.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = read_simple_yaml(args.config)
    out_dir = ROOT / config.get("output_dir", "admit_fwi/outputs/deep_time_preflight_v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    fwi_dir = ROOT / config.get("fwi_dir", "admit_fwi/outputs/FWI/full_salt_fwi_cg_audit0_train_ecg_v1")
    summary = json.loads((fwi_dir / "full_salt_fwi_summary.json").read_text(encoding="utf-8"))
    fwi_cfg = summary["config"]
    model_path = fwi_dir / "full_salt_initial_model.npy"
    model = np.load(model_path).astype(np.float32, copy=False)
    source_positions = [int(v) for v in summary["audit_split"]["audit_shots"]]
    receiver_positions = list(range(int(fwi_cfg["nx"])))
    dt = float(config.get("time_dt", fwi_cfg["dt"]))
    if dt != float(fwi_cfg["dt"]):
        raise ValueError("deep-time dt must equal existing stable dt")
    f0 = float(config.get("time_f0", 15.0))
    plan = compute_required_record_time(
        model,
        dx=float(fwi_cfg["dx"]),
        dz=float(fwi_cfg["dz"]),
        source_positions=source_positions,
        receiver_positions=receiver_positions,
        dt=dt,
        wavelet_peak_time=float(config.get("wavelet_peak_time", 1.0 / f0)),
        f0=f0,
        pml_thickness=int(fwi_cfg["absorb_cells"]),
        target_depth_fraction=float(config.get("target_depth_fraction", 0.95)),
        safety_factor=float(config.get("safety_factor", 1.15)),
        margin_seconds=yaml_none(config.get("margin_seconds", None)),
        v_ref_percentile=float(config.get("v_ref_percentile", 10)),
        current_nt=int(summary["config"]["nt"]),
        nt_floor=int(config.get("time_nt", 5000)),
        stride=100,
    )
    payload = {
        "status": "READY",
        "script": "plan_deep_time_window.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "git_commit": git_commit(),
        "config_path": str(args.config),
        "config_hash": file_hash(args.config),
        "model_path": str(model_path),
        "model_hash": file_hash(model_path),
        "dt": dt,
        "dt_unchanged": dt == float(fwi_cfg["dt"]),
        "current_nt": int(fwi_cfg["nt"]),
        "current_t_record": float(fwi_cfg["nt"]) * dt,
        **asdict(plan),
        "note": "This is a conservative kinematic arrival-time proxy, not exact ray tracing.",
    }
    write_json(out_dir / "deep_time_plan.json", payload)
    with (out_dir / "depth_arrival_proxy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plan.depth_rows[0]))
        writer.writeheader()
        writer.writerows(plan.depth_rows)
    depths = [row["depth_m"] for row in plan.depth_rows]
    max_times = [row["max_path_time_s"] for row in plan.depth_rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    ax.plot(max_times, depths, marker="o")
    ax.invert_yaxis()
    ax.set_xlabel("Conservative max path time (s)")
    ax.set_ylabel("Target depth (m)")
    ax.set_title("Deep-time arrival proxy")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "figure_arrival_time_proxy.png", dpi=220)
    plt.close(fig)
    lines = [
        "# Deep-Time Window Plan",
        "",
        "This is a conservative kinematic arrival-time proxy, not exact ray tracing.",
        "",
        f"- Existing dt: {dt} s",
        f"- Existing audit0 FWI nt: {fwi_cfg['nt']}, T={float(fwi_cfg['nt']) * dt:.3f} s",
        f"- v_ref P{config.get('v_ref_percentile', 10)}: {plan.v_ref:.2f} m/s",
        f"- Required time: {plan.required_time:.3f} s",
        f"- nt_required: {plan.nt_required}",
        f"- nt_recommended: {plan.nt_recommended}",
        f"- current_satisfies: {plan.current_satisfies}",
        "- dt must remain unchanged.",
    ]
    (out_dir / "deep_time_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"nt_recommended": plan.nt_recommended, "required_time": plan.required_time}, indent=2))


if __name__ == "__main__":
    main()
