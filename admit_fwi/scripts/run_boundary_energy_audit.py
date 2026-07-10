from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from admit_fwi.diagnostics.boundary_energy_audit import boundary_energy_ratio, classify_boundary_energy
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
    parser = argparse.ArgumentParser(description="Audit PML/boundary energy from deep-time smoke outputs.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = read_simple_yaml(args.config)
    preflight = ROOT / config.get("output_dir", "admit_fwi/outputs/deep_time_preflight_v1")
    smoke_dir = preflight / "wavefield_smoke"
    csv_path = smoke_dir / "deep_energy_timeseries.csv"
    summary_path = smoke_dir / "deep_energy_summary.json"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    out_dir = preflight / "boundary_energy"
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    timeseries_rows = []
    for key in sorted({(r["model"], r["shot"]) for r in rows}):
        subset = [r for r in rows if (r["model"], r["shot"]) == key]
        times = np.asarray([float(r["time_s"]) for r in subset])
        ratio = boundary_energy_ratio(np.asarray([float(r["E_boundary"]) for r in subset]), np.asarray([float(r["E_total"]) for r in subset]))
        deep = np.asarray([float(r["E_deep"]) for r in subset])
        deep_peak_time = float(times[int(np.argmax(deep))])
        cls = classify_boundary_energy(times, ratio, deep_peak_time)
        summaries.append({"model": key[0], "shot": int(key[1]), "deep_peak_time": deep_peak_time, **cls})
        for t, value in zip(times, ratio):
            timeseries_rows.append({"model": key[0], "shot": int(key[1]), "time_s": float(t), "boundary_energy_ratio": float(value)})
    with (out_dir / "boundary_energy_timeseries.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(timeseries_rows[0]))
        writer.writeheader()
        writer.writerows(timeseries_rows)
    fig, ax = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    for key in sorted({(r["model"], r["shot"]) for r in timeseries_rows}):
        subset = [r for r in timeseries_rows if (r["model"], r["shot"]) == key]
        ax.plot([r["time_s"] for r in subset], [r["boundary_energy_ratio"] for r in subset], label=f"{key[0]} {key[1]}", alpha=0.75)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Boundary / physical energy")
    ax.set_yscale("symlog", linthresh=1.0e-8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(out_dir / "figure_boundary_energy_ratio.png", dpi=220)
    plt.close(fig)
    manifest = {
        "status": "READY",
        "script": "run_boundary_energy_audit.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "git_commit": git_commit(),
        "config_path": str(args.config),
        "config_hash": file_hash(args.config),
        "input": str(csv_path),
        "input_hash": file_hash(csv_path),
        "summary": summaries,
    }
    write_json(out_dir / "boundary_energy_summary.json", manifest)
    lines = ["# Boundary Energy Audit", ""]
    for row in summaries:
        lines.append(f"- {row['model']} shot {row['shot']}: {row['status']}, max_ratio={row['max_boundary_ratio']:.6g}, final_window={row['final_window_boundary_ratio']:.6g}")
    (out_dir / "boundary_energy_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "READY", "summary": str(out_dir / "boundary_energy_summary.json")}, indent=2))


if __name__ == "__main__":
    main()
