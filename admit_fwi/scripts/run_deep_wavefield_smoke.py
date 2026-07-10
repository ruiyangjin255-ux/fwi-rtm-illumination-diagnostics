from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from admit_fwi.acoustic_rtm import (
    RTMConfig,
    _step_wavefield,
    make_absorbing_mask,
    pad_rtm_config,
    pad_velocity_model,
    ricker_wavelet,
)
from admit_fwi.diagnostics.boundary_energy_audit import boundary_energy_ratio, boundary_mask, classify_boundary_energy
from admit_fwi.diagnostics.deep_wavefield_coverage import build_depth_roi_masks, energy, summarize_deep_energy
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


def save_snapshot_png(path: Path, field: np.ndarray, dx: float, dz: float, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clip = float(np.percentile(np.abs(field), 99.5)) or 1.0
    fig, ax = plt.subplots(figsize=(8.0, 3.8), constrained_layout=True)
    im = ax.imshow(field, cmap="seismic", vmin=-clip, vmax=clip, extent=[0, field.shape[1] * dx / 1000.0, field.shape[0] * dz / 1000.0, 0], aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Depth (km)")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def simulate_wavefield_energy(model: np.ndarray, cfg: RTMConfig, shot: int, snapshot_steps: dict[int, float], output_dir: Path, label: str) -> tuple[list[dict[str, float | int | str]], list[dict[str, str | float | int]]]:
    mask = make_absorbing_mask(cfg)
    velocity2_dt2 = (model.astype(np.float32) ** 2) * np.float32(cfg.dt * cfg.dt)
    wavelet = ricker_wavelet(cfg.nt, cfg.dt, cfg.f0, cfg.source_delay)
    rois = build_depth_roi_masks(model.shape, cfg.absorb_cells)
    bmask = boundary_mask(model.shape, cfg.absorb_cells)
    prev = np.zeros((cfg.nz, cfg.nx), dtype=np.float32)
    curr = np.zeros_like(prev)
    rows: list[dict[str, float | int | str]] = []
    snapshots: list[dict[str, str | float | int]] = []
    for it in range(cfg.nt):
        curr[cfg.source_z, shot] += wavelet[it]
        curr *= mask
        t = float(it * cfg.dt)
        e_physical = energy(curr, rois.physical)
        row = {
            "model": label,
            "shot": int(shot),
            "step": int(it),
            "time_s": t,
            "E_total": e_physical,
            "E_shallow": energy(curr, rois.shallow),
            "E_middle": energy(curr, rois.middle),
            "E_deep": energy(curr, rois.deep),
            "E_subsalt_proxy": energy(curr, rois.subsalt_proxy),
            "E_boundary": energy(curr, bmask),
        }
        row["boundary_energy_ratio"] = float(row["E_boundary"]) / (float(row["E_total"]) + 1.0e-12)
        rows.append(row)
        if it in snapshot_steps:
            seconds = snapshot_steps[it]
            filename = f"wavefield_{label}_shot{shot:03d}_t{seconds:.3f}s.png"
            save_snapshot_png(output_dir / "wavefield_snapshots" / filename, curr, cfg.dx, cfg.dz, f"{label} shot {shot} t={seconds:.3f}s")
            snapshots.append({"model": label, "shot": int(shot), "time_s": seconds, "step": int(it), "path": str(output_dir / "wavefield_snapshots" / filename)})
        nxt = _step_wavefield(prev, curr, velocity2_dt2, mask, cfg)
        prev, curr = curr, nxt
    return rows, snapshots


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deep-time forward wavefield coverage smoke test.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=3)
    parser.add_argument("--nt", type=int, default=None)
    args = parser.parse_args()
    config = read_simple_yaml(args.config)
    out_dir = ROOT / config.get("output_dir", "admit_fwi/outputs/deep_time_preflight_v1") / "wavefield_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    fwi_dir = ROOT / config.get("fwi_dir", "admit_fwi/outputs/FWI/full_salt_fwi_cg_audit0_train_ecg_v1")
    summary = json.loads((fwi_dir / "full_salt_fwi_summary.json").read_text(encoding="utf-8"))
    fwi_cfg = summary["config"]
    dt = float(config.get("time_dt", fwi_cfg["dt"]))
    if dt != float(fwi_cfg["dt"]):
        raise ValueError("deep-time dt must equal existing stable dt")
    nt = int(args.nt or config.get("time_nt", 5000))
    f0 = float(config.get("time_f0", 15.0))
    base_cfg = RTMConfig(
        nx=int(fwi_cfg["nx"]),
        nz=int(fwi_cfg["nz"]),
        dx=float(fwi_cfg["dx"]),
        dz=float(fwi_cfg["dz"]),
        dt=dt,
        nt=nt,
        f0=f0,
        source_x=int(fwi_cfg["nx"]) // 2,
        source_z=int(fwi_cfg["source_z"]),
        receiver_z=int(fwi_cfg["receiver_z"]),
        absorb_cells=int(fwi_cfg["absorb_cells"]),
        absorb_strength=float(config.get("absorb_strength", 3.0)),
        fd_order=int(fwi_cfg["fd_order"]),
        absorb_top=bool(config.get("absorb_top", False)),
    )
    pad_x = int(config.get("pad_x", 0))
    pad_top = int(config.get("pad_top", 0))
    pad_bottom = int(config.get("pad_bottom", 0))
    base_cfg = pad_rtm_config(base_cfg, pad_x=pad_x, pad_top=pad_top, pad_bottom=pad_bottom)
    initial = np.load(fwi_dir / "full_salt_initial_model.npy").astype(np.float32, copy=False)
    true = np.load(fwi_dir / "full_salt_true_model.npy").astype(np.float32, copy=False)
    if pad_x or pad_top or pad_bottom:
        initial = pad_velocity_model(initial, pad_x=pad_x, pad_top=pad_top, pad_bottom=pad_bottom)
        true = pad_velocity_model(true, pad_x=pad_x, pad_top=pad_top, pad_bottom=pad_bottom)
    audit_shots = [int(v) for v in summary["audit_split"]["audit_shots"]]
    if args.shots <= 1:
        shots = [audit_shots[len(audit_shots) // 2]]
    else:
        idx = np.linspace(0, len(audit_shots) - 1, int(args.shots)).round().astype(int)
        shots = [audit_shots[int(i)] for i in sorted(set(idx.tolist()))]
    requested_times = [float(v) for v in config.get("snapshot_times_seconds", [0.5, 1.0, 2.0, 3.0, 4.0, 5.0])]
    snapshot_steps = {int(round(t / dt)): t for t in requested_times if int(round(t / dt)) < nt}
    all_rows: list[dict[str, float | int | str]] = []
    snapshots: list[dict[str, str | float | int]] = []
    for label, model in (("initial", initial), ("true", true)):
        for shot in shots:
            rows, snaps = simulate_wavefield_energy(model, replace(base_cfg, source_x=shot), shot, snapshot_steps, out_dir, label)
            all_rows.extend(rows)
            snapshots.extend(snaps)
            print(f"wavefield smoke {label} shot={shot}", flush=True)
    csv_path = out_dir / "deep_energy_timeseries.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    summary_rows = []
    for label in ("initial", "true"):
        for shot in shots:
            subset = [r for r in all_rows if r["model"] == label and int(r["shot"]) == int(shot)]
            times = np.asarray([float(r["time_s"]) for r in subset])
            deep = np.asarray([float(r["E_deep"]) for r in subset])
            deep_summary = summarize_deep_energy(times, deep)
            item = {"model": label, "shot": int(shot), **{f"deep_{k}" if k == "status" else k: v for k, v in deep_summary.items()}}
            ratio = boundary_energy_ratio(np.asarray([float(r["E_boundary"]) for r in subset]), np.asarray([float(r["E_total"]) for r in subset]))
            boundary_summary = classify_boundary_energy(times, ratio, float(item["deep_energy_peak_time"]))
            item.update({f"boundary_{k}" if k == "status" else k: v for k, v in boundary_summary.items()})
            summary_rows.append(item)
    write_json(out_dir / "deep_energy_summary.json", {"status": "READY", "dt": dt, "nt": nt, "t_record": nt * dt, "shots": shots, "summary": summary_rows})
    write_json(out_dir / "wavefield_snapshots" / "snapshot_manifest.json", {"snapshots": snapshots, "mode": "physical_time"})
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    for label in ("initial", "true"):
        subset = [r for r in all_rows if r["model"] == label and int(r["shot"]) == shots[len(shots) // 2]]
        ax.plot([float(r["time_s"]) for r in subset], [float(r["E_deep"]) for r in subset], label=f"{label} deep")
        ax.plot([float(r["time_s"]) for r in subset], [float(r["E_middle"]) for r in subset], linestyle="--", label=f"{label} middle")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Wavefield energy")
    ax.set_yscale("symlog", linthresh=1.0e-16)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "figure_energy_curves.png", dpi=220)
    plt.close(fig)
    report = ["# Deep-Time Wavefield Smoke", "", f"- dt={dt}", f"- nt={nt}", f"- T_record={nt * dt:.3f} s", f"- shots={shots}"]
    for row in summary_rows:
        report.append(f"- {row['model']} shot {row['shot']}: {row['deep_status']}, deep_peak={row['deep_energy_peak_time']} s, boundary={row['boundary_status']} max_ratio={row.get('max_boundary_ratio')}")
    (out_dir / "deep_coverage_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest = {
        "status": "READY",
        "script": "run_deep_wavefield_smoke.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "git_commit": git_commit(),
        "config_path": str(args.config),
        "config_hash": file_hash(args.config),
        "input_hashes": {"initial": file_hash(fwi_dir / "full_salt_initial_model.npy"), "true": file_hash(fwi_dir / "full_salt_true_model.npy")},
        "rtm_config": asdict(base_cfg),
        "padding": {"pad_x": pad_x, "pad_top": pad_top, "pad_bottom": pad_bottom},
        "shots": shots,
        "snapshot_times_seconds": requested_times,
    }
    write_json(out_dir / "wavefield_smoke_manifest.json", manifest)
    print(json.dumps({"status": "READY", "shots": shots, "summary": str(out_dir / "deep_energy_summary.json")}, indent=2))


if __name__ == "__main__":
    main()
