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
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from rtm_acoustic.acoustic_rtm import (
    RTMConfig,
    crop_padded_model,
    crop_padded_record,
    multishot_reverse_time_migrate_parallel,
    preprocess_migration_section,
    preprocess_stacked_record,
)
from rtm_acoustic.plot_paper_style import save_migration_figure, save_record_and_migration_figure
from rtm_acoustic.scripts._common import read_simple_yaml, write_json
from rtm_acoustic.scripts.run_holdout_gate_audit import MODEL_FILES


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


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).ravel()
    bv = np.asarray(b, dtype=np.float64).ravel()
    if av.size != bv.size or av.size == 0 or float(np.std(av)) == 0.0 or float(np.std(bv)) == 0.0:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def image_metrics(image: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    image64 = np.asarray(image, dtype=np.float64)
    ref64 = np.asarray(reference, dtype=np.float64)
    diff = image64 - ref64
    return {
        "abs_mean": float(np.mean(np.abs(image64))),
        "abs_p95": float(np.percentile(np.abs(image64), 95.0)),
        "rms": float(np.sqrt(np.mean(image64 * image64))),
        "reference_corr": safe_corr(image64, ref64),
        "reference_rmse": float(np.sqrt(np.mean(diff * diff))),
        "reference_mae": float(np.mean(np.abs(diff))),
    }


def save_case_arrays(
    *,
    case_dir: Path,
    result: Any,
    reference_filtered: np.ndarray,
    reference_source_normalized: np.ndarray,
    cfg: RTMConfig,
    original_shape: tuple[int, int],
    pad_x: int,
    pad_top: int,
) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "stacked_record_raw": result.stacked_record.astype(np.float32),
        "rtm_raw": result.image.astype(np.float32),
        "rtm_source_normalized": result.normalized_image.astype(np.float32),
        "rtm_source_receiver_normalized": result.source_receiver_normalized_image.astype(np.float32),
        "rtm_laplacian_filtered": result.filtered_image.astype(np.float32),
        "illumination": result.illumination.astype(np.float32),
        "receiver_illumination": result.receiver_illumination.astype(np.float32),
    }
    for name, array in arrays.items():
        np.save(case_dir / f"{name}.npy", array)
    cropped_filtered = crop_padded_model(arrays["rtm_laplacian_filtered"], original_shape, pad_x, pad_top)
    cropped_norm = crop_padded_model(arrays["rtm_source_normalized"], original_shape, pad_x, pad_top)
    cropped_record = crop_padded_record(arrays["stacked_record_raw"], original_nx=original_shape[1], pad_x=pad_x)
    display_record = preprocess_stacked_record(cropped_record, dt=cfg.dt, mute_time=0.0, time_power=0.2)
    display_migration = preprocess_migration_section(cropped_filtered, depth_power=0.15, clip_percentile=99.5, trace_balance=0.25, output_clip=0.80)
    np.save(case_dir / "rtm_laplacian_filtered_physical.npy", cropped_filtered.astype(np.float32))
    np.save(case_dir / "rtm_source_normalized_physical.npy", cropped_norm.astype(np.float32))
    np.save(case_dir / "stacked_record_physical.npy", cropped_record.astype(np.float32))
    np.save(case_dir / "rtm_display.npy", display_migration.astype(np.float32))
    save_migration_figure(case_dir / "rtm_display.png", display_migration, dx=cfg.dx, dz=cfg.dz, title=case_dir.name)
    save_record_and_migration_figure(case_dir / "record_and_rtm.png", display_record, display_migration, dx=cfg.dx, dz=cfg.dz, dt=cfg.dt)
    return {
        "filtered": image_metrics(cropped_filtered, reference_filtered),
        "source_normalized": image_metrics(cropped_norm, reference_source_normalized),
    }


def load_case_metrics(case_dir: Path, reference_filtered: np.ndarray, reference_source_normalized: np.ndarray) -> dict[str, Any]:
    filtered_path = case_dir / "rtm_laplacian_filtered_physical.npy"
    norm_path = case_dir / "rtm_source_normalized_physical.npy"
    if not filtered_path.exists() or not norm_path.exists():
        raise FileNotFoundError(f"missing completed RTM arrays in {case_dir}")
    return {
        "filtered": image_metrics(np.load(filtered_path), reference_filtered),
        "source_normalized": image_metrics(np.load(norm_path), reference_source_normalized),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_gate_rtm_audit(
    *,
    config_path: Path,
    fwi_dir: Path,
    model_dir: Path,
    output_dir: Path,
    nt: int,
    f0: float,
    max_shots: int,
    workers: int,
    smoke: bool,
) -> dict[str, Any]:
    config = read_simple_yaml(config_path)
    fwi_summary = json.loads((fwi_dir / "full_salt_fwi_summary.json").read_text(encoding="utf-8"))
    fwi_config = fwi_summary["config"]
    true_velocity = np.load(fwi_dir / "full_salt_true_model.npy").astype(np.float32, copy=False)
    if not np.isfinite(true_velocity).all():
        raise ValueError("true velocity contains NaN or Inf")
    cfg = RTMConfig(
        nx=int(fwi_config["nx"]),
        nz=int(fwi_config["nz"]),
        dx=float(fwi_config["dx"]),
        dz=float(fwi_config["dz"]),
        dt=float(fwi_config["dt"]),
        nt=int(nt),
        f0=float(f0),
        source_x=int(fwi_config["nx"]) // 2,
        source_z=int(fwi_config["source_z"]),
        receiver_z=int(fwi_config["receiver_z"]),
        absorb_cells=int(fwi_config["absorb_cells"]),
        fd_order=int(fwi_config["fd_order"]),
    )
    audit_shots = [int(value) for value in fwi_summary["audit_split"]["audit_shots"]]
    if smoke:
        audit_shots = audit_shots[:2]
    elif max_shots > 0:
        indices = np.linspace(0, len(audit_shots) - 1, int(max_shots)).round().astype(int)
        audit_shots = [audit_shots[int(index)] for index in sorted(set(indices.tolist()))]
    if not audit_shots:
        raise ValueError("no audit RTM shots selected")

    output_dir.mkdir(parents=True, exist_ok=True)
    reference_dir = output_dir / "reference_true_velocity"
    original_shape = true_velocity.shape
    ref_filtered_path = reference_dir / "rtm_laplacian_filtered_physical.npy"
    ref_norm_path = reference_dir / "rtm_source_normalized_physical.npy"
    if ref_filtered_path.exists() and ref_norm_path.exists():
        ref_filtered = np.load(ref_filtered_path)
        ref_norm = np.load(ref_norm_path)
    else:
        reference = multishot_reverse_time_migrate_parallel(
            true_velocity,
            cfg,
            audit_shots,
            work_dir=reference_dir / "work",
            workers=max(1, int(workers)),
            laplacian_power=1,
            migration_velocity=true_velocity,
            direct_mute_params={"direct_velocity": 2000.0, "padding_time": 0.03, "taper_time": 0.02},
        )
        ref_filtered = crop_padded_model(reference.filtered_image, original_shape, 0, 0)
        ref_norm = crop_padded_model(reference.normalized_image, original_shape, 0, 0)
        save_case_arrays(
            case_dir=reference_dir,
            result=reference,
            reference_filtered=ref_filtered,
            reference_source_normalized=ref_norm,
            cfg=cfg,
            original_shape=original_shape,
            pad_x=0,
            pad_top=0,
        )

    rows: list[dict[str, Any]] = []
    model_records: list[dict[str, Any]] = []
    for method, filename in MODEL_FILES.items():
        model_path = model_dir / filename
        if not model_path.exists():
            raise FileNotFoundError(f"missing model for {method}: {model_path}")
        model = np.load(model_path).astype(np.float32, copy=False)
        if model.shape != true_velocity.shape:
            raise ValueError(f"{method} model shape {model.shape} != {true_velocity.shape}")
        if not np.isfinite(model).all():
            raise ValueError(f"{method} model contains NaN or Inf")
        case_dir = output_dir / method
        if (case_dir / "rtm_laplacian_filtered_physical.npy").exists() and (case_dir / "rtm_source_normalized_physical.npy").exists():
            print(f"RTM candidate {method} already complete; loading metrics", flush=True)
            metrics = load_case_metrics(case_dir, ref_filtered, ref_norm)
        else:
            print(f"RTM candidate {method}", flush=True)
            result = multishot_reverse_time_migrate_parallel(
                true_velocity,
                cfg,
                audit_shots,
                work_dir=case_dir / "work",
                workers=max(1, int(workers)),
                laplacian_power=1,
                migration_velocity=model,
                direct_mute_params={"direct_velocity": 2000.0, "padding_time": 0.03, "taper_time": 0.02},
            )
            metrics = save_case_arrays(
                case_dir=case_dir,
                result=result,
                reference_filtered=ref_filtered,
                reference_source_normalized=ref_norm,
                cfg=cfg,
                original_shape=original_shape,
                pad_x=0,
                pad_top=0,
            )
        row = {
            "method": method,
            "filtered_reference_rmse": metrics["filtered"]["reference_rmse"],
            "filtered_reference_mae": metrics["filtered"]["reference_mae"],
            "filtered_reference_corr": metrics["filtered"]["reference_corr"],
            "source_norm_reference_rmse": metrics["source_normalized"]["reference_rmse"],
            "source_norm_reference_corr": metrics["source_normalized"]["reference_corr"],
        }
        rows.append(row)
        model_records.append({"method": method, "model": str(model_path), "hash": file_hash(model_path)})

    before = next(row for row in rows if row["method"] == "initial")
    for row in rows:
        row["filtered_rmse_improvement_vs_initial"] = float((before["filtered_reference_rmse"] - row["filtered_reference_rmse"]) / max(before["filtered_reference_rmse"], 1.0e-20))
    rows.sort(key=lambda row: float(row["filtered_reference_rmse"]))
    write_csv(output_dir / "gate_rtm_method_summary.csv", rows)
    lines = ["# Gate RTM audit summary", "", "| rank | method | filtered RMSE | filtered corr | improvement vs initial |", "|---:|---|---:|---:|---:|"]
    for rank, row in enumerate(rows, start=1):
        lines.append(f"| {rank} | {row['method']} | {row['filtered_reference_rmse']:.6g} | {row['filtered_reference_corr']:.6g} | {row['filtered_rmse_improvement_vs_initial']:.6g} |")
    (output_dir / "gate_rtm_method_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "status": "READY",
        "script": "run_gate_rtm_audit.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "git_commit": git_commit(),
        "smoke": bool(smoke),
        "true_model_use": "synthetic RTM reference and observed wavefield only; not used for gate selection",
        "config_path": str(config_path),
        "config_hash": file_hash(config_path),
        "fwi_dir": str(fwi_dir),
        "model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "rtm_config": asdict(cfg),
        "audit_shots": audit_shots,
        "shot_count": len(audit_shots),
        "workers": int(workers),
        "models": model_records,
        "summary": str(output_dir / "gate_rtm_method_summary.csv"),
    }
    write_json(output_dir / "gate_rtm_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run audit0 RTM for all materialized candidate gate models.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fwi-dir", type=Path, default=ROOT / "rtm_acoustic" / "outputs" / "FWI" / "full_salt_fwi_cg_audit0_train_ecg_v1")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "rtm_acoustic" / "outputs" / "salt_reliability_gate_audit0_v1" / "models")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "rtm_acoustic" / "outputs" / "RTM" / "audit0_gate_rtm_v1")
    parser.add_argument("--nt", type=int, default=1200)
    parser.add_argument("--f0", type=float, default=15.0)
    parser.add_argument("--max-shots", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output_dir = ROOT / "rtm_acoustic" / "outputs" / "RTM" / "smoke_gate_rtm_audit0" if args.smoke else args.output_dir
    manifest = run_gate_rtm_audit(
        config_path=args.config,
        fwi_dir=args.fwi_dir,
        model_dir=args.model_dir,
        output_dir=output_dir,
        nt=args.nt,
        f0=args.f0,
        max_shots=args.max_shots,
        workers=args.workers,
        smoke=args.smoke,
    )
    print(json.dumps({"status": manifest["status"], "shot_count": manifest["shot_count"], "summary": manifest["summary"]}, indent=2))


if __name__ == "__main__":
    main()
