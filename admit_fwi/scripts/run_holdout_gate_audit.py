from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from admit_fwi.acoustic_rtm import RTMConfig, forward_model
from admit_fwi.diagnostics.heldout_audit import (
    ERROR_METRICS,
    audit_record_pair,
    paired_bootstrap,
    summarize_metric_rows,
)
from admit_fwi.diagnostics.shot_partition import interleaved_audit_split
from admit_fwi.scripts._common import ensure_output_tree, read_simple_yaml, write_json


MODEL_FILES = {
    "initial": "initial_model.npy",
    "full_fwi": "full_fwi_model.npy",
    "global": "global_matched_model.npy",
    "illumination": "illumination_only_matched_model.npy",
    "consensus": "gradient_consensus_only_matched_model.npy",
    "depth": "depth_matched_model.npy",
    "inverse": "inverse_illumination_negative_control_model.npy",
    "ecg": "ecg_reliability_gate_model.npy",
    "random_seed_0": "random_matched_seed_0_model.npy",
    "random_seed_1": "random_matched_seed_1_model.npy",
    "random_seed_2": "random_matched_seed_2_model.npy",
    "random_seed_3": "random_matched_seed_3_model.npy",
    "random_seed_4": "random_matched_seed_4_model.npy",
}


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return result.stdout.strip() or "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def require_finite(name: str, array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    metric_cols = ["normalized_l2_residual_mean", "nrms_residual_mean", "trace_correlation_mean", "envelope_error_mean", "phase_error_mean"]
    lines = [
        "# Audit-shot data-space residual summary",
        "",
        "| method | shots | normalized_l2 | nrms | trace_corr | envelope | phase |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        values = [float(row[col]) for col in metric_cols]
        lines.append(
            f"| {row['method']} | {row['shot_count']} | "
            f"{values[0]:.6g} | {values[1]:.6g} | {values[2]:.6g} | {values[3]:.6g} | {values[4]:.6g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_models(model_dir: Path, expected_shape: tuple[int, int]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    models: dict[str, np.ndarray] = {}
    manifest_rows: list[dict[str, Any]] = []
    for method, filename in MODEL_FILES.items():
        path = model_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"missing candidate model for {method}: {path}")
        model = np.load(path).astype(np.float32, copy=False)
        require_finite(method, model)
        if model.shape != expected_shape:
            raise ValueError(f"{method} model shape {model.shape} != {expected_shape}")
        models[method] = model
        manifest_rows.append(
            {
                "method": method,
                "path": str(path),
                "hash": file_hash(path),
                "min": float(np.min(model)),
                "max": float(np.max(model)),
                "mean": float(np.mean(model)),
            }
        )
    return models, manifest_rows


def source_offset_stats(source_x: int, nx: int, dx: float) -> dict[str, float]:
    offsets = (np.arange(nx, dtype=np.float64) - float(source_x)) * float(dx)
    return {
        "offset_min_m": float(np.min(offsets)),
        "offset_max_m": float(np.max(offsets)),
        "offset_abs_mean_m": float(np.mean(np.abs(offsets))),
    }


def observation_path(fwi_dir: Path, source_x: int) -> Path:
    return fwi_dir / "observations" / f"shot_{int(source_x):05d}.npy"


def save_observation(path: Path, record: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("wb") as handle:
        np.save(handle, np.asarray(record, dtype=np.float32))
    tmp.replace(path)


def load_or_create_observation(true_model: np.ndarray, cfg: RTMConfig, fwi_dir: Path, source_x: int) -> np.ndarray:
    path = observation_path(fwi_dir, source_x)
    if path.exists():
        observed = np.load(path).astype(np.float32, copy=False)
    else:
        observed = forward_model(true_model, RTMConfig(**{**asdict(cfg), "source_x": int(source_x)}))
        save_observation(path, observed)
    require_finite(f"observed shot {source_x}", observed)
    return observed


def compute_method_shot_metrics(task: tuple[str, str, str, str, dict[str, Any], int, int]) -> dict[str, Any]:
    method, model_path, observed_path, fwi_dir, cfg_payload, source_x, local_index = task
    cfg = RTMConfig(**{**cfg_payload, "source_x": int(source_x)})
    model = np.load(model_path).astype(np.float32, copy=False)
    observed = np.load(observed_path).astype(np.float32, copy=False)
    require_finite(method, model)
    require_finite(f"observed shot {source_x}", observed)
    predicted = forward_model(model, cfg)
    metrics = audit_record_pair(predicted, observed)
    return {
        "method": method,
        "shot_index": int(local_index),
        "source_x": int(source_x),
        **source_offset_stats(int(source_x), cfg.nx, cfg.dx),
        **metrics,
    }


def pairwise_rows(rows: list[dict[str, Any]], comparator: str) -> list[dict[str, Any]]:
    by_key = {(row["method"], row["shot_index"]): row for row in rows}
    shots = sorted({int(row["shot_index"]) for row in rows if row["method"] == "ecg"})
    output: list[dict[str, Any]] = []
    for shot in shots:
        ecg = by_key[("ecg", shot)]
        cmp_row = by_key[(comparator, shot)]
        item: dict[str, Any] = {"shot_index": shot, "source_x": ecg["source_x"], "comparator": comparator}
        for metric in (*ERROR_METRICS, "trace_correlation"):
            item[f"ecg_{metric}"] = ecg[metric]
            item[f"{comparator}_{metric}"] = cmp_row[metric]
            item[f"delta_{metric}"] = float(ecg[metric]) - float(cmp_row[metric])
        output.append(item)
    return output


def run_data_space_audit(
    *,
    config_path: Path,
    fwi_dir: Path,
    output_dir: Path,
    model_dir: Path,
    smoke: bool,
    max_audit_shots: int | None,
    bootstrap_samples: int,
    workers: int,
) -> dict[str, Any]:
    config = read_simple_yaml(config_path)
    ensure_output_tree(output_dir)
    audit_dir = output_dir / "audit"
    summary_path = fwi_dir / "full_salt_fwi_summary.json"
    true_path = fwi_dir / "full_salt_true_model.npy"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing FWI summary: {summary_path}")
    if not true_path.exists():
        raise FileNotFoundError(f"missing synthetic true model for held-out observed data: {true_path}")
    fwi_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    fwi_config = fwi_summary["config"]
    true_model = np.load(true_path).astype(np.float32, copy=False)
    require_finite("true_model", true_model)

    cfg = RTMConfig(
        nx=int(fwi_config["nx"]),
        nz=int(fwi_config["nz"]),
        dx=float(fwi_config["dx"]),
        dz=float(fwi_config["dz"]),
        dt=float(fwi_config["dt"]),
        nt=int(fwi_config["nt"]),
        f0=float(fwi_config["f0"]),
        source_x=int(fwi_config["nx"]) // 2,
        source_z=int(fwi_config["source_z"]),
        receiver_z=int(fwi_config["receiver_z"]),
        absorb_cells=int(fwi_config["absorb_cells"]),
        fd_order=int(fwi_config["fd_order"]),
    )
    if true_model.shape != (cfg.nz, cfg.nx):
        raise ValueError(f"true model shape {true_model.shape} != {(cfg.nz, cfg.nx)}")
    models, model_manifest = load_models(model_dir, true_model.shape)
    model_paths = {method: str(model_dir / filename) for method, filename in MODEL_FILES.items()}

    audit_shots = [int(value) for value in fwi_summary["audit_split"]["audit_shots"]]
    if max_audit_shots is not None:
        audit_shots = audit_shots[: int(max_audit_shots)]
    if smoke:
        audit_shots = audit_shots[:4]
    if not audit_shots:
        raise ValueError("no audit shots selected")

    observed_paths: dict[int, str] = {}
    for local_index, source_x in enumerate(audit_shots):
        load_or_create_observation(true_model, cfg, fwi_dir, source_x)
        observed_paths[source_x] = str(observation_path(fwi_dir, source_x))

    tasks = [
        (method, model_path, observed_paths[source_x], str(fwi_dir), asdict(cfg), int(source_x), int(local_index))
        for local_index, source_x in enumerate(audit_shots)
        for method, model_path in model_paths.items()
    ]
    rows: list[dict[str, Any]] = []
    workers = max(1, int(workers))
    if workers == 1:
        for index, task in enumerate(tasks, start=1):
            rows.append(compute_method_shot_metrics(task))
            print(f"audit task {index}/{len(tasks)} method={task[0]} source_x={task[5]}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(compute_method_shot_metrics, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                rows.append(future.result())
                print(f"audit task {index}/{len(tasks)} method={task[0]} source_x={task[5]}", flush=True)
    rows.sort(key=lambda row: (int(row["shot_index"]), str(row["method"])))

    summary_rows = summarize_metric_rows(rows)
    write_csv(audit_dir / "audit_per_shot_metrics.csv", rows)
    write_csv(audit_dir / "audit_method_summary.csv", summary_rows)
    write_summary_md(audit_dir / "audit_method_summary.md", summary_rows)
    for comparator in ("illumination", "global"):
        write_csv(audit_dir / f"audit_pairwise_vs_{comparator}.csv", pairwise_rows(rows, comparator))

    comparators = ["illumination", "global", "random_seed_0", "random_seed_1", "random_seed_2", "random_seed_3", "random_seed_4"]
    bootstrap = paired_bootstrap(
        rows,
        reference_method="ecg",
        comparator_methods=comparators,
        samples=int(bootstrap_samples),
    )
    write_json(audit_dir / "audit_bootstrap.json", bootstrap)

    split = interleaved_audit_split(list(range(224)), int(config.get("audit_fold", 0)), int(config.get("audit_num_folds", 4)))
    manifest = {
        "status": "READY",
        "script": "run_holdout_gate_audit.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "git_commit": git_commit(),
        "smoke": bool(smoke),
        "true_model_use": "synthetic observed data generation only; not used for gate, budget, coverage or parameter selection",
        "config_path": str(config_path),
        "config_hash": file_hash(config_path),
        "fwi_dir": str(fwi_dir),
        "fwi_summary_hash": file_hash(summary_path),
        "model_dir": str(model_dir),
        "models": model_manifest,
        "rtm_config": asdict(cfg),
        "workers": int(workers),
        "audit_shots": audit_shots,
        "audit_shot_count": len(audit_shots),
        "index_split": {
            "audit_fold": split.audit_fold,
            "inversion_shots": split.inversion_shots,
            "audit_shots": split.audit_shots,
        },
        "outputs": {
            "per_shot_metrics": str(audit_dir / "audit_per_shot_metrics.csv"),
            "method_summary": str(audit_dir / "audit_method_summary.csv"),
            "method_summary_md": str(audit_dir / "audit_method_summary.md"),
            "bootstrap": str(audit_dir / "audit_bootstrap.json"),
        },
    }
    write_json(audit_dir / "audit_manifest.json", manifest)
    return manifest


def write_split_manifest(config_path: Path, output_dir: Path, audit_fold: int | None, all_audit_folds: bool, smoke: bool) -> None:
    ensure_output_tree(output_dir)
    config = read_simple_yaml(config_path)
    shots = list(range(16 if smoke else 224))
    folds = range(4) if all_audit_folds else [audit_fold if audit_fold is not None else int(config.get("audit_fold", 0))]
    payload = {
        "status": "SPLIT_PREPARED",
        "folds": [
            {"audit_fold": split.audit_fold, "inversion_shots": split.inversion_shots, "audit_shots": split.audit_shots}
            for split in (interleaved_audit_split(shots, int(fold), 4) for fold in folds)
        ],
        "note": "Split metadata only; run without --split-only for data-space residual audit.",
    }
    write_json(output_dir / "audit" / "heldout_audit_manifest.json", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run held-out audit-shot data-space residual diagnostics.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audit-fold", type=int, default=None)
    parser.add_argument("--all-audit-folds", action="store_true")
    parser.add_argument("--split-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-audit-shots", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--fwi-dir", type=Path, default=ROOT / "admit_fwi" / "outputs" / "FWI" / "full_salt_fwi_cg_audit0_train_ecg_v1")
    parser.add_argument("--model-dir", type=Path, default=None)
    args = parser.parse_args()

    config = read_simple_yaml(args.config)
    base_output_dir = ROOT / config.get("output_dir", "admit_fwi/outputs/salt_reliability_gate_audit0_v1")
    output_dir = ROOT / "admit_fwi" / "outputs" / "smoke_reliability_gate_audit0" if args.smoke else base_output_dir
    model_dir = args.model_dir if args.model_dir is not None else base_output_dir / "models"
    if args.split_only or args.all_audit_folds:
        write_split_manifest(args.config, output_dir, args.audit_fold, args.all_audit_folds, args.smoke)
        print(output_dir)
        return
    manifest = run_data_space_audit(
        config_path=args.config,
        fwi_dir=args.fwi_dir,
        output_dir=output_dir,
        model_dir=model_dir,
        smoke=args.smoke,
        max_audit_shots=args.max_audit_shots,
        bootstrap_samples=args.bootstrap_samples,
        workers=args.workers,
    )
    print(json.dumps({"status": manifest["status"], "audit_shots": manifest["audit_shot_count"], "manifest": manifest["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
