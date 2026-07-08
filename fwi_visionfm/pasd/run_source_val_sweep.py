"""Source-validation-only lambda_edge sweep for Phase-1b locked B4 selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .data import VelocityScaler, load_arrays
from .diagnostics import edge_mask_from_threshold, edge_prf, gradient_magnitude_np, gradient_metrics, write_csv
from .experiment import ProtocolSplits, TrainingConfig, run_single_experiment
from .metrics import per_sample_metrics
from .protocol import load_protocol


def _candidate_lambda(name: str, current: float) -> float:
    if name == "current":
        return float(current)
    if name.startswith("lambda_edge_"):
        return float(name.replace("lambda_edge_", ""))
    raise ValueError(f"Unsupported candidate: {name}")


def _config_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def _rank(values: dict[str, float], higher: bool = False) -> dict[str, int]:
    ordered = sorted(values, key=lambda key: values[key], reverse=higher)
    return {key: index + 1 for index, key in enumerate(ordered)}


def run_sweep(protocol: str | Path, output: str | Path, variant: str, candidates: list[str], selection_split: str, selection_seed: int, epochs: int, batch_size: int, torch_threads: int) -> dict[str, Any]:
    if selection_split != "source_val":
        raise ValueError("selection-split must be exactly 'source_val'.")
    if torch_threads > 0:
        __import__("torch").set_num_threads(torch_threads)
    manifest = load_protocol(protocol)
    source_max = max(manifest.train_indices + manifest.val_indices + manifest.in_family_test_indices) + 1
    source = load_arrays(
        manifest.source.records,
        manifest.source.models,
        max_samples=source_max,
        sample_ids_path=manifest.source.sample_ids,
        family=manifest.source.family,
        source_positions_path=manifest.source.source_positions,
        receiver_positions_path=manifest.source.receiver_positions,
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    current = 0.10
    phase1_summary = Path("outputs/pasd_phase1/formal_flat_to_curve/B4_pasd_fwi/seed_0/metrics_summary.json")
    if phase1_summary.exists():
        current = float(json.loads(phase1_summary.read_text(encoding="utf-8"))["training"].get("weight_edge", current))
    resolved: dict[str, float] = {}
    for name in candidates:
        value = _candidate_lambda(name, current)
        if value not in resolved.values():
            resolved[name] = value

    train_velocities = source.velocities[np.asarray(manifest.train_indices, dtype=np.int64)]
    tau_source_90 = float(np.percentile(gradient_magnitude_np(train_velocities).reshape(-1), 90.0))
    rows: list[dict[str, Any]] = []
    for name, lambda_edge in resolved.items():
        run_dir = output / name
        config = TrainingConfig(epochs=epochs, batch_size=batch_size, seed=selection_seed, weight_edge=lambda_edge)
        run_single_experiment(
            source,
            ProtocolSplits(train=manifest.train_indices, val=manifest.val_indices, in_family_test=manifest.in_family_test_indices),
            run_dir,
            variant,
            config,
            target=None,
        )
        with np.load(run_dir / "predictions_val.npz") as payload:
            sample_ids = payload["sample_id"].astype(int)
            prediction = payload["prediction"]
            target = payload["target"]
        metric_payload = per_sample_metrics(__import__("torch").from_numpy(prediction), __import__("torch").from_numpy(target))
        pred_threshold = float(np.percentile(gradient_magnitude_np(prediction).reshape(-1), 90.0))
        edge_f1_values = []
        grad_edge_values = []
        for idx in range(len(sample_ids)):
            mask = edge_mask_from_threshold(target[idx], tau_source_90)
            pred_mask = gradient_magnitude_np(prediction[idx]) >= pred_threshold
            edge_f1_values.append(edge_prf(pred_mask, mask, tolerance_pixels=1)["edge_F1"])
            grad_edge_values.append(gradient_metrics(prediction[idx], target[idx], mask)["gradient_l1_edge"])
        row = {
            "candidate": name,
            "lambda_edge": lambda_edge,
            "selection_split": "source_val",
            "selection_seed": selection_seed,
            "MAE": float(metric_payload["mae"].mean().item()),
            "SSIM": float(metric_payload["ssim"].mean().item()),
            "edge_F1": float(np.mean(edge_f1_values)),
            "gradient_l1_edge": float(np.mean(grad_edge_values)),
            "target_loaded": False,
            "config_hash": _config_hash({"candidate": name, "lambda_edge": lambda_edge, "variant": variant, "epochs": epochs, "batch_size": batch_size}),
        }
        rows.append(row)
    p0 = next(row for row in rows if row["candidate"] == "current")
    eligible = [row for row in rows if float(row["MAE"]) <= float(p0["MAE"]) * 1.01]
    ranks = {
        "MAE": _rank({row["candidate"]: float(row["MAE"]) for row in eligible}, higher=False),
        "SSIM": _rank({row["candidate"]: float(row["SSIM"]) for row in eligible}, higher=True),
        "edge_F1": _rank({row["candidate"]: float(row["edge_F1"]) for row in eligible}, higher=True),
        "gradient_l1_edge": _rank({row["candidate"]: float(row["gradient_l1_edge"]) for row in eligible}, higher=False),
    }
    for row in rows:
        if row in eligible:
            row["MAE_guard_pass"] = True
            row["rank_score"] = sum(ranks[name][row["candidate"]] for name in ranks)
        else:
            row["MAE_guard_pass"] = False
            row["rank_score"] = None
    selected = sorted(
        eligible,
        key=lambda row: (row["rank_score"], float(row["MAE"]), -float(row["SSIM"]), float(row["gradient_l1_edge"]), float(row["lambda_edge"])),
    )[0]
    decision = {
        "selected_candidate": selected["candidate"],
        "lambda_edge": selected["lambda_edge"],
        "selection_rule": "MAE <= P0_MAE*1.01, then rank(MAE)+rank(SSIM)+rank(edge_F1)+rank(gradient_l1_edge)",
        "selection_split": "source_val",
        "selection_seed": selection_seed,
        "CurveVel-A target involvement": "none; target bundle was not loaded, scored, printed, or saved",
        "tau_source_90": tau_source_90,
        "selected_config_hash": selected["config_hash"],
    }
    write_csv(output / "source_selection_table.csv", rows)
    (output / "source_selection_decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# PASD Phase-1b Source-only Selection",
        "",
        "CurveVel-A target 未参与候选比较、排序、选择或阈值拟合。",
        "",
        f"- selected_candidate: {decision['selected_candidate']}",
        f"- lambda_edge: {decision['lambda_edge']}",
        f"- selection_split: {selection_split}",
        f"- selection_seed: {selection_seed}",
    ]
    (output / "source_selection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PASD Phase-1b source validation sweep.")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", default="B4_pasd_fwi")
    parser.add_argument("--candidates", nargs="+", default=["current", "lambda_edge_0.03", "lambda_edge_0.06"])
    parser.add_argument("--selection-split", required=True)
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    decision = run_sweep(args.protocol, args.output, args.variant, args.candidates, args.selection_split, args.selection_seed, args.epochs, args.batch_size, args.torch_threads)
    print(json.dumps({"status": "SUCCESS", **decision}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
