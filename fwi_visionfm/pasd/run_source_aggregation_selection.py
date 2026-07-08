"""Phase-3 source-val-only PASD-Core aggregation selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .data import load_arrays
from .diagnostics import write_csv
from .experiment import ProtocolSplits, TrainingConfig, run_single_experiment
from .phase3_utils import (
    archive_sample_rows,
    candidate_to_variant,
    load_json,
    mean_metric_rows,
    select_source_candidate,
    source_threshold,
    write_json,
)
from .protocol import load_protocol


def _training_config(args: argparse.Namespace, seed: int, locked: dict[str, Any]) -> TrainingConfig:
    return TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=seed,
        base_channels=args.base_channels,
        latent_channels=args.latent_channels,
        num_workers=0,
        weight_background=float(locked.get("lambda_bg", 0.25)),
        weight_edge=float(locked.get("lambda_edge", 0.10)),
        weight_smooth=float(locked.get("lambda_smooth", 0.02)),
        background_sigma=float(locked.get("Gaussian_sigma", 1.5)),
        generate_figures=False,
    )


def run_selection(args: argparse.Namespace) -> Path:
    if not args.forbid_target_access:
        raise ValueError("--forbid-target-access is required for Phase-3 source aggregation selection.")
    manifest = load_protocol(args.protocol)
    locked = load_json(args.locked_config)
    max_source = max(manifest.train_indices + manifest.val_indices + manifest.in_family_test_indices) + 1
    source = load_arrays(
        manifest.source.records,
        manifest.source.models,
        max_samples=max_source,
        sample_ids_path=manifest.source.sample_ids,
        family=manifest.source.family,
        source_positions_path=manifest.source.source_positions,
        receiver_positions_path=manifest.source.receiver_positions,
    )
    splits = ProtocolSplits(
        train=manifest.train_indices,
        val=manifest.val_indices,
        in_family_test=manifest.in_family_test_indices,
    )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tau = source_threshold(source.velocities[np.asarray(splits.train, dtype=np.int64)])
    sample_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for candidate in args.candidates:
        variant = candidate_to_variant(candidate)
        for seed in args.seeds:
            run_dir = out / "runs" / candidate / f"seed_{seed}"
            if not (run_dir / "predictions_val.npz").exists():
                run_single_experiment(source, splits, run_dir, variant, _training_config(args, seed, locked), target=None)
            rows = archive_sample_rows(
                run_dir / "predictions_val.npz",
                source.velocities[np.asarray(splits.train, dtype=np.int64)],
                "source_val",
                candidate,
                seed,
            )
            sample_rows.extend(rows)
            means = mean_metric_rows(rows, ("variant", "seed"))[0]
            seed_rows.append(
                {
                    "candidate": candidate,
                    "mapped_variant": variant,
                    "seed": seed,
                    "selection_split": args.selection_split,
                    **{k: v for k, v in means.items() if k not in {"variant", "seed"}},
                }
            )
    candidate_means = mean_metric_rows(seed_rows, ("candidate",))
    decision = select_source_candidate(candidate_means)
    decision.update(
        {
            "selection_split": args.selection_split,
            "source_family": source.family,
            "source_train_edge_threshold_tau90": tau,
            "target_files_accessed": [],
            "locked_config": str(args.locked_config),
            "protocol": str(args.protocol),
            "seeds": [int(seed) for seed in args.seeds],
        }
    )
    write_csv(out / "source_aggregation_selection.csv", seed_rows + candidate_means)
    write_csv(out / "source_aggregation_sample_metrics.csv", sample_rows)
    write_json(out / "source_aggregation_decision.json", decision)
    bootstrap = _source_selection_bootstrap(sample_rows, args.seeds)
    write_json(out / "source_aggregation_bootstrap.json", bootstrap)
    _write_report(out / "SOURCE_AGGREGATION_SELECTION_REPORT.md", decision, candidate_means, bootstrap)
    return out


def _source_selection_bootstrap(rows: list[dict[str, Any]], seeds: list[int], n_resamples: int = 2000) -> dict[str, Any]:
    by_key: dict[tuple[str, int, int], dict[str, Any]] = {
        (str(row["variant"]), int(row["seed"]), int(row["sample_id"])): row for row in rows
    }
    results: list[dict[str, Any]] = []
    rng = np.random.default_rng(0)
    for seed in seeds:
        c1_ids = {sample_id for candidate, s, sample_id in by_key if candidate == "C1_pasd_core_mean" and s == seed}
        c2_ids = {sample_id for candidate, s, sample_id in by_key if candidate == "C2_pasd_core_attention" and s == seed}
        ids = np.asarray(sorted(c1_ids.intersection(c2_ids)), dtype=np.int64)
        if ids.size == 0:
            continue
        for metric in ("MAE", "SSIM", "edge_F1", "gradient_l1_edge"):
            c1 = np.asarray([float(by_key[("C1_pasd_core_mean", seed, int(sample_id))][metric]) for sample_id in ids])
            c2 = np.asarray([float(by_key[("C2_pasd_core_attention", seed, int(sample_id))][metric]) for sample_id in ids])
            diff = c2 - c1
            boot = diff[rng.integers(0, diff.size, size=(n_resamples, diff.size))].mean(axis=1)
            results.append(
                {
                    "seed": int(seed),
                    "metric": metric,
                    "candidate_minus_baseline": "C2_pasd_core_attention - C1_pasd_core_mean",
                    "n_samples": int(ids.size),
                    "mean_difference": float(diff.mean()),
                    "ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
                }
            )
    return {"n_resamples": n_resamples, "comparisons": results}


def _write_report(path: Path, decision: dict[str, Any], means: list[dict[str, Any]], bootstrap: dict[str, Any]) -> None:
    lines = [
        "# PASD Phase-3 Source Aggregation Selection",
        "",
        "本报告只使用 FlatVel-A source validation；target 文件访问记录为空。",
        "",
        f"- Selected candidate: `{decision['selected_candidate']}`",
        f"- Selected mapped variant: `{decision['selected_variant']}`",
        f"- Selection rule: {decision['selection_rule']}",
        f"- Target files accessed: `{decision['target_files_accessed']}`",
        "",
        "## Candidate Means",
        "",
        "| candidate | MAE | SSIM | edge_F1 | gradient_l1_edge |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in means:
        lines.append(
            f"| {row['candidate']} | {float(row['MAE']):.6g} | {float(row['SSIM']):.6g} | "
            f"{float(row['edge_F1']):.6g} | {float(row['gradient_l1_edge']):.6g} |"
        )
    lines.extend(["", f"Bootstrap comparisons: {len(bootstrap['comparisons'])}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--locked-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--selection-split", default="source_val")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--latent-channels", type=int, default=96)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--forbid-target-access", action="store_true")
    args = parser.parse_args()
    import torch

    torch.set_num_threads(args.torch_threads)
    out = run_selection(args)
    print(json.dumps({"status": "SUCCESS", "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
