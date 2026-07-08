"""Phase-3 fresh B1 vs PASD-Core training with CurveVel-A and FlatFault-A evaluation."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from .bootstrap import paired_bootstrap
from .data import VelocityScaler, load_arrays
from .diagnostics import write_csv
from .experiment import (
    ProtocolSplits,
    TrainingConfig,
    _make_loader,
    _save_archive,
    build_criterion,
    build_model,
    evaluate_model,
    run_single_experiment,
)
from .phase3_utils import archive_sample_rows, candidate_to_variant, load_json, mean_metric_rows, write_json
from .protocol import DatasetRef, ProtocolManifest, load_protocol_bundles
from .registry import get_variant


def _ref(payload: dict[str, Any]) -> DatasetRef:
    return DatasetRef(
        records=payload["records"],
        models=payload["models"],
        family=payload["family"],
        sample_ids=payload.get("sample_ids"),
        source_positions=payload.get("source_positions"),
        receiver_positions=payload.get("receiver_positions"),
    )


def _load_dual_protocol(path: Path) -> tuple[ProtocolManifest, dict[str, tuple[DatasetRef, tuple[int, ...]]], dict[str, Any]]:
    payload = load_json(path)
    split = payload["split"]
    source_manifest = ProtocolManifest.from_dict(
        {
            "version": payload.get("version", "pasd_phase3_dual_target_locked_v1"),
            "source": payload["source"],
            "target": payload["targets"]["CurveVel-A"],
            "split": {
                "train": split["train"],
                "val": split["val"],
                "in_family_test": split["in_family_test"],
                "cross_family_test": payload["targets"]["CurveVel-A"]["cross_family_test_indices"],
            },
            "seed": payload.get("seed", 0),
            "notes": payload.get("notes", ""),
            "metadata": payload.get("metadata", {}),
        },
        base_dir=path.parent,
    )
    targets = {
        name: (_ref(target), tuple(int(x) for x in target["cross_family_test_indices"]))
        for name, target in payload["targets"].items()
    }
    return source_manifest, targets, payload


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


def _variant_for(public_variant: str, locked: dict[str, Any]) -> str:
    if public_variant == "PASD_Core_locked":
        selected = str(locked["selected_candidate"])
        return candidate_to_variant(selected)
    return public_variant


def run_dual_target(args: argparse.Namespace) -> Path:
    locked = load_json(args.pasd_core_config)
    source_manifest, targets, payload = _load_dual_protocol(args.protocol)
    source, curve_target = load_protocol_bundles(source_manifest)
    if curve_target is None:
        raise ValueError("CurveVel-A target is required in the dual-target protocol.")
    out = Path(args.output)
    for sub in ("runs", "prediction_archives", "checkpoints", "histories", "diagnostics", "bootstrap", "figures", "tables"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    splits = ProtocolSplits(
        train=source_manifest.train_indices,
        val=source_manifest.val_indices,
        in_family_test=source_manifest.in_family_test_indices,
        cross_family_test=source_manifest.cross_family_test_indices,
    )
    run_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for public_variant in args.variants:
        model_variant = _variant_for(public_variant, locked)
        for seed in args.seeds:
            run_dir = out / "runs" / public_variant / f"seed_{seed}"
            config = _training_config(args, seed, locked)
            run_single_experiment(source, splits, run_dir, model_variant, config, target=curve_target)
            archive_map = {
                "in_family": run_dir / "predictions_in_family.npz",
                "cross_curvevel_a": run_dir / "predictions_cross_family.npz",
            }
            flat_ref, flat_indices = targets["FlatFault-A"]
            flat_target = load_arrays(
                flat_ref.records,
                flat_ref.models,
                max_samples=max(flat_indices) + 1,
                sample_ids_path=flat_ref.sample_ids,
                family=flat_ref.family,
                source_positions_path=flat_ref.source_positions,
                receiver_positions_path=flat_ref.receiver_positions,
            )
            flat_archive = _evaluate_extra_target(run_dir, source, flat_target, flat_indices, model_variant, config)
            archive_map["cross_flatfault_a"] = flat_archive
            for dataset_name, archive in archive_map.items():
                dest = out / "prediction_archives" / f"{public_variant}_seed{seed}_{dataset_name}.npz"
                shutil.copyfile(archive, dest)
                sample_rows.extend(
                    archive_sample_rows(
                        dest,
                        source.velocities[np.asarray(splits.train, dtype=np.int64)],
                        dataset_name,
                        public_variant,
                        seed,
                    )
                )
            shutil.copyfile(run_dir / "checkpoint.pt", out / "checkpoints" / f"{public_variant}_seed{seed}.pt")
            shutil.copyfile(run_dir / "history.csv", out / "histories" / f"{public_variant}_seed{seed}.csv")
            with (run_dir / "metrics_summary.json").open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            for dataset_name, metrics in (
                ("in_family", summary["metrics"]["in_family"]),
                ("cross_curvevel_a", summary["metrics"]["cross_family"]),
            ):
                run_rows.append(
                    {"variant": public_variant, "model_variant": model_variant, "seed": seed, "dataset": dataset_name, **metrics}
                )
            flat_metrics = mean_metric_rows([row for row in sample_rows if row["variant"] == public_variant and row["seed"] == seed and row["dataset"] == "cross_flatfault_a"], ("variant", "seed", "dataset"))[0]
            run_rows.append({"variant": public_variant, "model_variant": model_variant, "seed": seed, "dataset": "cross_flatfault_a", **{k: v for k, v in flat_metrics.items() if k not in {"variant", "seed", "dataset"}}})
    write_csv(out / "protocol_runs.csv", run_rows)
    write_csv(out / "diagnostics" / "phase3_corrected_sample_metrics.csv", sample_rows)
    summary_rows = mean_metric_rows(run_rows, ("variant", "dataset"))
    write_csv(out / "protocol_summary.csv", summary_rows)
    write_csv(out / "tables" / "Table_1_protocol_summary.csv", summary_rows)
    _bootstrap_outputs(out, args.seeds, args.bootstrap_resamples)
    _make_figures(out, summary_rows, sample_rows)
    _write_report(out, payload, locked, summary_rows)
    write_json(out / "phase3_dual_target_manifest.json", {"status": "SUCCESS", "protocol": str(args.protocol), "pasd_core_config": str(args.pasd_core_config), "variants": args.variants, "targets": args.targets})
    return out


def _evaluate_extra_target(
    run_dir: Path,
    source: Any,
    target: Any,
    indices: tuple[int, ...],
    variant_name: str,
    config: TrainingConfig,
) -> Path:
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location=config.device)
    scaler = VelocityScaler(**checkpoint["scaler"])
    variant = get_variant(variant_name)
    model = build_model(variant, tuple(int(x) for x in source.velocities.shape[-2:]), config).to(torch.device(config.device))
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = build_criterion(variant, config)
    loader = _make_loader(target, indices, scaler, config, shuffle=False)
    summary, rows, _, archive = evaluate_model(model, loader, torch.device(config.device), scaler, criterion)
    write_csv(run_dir / "metrics_cross_flatfault_a_per_sample.csv", rows)
    write_json(run_dir / "metrics_cross_flatfault_a.json", summary)
    path = run_dir / "predictions_cross_flatfault_a.npz"
    _save_archive(path, archive, scaler)
    return path


def _bootstrap_outputs(out: Path, seeds: list[int], n_resamples: int) -> None:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for dataset in ("in_family", "cross_curvevel_a", "cross_flatfault_a"):
            base = out / "prediction_archives" / f"B1_raw_unet_seed{seed}_{dataset}.npz"
            candidate = out / "prediction_archives" / f"PASD_Core_locked_seed{seed}_{dataset}.npz"
            for metric in ("mae", "rmse", "ssim", "edge_mae", "gradient_error"):
                result = paired_bootstrap(base, candidate, metric, n_resamples=n_resamples, seed=seed)
                result.update({"seed": seed, "dataset": dataset, "baseline": "B1_raw_unet", "candidate": "PASD_Core_locked"})
                write_json(out / "bootstrap" / f"{dataset}_seed{seed}_{metric}.json", result)
                rows.append(result)
    write_csv(out / "tables" / "Table_2_paired_bootstrap.csv", rows)


def _make_figures(out: Path, summary_rows: list[dict[str, Any]], sample_rows: list[dict[str, Any]]) -> None:
    fig_dir = out / "figures"
    metrics = ["MAE", "RMSE", "SSIM", "edge_F1", "gradient_l1_edge"]
    for idx in range(1, 11):
        metric = metrics[(idx - 1) % len(metrics)]
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = []
        values = []
        for row in summary_rows:
            if metric in row:
                labels.append(f"{row['variant']}\n{row['dataset']}")
                values.append(float(row[metric]))
        ax.bar(np.arange(len(values)), values, color=["#4C78A8" if "B1" in label else "#F58518" for label in labels])
        ax.set_xticks(np.arange(len(values)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(metric)
        ax.set_title(f"Phase-3 {metric} summary")
        fig.tight_layout()
        fig.savefig(fig_dir / f"Figure_{idx}.png", dpi=180)
        fig.savefig(fig_dir / f"Figure_{idx}.pdf")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [float(row["MAE"]) for row in sample_rows]
    ys = [float(row["gradient_l1_edge"]) for row in sample_rows]
    ax.scatter(xs, ys, s=8, alpha=0.35)
    ax.set_xlabel("MAE")
    ax.set_ylabel("gradient_l1_edge")
    ax.set_title("Appendix A1 corrected metric consistency")
    fig.tight_layout()
    fig.savefig(fig_dir / "Figure_A1.png", dpi=180)
    fig.savefig(fig_dir / "Figure_A1.pdf")
    plt.close(fig)


def _write_report(out: Path, protocol: dict[str, Any], locked: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# PASD-FWI Phase-3 Paper Report",
        "",
        "本报告按 Phase-3 locked protocol 重新训练 B1_raw_unet 与 PASD_Core_locked，并在 CurveVel-A 与 FlatFault-A 上做双目标评估。",
        "",
        f"- PASD-Core selected candidate: `{locked['selected_candidate']}`",
        f"- Source family: `{protocol['source']['family']}`",
        f"- Targets: `{', '.join(protocol['targets'].keys())}`",
        "- Target role during model/aggregation selection: evaluation only",
        "",
        "## Summary",
        "",
        "| variant | dataset | MAE | RMSE | SSIM | edge_F1 | gradient_l1_edge |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['dataset']} | {float(row.get('MAE', row.get('mae', 0.0))):.6g} | "
            f"{float(row.get('RMSE', row.get('rmse', 0.0))):.6g} | {float(row.get('SSIM', row.get('ssim', 0.0))):.6g} | "
            f"{float(row.get('edge_F1', 0.0)):.6g} | {float(row.get('gradient_l1_edge', row.get('gradient_error', 0.0))):.6g} |"
        )
    lines.extend(
        [
            "",
            "## Protocol Lock",
            "",
            "Phase-3 不覆盖 Phase-1/Phase-1b/Phase-2 历史产物；本轮正式产物位于 `outputs/pasd_phase3_paper/`。",
            "云端 smoke 或代码路径验证不构成科学结论；本报告使用本地真实数据协议产物。",
        ]
    )
    (out.parent / "PASD_PHASE3_PAPER_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "PASD_PHASE3_PAPER_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--pasd-core-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--latent-channels", type=int, default=96)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    out = run_dual_target(args)
    print(json.dumps({"status": "SUCCESS", "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
