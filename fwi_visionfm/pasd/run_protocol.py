"""Run the complete B1--B4 PASD matrix for one fixed source/cross-family protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .bootstrap import save_paired_bootstrap
from .experiment import ProtocolSplits, TrainingConfig, build_model, run_single_experiment
from .metrics import per_sample_metrics
from .plotting import plot_gradient_comparison, plot_profiles, plot_protocol_metric_summary, plot_velocity_comparison
from .protocol import load_protocol, load_protocol_bundles
from .registry import VARIANTS, get_variant
from .reporting import write_protocol_report


def _config_hash(payload: dict[str, object]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _param_count(module: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad))


def write_variant_audit(path: Path, output_size: tuple[int, int], config: TrainingConfig) -> Path:
    rows: dict[str, object] = {}
    for name, variant in VARIANTS.items():
        model = build_model(variant, output_size, config)
        payload = {
            "bridge_mode": variant.bridge_mode,
            "input_channels": 1 if variant.bridge_mode == "raw" else 3,
            "computes_envelope": variant.bridge_mode == "hybrid",
            "computes_low_high_band_energy": variant.bridge_mode == "hybrid",
            "geometry_enabled": variant.aggregator == "geometry_attention",
            "attention_enabled": variant.aggregator == "geometry_attention",
            "aggregation": variant.aggregator,
            "decoder_type": variant.decoder_mode,
            "loss_type": variant.criterion,
            "encoder_trainable_parameters": _param_count(model.encoder),
            "aggregator_trainable_parameters": _param_count(model.aggregator),
            "decoder_trainable_parameters": _param_count(model.decoder),
            "total_trainable_parameters": _param_count(model),
        }
        payload["config_hash"] = _config_hash(payload)
        rows[name] = payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {name: np.asarray(payload[name]) for name in ("sample_id", "prediction", "target")}


def _write_common_sample_figures(root: Path, variants: list[str], seed: int = 0) -> None:
    paths = {variant: root / variant / f"seed_{seed}" / "predictions_cross_family.npz" for variant in variants}
    if not all(path.exists() for path in paths.values()) or "B1_raw_unet" not in paths:
        return
    archives = {variant: _load_archive(path) for variant, path in paths.items()}
    ids = np.asarray(archives["B1_raw_unet"]["sample_id"], dtype=np.int64)
    ordered = np.argsort(ids)
    b1_pred = archives["B1_raw_unet"]["prediction"][ordered]
    b1_target = archives["B1_raw_unet"]["target"][ordered]
    mae = per_sample_metrics(torch.from_numpy(b1_pred), torch.from_numpy(b1_target))["mae"].numpy()
    choice = int(np.argsort(mae)[len(mae) // 2])
    sample_id = int(ids[ordered][choice])
    target = b1_target[choice]
    predictions: dict[str, np.ndarray] = {}
    for variant, archive in archives.items():
        mapping = {int(sample_id): index for index, sample_id in enumerate(archive["sample_id"].tolist())}
        predictions[variant] = archive["prediction"][mapping[sample_id]]
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_velocity_comparison(target, predictions, figures / "Figure_cross_family_median_comparison.png", title=f"cross-family median B1 MAE sample_id={sample_id}")
    selected = {name: predictions[name] for name in predictions if name in {"B1_raw_unet", "B4_pasd_fwi"}}
    plot_gradient_comparison(target, selected, figures / "Figure_cross_family_gradient_comparison.png")
    plot_profiles(target, selected, figures / "Figure_cross_family_profiles.png")
    (figures / "fixed_sample_selection.json").write_text(json.dumps({"seed": seed, "sample_id": sample_id, "selection": "B1 cross-family median_MAE"}, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run B1--B4 PASD-FWI variants under a fixed manifest and multi-seed protocol.")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=sorted(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--latent-channels", type=int, default=96)
    parser.add_argument("--lowpass-kernel", type=int, default=21)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--locked-config", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.torch_threads > 0:
        __import__("torch").set_num_threads(args.torch_threads)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    manifest = load_protocol(args.protocol)
    source, target = load_protocol_bundles(manifest)
    splits = ProtocolSplits(
        train=manifest.train_indices,
        val=manifest.val_indices,
        in_family_test=manifest.in_family_test_indices,
        cross_family_test=manifest.cross_family_test_indices,
    )
    manifest.save(root / "protocol_manifest.json")
    audit_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=0,
        device="cpu",
        num_workers=0,
        base_channels=args.base_channels,
        latent_channels=args.latent_channels,
        lowpass_kernel=args.lowpass_kernel,
    )
    write_variant_audit(root / "variant_audit.json", tuple(int(x) for x in source.velocities.shape[-2:]), audit_config)

    locked_config_path = args.locked_config or (manifest.metadata or {}).get("locked_config_path")
    locked_config: dict[str, object] | None = None
    if locked_config_path:
        candidate = Path(str(locked_config_path))
        if not candidate.is_absolute():
            candidate = Path(args.protocol).parent / candidate
        if not candidate.exists():
            candidate = Path(str(locked_config_path))
        locked_config = json.loads(candidate.read_text(encoding="utf-8"))

    result_rows: list[dict[str, object]] = []
    for variant in args.variants:
        for seed in args.seeds:
            run_dir = root / variant / f"seed_{seed}"
            config = TrainingConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                seed=seed,
                device=args.device,
                num_workers=args.num_workers,
                base_channels=args.base_channels,
                latent_channels=args.latent_channels,
                lowpass_kernel=args.lowpass_kernel,
            )
            if locked_config is not None and variant in {"B4_pasd_fwi", "B4_no_geometry_attention"}:
                config = TrainingConfig(
                    **{
                        **config.__dict__,
                        "weight_edge": float(locked_config.get("lambda_edge", config.weight_edge)),
                        "weight_background": float(locked_config.get("lambda_bg", config.weight_background)),
                        "weight_smooth": float(locked_config.get("lambda_smooth", config.weight_smooth)),
                        "background_sigma": float(locked_config.get("Gaussian_sigma", config.background_sigma)),
                    }
                )
            result = run_single_experiment(source, splits, run_dir, variant, config, target=target)
            result_rows.append({"variant": variant, "seed": seed, "status": "SUCCESS", "output": result.output_dir})

    bootstrap_dir = root / "bootstrap"
    if "B1_raw_unet" in args.variants and "B4_pasd_fwi" in args.variants:
        for seed in args.seeds:
            baseline_dir = root / "B1_raw_unet" / f"seed_{seed}"
            candidate_dir = root / "B4_pasd_fwi" / f"seed_{seed}"
            for split in ("in_family", "cross_family"):
                baseline = baseline_dir / f"predictions_{split}.npz"
                candidate = candidate_dir / f"predictions_{split}.npz"
                if not baseline.exists() or not candidate.exists():
                    continue
                for metric in ("mae", "rmse", "ssim", "edge_mae", "gradient_error"):
                    save_paired_bootstrap(
                        baseline,
                        candidate,
                        bootstrap_dir / f"B4_vs_B1_seed{seed}_{split}_{metric}.json",
                        metric=metric,
                        n_resamples=args.bootstrap_resamples,
                        seed=seed,
                    )

    _write_common_sample_figures(root, list(args.variants), seed=args.seeds[0])
    raw_csv, summary_csv, report = write_protocol_report(root)
    with (root / "matrix_status.json").open("w", encoding="utf-8") as handle:
        json.dump({"status": "SUCCESS", "runs": result_rows, "report": str(report), "summary_csv": str(summary_csv)}, handle, indent=2, ensure_ascii=False)
    print(json.dumps({"status": "SUCCESS", "runs": len(result_rows), "output": str(root), "report": str(report)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
