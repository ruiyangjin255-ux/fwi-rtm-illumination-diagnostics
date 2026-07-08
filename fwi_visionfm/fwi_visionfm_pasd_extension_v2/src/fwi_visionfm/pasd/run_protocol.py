"""Run the complete B1--B4 PASD matrix for one fixed source/cross-family protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bootstrap import save_paired_bootstrap
from .experiment import ProtocolSplits, TrainingConfig, run_single_experiment
from .protocol import load_protocol, load_protocol_bundles
from .registry import VARIANTS
from .reporting import write_protocol_report


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

    raw_csv, summary_csv, report = write_protocol_report(root)
    with (root / "matrix_status.json").open("w", encoding="utf-8") as handle:
        json.dump({"status": "SUCCESS", "runs": result_rows, "report": str(report), "summary_csv": str(summary_csv)}, handle, indent=2, ensure_ascii=False)
    print(json.dumps({"status": "SUCCESS", "runs": len(result_rows), "output": str(root), "report": str(report)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
