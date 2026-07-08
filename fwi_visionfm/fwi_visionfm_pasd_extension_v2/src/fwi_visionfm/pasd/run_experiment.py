"""Run one PASD B1--B4 experiment from a protocol manifest, NPY paths, or synthetic smoke data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import ArrayBundle, deterministic_subset_indices, fixed_split_indices, load_arrays, synthetic_openfwi_like
from .experiment import ProtocolSplits, TrainingConfig, run_single_experiment
from .protocol import DatasetRef, build_protocol, load_protocol, load_protocol_bundles
from .registry import VARIANTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one PASD-FWI variant with source-only normalization and optional target-family evaluation.")
    parser.add_argument("--protocol", type=str, default=None, help="Fixed protocol JSON produced by make_protocol.py.")
    parser.add_argument("--records", type=str, default=None, help="Source records .npy [N,S,T,R].")
    parser.add_argument("--models", type=str, default=None, help="Source velocity .npy [N,H,W] or [N,1,H,W].")
    parser.add_argument("--source-family", default="FlatVel-A")
    parser.add_argument("--source-positions", default=None)
    parser.add_argument("--receiver-positions", default=None)
    parser.add_argument("--target-records", default=None)
    parser.add_argument("--target-models", default=None)
    parser.add_argument("--target-family", default="CurveVel-A")
    parser.add_argument("--target-source-positions", default=None)
    parser.add_argument("--target-receiver-positions", default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-shots", type=int, default=5)
    parser.add_argument("--synthetic-time", type=int, default=128)
    parser.add_argument("--synthetic-receivers", type=int, default=48)
    parser.add_argument("--synthetic-model-size", type=int, default=32)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="B4_pasd_fwi")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--target-size", type=int, default=None)
    parser.add_argument("--train-size", type=int, default=350)
    parser.add_argument("--val-size", type=int, default=75)
    parser.add_argument("--test-size", type=int, default=75)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--latent-channels", type=int, default=96)
    parser.add_argument("--latent-height", type=int, default=9)
    parser.add_argument("--latent-width", type=int, default=9)
    parser.add_argument("--lowpass-kernel", type=int, default=21)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=1, help="CPU thread cap for deterministic low-overhead smoke runs.")
    return parser.parse_args()


def _synthetic_protocol(args: argparse.Namespace):
    records, velocities = synthetic_openfwi_like(
        n=args.max_samples, shots=args.synthetic_shots, time=args.synthetic_time, receivers=args.synthetic_receivers,
        model_size=args.synthetic_model_size, seed=args.seed
    )
    n = len(records)
    requested = args.train_size + args.val_size + args.test_size
    if requested > n:
        train_size = max(1, int(n * 0.6))
        val_size = max(1, int(n * 0.2))
        test_size = n - train_size - val_size
    else:
        train_size, val_size, test_size = args.train_size, args.val_size, args.test_size
    train, val, in_test = fixed_split_indices(n, train_size, val_size, test_size, seed=args.seed)
    source = ArrayBundle(records=records, velocities=velocities, sample_ids=__import__("numpy").arange(n), family="synthetic_source")
    return source, None, ProtocolSplits(tuple(train), tuple(val), tuple(in_test))


def _path_protocol(args: argparse.Namespace):
    if not args.records or not args.models:
        raise SystemExit("Provide --records and --models, --protocol, or use --synthetic.")
    if bool(args.target_records) != bool(args.target_models):
        raise SystemExit("Provide both --target-records and --target-models, or neither.")
    target = DatasetRef(args.target_records, args.target_models, args.target_family, None, args.target_source_positions, args.target_receiver_positions) if args.target_records else None
    manifest = build_protocol(
        source=DatasetRef(args.records, args.models, args.source_family, None, args.source_positions, args.receiver_positions),
        target=target,
        train_size=args.train_size,
        val_size=args.val_size,
        in_family_test_size=args.test_size,
        cross_family_test_size=args.target_size or args.test_size,
        seed=args.seed,
        max_source_samples=args.max_samples,
        notes="Ad-hoc protocol written by run_experiment; use make_protocol.py for a persistent study manifest.",
    )
    source, target_bundle = load_protocol_bundles(manifest)
    splits = ProtocolSplits(
        train=manifest.train_indices,
        val=manifest.val_indices,
        in_family_test=manifest.in_family_test_indices,
        cross_family_test=manifest.cross_family_test_indices,
    )
    return source, target_bundle, splits


def main() -> None:
    args = parse_args()
    if args.torch_threads > 0:
        __import__("torch").set_num_threads(args.torch_threads)
    if args.protocol:
        manifest = load_protocol(args.protocol)
        source, target = load_protocol_bundles(manifest)
        splits = ProtocolSplits(
            train=manifest.train_indices,
            val=manifest.val_indices,
            in_family_test=manifest.in_family_test_indices,
            cross_family_test=manifest.cross_family_test_indices,
        )
    elif args.synthetic:
        source, target, splits = _synthetic_protocol(args)
    else:
        source, target, splits = _path_protocol(args)

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        base_channels=args.base_channels,
        latent_channels=args.latent_channels,
        latent_size=(args.latent_height, args.latent_width),
        lowpass_kernel=args.lowpass_kernel,
    )
    result = run_single_experiment(source, splits, args.output, args.variant, config, target=target)
    print(json.dumps({"status": "SUCCESS", "variant": result.variant, "seed": result.seed, "in_family": result.in_family, "cross_family": result.cross_family, "output": result.output_dir}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
