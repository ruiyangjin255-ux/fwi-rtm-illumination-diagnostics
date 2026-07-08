"""CLI for building a frozen PASD source/cross-family protocol manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from .protocol import DatasetRef, build_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a fixed PASD-FWI protocol manifest from NPY data paths.")
    parser.add_argument("--source-records", required=True)
    parser.add_argument("--source-models", required=True)
    parser.add_argument("--source-family", default="FlatVel-A")
    parser.add_argument("--source-sample-ids", default=None)
    parser.add_argument("--source-positions", default=None)
    parser.add_argument("--receiver-positions", default=None)
    parser.add_argument("--target-records", default=None)
    parser.add_argument("--target-models", default=None)
    parser.add_argument("--target-family", default="CurveVel-A")
    parser.add_argument("--target-sample-ids", default=None)
    parser.add_argument("--target-source-positions", default=None)
    parser.add_argument("--target-receiver-positions", default=None)
    parser.add_argument("--train-size", type=int, default=350)
    parser.add_argument("--val-size", type=int, default=75)
    parser.add_argument("--in-family-test-size", type=int, default=75)
    parser.add_argument("--cross-family-test-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.target_records) != bool(args.target_models):
        raise SystemExit("Provide both --target-records and --target-models, or neither.")
    source = DatasetRef(args.source_records, args.source_models, args.source_family, args.source_sample_ids, args.source_positions, args.receiver_positions)
    target = None
    if args.target_records:
        target = DatasetRef(args.target_records, args.target_models, args.target_family, args.target_sample_ids, args.target_source_positions, args.target_receiver_positions)
    manifest = build_protocol(
        source=source,
        target=target,
        train_size=args.train_size,
        val_size=args.val_size,
        in_family_test_size=args.in_family_test_size,
        cross_family_test_size=args.cross_family_test_size,
        seed=args.seed,
        notes=args.notes,
    )
    saved = manifest.save(args.output)
    print(f"SUCCESS: wrote fixed PASD protocol to {Path(saved).resolve()}")


if __name__ == "__main__":
    main()
