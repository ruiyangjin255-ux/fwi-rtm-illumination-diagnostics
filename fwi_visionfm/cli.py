from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fwi_visionfm.config import BridgeConfig, DataConfig, ModelConfig, RunConfig
from fwi_visionfm.train import run_npz_experiment, run_smoke_experiment
from fwi_visionfm.torch_backend import (
    run_openfwi_small_experiment,
    run_openfwi_scale_study,
    run_torch_ablation_experiment,
    run_torch_cpu_experiment,
    run_torch_smoke_experiment,
)


def _build_legacy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a minimal FWI-VisionFM smoke experiment.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Optional local npz sample directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/smoke"))
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--receivers", type=int, default=70)
    parser.add_argument("--time-samples", type=int, default=200)
    parser.add_argument("--depth", type=int, default=70)
    parser.add_argument("--width", type=int, default=70)
    parser.add_argument("--channels", default="raw,envelope")
    parser.add_argument("--aggregation", choices=("mean", "attention", "source_attention"), default="mean")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--fit-linear-calibration", action="store_true", help="在 train split 上拟合线性校准 a*x+b。")
    parser.add_argument("--train-linear-epochs", type=int, default=0, help="使用 NumPy 梯度下降训练线性校准的 epoch 数。")
    parser.add_argument("--linear-learning-rate", type=float, default=1.0e-8, help="线性校准训练学习率。")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _build_subcommand_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FWI-VisionFM CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke", help="运行 NumPy smoke。")
    smoke.add_argument("--output-dir", type=Path, default=Path("outputs/smoke"))
    smoke.add_argument("--samples", type=int, default=4)
    smoke.add_argument("--shots", type=int, default=5)
    smoke.add_argument("--receivers", type=int, default=70)
    smoke.add_argument("--time-samples", type=int, default=200)
    smoke.add_argument("--depth", type=int, default=70)
    smoke.add_argument("--width", type=int, default=70)
    smoke.add_argument("--channels", default="raw,envelope")
    smoke.add_argument("--aggregation", choices=("mean", "attention", "source_attention"), default="mean")
    smoke.add_argument("--seed", type=int, default=0)
    torch_smoke = subparsers.add_parser("torch-smoke", help="运行最小 PyTorch CPU smoke 训练。")
    torch_smoke.add_argument("--output-dir", type=Path, default=Path("outputs/torch_smoke"))
    torch_smoke.add_argument("--samples", type=int, default=4)
    torch_smoke.add_argument("--shots", type=int, default=3)
    torch_smoke.add_argument("--receivers", type=int, default=8)
    torch_smoke.add_argument("--time-samples", type=int, default=12)
    torch_smoke.add_argument("--depth", type=int, default=6)
    torch_smoke.add_argument("--width", type=int, default=7)
    torch_smoke.add_argument("--channels", default="raw,offset")
    torch_smoke.add_argument("--aggregation", choices=("mean", "source_attention"), default="source_attention")
    torch_smoke.add_argument("--batch-size", type=int, default=2)
    torch_smoke.add_argument("--epochs", type=int, default=1)
    torch_smoke.add_argument("--learning-rate", type=float, default=1.0e-3)
    torch_smoke.add_argument("--seed", type=int, default=0)
    torch_cpu = subparsers.add_parser("torch-cpu-experiment", help="运行 CPU 小规模 PyTorch 研究实验。")
    torch_cpu.add_argument("--output-dir", type=Path, default=Path("outputs/torch_cpu_experiment"))
    torch_cpu.add_argument("--samples", type=int, default=6)
    torch_cpu.add_argument("--shots", type=int, default=3)
    torch_cpu.add_argument("--receivers", type=int, default=8)
    torch_cpu.add_argument("--time-samples", type=int, default=12)
    torch_cpu.add_argument("--depth", type=int, default=6)
    torch_cpu.add_argument("--width", type=int, default=7)
    torch_cpu.add_argument("--channels", default="raw,offset")
    torch_cpu.add_argument("--aggregation", choices=("mean", "source_attention"), default="source_attention")
    torch_cpu.add_argument("--batch-size", type=int, default=2)
    torch_cpu.add_argument("--epochs", type=int, default=3)
    torch_cpu.add_argument("--learning-rate", type=float, default=1.0e-3)
    torch_cpu.add_argument("--seed", type=int, default=0)
    torch_ablation = subparsers.add_parser("torch-ablation", help="运行 CPU 小规模消融实验。")
    torch_ablation.add_argument("--output-dir", type=Path, default=Path("outputs/torch_ablation"))
    torch_ablation.add_argument("--samples", type=int, default=6)
    torch_ablation.add_argument("--shots", type=int, default=3)
    torch_ablation.add_argument("--receivers", type=int, default=8)
    torch_ablation.add_argument("--time-samples", type=int, default=12)
    torch_ablation.add_argument("--depth", type=int, default=6)
    torch_ablation.add_argument("--width", type=int, default=7)
    torch_ablation.add_argument("--channels", default="raw,offset")
    torch_ablation.add_argument("--batch-size", type=int, default=2)
    torch_ablation.add_argument("--epochs", type=int, default=1)
    torch_ablation.add_argument("--learning-rate", type=float, default=1.0e-3)
    torch_ablation.add_argument("--seed", type=int, default=0)
    openfwi_small = subparsers.add_parser("openfwi-small-experiment", help="运行 OpenFWI-style 小规模 CPU 实验。")
    openfwi_small.add_argument("--data-root", type=Path, default=None)
    openfwi_small.add_argument("--split-dir", type=Path, default=None)
    openfwi_small.add_argument("--stats-file", type=Path, default=None)
    openfwi_small.add_argument("--output-dir", type=Path, default=Path("outputs/openfwi_small_experiment"))
    openfwi_small.add_argument("--epochs", type=int, default=2)
    openfwi_small.add_argument("--batch-size", type=int, default=1)
    openfwi_small.add_argument("--learning-rate", type=float, default=1.0e-3)
    openfwi_small.add_argument("--max-train-samples", type=int, default=None)
    openfwi_small.add_argument("--max-val-samples", type=int, default=None)
    openfwi_small.add_argument("--max-test-samples", type=int, default=None)
    openfwi_small.add_argument("--seed", type=int, default=0)
    openfwi_scale = subparsers.add_parser("openfwi-scale-study", help="运行 OpenFWI 同口径 CPU 规模递增实验。")
    openfwi_scale.add_argument("--data-root", type=Path, default=None)
    openfwi_scale.add_argument("--split-dir", type=Path, default=None)
    openfwi_scale.add_argument("--stats-file", type=Path, default=None)
    openfwi_scale.add_argument("--output-dir", type=Path, default=Path("outputs/openfwi_scale_study"))
    openfwi_scale.add_argument("--epochs", type=int, default=3)
    openfwi_scale.add_argument("--batch-size", type=int, default=1)
    openfwi_scale.add_argument("--learning-rate", type=float, default=1.0e-3)
    openfwi_scale.add_argument("--sizes", default="8:2:2,32:8:8,64:8:8")
    openfwi_scale.add_argument("--seed", type=int, default=0)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in {"smoke", "torch-smoke", "torch-cpu-experiment", "torch-ablation", "openfwi-small-experiment", "openfwi-scale-study"}:
        return _build_subcommand_parser().parse_args(argv)
    return _build_legacy_parser().parse_args(argv)


def main() -> None:
    args = parse_args()
    if getattr(args, "command", None) == "smoke":
        channels = tuple(part.strip() for part in args.channels.split(",") if part.strip())
        cfg = RunConfig(
            data=DataConfig(
                num_shots=args.shots,
                num_receivers=args.receivers,
                num_time_samples=args.time_samples,
                velocity_depth=args.depth,
                velocity_width=args.width,
            ),
            bridge=BridgeConfig(channels=channels),
            model=ModelConfig(velocity_depth=args.depth, velocity_width=args.width, aggregation=args.aggregation),
            samples=args.samples,
            seed=args.seed,
        )
        summary = run_smoke_experiment(args.output_dir, cfg)
        print(f"Wrote summary to {args.output_dir / 'summary.json'}")
        print(f"Total loss: {summary['loss']['total']:.6f}")
        return
    if getattr(args, "command", None) == "torch-smoke":
        channels = tuple(part.strip() for part in args.channels.split(",") if part.strip())
        summary = run_torch_smoke_experiment(
            args.output_dir,
            samples=args.samples,
            shots=args.shots,
            receivers=args.receivers,
            time_samples=args.time_samples,
            depth=args.depth,
            width=args.width,
            channels=channels,
            aggregation=args.aggregation,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device="cpu",
        )
        print(f"Wrote summary to {args.output_dir / 'torch_experiment_summary.json'}")
        print(f"Torch smoke epochs: {summary['epochs']}")
        return
    if getattr(args, "command", None) == "torch-cpu-experiment":
        channels = tuple(part.strip() for part in args.channels.split(",") if part.strip())
        summary = run_torch_cpu_experiment(
            args.output_dir,
            samples=args.samples,
            shots=args.shots,
            receivers=args.receivers,
            time_samples=args.time_samples,
            depth=args.depth,
            width=args.width,
            channels=channels,
            aggregation=args.aggregation,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device="cpu",
        )
        print(f"Wrote summary to {args.output_dir / 'experiment_summary.json'}")
        print(f"CPU experiment final val loss: {summary['final_val_loss']:.6f}")
        return
    if getattr(args, "command", None) == "torch-ablation":
        channels = tuple(part.strip() for part in args.channels.split(",") if part.strip())
        summary = run_torch_ablation_experiment(
            args.output_dir,
            samples=args.samples,
            shots=args.shots,
            receivers=args.receivers,
            time_samples=args.time_samples,
            depth=args.depth,
            width=args.width,
            channels=channels,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device="cpu",
        )
        print(f"Wrote summary to {args.output_dir / 'ablation_summary.json'}")
        print(f"Best experiment: {summary['best_experiment_id']}")
        return
    if getattr(args, "command", None) == "openfwi-small-experiment":
        if args.data_root is None:
            raise SystemExit("openfwi-small-experiment requires --data-root")
        if args.split_dir is None:
            raise SystemExit("openfwi-small-experiment requires --split-dir")
        if args.stats_file is None:
            raise SystemExit("openfwi-small-experiment requires --stats-file")
        summary = run_openfwi_small_experiment(
            args.output_dir,
            data_root=args.data_root,
            split_dir=args.split_dir,
            stats_file=args.stats_file,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device="cpu",
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            max_test_samples=args.max_test_samples,
        )
        print(f"Wrote summary to {args.output_dir / 'experiment_summary.json'}")
        print(f"OpenFWI small final val loss: {summary['final_val_loss']:.6f}")
        return
    if getattr(args, "command", None) == "openfwi-scale-study":
        if args.data_root is None:
            raise SystemExit("openfwi-scale-study requires --data-root")
        if args.split_dir is None:
            raise SystemExit("openfwi-scale-study requires --split-dir")
        if args.stats_file is None:
            raise SystemExit("openfwi-scale-study requires --stats-file")
        summary = run_openfwi_scale_study(
            args.output_dir,
            data_root=args.data_root,
            split_dir=args.split_dir,
            stats_file=args.stats_file,
            sizes=args.sizes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device="cpu",
        )
        print(f"Wrote summary to {args.output_dir / 'scale_summary.json'}")
        print(f"Best scale experiment: {summary['best_experiment_id']}")
        return
    channels = tuple(part.strip() for part in args.channels.split(",") if part.strip())
    data = DataConfig(
        num_shots=args.shots,
        num_receivers=args.receivers,
        num_time_samples=args.time_samples,
        velocity_depth=args.depth,
        velocity_width=args.width,
    )
    bridge = BridgeConfig(channels=channels)
    model = ModelConfig(velocity_depth=args.depth, velocity_width=args.width, aggregation=args.aggregation)
    if args.data_dir is not None:
        summary = run_npz_experiment(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            bridge=bridge,
            model_config=model,
            batch_size=args.batch_size,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
            seed=args.seed,
            fit_linear_calibration=args.fit_linear_calibration,
            train_linear_epochs=args.train_linear_epochs,
            linear_learning_rate=args.linear_learning_rate,
        )
        print(f"Wrote summary to {args.output_dir / 'npz_experiment_summary.json'}")
        print(f"Dataset samples: {summary['数据集']['样本数']}")
        return
    cfg = RunConfig(
        data=data,
        bridge=bridge,
        model=model,
        samples=args.samples,
        seed=args.seed,
    )
    summary = run_smoke_experiment(args.output_dir, cfg)
    print(f"Wrote summary to {args.output_dir / 'summary.json'}")
    print(f"Total loss: {summary['loss']['total']:.6f}")


if __name__ == "__main__":
    main()
