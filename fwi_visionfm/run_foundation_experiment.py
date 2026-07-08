from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fwi_visionfm.foundation_train import (
    generate_foundation_synthetic_npz,
    run_foundation_npz_experiment,
    run_foundation_openfwi_experiment,
)


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def validate_foundation_args(args: argparse.Namespace) -> None:
    effective_type = getattr(args, "backbone_type", None)
    effective_name = getattr(args, "model_name", None) or args.foundation_backbone
    if effective_type == "dummy" or effective_name == "dummy_dinov2":
        return
    if effective_name == "vit_small_patch14_dinov2.lvd142m" and int(args.image_size) != 518:
        raise ValueError(
            "foundation-backbone=vit_small_patch14_dinov2.lvd142m 期望输入尺寸 518x518，"
            f"当前收到 --image-size {args.image_size}。请改用 --image-size 518。"
        )
    if args.transfer_mode in {"adapter", "lora"} and args.peft not in {"none", args.transfer_mode}:
        raise ValueError(
            f"--transfer-mode {args.transfer_mode} 与 --peft {args.peft} 不一致。"
            f"请改用 --peft {args.transfer_mode} 或移除 --peft。"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 frozen Vision Foundation FWI baseline。")
    parser.add_argument("--data-dir", type=Path, required=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depth", type=int, required=False)
    parser.add_argument("--width", type=int, required=False)
    parser.add_argument("--model-type", choices=("baseline", "foundation_fwi"), default="baseline")
    parser.add_argument("--foundation-backbone", default="dummy_dinov2")
    parser.add_argument("--backbone-type", choices=("dummy", "timm", "hf_dinov2"), default=None)
    parser.add_argument("--model-name", "--backbone-name", dest="model_name", default=None, help="真实 backbone 名称，未提供时回退到 --foundation-backbone。")
    parser.add_argument("--transfer-mode", choices=("scratch", "frozen", "full", "adapter", "lora"), default=None)
    parser.add_argument("--peft", choices=("none", "lora", "adapter"), default="none")
    parser.add_argument("--pretrained", dest="pretrained", nargs="?", const=True, type=_parse_bool)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=False)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--in-chans", type=int, default=3)
    parser.add_argument("--norm-mode", "--bridge-norm", dest="norm_mode", choices=("none", "zscore", "minmax"), default="zscore")
    parser.add_argument(
        "--bridge-feature-mode",
        choices=("raw_repeat3", "raw_envelope_spectrogram", "raw_envelope", "raw_spectrogram", "envelope_repeat3", "spectrogram_repeat3"),
        default="raw_repeat3",
    )
    parser.add_argument("--spectrogram-n-fft", type=int, default=64)
    parser.add_argument("--spectrogram-hop-length", type=int, default=16)
    parser.add_argument("--spectrogram-win-length", type=int, default=64)
    parser.add_argument("--spectrogram-power", type=float, default=1.0)
    parser.add_argument("--remove-cls-token", nargs="?", const=True, default=False, type=_parse_bool)
    parser.add_argument("--local-files-only", nargs="?", const=True, default=False, type=_parse_bool)
    parser.add_argument("--print-parameter-report", nargs="?", const=True, default=True, type=_parse_bool)
    parser.add_argument("--aggregation", choices=("mean", "attention", "source_attention"), default="mean")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-target-modules", default="qkv,proj,fc1,fc2")
    parser.add_argument("--train-bias", action="store_true")
    parser.add_argument("--adapter-bottleneck-dim", type=int, default=64)
    parser.add_argument("--adapter-dropout", type=float, default=0.0)
    parser.add_argument("--vmin", type=float, default=1500.0)
    parser.add_argument("--vmax", type=float, default=4500.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--synthetic-smoke", action="store_true", help="当 data-dir 不存在时自动生成最小 synthetic npz 数据。")
    parser.add_argument("--synthetic-samples", type=int, default=4)
    parser.add_argument("--synthetic-shots", type=int, default=3)
    parser.add_argument("--synthetic-receivers", type=int, default=8)
    parser.add_argument("--synthetic-time-samples", type=int, default=12)
    parser.add_argument("--openfwi-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--train-split", type=Path, default=None)
    parser.add_argument("--val-split", type=Path, default=None)
    parser.add_argument("--test-split", type=Path, default=None)
    parser.add_argument("--stats-file", type=Path, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--input-norm", choices=("zscore", "minmax", "none"), default="zscore")
    parser.add_argument("--target-norm", choices=("minmax", "zscore", "none"), default="minmax")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--num-preview", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        validate_foundation_args(args)
    except ValueError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    try:
        transfer_mode = args.transfer_mode
        if transfer_mode is None:
            if args.peft == "adapter":
                transfer_mode = "adapter"
            elif args.peft == "lora":
                transfer_mode = "lora"
            elif args.freeze_backbone:
                transfer_mode = "frozen"
            else:
                transfer_mode = "full"
        peft_type = args.peft
        if peft_type == "none" and transfer_mode in {"adapter", "lora"}:
            peft_type = transfer_mode
        if args.synthetic_smoke:
            if args.depth is None or args.width is None:
                raise ValueError("synthetic-smoke 模式要求显式提供 --depth 和 --width。")
            data_dir = args.data_dir or (args.output_dir / "foundation_smoke_inputs")
            if not data_dir.exists():
                data_dir = generate_foundation_synthetic_npz(
                    args.output_dir / "foundation_smoke_inputs",
                    samples=args.synthetic_samples,
                    shots=args.synthetic_shots,
                    receivers=args.synthetic_receivers,
                    time_samples=args.synthetic_time_samples,
                    depth=args.depth,
                    width=args.width,
                    seed=args.seed,
                )
            summary = run_foundation_npz_experiment(
                data_dir=data_dir,
                output_dir=args.output_dir,
                depth=args.depth,
                width=args.width,
                model_type=args.model_type,
                foundation_backbone=args.foundation_backbone,
                backbone_type=args.backbone_type,
                model_name=args.model_name,
                peft_type=peft_type,
                pretrained=args.pretrained,
                freeze_backbone=args.freeze_backbone,
                transfer_mode=transfer_mode,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                lora_target_modules=tuple(part.strip() for part in args.lora_target_modules.split(",") if part.strip()),
                train_bias=args.train_bias,
                adapter_bottleneck_dim=args.adapter_bottleneck_dim,
                adapter_dropout=args.adapter_dropout,
                image_size=args.image_size,
                in_chans=args.in_chans,
                norm_mode=args.norm_mode,
                bridge_feature_mode=args.bridge_feature_mode,
                spectrogram_n_fft=args.spectrogram_n_fft,
                spectrogram_hop_length=args.spectrogram_hop_length,
                spectrogram_win_length=args.spectrogram_win_length,
                spectrogram_power=args.spectrogram_power,
                remove_cls_token=args.remove_cls_token,
                local_files_only=args.local_files_only,
                print_parameter_report=args.print_parameter_report,
                aggregation=args.aggregation,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                vmin=args.vmin,
                vmax=args.vmax,
                device=args.device,
                seed=args.seed,
            )
        else:
            if args.train_split is None or args.val_split is None or args.stats_file is None:
                if args.data_dir is None:
                    raise ValueError(
                        "真实 OpenFWI 模式要求提供 --openfwi-root/--train-split/--val-split/--stats-file；"
                        "若要使用旧 npz 流程，请传入 --data-dir；若要 smoke，请追加 --synthetic-smoke。"
                    )
                if args.depth is None or args.width is None:
                    raise ValueError("旧 npz 流程要求显式提供 --depth 和 --width。")
                summary = run_foundation_npz_experiment(
                    data_dir=args.data_dir,
                    output_dir=args.output_dir,
                    depth=args.depth,
                    width=args.width,
                    model_type=args.model_type,
                    foundation_backbone=args.foundation_backbone,
                    backbone_type=args.backbone_type,
                    model_name=args.model_name,
                    peft_type=peft_type,
                    pretrained=args.pretrained,
                    freeze_backbone=args.freeze_backbone,
                    transfer_mode=transfer_mode,
                    lora_rank=args.lora_rank,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    lora_target_modules=tuple(part.strip() for part in args.lora_target_modules.split(",") if part.strip()),
                    train_bias=args.train_bias,
                    adapter_bottleneck_dim=args.adapter_bottleneck_dim,
                    adapter_dropout=args.adapter_dropout,
                    image_size=args.image_size,
                    in_chans=args.in_chans,
                    norm_mode=args.norm_mode,
                    bridge_feature_mode=args.bridge_feature_mode,
                    spectrogram_n_fft=args.spectrogram_n_fft,
                    spectrogram_hop_length=args.spectrogram_hop_length,
                    spectrogram_win_length=args.spectrogram_win_length,
                    spectrogram_power=args.spectrogram_power,
                    remove_cls_token=args.remove_cls_token,
                    local_files_only=args.local_files_only,
                    print_parameter_report=args.print_parameter_report,
                    aggregation=args.aggregation,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    vmin=args.vmin,
                    vmax=args.vmax,
                    device=args.device,
                    seed=args.seed,
                )
            else:
                if args.openfwi_root is None:
                    raise ValueError("真实 OpenFWI 模式必须提供 --openfwi-root。")
                summary = run_foundation_openfwi_experiment(
                    openfwi_root=args.openfwi_root,
                    manifest_path=args.manifest or args.train_split,
                    train_split=args.train_split,
                    val_split=args.val_split,
                    test_split=args.test_split,
                    stats_file=args.stats_file,
                    output_dir=args.output_dir,
                    model_type=args.model_type,
                    foundation_backbone=args.foundation_backbone,
                    backbone_type=args.backbone_type,
                    model_name=args.model_name,
                    peft_type=peft_type,
                    pretrained=args.pretrained,
                    freeze_backbone=args.freeze_backbone,
                    transfer_mode=transfer_mode,
                    lora_rank=args.lora_rank,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    lora_target_modules=tuple(part.strip() for part in args.lora_target_modules.split(",") if part.strip()),
                    train_bias=args.train_bias,
                    adapter_bottleneck_dim=args.adapter_bottleneck_dim,
                    adapter_dropout=args.adapter_dropout,
                    image_size=args.image_size,
                    in_chans=args.in_chans,
                    norm_mode=args.norm_mode,
                    bridge_feature_mode=args.bridge_feature_mode,
                    spectrogram_n_fft=args.spectrogram_n_fft,
                    spectrogram_hop_length=args.spectrogram_hop_length,
                    spectrogram_win_length=args.spectrogram_win_length,
                    spectrogram_power=args.spectrogram_power,
                    remove_cls_token=args.remove_cls_token,
                    local_files_only=args.local_files_only,
                    print_parameter_report=args.print_parameter_report,
                    aggregation=args.aggregation,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    vmin=args.vmin,
                    vmax=args.vmax,
                    device=args.device,
                    seed=args.seed,
                    max_train_samples=args.max_train_samples,
                    max_val_samples=args.max_val_samples,
                    max_test_samples=args.max_test_samples,
                    input_norm=args.input_norm,
                    target_norm=args.target_norm,
                    save_predictions=args.save_predictions,
                    num_preview=args.num_preview,
                )
    except (RuntimeError, NotImplementedError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"写出摘要: {args.output_dir / 'foundation_experiment_summary.json'}")
    print(f"训练轮数: {summary['epochs']}")


if __name__ == "__main__":
    main()
