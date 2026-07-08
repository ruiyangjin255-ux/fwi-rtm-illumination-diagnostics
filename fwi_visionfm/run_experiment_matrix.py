from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fwi_visionfm.compare_experiments import compare_experiments
from fwi_visionfm.foundation_train import run_foundation_npz_experiment
from fwi_visionfm.shape_utils import (
    assert_requested_shape_matches,
    infer_npz_dataset_shape,
    infer_split_manifest_shape,
)
from fwi_visionfm.split_utils import load_split_paths
from fwi_visionfm.torch_backend import run_torch_npz_experiment


def build_default_matrix_config(
    *,
    data_dir: str | Path | None,
    output_root: str | Path,
    depth: int,
    width: int,
    device: str,
    epochs: int = 3,
    learning_rate: float = 1.0e-3,
    batch_size: int = 2,
    split_manifest: str | Path | None = None,
    inferred_depth: int | None = None,
    inferred_width: int | None = None,
    auto_shape: bool = False,
    shape_source: str = "",
    dataset_shape_verified: bool = False,
    model_type: str | None = None,
    backbone_type: str | None = None,
    backbone_name: str | None = None,
    pretrained: bool | None = None,
    transfer_mode: str | None = None,
    image_size: int | None = None,
    bridge_norm: str | None = None,
    remove_cls_token: bool = False,
    adapter_bottleneck: int = 64,
    lora_rank: int = 4,
    lora_alpha: float = 8.0,
    lora_dropout: float = 0.0,
    local_files_only: bool = False,
    print_parameter_report: bool = True,
    foundation_aggregation: str | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root)
    return {
        "data_dir": "" if data_dir is None else str(Path(data_dir)),
        "output_root": str(output_root),
        "depth": int(depth),
        "width": int(width),
        "device": device,
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "split_manifest": "" if split_manifest is None else str(Path(split_manifest)),
        "auto_shape": bool(auto_shape),
        "inferred_depth": None if inferred_depth is None else int(inferred_depth),
        "inferred_width": None if inferred_width is None else int(inferred_width),
        "shape_source": shape_source,
        "dataset_shape_verified": bool(dataset_shape_verified),
        "vision_transfer": {
            "model_type": model_type,
            "backbone_type": backbone_type,
            "backbone_name": backbone_name,
            "pretrained": pretrained,
            "transfer_mode": transfer_mode,
            "image_size": image_size,
            "bridge_norm": bridge_norm,
            "remove_cls_token": bool(remove_cls_token),
            "adapter_bottleneck": int(adapter_bottleneck),
            "lora_rank": int(lora_rank),
            "lora_alpha": float(lora_alpha),
            "lora_dropout": float(lora_dropout),
            "local_files_only": bool(local_files_only),
            "print_parameter_report": bool(print_parameter_report),
            "foundation_aggregation": foundation_aggregation,
        },
        "experiments": [
            {
                "name": "torch_cnn_baseline",
                "kind": "torch",
                "output_dir": str(output_root / "torch_cnn_baseline"),
                "channels": ("raw", "offset"),
                "aggregation": "source_attention",
                "batch_size": int(batch_size),
                "epochs": int(epochs),
                "learning_rate": float(learning_rate),
            },
            {
                "name": "dummy_dinov2_frozen",
                "kind": "foundation",
                "output_dir": str(output_root / "dummy_dinov2_frozen"),
                "foundation_backbone": "dummy_dinov2",
                "model_type": model_type or "baseline",
                "backbone_type": backbone_type,
                "model_name": backbone_name,
                "pretrained": False,
                "freeze_backbone": True,
                "transfer_mode": transfer_mode,
                "image_size": int(image_size or 64),
                "norm_mode": bridge_norm or "zscore",
                "remove_cls_token": bool(remove_cls_token),
                "local_files_only": bool(local_files_only),
                "aggregation": foundation_aggregation or "source_attention",
                "batch_size": int(batch_size),
                "epochs": int(epochs),
                "learning_rate": float(learning_rate),
                "peft_type": "none",
                "adapter_bottleneck_dim": int(adapter_bottleneck),
                "print_parameter_report": bool(print_parameter_report),
            },
            {
                "name": "dummy_dinov2_lora",
                "kind": "foundation",
                "output_dir": str(output_root / "dummy_dinov2_lora"),
                "foundation_backbone": "dummy_dinov2",
                "model_type": model_type or "baseline",
                "backbone_type": backbone_type,
                "model_name": backbone_name,
                "pretrained": False,
                "freeze_backbone": True,
                "transfer_mode": transfer_mode,
                "image_size": int(image_size or 64),
                "norm_mode": bridge_norm or "zscore",
                "remove_cls_token": bool(remove_cls_token),
                "local_files_only": bool(local_files_only),
                "aggregation": foundation_aggregation or "source_attention",
                "batch_size": int(batch_size),
                "epochs": int(epochs),
                "learning_rate": float(learning_rate),
                "peft_type": "lora",
                "adapter_bottleneck_dim": int(adapter_bottleneck),
                "lora_rank": int(lora_rank),
                "lora_alpha": float(lora_alpha),
                "lora_dropout": float(lora_dropout),
                "lora_target_modules": ("qkv", "proj", "fc1", "fc2"),
                "print_parameter_report": bool(print_parameter_report),
            },
        ],
    }


def run_experiment_matrix(
    *,
    data_dir: str | Path | None,
    output_root: str | Path,
    depth: int | None,
    width: int | None,
    device: str = "cpu",
    epochs: int = 3,
    learning_rate: float = 1.0e-3,
    batch_size: int = 2,
    split_manifest: str | Path | None = None,
    auto_shape: bool = False,
    model_type: str | None = None,
    backbone_type: str | None = None,
    backbone_name: str | None = None,
    pretrained: bool | None = None,
    transfer_mode: str | None = None,
    image_size: int | None = None,
    bridge_norm: str | None = None,
    remove_cls_token: bool = False,
    adapter_bottleneck: int = 64,
    lora_rank: int = 4,
    lora_alpha: float = 8.0,
    lora_dropout: float = 0.0,
    local_files_only: bool = False,
    print_parameter_report: bool = True,
    foundation_aggregation: str | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if data_dir is None and split_manifest is None:
        raise ValueError("data_dir 和 split_manifest 至少提供一个")
    if split_manifest is not None:
        shape_summary = infer_split_manifest_shape(split_manifest)
        shape_source = "split_manifest"
    else:
        if data_dir is None:
            raise ValueError("data_dir is required when split_manifest is not provided")
        shape_summary = infer_npz_dataset_shape(data_dir)
        shape_source = "data_dir"
    inferred_depth = shape_summary["inferred_depth"]
    inferred_width = shape_summary["inferred_width"]
    dataset_shape_verified = bool(inferred_depth == 70 and inferred_width == 70)
    if not shape_summary["is_uniform_velocity_shape"]:
        raise ValueError(f"dataset velocity shapes are not uniform: {shape_summary['velocity_shape_set']}")
    if auto_shape:
        if inferred_depth is None or inferred_width is None:
            raise ValueError("could not infer dataset depth/width; velocity shapes are not uniform")
        depth = inferred_depth
        width = inferred_width
    else:
        if depth is None or width is None:
            raise ValueError("depth/width 未提供。请使用 --auto-shape 或显式传入 --depth --width。")
        assert_requested_shape_matches(inferred_depth, inferred_width, depth, width)
    config = build_default_matrix_config(
        data_dir=data_dir,
        output_root=output_root,
        depth=depth,
        width=width,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        split_manifest=split_manifest,
        inferred_depth=inferred_depth,
        inferred_width=inferred_width,
        auto_shape=auto_shape,
        shape_source=shape_source,
        dataset_shape_verified=dataset_shape_verified,
        model_type=model_type,
        backbone_type=backbone_type,
        backbone_name=backbone_name,
        pretrained=pretrained,
        transfer_mode=transfer_mode,
        image_size=image_size,
        bridge_norm=bridge_norm,
        remove_cls_token=remove_cls_token,
        adapter_bottleneck=adapter_bottleneck,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        local_files_only=local_files_only,
        print_parameter_report=print_parameter_report,
        foundation_aggregation=foundation_aggregation,
    )
    (output_root / "matrix_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    split_paths = None if split_manifest is None else load_split_paths(split_manifest)

    results: list[dict[str, Any]] = []
    successful_dirs: list[Path] = []
    for experiment in config["experiments"]:
        name = experiment["name"]
        try:
            if experiment["kind"] == "torch":
                summary = run_torch_npz_experiment(
                    data_dir=data_dir,
                    output_dir=experiment["output_dir"],
                    depth=depth,
                    width=width,
                    channels=tuple(experiment["channels"]),
                    aggregation=experiment["aggregation"],
                    batch_size=experiment["batch_size"],
                    epochs=experiment["epochs"],
                    learning_rate=experiment["learning_rate"],
                    device=device,
                    split_paths=split_paths,
                )
            else:
                summary = run_foundation_npz_experiment(
                    data_dir=data_dir,
                    output_dir=experiment["output_dir"],
                    depth=depth,
                    width=width,
                    model_type=experiment.get("model_type", "baseline"),
                    foundation_backbone=experiment["foundation_backbone"],
                    pretrained=bool(experiment["pretrained"]),
                    freeze_backbone=bool(experiment["freeze_backbone"]),
                    backbone_type=experiment.get("backbone_type"),
                    model_name=experiment.get("model_name"),
                    transfer_mode=experiment.get("transfer_mode"),
                    image_size=int(experiment["image_size"]),
                    norm_mode=experiment.get("norm_mode", "zscore"),
                    remove_cls_token=bool(experiment.get("remove_cls_token", False)),
                    local_files_only=bool(experiment.get("local_files_only", False)),
                    print_parameter_report=bool(experiment.get("print_parameter_report", True)),
                    aggregation=experiment["aggregation"],
                    batch_size=experiment["batch_size"],
                    epochs=experiment["epochs"],
                    learning_rate=experiment["learning_rate"],
                    device=device,
                    split_paths=split_paths,
                    peft_type=experiment.get("peft_type", "none"),
                    adapter_bottleneck_dim=int(experiment.get("adapter_bottleneck_dim", 64)),
                    lora_rank=int(experiment.get("lora_rank", 4)),
                    lora_alpha=float(experiment.get("lora_alpha", 8.0)),
                    lora_dropout=float(experiment.get("lora_dropout", 0.0)),
                    lora_target_modules=tuple(experiment.get("lora_target_modules", ("qkv", "proj", "fc1", "fc2"))),
                )
            results.append(
                {
                    "name": name,
                    "status": "completed",
                    "output_dir": experiment["output_dir"],
                    "summary_file": "torch_experiment_summary.json" if experiment["kind"] == "torch" else "foundation_experiment_summary.json",
                    "error": "",
                    "final_val_mae": summary.get("final_val_mae"),
                    "trainable_ratio": summary.get("trainable_ratio"),
                }
            )
            successful_dirs.append(Path(experiment["output_dir"]))
        except Exception as exc:
            results.append(
                {
                    "name": name,
                    "status": "failed",
                    "output_dir": experiment["output_dir"],
                    "summary_file": "",
                    "error": str(exc),
                    "final_val_mae": None,
                    "trainable_ratio": None,
                }
            )

    comparison_output = output_root / "comparison"
    comparison_payload = None
    if successful_dirs:
        comparison_payload = compare_experiments(successful_dirs, comparison_output)
    matrix_summary = {
        "data_dir": "" if data_dir is None else str(Path(data_dir)),
        "output_root": str(output_root),
        "depth": int(depth),
        "width": int(width),
        "device": device,
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "split_manifest": "" if split_manifest is None else str(Path(split_manifest)),
        "used_split_manifest": split_manifest is not None,
        "auto_shape": bool(auto_shape),
        "inferred_depth": inferred_depth,
        "inferred_width": inferred_width,
        "shape_source": shape_source,
        "dataset_shape_verified": dataset_shape_verified,
        "results": results,
        "comparison_output": str(comparison_output),
        "successful_experiments": len(successful_dirs),
        "comparison_count": int(comparison_payload["count"]) if comparison_payload else 0,
    }
    (output_root / "matrix_run_summary.json").write_text(json.dumps(matrix_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return matrix_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行标准 FWI-VisionFM 实验矩阵。")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--auto-shape", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--model-type", choices=("baseline", "foundation_fwi"))
    parser.add_argument("--backbone-type", choices=("dummy", "timm", "hf_dinov2"))
    parser.add_argument("--backbone-name")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--transfer-mode", choices=("scratch", "frozen", "full", "adapter", "lora"))
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--bridge-norm", choices=("none", "zscore", "minmax"))
    parser.add_argument("--remove-cls-token", action="store_true")
    parser.add_argument("--adapter-bottleneck", type=int, default=64)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--print-parameter-report", action="store_true")
    parser.add_argument("--foundation-aggregation", choices=("mean", "attention", "source_attention"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.data_dir is None and args.split_manifest is None:
        raise SystemExit("错误: --data-dir 和 --split-manifest 至少需要提供一个。")
    summary = run_experiment_matrix(
        data_dir=args.data_dir,
        output_root=args.output_root,
        depth=args.depth,
        width=args.width,
        device=args.device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        split_manifest=args.split_manifest,
        auto_shape=args.auto_shape,
        model_type=args.model_type,
        backbone_type=args.backbone_type,
        backbone_name=args.backbone_name,
        pretrained=True if args.pretrained else None,
        transfer_mode=args.transfer_mode,
        image_size=args.image_size,
        bridge_norm=args.bridge_norm,
        remove_cls_token=args.remove_cls_token,
        adapter_bottleneck=args.adapter_bottleneck,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        local_files_only=args.local_files_only,
        print_parameter_report=True if args.print_parameter_report else None,
        foundation_aggregation=args.foundation_aggregation,
    )
    print(f"写出矩阵配置: {args.output_root / 'matrix_config.json'}")
    print(f"成功实验数: {summary['successful_experiments']}")


if __name__ == "__main__":
    main()
