from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from fwi_visionfm.data.openfwi_npy_dataset import OpenFWINpyDataset
from fwi_visionfm.evaluation.metrics import compute_velocity_metrics
from fwi_visionfm.foundation_train import FrozenFoundationFWI, _extract_batch, _normalize_meta_batch
from fwi_visionfm.peft import AdapterConfig, LoRAConfig
from fwi_visionfm.torch_backend import require_torch_backend


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_config_path(experiment_dir: Path) -> Path:
    for name in ("config_resolved.json", "resolved_foundation_config.json"):
        candidate = experiment_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find resolved config under {experiment_dir}")


def _build_dataloader(dataset: Any, *, batch_size: int, seed: int = 0) -> Any:
    torch = require_torch_backend()
    from torch.utils.data import DataLoader

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, generator=generator)


def _load_model_from_experiment(*, experiment_dir: Path, checkpoint: Path, device: str) -> tuple[FrozenFoundationFWI, dict[str, Any]]:
    torch = require_torch_backend()
    config = _load_json(_find_config_path(experiment_dir))
    summary_path = experiment_dir / "foundation_experiment_summary.json"
    summary = _load_json(summary_path) if summary_path.exists() else {}
    model = FrozenFoundationFWI(
        foundation_backbone=str(config.get("foundation_backbone", summary.get("backbone_name", "dummy_dinov2"))),
        backbone_type=config.get("backbone_type", summary.get("backbone_type")),
        model_name=config.get("model_name", summary.get("backbone_name")),
        pretrained=bool(config.get("pretrained", summary.get("pretrained", False))),
        freeze_backbone=bool(config.get("freeze_backbone", summary.get("freeze_backbone", True))),
        peft_type=str(config.get("peft_type", summary.get("peft_type", "none"))),
        lora_config=LoRAConfig(
            enabled=str(config.get("peft_type", summary.get("peft_type", "none"))) == "lora",
            rank=int(config.get("lora_rank", summary.get("lora_rank", 4))),
            alpha=float(config.get("lora_alpha", summary.get("lora_alpha", 8.0))),
            dropout=float(config.get("lora_dropout", summary.get("lora_dropout", 0.0))),
            target_modules=tuple(config.get("lora_target_modules", summary.get("lora_target_modules", ["qkv", "proj", "fc1", "fc2"]))),
            train_bias=bool(config.get("train_bias", False)),
        ),
        adapter_config=AdapterConfig(
            enabled=str(config.get("peft_type", summary.get("peft_type", "none"))) == "adapter",
            bottleneck_dim=int(config.get("adapter_bottleneck_dim", 64)),
            dropout=float(config.get("adapter_dropout", 0.0)),
        ),
        image_size=int(config.get("image_size", 224)),
        in_chans=int(config.get("in_chans", 3)),
        norm_mode=str(config.get("norm_mode", "zscore")),
        bridge_feature_mode=str(config.get("bridge_feature_mode", "raw_repeat3")),
        spectrogram_n_fft=int(config.get("spectrogram_n_fft", 64)),
        spectrogram_hop_length=int(config.get("spectrogram_hop_length", 16)),
        spectrogram_win_length=int(config.get("spectrogram_win_length", 64)),
        spectrogram_power=float(config.get("spectrogram_power", 1.0)),
        remove_cls_token=bool(config.get("remove_cls_token", False)),
        local_files_only=bool(config.get("local_files_only", False)),
        depth=int(config["depth"]),
        width=int(config["width"]),
        aggregation=str(config.get("aggregation", summary.get("aggregation", "mean"))),
        vmin=float(config.get("vmin", 1500.0)),
        vmax=float(config.get("vmax", 4500.0)),
        device=device,
        transfer_mode=config.get("transfer_mode", summary.get("transfer_mode")),
        print_parameter_report=False,
    ).to(device)
    payload = torch.load(checkpoint, map_location=device)
    if "trainable_state_dict" in payload:
        model.module.load_state_dict(payload["trainable_state_dict"], strict=False)
    elif "model_state_dict" in payload:
        model.module.load_state_dict(payload["model_state_dict"], strict=False)
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint}")
    model.eval()
    return model, config


def _collect_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, Any]:
    return compute_velocity_metrics(pred, true, data_range=1.0)


def evaluate_foundation_checkpoint(
    *,
    experiment_dir: str | Path,
    checkpoint: str | Path,
    openfwi_root: str | Path,
    manifest: str | Path,
    split: str | Path,
    stats_file: str | Path,
    output_metrics: str | Path,
    save_predictions: bool = False,
    output_predictions: str | Path | None = None,
    num_preview: int = 4,
    max_samples: int | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    experiment_dir = Path(experiment_dir)
    checkpoint = Path(checkpoint)
    output_metrics = Path(output_metrics)
    output_predictions = None if output_predictions is None else Path(output_predictions)
    model, config = _load_model_from_experiment(experiment_dir=experiment_dir, checkpoint=checkpoint, device=device)

    dataset = OpenFWINpyDataset(
        root=str(openfwi_root),
        split_file=str(split),
        stats_file=str(stats_file),
        fit_stats=False,
        input_norm=str(config.get("input_norm", "zscore")),
        target_norm=str(config.get("target_norm", "minmax")),
        max_samples=max_samples,
    )
    loader = _build_dataloader(dataset, batch_size=int(config.get("batch_size", 1)), seed=int(config.get("seed", 0)))
    torch = require_torch_backend()

    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    preview_records: list[np.ndarray] = []
    preview_preds: list[np.ndarray] = []
    preview_trues: list[np.ndarray] = []
    preview_errors: list[np.ndarray] = []
    preview_meta: list[str] = []

    with torch.no_grad():
        for batch in loader:
            records, velocity, source_positions, meta = _extract_batch(batch, device=device)
            prediction = model(records, source_positions).cpu().numpy().astype(np.float32)
            target = velocity.cpu().numpy().astype(np.float32)
            preds.append(prediction)
            trues.append(target)
            if save_predictions and output_predictions is not None and len(preview_preds) < int(num_preview):
                records_np = records.cpu().numpy().astype(np.float32)
                meta_items = _normalize_meta_batch(meta, prediction.shape[0])
                for index in range(prediction.shape[0]):
                    if len(preview_preds) >= int(num_preview):
                        break
                    preview_records.append(records_np[index])
                    preview_preds.append(prediction[index])
                    preview_trues.append(target[index])
                    preview_errors.append((prediction[index] - target[index]).astype(np.float32))
                    preview_meta.append(json.dumps(meta_items[index], ensure_ascii=False))

    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(trues, axis=0)
    metrics = _collect_metrics(pred, true)
    metrics.update(
        {
            "experiment_dir": str(experiment_dir),
            "checkpoint": str(checkpoint),
            "openfwi_root": str(Path(openfwi_root)),
            "manifest": str(Path(manifest)),
            "split": str(Path(split)),
            "stats_file": str(Path(stats_file)),
            "sample_count": int(pred.shape[0]),
            "eval_max_samples": None if max_samples is None else int(max_samples),
        }
    )
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    if save_predictions and output_predictions is not None:
        output_predictions.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_predictions,
            seismic_preview=np.asarray(preview_records, dtype=np.float32),
            velocity_true=np.asarray(preview_trues, dtype=np.float32),
            velocity_pred=np.asarray(preview_preds, dtype=np.float32),
            error_map=np.asarray(preview_errors, dtype=np.float32),
            meta=np.asarray(preview_meta, dtype=object),
        )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估已训练 foundation_fwi checkpoint，不重新训练。")
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--openfwi-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--stats-file", required=True, type=Path)
    parser.add_argument("--output-metrics", required=True, type=Path)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--output-predictions", type=Path, default=None)
    parser.add_argument("--num-preview", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_foundation_checkpoint(
        experiment_dir=args.experiment_dir,
        checkpoint=args.checkpoint,
        openfwi_root=args.openfwi_root,
        manifest=args.manifest,
        split=args.split,
        stats_file=args.stats_file,
        output_metrics=args.output_metrics,
        save_predictions=args.save_predictions,
        output_predictions=args.output_predictions,
        num_preview=args.num_preview,
        max_samples=args.max_samples,
        device=args.device,
    )
    print(f"写出评估指标: {args.output_metrics}")
    print(f"样本数: {metrics['sample_count']}")


if __name__ == "__main__":
    main()
