from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from fwi_visionfm.foundation_train import FrozenFoundationFWI
from fwi_visionfm.peft import LoRAConfig
from fwi_visionfm.split_utils import load_split_paths, read_json
from fwi_visionfm.torch_backend import require_torch_backend
from fwi_visionfm.torch_backend.data import build_torch_dataloader
from fwi_visionfm.torch_backend.model import FwiVisionFmTorchBaseline


MODEL_DIR_NAMES = ("torch_cnn_baseline", "dummy_dinov2_frozen", "dummy_dinov2_lora")


def _latest_checkpoint(model_dir: Path) -> Path | None:
    candidates = sorted(list(model_dir.rglob("*.pt")) + list(model_dir.rglob("*.pth")))
    return candidates[-1] if candidates else None


def _collect_array_stats(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    error = pred - true
    abs_error = np.abs(error)
    mse = float(np.mean(error**2))
    pred_norm = float(np.mean(np.abs(true))) if np.mean(np.abs(true)) > 0 else 1.0
    pred_energy = float(np.sqrt(np.mean(true**2))) if np.mean(true**2) > 0 else 1.0
    return {
        "mae": float(np.mean(abs_error)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "relative_l1": float(np.mean(abs_error) / pred_norm),
        "relative_l2": float(np.sqrt(mse) / pred_energy),
        "pred_min": float(np.min(pred)),
        "pred_max": float(np.max(pred)),
        "pred_mean": float(np.mean(pred)),
        "true_min": float(np.min(true)),
        "true_max": float(np.max(true)),
        "true_mean": float(np.mean(true)),
        "error_mean": float(np.mean(error)),
        "error_max": float(np.max(abs_error)),
        "error_p95": float(np.percentile(abs_error, 95.0)),
    }


def _load_torch_model(summary: dict[str, Any], checkpoint_path: Path, *, device: str):
    torch = require_torch_backend()
    config = summary.get("config", {})
    model = FwiVisionFmTorchBaseline(
        channels=tuple(config.get("channels", ["raw"])),
        depth=int(summary.get("depth", config.get("depth", 70))),
        width=int(summary.get("width", config.get("width", 70))),
        aggregation=str(summary.get("aggregation", config.get("aggregation", "mean"))),
        vmin=float(config.get("vmin", 1500.0)),
        vmax=float(config.get("vmax", 4500.0)),
    ).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    model.module.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def _load_foundation_model(summary: dict[str, Any], checkpoint_path: Path, *, device: str):
    torch = require_torch_backend()
    config = summary.get("config", {})
    model = FrozenFoundationFWI(
        foundation_backbone=str(summary.get("backbone_name", config.get("foundation_backbone", "dummy_dinov2"))),
        pretrained=bool(summary.get("pretrained", config.get("pretrained", False))),
        freeze_backbone=bool(summary.get("freeze_backbone", config.get("freeze_backbone", True))),
        peft_type=str(summary.get("peft_type", config.get("peft_type", "none"))),
        lora_config=LoRAConfig(
            enabled=str(summary.get("peft_type", config.get("peft_type", "none"))) == "lora",
            rank=int(summary.get("lora_rank", config.get("lora_rank", 4))),
            alpha=float(summary.get("lora_alpha", config.get("lora_alpha", 8.0))),
            dropout=float(summary.get("lora_dropout", config.get("lora_dropout", 0.0))),
            target_modules=tuple(summary.get("lora_target_modules", config.get("lora_target_modules", ["qkv", "proj", "fc1", "fc2"]))),
        ),
        image_size=int(config.get("image_size", 64)),
        depth=int(summary.get("depth", config.get("depth", 70))),
        width=int(summary.get("width", config.get("width", 70))),
        aggregation=str(summary.get("aggregation", config.get("aggregation", "mean"))),
        vmin=float(config.get("vmin", 1500.0)),
        vmax=float(config.get("vmax", 4500.0)),
        device=device,
    ).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    if "trainable_state_dict" in payload:
        model.module.load_state_dict(payload["trainable_state_dict"], strict=False)
    elif "model_state_dict" in payload:
        model.module.load_state_dict(payload["model_state_dict"], strict=False)
    else:
        raise ValueError(f"unsupported foundation checkpoint format: {checkpoint_path}")
    model.eval()
    return model


def _evaluate_model(model: Any, test_paths: list[Path], *, batch_size: int, device: str) -> dict[str, float]:
    torch = require_torch_backend()
    loader = build_torch_dataloader(test_paths, batch_size=batch_size, shuffle=False, seed=0)
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            prediction = model(batch["records"].to(device), batch["source_positions"].to(device)).cpu().numpy()
            target = batch["velocity"].cpu().numpy()
            preds.append(prediction.astype(np.float32))
            trues.append(target.astype(np.float32))
    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(trues, axis=0)
    return _collect_array_stats(pred, true)


def _write_metrics_outputs(output_dir: Path, metrics: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics_test.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_dir / "metrics_test.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    lines = ["# Test Metrics", ""]
    for key, value in metrics.items():
        lines.append(f"- `{key}`: `{value}`")
    (output_dir / "metrics_test.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate_existing_checkpoints(
    *,
    split_dir: str | Path,
    outputs_root: str | Path,
    output_dir: str | Path,
    device: str = "cpu",
) -> dict[str, Any]:
    split_root = Path(split_dir)
    outputs_root = Path(outputs_root)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for split_manifest in sorted(path for path in split_root.glob("protocol_v1_*.json") if path.name != "protocol_v1_summary.json"):
        split_name = split_manifest.stem
        split_paths = load_split_paths(split_manifest)
        test_paths = [Path(path) for path in split_paths["test"]]
        experiment_root = outputs_root / split_name
        for model_name in MODEL_DIR_NAMES:
            model_dir = experiment_root / model_name
            row = {
                "experiment": split_name,
                "model_name": model_name,
                "model_dir": str(model_dir),
                "status": "missing_checkpoint",
            }
            summary_path = model_dir / ("torch_experiment_summary.json" if model_name == "torch_cnn_baseline" else "foundation_experiment_summary.json")
            checkpoint_path = _latest_checkpoint(model_dir)
            if not model_dir.exists():
                row["status"] = "missing_checkpoint"
                rows.append(row)
                continue
            if not summary_path.exists() or checkpoint_path is None:
                row["status"] = "missing_checkpoint"
                rows.append(row)
                continue
            summary = read_json(summary_path)
            batch_size = int(summary.get("batch_size", 1))
            if model_name == "torch_cnn_baseline":
                model = _load_torch_model(summary, checkpoint_path, device=device)
            else:
                model = _load_foundation_model(summary, checkpoint_path, device=device)
            metrics = _evaluate_model(model, test_paths, batch_size=batch_size, device=device)
            row.update(metrics)
            row["status"] = "complete"
            row["checkpoint_path"] = str(checkpoint_path)
            model_output_dir = output_root / split_name / model_name
            _write_metrics_outputs(model_output_dir, row)
            rows.append(row)

    if not rows:
        raise ValueError(f"no protocol_v1 split manifests found in {split_root}")
    csv_path = output_root / "all_test_metrics.csv"
    md_path = output_root / "all_test_metrics.md"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Protocol v1 Test Metrics",
        "",
        "| experiment | model_name | status | mae | rmse | relative_l1 | relative_l2 | error_p95 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('experiment', '')} | {row.get('model_name', '')} | {row.get('status', '')} | "
            f"{row.get('mae', '')} | {row.get('rmse', '')} | {row.get('relative_l1', '')} | "
            f"{row.get('relative_l2', '')} | {row.get('error_p95', '')} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    payload = {"count": len(rows), "rows": rows, "csv_path": str(csv_path), "md_path": str(md_path)}
    (output_root / "all_test_metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="加载已有 checkpoint，在 protocol_v1 test split 上统一评估。")
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--outputs-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_existing_checkpoints(
        split_dir=args.split_dir,
        outputs_root=args.outputs_root,
        output_dir=args.output_dir,
        device=args.device,
    )
    print(f"写出测试指标: {result['csv_path']}")
    print(f"记录数量: {result['count']}")


if __name__ == "__main__":
    main()
