from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from fwi_visionfm.torch_backend import require_torch_backend
from fwi_visionfm.torch_backend.data import build_torch_dataloader
from fwi_visionfm.torch_backend.model import FrozenFoundationFWI


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_paths(rows: list[dict[str, Any]]) -> list[Path]:
    return [Path(row.get("path") or row["data_file"]) for row in rows]


def _split_paths(manifest: dict[str, Any]) -> dict[str, list[Path]]:
    return {
        "train": _sample_paths(manifest["train_samples"]),
        "val": _sample_paths(manifest["val_samples"]),
        "in_family_test": _sample_paths(manifest["in_family_test_samples"]),
        "cross_family_test": _sample_paths(manifest["cross_family_test_samples"]),
    }


def _first_velocity_shape(path: Path) -> tuple[int, int]:
    with np.load(path) as payload:
        velocity = np.asarray(payload["velocity"], dtype=np.float32)
    return int(velocity.shape[-2]), int(velocity.shape[-1])


def _extract_split(model: FrozenFoundationFWI, paths: list[Path], *, device: str) -> dict[str, np.ndarray]:
    torch = require_torch_backend()
    loader = build_torch_dataloader(paths, batch_size=min(4, max(1, len(paths))), shuffle=False, seed=0)
    features = []
    targets = []
    records = []
    source_positions = []
    with torch.no_grad():
        for batch in loader:
            batch_records = batch["records"].to(device)
            batch_positions = batch["source_positions"].to(device)
            aggregated = model.module.extract_aggregated_features(batch_records, source_positions=batch_positions)
            features.append(aggregated.detach().cpu().numpy().astype(np.float32))
            targets.append(batch["velocity"].cpu().numpy().astype(np.float32))
            records.append(batch["records"].cpu().numpy().astype(np.float32))
            source_positions.append(batch["source_positions"].cpu().numpy().astype(np.float32))
    return {
        "features": np.concatenate(features, axis=0),
        "target": np.concatenate(targets, axis=0),
        "records_preview": np.concatenate(records, axis=0),
        "source_positions": np.concatenate(source_positions, axis=0),
    }


def cache_foundation_features(
    *,
    manifest_path: str | Path,
    output_root: str | Path,
    backbone_type: str = "dummy",
    model_name: str = "dummy_dinov2",
    bridge: str = "raw_spectrogram",
    device: str = "cpu",
) -> dict[str, Any]:
    manifest = _load_manifest(Path(manifest_path))
    splits = _split_paths(manifest)
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        depth, width = _first_velocity_shape(splits["train"][0])
        model = FrozenFoundationFWI(
            foundation_backbone=model_name,
            backbone_type=backbone_type,
            model_name=model_name,
            pretrained=False if backbone_type == "dummy" else True,
            freeze_backbone=True,
            peft_type="none",
            bridge_feature_mode=bridge,
            depth=depth,
            width=width,
            device=device,
            print_parameter_report=False,
        ).to(device)
    except Exception as exc:
        status = "SKIPPED_REAL_DINOV2" if "dinov2" in str(model_name).lower() else "FAILED"
        result = {"status": status, "reason": f"{type(exc).__name__}: {exc}"}
        (output_dir / "metadata.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return result

    split_counts: dict[str, int] = {}
    feature_dim = None
    for split_name, paths in splits.items():
        payload = _extract_split(model, paths, device=device)
        np.savez(output_dir / f"{split_name}_features.npz", **payload)
        split_counts[split_name] = int(payload["features"].shape[0])
        feature_dim = int(payload["features"].shape[-1])
    result = {
        "status": "SUCCESS",
        "manifest_path": str(manifest_path),
        "output_root": str(output_root),
        "backbone_type": backbone_type,
        "model_name": model_name,
        "bridge": bridge,
        "device": device,
        "feature_dim": feature_dim,
        "split_counts": split_counts,
    }
    (output_dir / "metadata.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache frozen foundation features for Protocol V3.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--backbone-type", default="dummy")
    parser.add_argument("--model-name", default="dummy_dinov2")
    parser.add_argument("--bridge", default="raw_spectrogram")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = cache_foundation_features(
        manifest_path=args.manifest,
        output_root=args.output_root,
        backbone_type=args.backbone_type,
        model_name=args.model_name,
        bridge=args.bridge,
        device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
