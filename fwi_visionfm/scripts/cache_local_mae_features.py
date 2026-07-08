from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from fwi_visionfm.models.seismic_backbones.local_mae import LocalSeismicMAE
from fwi_visionfm.models.tokenizers.mae_patch_tokenizer import batch_bridge_images
from fwi_visionfm.scripts.build_protocol_v2_splits import build_protocol_v2_splits
from fwi_visionfm.torch_backend import require_torch_backend


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_model(checkpoint_path: Path, *, device: str) -> tuple[LocalSeismicMAE, dict[str, Any]]:
    torch = require_torch_backend()
    payload = torch.load(checkpoint_path, map_location=device)
    model = LocalSeismicMAE(
        input_size=int(payload.get("input_size", 64)),
        patch_size=int(payload.get("patch_size", 8)),
        embed_dim=int(payload.get("embed_dim", 128)),
        depth=int(payload.get("depth", 4)),
        num_heads=int(payload.get("num_heads", 4)),
        decoder_embed_dim=int(payload.get("decoder_embed_dim", 64)),
        decoder_depth=int(payload.get("decoder_depth", 2)),
        decoder_heads=int(payload.get("decoder_heads", 4)),
        mask_ratio=float(payload.get("mask_ratio", 0.75)),
    ).to(device)
    state = payload.get("encoder_state") or payload.get("model_state") or {}
    model.module.load_state_dict(state, strict=False)
    return model, payload


def _sample_ids(rows: list[dict[str, Any]]) -> list[str]:
    return [f"{Path(row['data_file']).name}:{int(row.get('local_index', 0))}" for row in rows]


def _write_split(model: LocalSeismicMAE, checkpoint_meta: dict[str, Any], rows: list[dict[str, Any]], bridge: str, output_path: Path, *, device: str) -> dict[str, Any]:
    torch = require_torch_backend()
    batch = batch_bridge_images([row["path"] for row in rows], bridge, output_size=int(checkpoint_meta.get("input_size", 64)))
    images = torch.as_tensor(batch["images"], dtype=torch.float32, device=device)
    with torch.no_grad():
        features = model.encode_features(images).detach().cpu()
    payload = {
        "features": features,
        "target": torch.as_tensor(batch["velocity"], dtype=torch.float32),
        "records_preview": torch.as_tensor(np.stack([np.load(Path(row["path"]))["records"] for row in rows], axis=0).astype(np.float32)),
        "sample_ids": list(batch["sample_ids"]),
        "target_shape": list(batch["velocity"].shape[-2:]),
    }
    torch.save(payload, output_path)
    return {"sample_ids": payload["sample_ids"], "feature_shape": list(features.shape[1:]), "target_shape": payload["target_shape"]}


def write_local_mae_feature_cache(
    *,
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    output_root: str | Path,
    bridge: str,
    model_type: str = "pretrained_local_mae",
    mask_type: str = "random_patch",
    device: str = "cpu",
) -> dict[str, Any]:
    torch = require_torch_backend()
    manifest = _load_manifest(Path(manifest_path))
    model, checkpoint_meta = _load_model(Path(checkpoint_path), device=device)
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_meta = {}
    for split_name, rows in {
        "train": manifest["train_samples"],
        "val": manifest["val_samples"],
        "in_family_test": manifest["in_family_test_samples"],
        "cross_family_test": manifest["cross_family_test_samples"],
    }.items():
        sample_meta[split_name] = _write_split(model, checkpoint_meta, rows, bridge, output_dir / f"{split_name}_features.pt", device=device)
    metadata = {
        "bridge": bridge,
        "mask_type": mask_type,
        "model_type": model_type,
        "encoder_type": "local_seismic_mae",
        "feature_shape": sample_meta["train"]["feature_shape"],
        "target_shape": sample_meta["train"]["target_shape"],
        "sample_ids": {name: meta["sample_ids"] for name, meta in sample_meta.items()},
        "metric_space": "physical_velocity",
        "pretrain_checkpoint": str(checkpoint_path),
        "is_pretrained": True,
        "status": "SUCCESS",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"status": "SUCCESS", "output_root": str(output_dir)}


def cache_local_mae_features_matrix(
    *,
    data_root: str | Path,
    output_root: str | Path,
    source: str,
    target: str,
    bridges: list[str],
    seed: int,
    train_size: int,
    val_size: int,
    test_size: int,
    device: str,
    mask_types: list[str] | None = None,
) -> dict[str, Any]:
    root = Path(output_root)
    build_protocol_v2_splits(data_root=data_root, output_root=root, train_size=train_size, val_size=val_size, test_size=test_size, seeds=[seed])
    manifest_path = root / "manifests" / f"{source}_to_{target}_seed{seed}_manifest.json"
    rows = []
    selected_masks = list(mask_types or ["random_patch"])
    for bridge in bridges:
        for mask_type in selected_masks:
            ckpt = root / "pretrain" / bridge / mask_type / f"seed_{seed}" / "best_mae_encoder.pt"
            if not ckpt.exists():
                rows.append({"bridge": bridge, "mask_type": mask_type, "status": "SKIPPED", "skip_reason": "missing pretrain checkpoint"})
                continue
            out = root / "feature_cache" / "pretrained_local_mae" / bridge / mask_type / f"seed_{seed}"
            rows.append({"bridge": bridge, "mask_type": mask_type, **write_local_mae_feature_cache(manifest_path=manifest_path, checkpoint_path=ckpt, output_root=out, bridge=bridge, model_type="pretrained_local_mae", mask_type=mask_type, device=device)})
    summary = {"status": "SUCCESS", "rows": rows}
    (root / "feature_cache_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache frozen local seismic MAE features.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source", default="flatvel_a_subset2k")
    parser.add_argument("--target", default="curvevel_a_subset500")
    parser.add_argument("--bridges", nargs="+", default=["raw_envelope_spectrum3", "spectrogram_multiband", "raw_spectrogram"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-size", type=int, default=300)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--mask-types", nargs="+", default=["random_patch"])
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = cache_local_mae_features_matrix(
        data_root=args.data_root,
        output_root=args.output_root,
        source=args.source,
        target=args.target,
        bridges=args.bridges,
        seed=args.seed,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        device=args.device,
        mask_types=args.mask_types,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
