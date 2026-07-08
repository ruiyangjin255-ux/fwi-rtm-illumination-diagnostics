from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from fwi_visionfm.datasets import load_npz_sample
from fwi_visionfm.models.seismic_backbones.ncs_backbone import NCS2DBackboneAdapter
from fwi_visionfm.models.tokenizers.mae_patch_tokenizer import batch_bridge_images
from fwi_visionfm.scripts.build_protocol_v2_splits import build_protocol_v2_splits
from fwi_visionfm.scripts.check_seismic_fm_availability import check_seismic_fm_availability
from fwi_visionfm.torch_backend import require_torch_backend


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}


def _manifest_path(root: Path, source: str, target: str, seed: int) -> Path:
    return root / "manifests" / f"{source}_to_{target}_seed{seed}_manifest.json"


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_manifest(*, protocol_root: Path, data_root: Path, source: str, target: str, train_size: int, val_size: int, test_size: int, seed: int) -> Path:
    path = _manifest_path(protocol_root, source, target, seed)
    if path.exists():
        return path
    build_protocol_v2_splits(
        data_root=data_root,
        output_root=protocol_root,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        seeds=[seed],
    )
    return path


def _split_rows(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        "train": manifest["train_samples"],
        "val": manifest["val_samples"],
        "cross_family_test": manifest["cross_family_test_samples"],
    }


def _feature_from_sample_stats(sample_path: str | Path, *, feature_dim: int) -> np.ndarray:
    sample = load_npz_sample(sample_path)
    records = np.asarray(sample.records, dtype=np.float32)
    stats = np.array(
        [
            float(records.mean()),
            float(records.std()),
            float(np.min(records)),
            float(np.max(records)),
            float(np.mean(np.abs(records))),
            float(np.sqrt(np.mean(records * records))),
        ],
        dtype=np.float32,
    )
    repeats = int(np.ceil(int(feature_dim) / stats.shape[0]))
    return np.tile(stats, repeats)[: int(feature_dim)].astype(np.float32)


def write_feature_npz(
    *,
    rows: list[dict[str, Any]],
    output_path: str | Path,
    backbone_name: str,
    bridge_name: str,
    tokenizer_name: str,
    feature_builder: Callable[[str | Path], np.ndarray],
    source_split: str,
    target_split: str,
    status: str,
    is_real_feature: bool,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    features = []
    sample_ids = []
    targets = []
    previews = []
    for row in rows:
        path = row.get("path") or row["data_file"]
        sample = load_npz_sample(path)
        features.append(np.asarray(feature_builder(path), dtype=np.float32).reshape(-1))
        sample_ids.append(f"{Path(str(path)).name}:{int(row.get('local_index', 0))}")
        targets.append(np.asarray(sample.velocity, dtype=np.float32))
        previews.append(np.asarray(sample.records, dtype=np.float32))
    feature_array = np.stack(features, axis=0).astype(np.float32) if features else np.zeros((0, 1), dtype=np.float32)
    target_array = np.stack(targets, axis=0).astype(np.float32) if targets else np.zeros((0, 1, 1), dtype=np.float32)
    preview_array = np.stack(previews, axis=0).astype(np.float32) if previews else np.zeros((0, 1, 1, 1), dtype=np.float32)
    metadata = {
        "backbone_name": backbone_name,
        "bridge_name": bridge_name,
        "tokenizer_name": tokenizer_name,
        "feature_shape": list(feature_array.shape[1:]),
        "target_shape": list(target_array.shape[1:]),
        "source_split": source_split,
        "target_split": target_split,
        "status": status,
        "is_real_feature": bool(is_real_feature),
        "sample_id_count": len(sample_ids),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    np.savez(
        output_path,
        features=feature_array,
        target=target_array,
        records_preview=preview_array,
        sample_id=np.asarray(sample_ids, dtype=object),
        backbone_name=np.asarray(str(backbone_name)),
        bridge_name=np.asarray(str(bridge_name)),
        tokenizer_name=np.asarray(str(tokenizer_name)),
        feature_shape=np.asarray(metadata["feature_shape"], dtype=np.int32),
        source_split=np.asarray(str(source_split)),
        target_split=np.asarray(str(target_split)),
        status=np.asarray(str(status)),
        is_real_feature=np.asarray(bool(is_real_feature)),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    return metadata


def _load_real_ncs_2d_model(path: Path):
    return NCS2DBackboneAdapter.from_pretrained(path, feature_mode="mean_patch", device="cpu")


def _load_real_mae_model(path: Path):
    from transformers import AutoConfig, ViTMAEModel

    config = AutoConfig.from_pretrained(str(path))
    model_type = str(getattr(config, "model_type", "")).lower()
    if "mae" not in model_type:
        raise ValueError(f"invalid model_type for MAE: {model_type}")
    model = ViTMAEModel.from_pretrained(str(path))
    model.eval()
    return model


def _real_feature_builder_for_model(model: Any, *, bridge_name: str, output_size: int = 224):
    torch = require_torch_backend()

    def build(sample_path: str | Path) -> np.ndarray:
        batch = batch_bridge_images([sample_path], bridge_name, output_size=output_size)
        images = np.asarray(batch["images"], dtype=np.float32)
        if hasattr(model, "encode"):
            return np.asarray(model.encode(images), dtype=np.float32)[0]
        pixel_values = torch.as_tensor(images, dtype=torch.float32)
        if pixel_values.shape[-1] != output_size or pixel_values.shape[-2] != output_size:
            pixel_values = torch.nn.functional.interpolate(pixel_values, size=(output_size, output_size), mode="bilinear", align_corners=False)
        with torch.no_grad():
            out = model(pixel_values=pixel_values)
        if hasattr(out, "last_hidden_state"):
            tensor = out.last_hidden_state.mean(dim=1)
        elif isinstance(out, (tuple, list)) and len(out) > 0:
            tensor = out[0].mean(dim=1)
        else:
            raise ValueError("unsupported model output for feature extraction")
        return tensor.detach().cpu().numpy()[0].astype(np.float32)

    return build


def cache_v9_real_frozen_features(
    *,
    repo_root: str | Path,
    config_path: str | Path,
    data_root: str | Path,
    source: str,
    target: str,
    bridge: str,
    output_dir: str | Path,
    train_size: int,
    val_size: int,
    test_size: int,
    device: str,
    backbone: str | None = None,
    feature_mode: str = "mean_patch",
    input_size: int = 224,
    allow_fallback: bool = True,
) -> dict[str, Any]:
    root = Path(repo_root)
    cfg = _load_yaml(Path(config_path))
    availability_root = Path(output_dir).parent
    availability = check_seismic_fm_availability(repo_root=root, output_dir=availability_root, config_path=config_path, device=device)
    availability_map = {row["name"]: row for row in availability["models"]}

    protocol_root = Path(output_dir).parent
    manifest_path = _ensure_manifest(protocol_root=protocol_root, data_root=Path(data_root), source=source, target=target, train_size=train_size, val_size=val_size, test_size=test_size, seed=0)
    manifest = _load_manifest(manifest_path)
    split_map = _split_rows(manifest)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    priority = [backbone] if backbone else ["ncs_2d", "ncs_2p5d", "vit_mae_base"]
    for backbone_name in priority:
        backbone_dir = out / backbone_name
        backbone_dir.mkdir(parents=True, exist_ok=True)
        avail = availability_map.get(backbone_name, {"name": backbone_name, "status": "SKIPPED"})
        status = str(avail.get("status", "SKIPPED"))
        is_real = False
        tokenizer_name = "fallback_tokenizer"
        feature_builder: Callable[[str | Path], np.ndarray] = lambda sample_path: _feature_from_sample_stats(sample_path, feature_dim=16)
        cache_status = "FALLBACK_FEATURE_ONLY"
        message = str(avail.get("message", ""))
        adapter_metadata: dict[str, Any] = {}
        try:
            if backbone_name == "ncs_2d" and status == "AVAILABLE":
                model_path = Path(str(avail["path"]))
                model = NCS2DBackboneAdapter.from_pretrained(model_path, feature_mode=feature_mode, device=device)
                feature_builder = _real_feature_builder_for_model(model, bridge_name=bridge, output_size=input_size)
                tokenizer_name = "vit_pixel_values"
                is_real = True
                cache_status = "AVAILABLE"
                adapter_metadata = dict(model.metadata)
            elif backbone_name == "vit_mae_base" and status == "AVAILABLE":
                model_path = Path(str(avail["path"]))
                model = _load_real_mae_model(model_path)
                feature_builder = _real_feature_builder_for_model(model, bridge_name=bridge, output_size=input_size)
                tokenizer_name = "vit_mae_pixel_values"
                is_real = True
                cache_status = "AVAILABLE"
            elif backbone_name == "ncs_2p5d" and status == "WEIGHTS_PRESENT_ADAPTER_PENDING":
                cache_status = "WEIGHTS_PRESENT_ADAPTER_PENDING"
                feature_builder = lambda sample_path: _feature_from_sample_stats(sample_path, feature_dim=24)
            elif not allow_fallback:
                cache_status = status
        except Exception as exc:
            message = f"{message}; feature extraction init failed: {type(exc).__name__}: {exc}".strip("; ")
            is_real = False
            cache_status = "FALLBACK_FEATURE_ONLY" if allow_fallback else "CHECKPOINT_LOAD_ERROR"
        if not allow_fallback and not is_real:
            config_payload = {
                "backbone_name": backbone_name,
                "bridge_name": bridge,
                "status": cache_status,
                "is_real_feature": False,
                "feature_shape": [],
                "target_shape": [],
                "sample_id_count": 0,
                "availability_status": status,
                "message": message,
                "feature_mode": feature_mode,
                "input_size": int(input_size),
                "metric_space": "physical_velocity",
            }
            (backbone_dir / "cache_config.json").write_text(json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8")
            (backbone_dir / "metadata.json").write_text(json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8")
            (backbone_dir / "cache_log.txt").write_text(f"status={cache_status}\nis_real_feature=False\nmessage={message}\n", encoding="utf-8")
            rows.append(config_payload)
            continue
        split_meta = {}
        for split_name, split_rows in split_map.items():
            split_meta[split_name] = write_feature_npz(
                rows=split_rows,
                output_path=backbone_dir / f"{split_name}_features.npz",
                backbone_name=backbone_name,
                bridge_name=bridge,
                tokenizer_name=tokenizer_name,
                feature_builder=feature_builder,
                source_split=split_name,
                target_split=manifest["target_family"],
                status=cache_status,
                is_real_feature=is_real,
                extra_metadata=adapter_metadata | {"metric_space": "physical_velocity", "feature_mode": feature_mode, "input_size": int(input_size)},
            )
        config_payload = {
            "backbone_name": backbone_name,
            "bridge_name": bridge,
            "status": cache_status,
            "is_real_feature": is_real,
            "feature_shape": split_meta["train"]["feature_shape"],
            "target_shape": split_meta["train"]["target_shape"],
            "sample_id_count": split_meta["train"]["sample_id_count"],
            "availability_status": status,
            "message": message,
            "feature_mode": feature_mode,
            "input_size": int(input_size),
            "metric_space": "physical_velocity",
        }
        (backbone_dir / "cache_config.json").write_text(json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        (backbone_dir / "metadata.json").write_text(json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        (backbone_dir / "cache_log.txt").write_text(f"status={cache_status}\nis_real_feature={is_real}\nmessage={message}\n", encoding="utf-8")
        rows.append(config_payload)
    return {"rows": rows, "output_dir": str(out)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache V9 real frozen features when local weights are available.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", dest="config_path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--bridge", default="raw_envelope_spectrum3")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=100)
    parser.add_argument("--val-size", type=int, default=50)
    parser.add_argument("--test-size", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--feature-mode", default="mean_patch")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--allow-fallback", type=str, default="true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = cache_v9_real_frozen_features(
        repo_root=args.repo_root,
        config_path=args.config_path,
        data_root=args.data_root,
        source=args.source,
        target=args.target,
        bridge=args.bridge,
        output_dir=args.output_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        device=args.device,
        backbone=args.backbone,
        feature_mode=args.feature_mode,
        input_size=args.input_size,
        allow_fallback=str(args.allow_fallback).lower() not in {"0", "false", "no"},
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
