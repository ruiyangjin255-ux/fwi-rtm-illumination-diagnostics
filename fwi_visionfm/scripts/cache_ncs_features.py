from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from fwi_visionfm.datasets import load_npz_sample
from fwi_visionfm.models.seismic_backbones.ncs_backbone import DummyNCSModel, load_ncs_model
from fwi_visionfm.models.tokenizers.seismic_tokenization import fallback_tokenization, ncs_2d_tokenization, ncs_2p5d_tokenization
from fwi_visionfm.torch_backend import require_torch_backend


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_id(row: dict[str, Any]) -> str:
    return f"{Path(row['data_file']).name}:{int(row.get('local_index', 0))}"


def _tokenize(records: np.ndarray, *, variant: str, bridge: str) -> dict[str, Any]:
    if variant == "ncs_2d":
        return ncs_2d_tokenization(records, bridge_name=bridge)
    if variant == "ncs_2p5d":
        return ncs_2p5d_tokenization(records, bridge_name=bridge, view_mode="shot_view")
    return fallback_tokenization(records, variant=variant, bridge_name=bridge)


def _feature_from_tokens(model_payload: dict[str, Any], token_payload: dict[str, Any]) -> np.ndarray:
    tokens = np.asarray(token_payload["tokens"], dtype=np.float32)
    if model_payload["status"] == "READY" and model_payload.get("model") is not None:
        model = model_payload["model"]
        if hasattr(model, "encode"):
            encoded = model.encode(tokens)
            return np.asarray(encoded, dtype=np.float32).reshape(-1)
    fallback = DummyNCSModel(variant=str(model_payload["variant"]))
    return np.asarray(fallback.encode(tokens), dtype=np.float32).reshape(-1)


def _collect_split(
    rows: list[dict[str, Any]],
    *,
    variant: str,
    bridge: str,
    model_payload: dict[str, Any],
) -> dict[str, Any]:
    features = []
    targets = []
    records_preview = []
    sample_ids: list[str] = []
    tokenization = ""
    for row in rows:
        sample = load_npz_sample(Path(row["path"]))
        token_payload = _tokenize(sample.records, variant=variant, bridge=bridge)
        feature = _feature_from_tokens(model_payload, token_payload)
        features.append(feature.astype(np.float32))
        targets.append(sample.velocity.astype(np.float32))
        records_preview.append(sample.records.astype(np.float32))
        sample_ids.append(_sample_id(row))
        tokenization = token_payload["metadata"]["tokenization_name"]
    feature_array = np.stack(features, axis=0).astype(np.float32)
    target_array = np.stack(targets, axis=0).astype(np.float32)
    preview_array = np.stack(records_preview, axis=0).astype(np.float32)
    return {
        "features": feature_array,
        "target": target_array,
        "records_preview": preview_array,
        "sample_ids": sample_ids,
        "target_shape": list(target_array.shape[-2:]),
        "feature_shape": list(feature_array.shape[1:]),
        "tokenization": tokenization,
    }


def cache_ncs_features(
    *,
    manifest_path: str | Path,
    output_root: str | Path,
    variant: str,
    bridge: str,
    repo_path: str | Path | None = None,
    weights_path: str | Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    torch = require_torch_backend()
    manifest = _load_manifest(Path(manifest_path))
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_payload = load_ncs_model(variant, repo_path=repo_path, weights_path=weights_path, device=device)
    split_map = {
        "train": manifest["train_samples"],
        "val": manifest["val_samples"],
        "in_family_test": manifest["in_family_test_samples"],
        "cross_family_test": manifest["cross_family_test_samples"],
    }
    split_meta: dict[str, Any] = {}
    tokenization_name = ""
    feature_shape: list[int] = []
    target_shape: list[int] = []
    sample_ids_by_split: dict[str, list[str]] = {}
    for split_name, rows in split_map.items():
        payload = _collect_split(rows, variant=variant, bridge=bridge, model_payload=model_payload)
        torch_payload = {
            "features": torch.as_tensor(payload["features"], dtype=torch.float32),
            "target": torch.as_tensor(payload["target"], dtype=torch.float32),
            "records_preview": torch.as_tensor(payload["records_preview"], dtype=torch.float32),
            "sample_ids": list(payload["sample_ids"]),
            "target_shape": list(payload["target_shape"]),
            "feature_shape": list(payload["feature_shape"]),
            "tokenization": payload["tokenization"],
            "ncs_status": dict(model_payload.get("ncs_status", {})),
            "is_real_ncs_feature": model_payload["status"] == "READY",
            "status": model_payload["status"],
        }
        torch.save(torch_payload, output_dir / f"{split_name}_features.pt")
        split_meta[split_name] = int(len(rows))
        tokenization_name = payload["tokenization"]
        feature_shape = payload["feature_shape"]
        target_shape = payload["target_shape"]
        sample_ids_by_split[split_name] = payload["sample_ids"]
    status = "SUCCESS" if model_payload["status"] == "READY" else "SKIPPED_NCS_UNAVAILABLE" if model_payload["status"] == "SKIPPED_NCS_UNAVAILABLE" else model_payload["status"]
    metadata = {
        "variant": variant,
        "bridge": bridge,
        "tokenization": tokenization_name or "fallback_tokenization",
        "feature_shape": feature_shape,
        "target_shape": target_shape,
        "sample_ids": sample_ids_by_split,
        "ncs_status": model_payload.get("ncs_status", {}),
        "is_real_ncs_feature": model_payload["status"] == "READY",
        "status": status,
        "split_counts": split_meta,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"status": status, "output_root": str(output_dir), "metadata": metadata}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache NCS frozen features or dummy fallback features.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variant", required=True, choices=["ncs_2d", "ncs_2p5d"])
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--ncs-repo", type=Path, default=None)
    parser.add_argument("--ncs-weights", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = cache_ncs_features(
        manifest_path=args.manifest,
        output_root=args.output_root,
        variant=args.variant,
        bridge=args.bridge,
        repo_path=args.ncs_repo,
        weights_path=args.ncs_weights,
        device=args.device,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
