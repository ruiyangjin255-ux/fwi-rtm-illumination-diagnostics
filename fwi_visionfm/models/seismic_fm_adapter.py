from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fwi_visionfm.datasets import load_npz_sample
from fwi_visionfm.models.seismic_backbones.ncs_backbone import DummyNCSModel, detect_ncs_repo, detect_ncs_weights, load_ncs_model
from fwi_visionfm.models.tokenizers.seismic_tokenization import fallback_tokenization, ncs_2d_tokenization, ncs_2p5d_tokenization


SUPPORTED_BACKBONES = {"ncs_2d", "ncs_2p5d", "ncs_3d", "local_mae", "fallback_random"}
AVAILABILITY_STATUSES = {
    "AVAILABLE",
    "UNAVAILABLE_REPO",
    "UNAVAILABLE_WEIGHTS",
    "IMPORT_ERROR",
    "CHECKPOINT_LOAD_ERROR",
    "FORWARD_ERROR",
    "WEIGHTS_PRESENT_ADAPTER_PENDING",
    "SKIPPED",
}


def _sample_id(row: dict[str, Any]) -> str:
    return f"{Path(row['data_file']).name}:{int(row.get('local_index', 0))}"


def load_sample_arrays(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample = load_npz_sample(Path(row["path"]))
    return (
        np.asarray(sample.records, dtype=np.float32),
        np.asarray(sample.velocity, dtype=np.float32),
        np.asarray(sample.source_positions, dtype=np.float32),
    )


def get_tokenizer_name(backbone_name: str) -> str:
    if backbone_name == "ncs_2d":
        return "ncs_2d_tokenization"
    if backbone_name in {"ncs_2p5d", "ncs_3d"}:
        return "ncs_2p5d_tokenization"
    if backbone_name == "local_mae":
        return "local_mae_summary_tokenization"
    return "fallback_tokenization"


def tokenize_records(records: np.ndarray, *, backbone_name: str, bridge_name: str) -> dict[str, Any]:
    if backbone_name == "ncs_2d":
        return ncs_2d_tokenization(records, bridge_name=bridge_name)
    if backbone_name in {"ncs_2p5d", "ncs_3d"}:
        return ncs_2p5d_tokenization(records, bridge_name=bridge_name, view_mode="shot_view")
    payload = fallback_tokenization(records, variant=backbone_name, bridge_name=bridge_name)
    payload["metadata"]["tokenization_name"] = get_tokenizer_name(backbone_name)
    return payload


def probe_ncs_availability(
    *,
    variant: str,
    repo_path: str | Path | None = None,
    weights_path: str | Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    repo = detect_ncs_repo(repo_path)
    if not repo.get("available"):
        return {
            "variant": variant,
            "status": "UNAVAILABLE_REPO",
            "message": repo.get("reason", "NCS repo not found"),
            "repo": repo,
            "weights": detect_ncs_weights(weights_path),
        }
    weights = detect_ncs_weights(weights_path)
    if not weights.get("available"):
        return {
            "variant": variant,
            "status": "UNAVAILABLE_WEIGHTS",
            "message": weights.get("reason", "NCS weights not found"),
            "repo": repo,
            "weights": weights,
        }
    try:
        model_payload = load_ncs_model(variant, repo_path=repo_path, weights_path=weights_path, device=device)
    except Exception as exc:  # pragma: no cover - defensive path
        return {
            "variant": variant,
            "status": "CHECKPOINT_LOAD_ERROR",
            "message": f"{type(exc).__name__}: {exc}",
            "repo": repo,
            "weights": weights,
        }
    if model_payload.get("status") == "WEIGHTS_PRESENT_ADAPTER_PENDING":
        return {
            "variant": variant,
            "status": "WEIGHTS_PRESENT_ADAPTER_PENDING",
            "message": str(model_payload.get("ncs_status", {}).get("reason") or "adapter pending"),
            "repo": repo,
            "weights": weights,
        }
    if model_payload.get("status") != "READY":
        return {
            "variant": variant,
            "status": "IMPORT_ERROR",
            "message": str(model_payload.get("ncs_status", {}).get("reason") or model_payload.get("status")),
            "repo": repo,
            "weights": weights,
        }
    try:
        model = model_payload["model"]
        if not hasattr(model, "encode"):
            raise AttributeError("loaded NCS model does not expose encode()")
        if variant == "ncs_2d":
            encoded = model.encode(np.zeros((1, 3, 64, 64), dtype=np.float32))
        elif variant == "ncs_2p5d":
            encoded = model.encode(np.zeros((1, 4, 3, 224, 224), dtype=np.float32))
        else:
            token_payload = fallback_tokenization(np.zeros((5, 18, 14), dtype=np.float32), variant=variant, bridge_name="availability_probe")
            encoded = model.encode(np.asarray(token_payload["tokens"], dtype=np.float32))
        feature_shape = list(np.asarray(encoded, dtype=np.float32).reshape(-1).shape)
    except Exception as exc:
        return {
            "variant": variant,
            "status": "FORWARD_ERROR",
            "message": f"{type(exc).__name__}: {exc}",
            "repo": repo,
            "weights": weights,
        }
    return {
        "variant": variant,
        "status": "AVAILABLE",
        "message": "repo, weights, import, and forward probe passed",
        "repo": repo,
        "weights": weights,
        "feature_shape": feature_shape,
    }


def load_backbone(
    backbone_name: str,
    *,
    repo_path: str | Path | None = None,
    weights_path: str | Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    if backbone_name not in SUPPORTED_BACKBONES:
        raise ValueError(f"unsupported seismic FM backbone: {backbone_name}")
    if backbone_name in {"local_mae", "fallback_random"}:
        return {
            "backbone_name": backbone_name,
            "status": "READY" if backbone_name == "local_mae" else "FALLBACK_FEATURE_ONLY",
            "model": None,
            "availability": {
                "variant": backbone_name,
                "status": "SKIPPED",
                "message": "non-NCS local fallback backbone",
            },
        }
    availability = probe_ncs_availability(variant=backbone_name, repo_path=repo_path, weights_path=weights_path, device=device)
    if availability["status"] != "AVAILABLE":
        return {
            "backbone_name": backbone_name,
            "status": "SKIPPED_NCS_UNAVAILABLE",
            "model": None,
            "availability": availability,
        }
    model_payload = load_ncs_model(backbone_name, repo_path=repo_path, weights_path=weights_path, device=device)
    return {
        "backbone_name": backbone_name,
        "status": "READY" if model_payload.get("status") == "READY" else "SKIPPED_NCS_UNAVAILABLE",
        "model": model_payload.get("model"),
        "availability": availability,
        "raw_model_payload": model_payload,
    }


def extract_feature(
    records: np.ndarray,
    *,
    backbone_payload: dict[str, Any],
    bridge_name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    backbone_name = str(backbone_payload["backbone_name"])
    token_payload = tokenize_records(records, backbone_name=backbone_name, bridge_name=bridge_name)
    tokens = np.asarray(token_payload["tokens"], dtype=np.float32)
    tokenizer_name = str(token_payload["metadata"]["tokenization_name"])
    if backbone_payload.get("status") == "READY" and backbone_payload.get("model") is not None:
        model = backbone_payload["model"]
        if hasattr(model, "encode"):
            encoded = np.asarray(model.encode(tokens), dtype=np.float32).reshape(-1)
        else:
            raise AttributeError(f"{backbone_name} model does not provide encode()")
        return encoded, {
            "status": "SUCCESS",
            "tokenizer_name": tokenizer_name,
            "used_fallback": False,
        }
    if backbone_name == "local_mae":
        encoded = np.asarray(fallback_tokenization(records, variant="local_mae", bridge_name=bridge_name, feature_dim=192)["tokens"], dtype=np.float32).reshape(-1)
        return encoded, {
            "status": "SUCCESS",
            "tokenizer_name": tokenizer_name,
            "used_fallback": True,
        }
    fallback = DummyNCSModel(variant=backbone_name)
    encoded = np.asarray(fallback.encode(tokens), dtype=np.float32).reshape(-1)
    status = "SKIPPED_NCS_UNAVAILABLE" if backbone_name.startswith("ncs_") else "FALLBACK_FEATURE_ONLY"
    return encoded, {
        "status": status,
        "tokenizer_name": tokenizer_name,
        "used_fallback": True,
    }


def build_split_npz_payload(
    rows: list[dict[str, Any]],
    *,
    backbone_payload: dict[str, Any],
    bridge_name: str,
    split_name: str,
    source_family: str,
    target_family: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    previews: list[np.ndarray] = []
    sample_ids: list[str] = []
    tokenizer_name = get_tokenizer_name(str(backbone_payload["backbone_name"]))
    split_status = "SUCCESS"
    used_fallback = False
    for row in rows:
        records, velocity, _ = load_sample_arrays(row)
        feature, feature_meta = extract_feature(records, backbone_payload=backbone_payload, bridge_name=bridge_name)
        features.append(feature.astype(np.float32))
        targets.append(velocity.astype(np.float32))
        previews.append(records.astype(np.float32))
        sample_ids.append(_sample_id(row))
        tokenizer_name = str(feature_meta["tokenizer_name"])
        used_fallback = used_fallback or bool(feature_meta["used_fallback"])
        if feature_meta["status"] != "SUCCESS":
            split_status = str(feature_meta["status"])
    feature_array = np.stack(features, axis=0).astype(np.float32)
    target_array = np.stack(targets, axis=0).astype(np.float32)
    preview_array = np.stack(previews, axis=0).astype(np.float32)
    metadata = {
        "split_name": split_name,
        "source_family": source_family,
        "target_family": target_family,
        "backbone_name": backbone_payload["backbone_name"],
        "bridge_name": bridge_name,
        "tokenizer_name": tokenizer_name,
        "feature_shape": list(feature_array.shape[1:]),
        "target_shape": list(target_array.shape[1:]),
        "status": split_status,
        "used_fallback": used_fallback,
        "sample_count": int(feature_array.shape[0]),
        "metric_space": "physical_velocity",
    }
    payload = {
        "features": feature_array,
        "target": target_array,
        "records_preview": preview_array,
        "sample_id": np.asarray(sample_ids, dtype=object),
        "backbone_name": np.asarray(str(backbone_payload["backbone_name"])),
        "bridge_name": np.asarray(str(bridge_name)),
        "tokenizer_name": np.asarray(str(tokenizer_name)),
        "feature_shape": np.asarray(metadata["feature_shape"], dtype=np.int32),
        "source_split": np.asarray(str(source_family)),
        "target_split": np.asarray(str(target_family)),
        "status": np.asarray(str(split_status)),
        "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False)),
        "metric_space": np.asarray("physical_velocity"),
    }
    return payload, metadata
