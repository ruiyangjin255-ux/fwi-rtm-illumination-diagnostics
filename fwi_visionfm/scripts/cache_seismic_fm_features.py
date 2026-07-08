from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from fwi_visionfm.models.seismic_fm_adapter import build_split_npz_payload, load_backbone
from fwi_visionfm.scripts.build_protocol_v2_splits import build_protocol_v2_splits


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_manifest(root: Path, *, source: str, target: str, seed: int) -> Path:
    return root / "manifests" / f"{source}_to_{target}_seed{seed}_manifest.json"


def _ensure_manifest(
    *,
    repo_root: Path,
    output_dir: Path,
    data_root: str | Path,
    source: str,
    target: str,
    seed: int,
    train_size: int,
    val_size: int,
    test_size: int,
    manifest_path: str | Path | None,
) -> Path:
    if manifest_path is not None:
        path = Path(manifest_path)
        if path.exists():
            return path
    protocol_root = output_dir.parent
    path = _find_manifest(protocol_root, source=source, target=target, seed=seed)
    if path.exists():
        return path
    build_protocol_v2_splits(
        data_root=data_root,
        output_root=protocol_root,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        seeds=[int(seed)],
    )
    if not path.exists():
        raise FileNotFoundError(f"manifest not found after build: {path}")
    return path


def _split_rows(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        "train": manifest["train_samples"],
        "val": manifest["val_samples"],
        "in_family_test": manifest["in_family_test_samples"],
        "cross_family_test": manifest["cross_family_test_samples"],
    }


def cache_seismic_fm_features(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    backbone_name: str,
    bridge_name: str,
    repo_path: str | Path | None = None,
    weights_path: str | Path | None = None,
    allow_fallback: bool = True,
    device: str = "cpu",
) -> dict[str, Any]:
    manifest = _load_manifest(Path(manifest_path))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    backbone_payload = load_backbone(backbone_name, repo_path=repo_path, weights_path=weights_path, device=device)
    if backbone_payload["status"] == "SKIPPED_NCS_UNAVAILABLE" and not allow_fallback:
        metadata = {
            "status": "SKIPPED_NCS_UNAVAILABLE",
            "backbone_name": backbone_name,
            "bridge_name": bridge_name,
            "tokenizer_name": "fallback_tokenization",
            "source_family": manifest["source_family"],
            "target_family": manifest["target_family"],
            "metric_space": "physical_velocity",
            "skip_reason": "NCS unavailable and fallback disabled",
        }
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        return metadata
    split_meta: dict[str, Any] = {}
    final_status = "SUCCESS"
    tokenizer_name = ""
    feature_shape: list[int] = []
    for split_name, rows in _split_rows(manifest).items():
        payload, meta = build_split_npz_payload(
            rows,
            backbone_payload=backbone_payload,
            bridge_name=bridge_name,
            split_name=split_name,
            source_family=manifest["source_family"],
            target_family=manifest["target_family"],
        )
        np.savez(out / f"{split_name}_features.npz", **payload)
        split_meta[split_name] = meta
        tokenizer_name = meta["tokenizer_name"]
        feature_shape = meta["feature_shape"]
        if meta["status"] != "SUCCESS":
            final_status = str(meta["status"])
    if backbone_name == "fallback_random" and final_status == "SUCCESS":
        final_status = "FALLBACK_FEATURE_ONLY"
    metadata = {
        "status": final_status,
        "backbone_name": backbone_name,
        "bridge_name": bridge_name,
        "tokenizer_name": tokenizer_name,
        "feature_shape": feature_shape,
        "source_family": manifest["source_family"],
        "target_family": manifest["target_family"],
        "seed": int(manifest["seed"]),
        "metric_space": "physical_velocity",
        "availability": backbone_payload.get("availability", {}),
        "splits": split_meta,
        "manifest_path": str(manifest_path),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache seismic FM features with graceful fallback.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source", default="flatvel_a_subset2k")
    parser.add_argument("--target", default="curvevel_a_subset500")
    parser.add_argument("--bridge", dest="bridge_name", default="raw_envelope_spectrum3")
    parser.add_argument("--backbone", dest="backbone_name", default="ncs_2p5d")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-size", type=int, default=100)
    parser.add_argument("--val-size", type=int, default=50)
    parser.add_argument("--test-size", type=int, default=50)
    parser.add_argument("--repo-path", type=Path, default=None)
    parser.add_argument("--weights-path", type=Path, default=None)
    parser.add_argument("--allow-fallback", type=str, default="true")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = _ensure_manifest(
        repo_root=args.repo_root,
        output_dir=args.output_dir,
        data_root=args.data_root,
        source=args.source,
        target=args.target,
        seed=int(args.seed),
        train_size=int(args.train_size),
        val_size=int(args.val_size),
        test_size=int(args.test_size),
        manifest_path=args.manifest_path,
    )
    result = cache_seismic_fm_features(
        manifest_path=manifest_path,
        output_dir=args.output_dir,
        backbone_name=args.backbone_name,
        bridge_name=args.bridge_name,
        repo_path=args.repo_path,
        weights_path=args.weights_path,
        allow_fallback=str(args.allow_fallback).lower() not in {"0", "false", "no"},
        device=args.device,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

