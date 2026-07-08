"""Audit PASD variants for Phase-2 locked experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .experiment import TrainingConfig, build_model
from .registry import get_variant


def _hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def _count(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def audit_variants(config: str | Path, variants: list[str], output: str | Path) -> dict[str, Any]:
    locked = json.loads(Path(config).read_text(encoding="utf-8"))
    cfg = TrainingConfig(
        weight_edge=float(locked["lambda_edge"]),
        weight_background=float(locked["lambda_bg"]),
        weight_smooth=float(locked["lambda_smooth"]),
        background_sigma=float(locked["Gaussian_sigma"]),
    )
    rows: dict[str, Any] = {}
    for name in variants:
        variant = get_variant(name)
        model = build_model(variant, (70, 70), cfg)
        payload = {
            "variant": name,
            "bridge mode": variant.bridge_mode,
            "input channels": 1 if variant.bridge_mode == "raw" else 3,
            "envelope enabled": variant.bridge_mode == "hybrid",
            "band-energy enabled": variant.bridge_mode == "hybrid",
            "geometry channels enabled": variant.aggregator == "geometry_attention",
            "geometry-aware attention enabled": variant.aggregator == "geometry_attention",
            "aggregation mode": variant.aggregator,
            "decoder type": variant.decoder_mode,
            "loss type": variant.criterion,
            "lambda_l1": 1.0,
            "lambda_bg": cfg.weight_background if variant.criterion == "background_edge" else 0.0,
            "lambda_edge": cfg.weight_edge if variant.criterion == "background_edge" else 0.0,
            "lambda_smooth": cfg.weight_smooth if variant.criterion == "background_edge" else 0.0,
            "Gaussian sigma": cfg.background_sigma if variant.criterion == "background_edge" else None,
            "edge threshold source": locked.get("edge_threshold_source"),
            "trainable parameters": _count(model),
            "total parameters": int(sum(p.numel() for p in model.parameters())),
        }
        payload["config hash"] = _hash(payload)
        rows[name] = payload
    ok = True
    reason = ""
    if {"B4_no_geometry_attention", "B4_pasd_fwi"}.issubset(rows):
        a = dict(rows["B4_no_geometry_attention"])
        b = dict(rows["B4_pasd_fwi"])
        for key in ["geometry channels enabled", "geometry-aware attention enabled", "aggregation mode", "config hash", "variant", "trainable parameters", "total parameters"]:
            a.pop(key, None)
            b.pop(key, None)
        ok = a == b
        if not ok:
            reason = "B4_no_geometry_attention and B4_pasd_fwi differ beyond aggregation/geometry attention."
    result = {"status": "PASS" if ok else "FAIL", "variants": rows, "no_geometry_diff_check": ok, "failure_reason": reason}
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit PASD variant configs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit_variants(args.config, args.variants, args.output)
    print(json.dumps({"status": result["status"], "output": args.output}, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
