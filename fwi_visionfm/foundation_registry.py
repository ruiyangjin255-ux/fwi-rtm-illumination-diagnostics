from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fwi_visionfm.foundation_models import create_foundation_backbone
from fwi_visionfm.peft import LoRAConfig
from fwi_visionfm.torch_backend import require_torch_backend
from fwi_visionfm.torch_backend.model import FrozenFoundationFWI, build_pseudo_vision_images


BACKBONES = [
    {"name": "dummy_dinov2", "kind": "offline_dummy", "pretrained_supported": False},
    {"name": "vit_tiny_patch16_224", "kind": "timm_vit", "pretrained_supported": True},
    {"name": "vit_small_patch16_224", "kind": "timm_vit", "pretrained_supported": True},
    {"name": "vit_base_patch16_224", "kind": "timm_vit", "pretrained_supported": True},
    {"name": "vit_small_patch14_dinov2.lvd142m", "kind": "timm_dinov2", "pretrained_supported": True},
    {"name": "vit_base_patch14_dinov2.lvd142m", "kind": "timm_dinov2", "pretrained_supported": True},
    {"name": "facebook/dinov2-small", "kind": "hf_dinov2", "pretrained_supported": True},
    {"name": "mae_vit_base_patch16", "kind": "placeholder", "pretrained_supported": False},
    {"name": "sam_resnet_or_vit_placeholder", "kind": "placeholder", "pretrained_supported": False},
]


def foundation_registry() -> dict[str, Any]:
    return {"backbones": BACKBONES}


def run_foundation_smoke(*, config: dict[str, Any], device: str = "cpu") -> dict[str, Any]:
    torch = require_torch_backend()
    backbone_name = str(config.get("backbone_name", "dummy_dinov2"))
    output_dir = Path(str(config.get("output_dir", Path.cwd() / "foundation_smoke")))
    output_dir.mkdir(parents=True, exist_ok=True)
    readme_path = output_dir / "README_result.md"
    try:
        model = FrozenFoundationFWI(
            foundation_backbone=backbone_name,
            pretrained=bool(config.get("pretrained", False)),
            freeze_backbone=bool(config.get("freeze_backbone", True)),
            peft_type=str(config.get("peft_type", "none")),
            lora_config=LoRAConfig(enabled=False),
            image_size=int(config.get("image_size", 64)),
            depth=int(config.get("depth", 70)),
            width=int(config.get("width", 70)),
            aggregation=str(config.get("aggregation", "source_attention")),
            device=device,
        ).to(device)
        records = torch.ones((1, 2, 6, 8), dtype=torch.float32)
        source_positions = torch.tensor([[0.2, 0.8]], dtype=torch.float32)
        pseudo = build_pseudo_vision_images(records, source_positions, image_size=int(config.get("image_size", 64)))
        prediction = model(records.to(device), source_positions.to(device))
        target = torch.ones_like(prediction) * 2200.0
        loss = torch.nn.functional.mse_loss(prediction, target)
        payload = {
            "status": "ok",
            "backbone_name": backbone_name,
            "pseudo_image_shape": list(pseudo.shape),
            "prediction_shape": list(prediction.shape),
            "loss": float(loss.detach().cpu()),
            "readme_path": str(readme_path),
        }
        note = "interface smoke only, not performance conclusion"
    except Exception as exc:
        payload = {
            "status": "failed",
            "backbone_name": backbone_name,
            "error": str(exc),
            "readme_path": str(readme_path),
        }
        note = "interface smoke only, not performance conclusion"
    readme_path.write_text(
        f"# Foundation Smoke\n\nbackbone: {backbone_name}\n\nstatus: {payload['status']}\n\n{note}\n",
        encoding="utf-8",
    )
    return payload
