from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fwi_visionfm.models.vision_backbones import build_vision_backbone
from fwi_visionfm.optional_deps import missing_dependencies

FOUNDATION_BACKBONES = ("dinov2", "dummy_dinov2", "dummy", "timm", "hf_dinov2", "mae", "sam")


def _require_torch() -> Any:
    if missing_dependencies("torch"):
        raise RuntimeError(
            "PyTorch backend is unavailable. Install PyTorch first, then rerun this experiment. "
            "Suggested CPU install: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
        )
    import torch

    return torch


@dataclass(frozen=True)
class FoundationBackboneSpec:
    name: str
    dependency_hint: tuple[str, ...]
    description: str


SPECS = {
    "dinov2": FoundationBackboneSpec("dinov2", ("torch", "timm"), "DINOv2 / self-supervised ViT feature extractor."),
    "dummy_dinov2": FoundationBackboneSpec("dummy_dinov2", ("torch",), "Small offline dummy encoder used for CPU smoke tests."),
    "dummy": FoundationBackboneSpec("dummy", ("torch",), "Unified offline dummy vision backbone."),
    "timm": FoundationBackboneSpec("timm", ("torch", "timm"), "Generic timm vision backbone wrapper."),
    "hf_dinov2": FoundationBackboneSpec("hf_dinov2", ("torch", "transformers"), "Hugging Face DINOv2 wrapper."),
    "mae": FoundationBackboneSpec("mae", ("torch", "timm"), "MAE-style ViT feature extractor."),
    "sam": FoundationBackboneSpec("sam", ("torch", "segment_anything"), "SAM image encoder style feature extractor."),
}


def available_foundation_backbones() -> tuple[str, ...]:
    return FOUNDATION_BACKBONES


def foundation_backbone_spec(name: str) -> FoundationBackboneSpec:
    key = name.lower()
    if key not in SPECS:
        raise ValueError(f"unsupported foundation backbone: {name}")
    return SPECS[key]


class TorchFoundationBackbone:
    def __init__(
        self,
        name: str,
        model_id: str | None = None,
        *,
        backbone_name: str | None = None,
        pretrained: bool = True,
        freeze: bool = True,
        output_dim: int | None = None,
        device: str = "cpu",
    ) -> None:
        self.spec = foundation_backbone_spec(name)
        if self.spec.name not in {"dummy_dinov2", "dummy", "dinov2"}:
            missing = missing_dependencies(*self.spec.dependency_hint)
            if missing:
                raise RuntimeError(f"{self.spec.name} backend requires missing dependencies: {', '.join(missing)}")
        self.torch = _require_torch()
        self.name = self.spec.name
        self.model_id = model_id
        self.backbone_name = backbone_name
        self.pretrained = bool(pretrained)
        self.freeze = bool(freeze)
        self.requested_output_dim = output_dim
        self.device = device
        self.wrapper = self._build_wrapper()
        self.module = self._build_module()
        self.module.to(device)
        self.output_dim = int(getattr(self.module, "output_dim"))
        if self.freeze:
            self.module.eval()

    def _resolve_wrapper_args(self) -> dict[str, Any]:
        if self.name in {"dummy_dinov2", "dummy"}:
            return {
                "backbone_type": "dummy",
                "model_name": self.backbone_name or self.model_id or "dummy_dinov2",
                "pretrained": False,
                "freeze": self.freeze,
            }
        if self.name == "timm":
            return {
                "backbone_type": "timm",
                "model_name": self.backbone_name or self.model_id or "vit_tiny_patch16_224",
                "pretrained": self.pretrained,
                "freeze": self.freeze,
            }
        if self.name == "hf_dinov2":
            return {
                "backbone_type": "hf_dinov2",
                "model_name": self.backbone_name or self.model_id or "facebook/dinov2-small",
                "pretrained": self.pretrained,
                "freeze": self.freeze,
            }
        if self.name == "dinov2":
            backbone_name = self.backbone_name or self.model_id or "vit_small_patch14_dinov2.lvd142m"
            backbone_type = "hf_dinov2" if str(backbone_name).startswith("facebook/") else "timm"
            return {
                "backbone_type": backbone_type,
                "model_name": backbone_name,
                "pretrained": self.pretrained,
                "freeze": self.freeze,
            }
        raise NotImplementedError(
            f"{self.name} loading is not implemented in this round. Only dummy_dinov2 and DINOv2 wrappers are available."
        )

    def _build_wrapper(self):
        args = self._resolve_wrapper_args()
        return build_vision_backbone(**args)

    def _build_module(self) -> Any:
        torch = self.torch
        nn = torch.nn
        base_dim = int(self.wrapper.embed_dim)
        target_dim = int(self.requested_output_dim or base_dim)

        class _PooledBackboneModule(nn.Module):
            def __init__(self, wrapper: Any) -> None:
                super().__init__()
                self.backbone = wrapper.module
                self.projection = nn.Identity() if target_dim == base_dim else nn.Linear(base_dim, target_dim)
                self.output_dim = target_dim

            def forward(self, tensor: Any) -> Any:
                tokens = self.backbone(tensor)
                if tokens.ndim != 3:
                    raise RuntimeError(f"expected token tensor [B, N, C], got {tuple(tokens.shape)}")
                pooled = tokens.mean(dim=1)
                return self.projection(pooled)

        return _PooledBackboneModule(self.wrapper)

    def encode(self, tensor: Any) -> Any:
        return self.module(tensor)


def create_foundation_backbone(
    name: str,
    model_id: str | None = None,
    *,
    backbone_name: str | None = None,
    pretrained: bool = True,
    freeze: bool = True,
    output_dim: int | None = None,
    device: str = "cpu",
) -> TorchFoundationBackbone:
    return TorchFoundationBackbone(
        name=name,
        model_id=model_id,
        backbone_name=backbone_name,
        pretrained=pretrained,
        freeze=freeze,
        output_dim=output_dim,
        device=device,
    )
