from __future__ import annotations

from typing import Any

from fwi_visionfm.optional_deps import missing_dependencies

try:
    import torch

    _VisionBackboneBase = torch.nn.Module
except ImportError:
    _VisionBackboneBase = object


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch backend is unavailable. Install PyTorch first, then rerun this experiment. "
            "Suggested CPU install: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
        ) from exc

    return torch


def _as_kwargs(cfg_or_kwargs: Any, **overrides: Any) -> dict[str, Any]:
    if cfg_or_kwargs is None:
        payload: dict[str, Any] = {}
    elif isinstance(cfg_or_kwargs, dict):
        payload = dict(cfg_or_kwargs)
    else:
        payload = {
            key: getattr(cfg_or_kwargs, key)
            for key in dir(cfg_or_kwargs)
            if not key.startswith("_") and not callable(getattr(cfg_or_kwargs, key))
        }
    payload.update({key: value for key, value in overrides.items() if value is not None})
    return payload


def _ensure_dependency(name: str, install_hint: str) -> None:
    missing = missing_dependencies(name)
    if missing:
        raise RuntimeError(f"{name} is required for this backbone. Install it with: {install_hint}")


class VisionBackboneWrapper(_VisionBackboneBase):
    def __init__(
        self,
        backbone_type: str = "dummy",
        model_name: str = "vit_tiny_patch16_224",
        pretrained: bool = False,
        image_size: int = 224,
        in_chans: int = 3,
        freeze: bool = False,
        remove_cls_token: bool = False,
        local_files_only: bool = False,
    ) -> None:
        torch = _require_torch()
        super().__init__()

        self.backbone_type = str(backbone_type)
        self.model_name = str(model_name)
        self.pretrained = bool(pretrained)
        self.image_size = int(image_size)
        self.in_chans = int(in_chans)
        self.freeze = bool(freeze)
        self.remove_cls_token = bool(remove_cls_token)
        self.local_files_only = bool(local_files_only)
        self.backbone, self._embed_dim = self._build_backbone()
        if self.freeze:
            for parameter in self.parameters():
                parameter.requires_grad = False
            self.eval()

    @property
    def embed_dim(self) -> int:
        return int(self._embed_dim)

    @property
    def module(self):
        return self

    def to(self, device: str):
        return super().to(device)

    def eval(self):
        return super().eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            super().train(False)
        return self

    def _build_backbone(self):
        if self.backbone_type == "dummy":
            return self._build_dummy_backbone()
        if self.backbone_type == "timm":
            return self._build_timm_backbone()
        if self.backbone_type == "hf_dinov2":
            return self._build_hf_dinov2_backbone()
        raise ValueError(f"unsupported backbone_type: {self.backbone_type}")

    def _build_dummy_backbone(self):
        torch = _require_torch()
        nn = torch.nn
        embed_dim = 128

        class _DummyBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm1 = nn.LayerNorm(embed_dim)
                self.qkv = nn.Linear(embed_dim, embed_dim)
                self.proj = nn.Linear(embed_dim, embed_dim)
                self.norm2 = nn.LayerNorm(embed_dim)
                self.fc1 = nn.Linear(embed_dim, embed_dim)
                self.fc2 = nn.Linear(embed_dim, embed_dim)

            def forward(self, x):
                residual = x
                x = self.norm1(x)
                x = torch.nn.functional.gelu(self.qkv(x))
                x = residual + self.proj(x)
                residual = x
                x = self.norm2(x)
                x = torch.nn.functional.gelu(self.fc1(x))
                return residual + self.fc2(x)

        class _DummyBackbone(nn.Module):
            def __init__(self, in_chans: int, image_size: int) -> None:
                super().__init__()
                self.patch = nn.Conv2d(in_chans, embed_dim, kernel_size=16, stride=16)
                self.blocks = nn.ModuleList([_DummyBlock(), _DummyBlock()])
                self.image_size = image_size

            def forward_features(self, x):
                tokens = self.patch(x).flatten(2).transpose(1, 2)
                for block in self.blocks:
                    tokens = block(tokens)
                return tokens

        return _DummyBackbone(self.in_chans, self.image_size), embed_dim

    def _build_timm_backbone(self):
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError("timm is required for this backbone. Install it with: pip install timm") from exc

        try:
            model = timm.create_model(
                self.model_name,
                pretrained=self.pretrained,
                in_chans=self.in_chans,
                img_size=self.image_size,
                num_classes=0,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create timm backbone '{self.model_name}'. "
                "Check model name, local environment, and pretrained availability."
            ) from exc
        embed_dim = int(getattr(model, "num_features", 0) or 0)
        if embed_dim <= 0:
            raise RuntimeError(f"Unable to infer embed_dim for timm backbone '{self.model_name}'")
        return model, embed_dim

    def _build_hf_dinov2_backbone(self):
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for hf_dinov2. Install it with: pip install transformers"
            ) from exc
        try:
            model = AutoModel.from_pretrained(self.model_name, local_files_only=self.local_files_only)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Hugging Face DINOv2 backbone '{self.model_name}'. "
                "Use a local cached model, set --local-files-only only when cache exists, or switch to dummy/timm."
            ) from exc
        embed_dim = int(getattr(getattr(model, "config", None), "hidden_size", 0) or 0)
        if embed_dim <= 0:
            raise RuntimeError(f"Unable to infer hidden_size for Hugging Face backbone '{self.model_name}'")
        return model, embed_dim

    def _standardize_tokens(self, features):
        torch = _require_torch()
        if isinstance(features, dict):
            if "last_hidden_state" in features:
                features = features["last_hidden_state"]
            elif "x_norm_patchtokens" in features:
                patch = features["x_norm_patchtokens"]
                cls = features.get("x_norm_clstoken")
                if cls is not None and not self.remove_cls_token:
                    features = torch.cat([cls.unsqueeze(1), patch], dim=1)
                else:
                    features = patch
            elif "x_prenorm" in features:
                features = features["x_prenorm"]
            else:
                first_tensor = next((value for value in features.values() if hasattr(value, "ndim")), None)
                if first_tensor is None:
                    raise RuntimeError("forward_features returned dict without tensor-like features")
                features = first_tensor
        if isinstance(features, (list, tuple)):
            if not features:
                raise RuntimeError("backbone returned an empty tuple/list")
            features = features[0]
        if features.ndim == 4:
            features = features.flatten(2).transpose(1, 2)
        elif features.ndim == 2:
            features = features.unsqueeze(1)
        elif features.ndim != 3:
            raise RuntimeError(f"unsupported feature shape from backbone: {tuple(features.shape)}")
        if self.remove_cls_token and features.shape[1] > 1:
            features = features[:, 1:, :]
        return features

    def _forward_impl(self, x):
        if self.backbone_type == "hf_dinov2":
            outputs = self.backbone(pixel_values=x)
            return self._standardize_tokens(outputs.last_hidden_state)
        if hasattr(self.backbone, "forward_features"):
            return self._standardize_tokens(self.backbone.forward_features(x))
        return self._standardize_tokens(self.backbone(x))

    def forward(self, x):
        return self._forward_impl(x)


def build_vision_backbone(cfg_or_kwargs: Any = None, **overrides: Any) -> VisionBackboneWrapper:
    kwargs = _as_kwargs(cfg_or_kwargs, **overrides)
    return VisionBackboneWrapper(**kwargs)
