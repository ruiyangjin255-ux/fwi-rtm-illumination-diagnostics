from __future__ import annotations

import warnings
from typing import Any, Iterable

from fwi_visionfm.optional_deps import missing_dependencies

if not missing_dependencies("torch"):
    import torch

    _AdapterBase = torch.nn.Module
else:
    _AdapterBase = object


def _require_torch():
    if missing_dependencies("torch"):
        raise RuntimeError(
            "PyTorch backend is unavailable. Install PyTorch first, then rerun this experiment. "
            "Suggested CPU install: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
        )
    import torch

    return torch


class BottleneckAdapter(_AdapterBase):
    def __init__(self, dim: int, bottleneck_dim: int = 64, dropout: float = 0.0, scale: float = 1.0) -> None:
        torch = _require_torch()
        nn = torch.nn
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be positive")
        self.scale = float(scale)
        self.down = nn.Linear(int(dim), int(bottleneck_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))
        self.up = nn.Linear(int(bottleneck_dim), int(dim))
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.scale * self.up(self.dropout(self.activation(self.down(x))))


class _AdapterWrappedBlock(_AdapterBase):
    def __init__(self, block: Any, adapter: BottleneckAdapter) -> None:
        super().__init__()
        self.block = block
        self.adapter = adapter

    def forward(self, *args: Any, **kwargs: Any):
        torch = _require_torch()
        output = self.block(*args, **kwargs)
        if torch.is_tensor(output):
            return self.adapter(output)
        if isinstance(output, tuple) and output:
            first = output[0]
            if torch.is_tensor(first):
                return (self.adapter(first), *output[1:])
        return output


def _resolve_container_candidates(model: Any) -> list[tuple[Any, str, Iterable[Any]]]:
    candidates: list[tuple[Any, str, Iterable[Any]]] = []
    paths = (
        ("blocks",),
        ("encoder", "layer"),
        ("backbone", "blocks"),
        ("model", "blocks"),
    )
    for path in paths:
        parent = model
        for attr in path[:-1]:
            parent = getattr(parent, attr, None)
            if parent is None:
                break
        if parent is None:
            continue
        leaf = path[-1]
        container = getattr(parent, leaf, None)
        if container is not None:
            candidates.append((parent, leaf, container))
    return candidates


def _infer_hidden_dim(block: Any) -> int | None:
    direct = getattr(block, "embed_dim", None) or getattr(block, "hidden_size", None)
    if isinstance(direct, int) and direct > 0:
        return direct
    for norm_name in ("norm1", "norm2", "layernorm_before", "layernorm_after"):
        norm = getattr(block, norm_name, None)
        shape = getattr(norm, "normalized_shape", None)
        if isinstance(shape, int) and shape > 0:
            return shape
        if isinstance(shape, (tuple, list)) and shape and int(shape[-1]) > 0:
            return int(shape[-1])
    for path in (("mlp", "fc2"), ("mlp", "proj"), ("output", "dense"), ("intermediate", "dense")):
        node = block
        for attr in path:
            node = getattr(node, attr, None)
            if node is None:
                break
        out_features = getattr(node, "out_features", None)
        if isinstance(out_features, int) and out_features > 0:
            return out_features
    return None


def attach_adapters_to_vit(model, bottleneck_dim: int = 64, dropout: float = 0.0, scale: float = 1.0) -> int:
    """
    Attach lightweight adapters to common ViT / DINOv2 transformer block layouts.

    The function prefers wrapping each transformer block output. If no supported
    block container or hidden dimension can be inferred, it emits a warning and
    leaves the model unchanged.
    """

    candidates = _resolve_container_candidates(model)
    if not candidates:
        warnings.warn(
            "attach_adapters_to_vit: no supported transformer block container found "
            "(expected model.blocks / model.encoder.layer / model.backbone.blocks / model.model.blocks). "
            "Falling back to no adapter insertion.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 0
    attached = 0
    for parent, leaf, container in candidates:
        if not hasattr(container, "__len__"):
            continue
        wrapped_blocks = []
        changed = False
        for block in list(container):
            dim = _infer_hidden_dim(block)
            if dim is None:
                wrapped_blocks.append(block)
                continue
            adapter = BottleneckAdapter(dim=dim, bottleneck_dim=bottleneck_dim, dropout=dropout, scale=scale)
            wrapped_blocks.append(_AdapterWrappedBlock(block, adapter))
            attached += 1
            changed = True
        if changed:
            try:
                container_type = type(container)
                setattr(parent, leaf, container_type(wrapped_blocks))
            except Exception:
                try:
                    for index, block in enumerate(wrapped_blocks):
                        container[index] = block
                except Exception:
                    warnings.warn(
                        f"attach_adapters_to_vit: failed to replace blocks in container '{leaf}'. "
                        "Falling back to the original model.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    return 0
            return attached
    warnings.warn(
        "attach_adapters_to_vit: found block container but could not infer hidden dimension for any block. "
        "Falling back to no adapter insertion.",
        RuntimeWarning,
        stacklevel=2,
    )
    return 0


def count_adapter_parameters(model) -> dict[str, float]:
    total = 0
    trainable = 0
    for name, parameter in model.named_parameters():
        if "adapter." not in name and ".adapter." not in name:
            continue
        count = int(parameter.numel())
        total += count
        if bool(getattr(parameter, "requires_grad", False)):
            trainable += count
    ratio = float(trainable / total) if total > 0 else 0.0
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_ratio": ratio,
    }
