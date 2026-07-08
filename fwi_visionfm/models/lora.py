from __future__ import annotations

import warnings
from typing import Any

from fwi_visionfm.optional_deps import missing_dependencies

if not missing_dependencies("torch"):
    import torch

    _LoRABase = torch.nn.Module
else:
    _LoRABase = object


def _require_torch():
    if missing_dependencies("torch"):
        raise RuntimeError(
            "PyTorch backend is unavailable. Install PyTorch first, then rerun this experiment. "
            "Suggested CPU install: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
        )
    import torch

    return torch


class LoRALinear(_LoRABase):
    """
    Wrap nn.Linear with a low-rank residual path.
    """

    def __init__(self, base: Any, r: int = 4, alpha: float = 8.0, dropout: float = 0.0, freeze_base: bool = True) -> None:
        torch = _require_torch()
        nn = torch.nn
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear only supports torch.nn.Linear")
        if r <= 0:
            raise ValueError("LoRA rank r must be positive")
        self.base = base
        self.dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Linear(base.in_features, int(r), bias=False)
        self.lora_B = nn.Linear(int(r), base.out_features, bias=False)
        self.scaling = float(alpha) / float(r)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
        if freeze_base:
            for parameter in self.base.parameters():
                parameter.requires_grad = False

    def forward(self, x):
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(self.dropout(x)))


def _replace_child(parent: Any, child_name: str, new_module: Any) -> None:
    if hasattr(parent, child_name):
        setattr(parent, child_name, new_module)
        return
    try:
        parent._modules[child_name] = new_module
    except Exception as exc:
        raise RuntimeError(f"failed to replace child module '{child_name}' with LoRALinear") from exc


def replace_linear_with_lora(
    model,
    target_keywords=("qkv", "query", "key", "value", "proj", "fc1", "fc2", "dense"),
    r: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.0,
    freeze_base: bool = True,
) -> int:
    """
    Replace matched nn.Linear modules with LoRALinear wrappers.
    """

    torch = _require_torch()
    nn = torch.nn
    replaced = 0
    lowered_keywords = tuple(str(keyword).lower() for keyword in target_keywords)

    def _visit(parent: Any, prefix: str = "") -> None:
        nonlocal replaced
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            lowered_name = full_name.lower()
            if isinstance(child, nn.Linear) and any(keyword in lowered_name or keyword == child_name.lower() for keyword in lowered_keywords):
                _replace_child(
                    parent,
                    child_name,
                    LoRALinear(child, r=r, alpha=alpha, dropout=dropout, freeze_base=freeze_base),
                )
                replaced += 1
                continue
            _visit(child, full_name)

    _visit(model)
    if replaced == 0:
        warnings.warn(
            "replace_linear_with_lora: matched 0 Linear modules for target_keywords="
            f"{target_keywords}. The model is left unchanged.",
            RuntimeWarning,
            stacklevel=2,
        )
    return replaced


def count_lora_parameters(model) -> dict[str, float]:
    total = 0
    trainable = 0
    for name, parameter in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
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
