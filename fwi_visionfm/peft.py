from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fwi_visionfm.optional_deps import missing_dependencies

if not missing_dependencies("torch"):
    import torch

    _LoRABaseModule = torch.nn.Module
else:
    _LoRABaseModule = object


@dataclass(frozen=True)
class LoRAConfig:
    enabled: bool = False
    rank: int = 4
    alpha: float = 8.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("qkv", "proj", "fc1", "fc2")
    train_bias: bool = False


@dataclass(frozen=True)
class AdapterConfig:
    enabled: bool = False
    bottleneck_dim: int = 64
    dropout: float = 0.0
    target_blocks: str = "all"


class LoRALinear(_LoRABaseModule):
    def __init__(self, base_linear: Any, config: LoRAConfig) -> None:
        torch = require_peft_torch()
        nn = torch.nn
        super().__init__()
        if config.rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("LoRALinear only supports torch.nn.Linear")
        self.base_linear = base_linear
        self.dropout = nn.Dropout(config.dropout)
        self.lora_A = nn.Linear(base_linear.in_features, config.rank, bias=False)
        self.lora_B = nn.Linear(config.rank, base_linear.out_features, bias=False)
        self.scaling = float(config.alpha) / float(config.rank)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
        for parameter in self.base_linear.parameters():
            parameter.requires_grad = False

    def forward(self, tensor: Any) -> Any:
        base = self.base_linear(tensor)
        lora = self.lora_B(self.lora_A(self.dropout(tensor)))
        return base + self.scaling * lora


class AdapterLinear(_LoRABaseModule):
    def __init__(self, base_linear: Any, config: AdapterConfig) -> None:
        torch = require_peft_torch()
        nn = torch.nn
        super().__init__()
        if config.bottleneck_dim <= 0:
            raise ValueError("Adapter bottleneck_dim must be positive")
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("AdapterLinear only supports torch.nn.Linear")
        self.base_linear = base_linear
        self.down = nn.Linear(base_linear.out_features, config.bottleneck_dim)
        self.up = nn.Linear(config.bottleneck_dim, base_linear.out_features)
        self.dropout = nn.Dropout(config.dropout)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        for parameter in self.base_linear.parameters():
            parameter.requires_grad = False

    def forward(self, tensor: Any) -> Any:
        base = self.base_linear(tensor)
        adapted = self.up(self.dropout(torch.nn.functional.gelu(self.down(base))))
        return base + adapted


def require_peft_torch() -> Any:
    if missing_dependencies("torch"):
        raise RuntimeError(
            "PyTorch backend is unavailable. Install PyTorch first, then rerun this experiment. "
            "Suggested CPU install: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
        )
    import torch

    return torch


def peft_status() -> dict[str, object]:
    try:
        require_peft_torch()
    except RuntimeError as exc:
        return {"available": False, "reason": str(exc)}
    return {"available": True, "reason": ""}


def _replace_module(parent: Any, child_name: str, new_module: Any) -> None:
    if hasattr(parent, child_name):
        setattr(parent, child_name, new_module)
        return
    if isinstance(parent, (list, tuple)):
        parent[int(child_name)] = new_module
        return
    try:
        parent._modules[child_name] = new_module
    except Exception as exc:
        raise RuntimeError(f"failed to replace module '{child_name}' with LoRA wrapper") from exc


def inject_lora_modules(model: Any, config: LoRAConfig) -> int:
    torch = require_peft_torch()
    nn = torch.nn
    if not config.enabled:
        return 0
    injected_count = 0
    available_linear_names: list[str] = []

    def _visit(parent: Any, prefix: str = "") -> None:
        nonlocal injected_count
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear):
                available_linear_names.append(full_name)
                if any(token in full_name or token == child_name for token in config.target_modules):
                    lora_module = LoRALinear(child, config)
                    _replace_module(parent, child_name, lora_module)
                    injected_count += 1
                    continue
            _visit(child, full_name)

    _visit(model)
    if injected_count == 0:
        available = ", ".join(available_linear_names[:32]) if available_linear_names else "none"
        raise ValueError(
            "LoRA injection matched 0 Linear modules. "
            f"Requested targets: {config.target_modules}. Available Linear module names: {available}"
        )
    return injected_count


def mark_only_lora_as_trainable(model: Any, train_bias: bool = False) -> None:
    for name, parameter in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            parameter.requires_grad = True
        elif train_bias and name.endswith("bias"):
            parameter.requires_grad = True
        else:
            parameter.requires_grad = False


def inject_adapter_modules(model: Any, config: AdapterConfig) -> int:
    torch = require_peft_torch()
    nn = torch.nn
    if not config.enabled:
        return 0
    injected_count = 0

    def _match(name: str) -> bool:
        if config.target_blocks == "all":
            return True
        targets = tuple(part.strip() for part in str(config.target_blocks).split(",") if part.strip())
        return any(token in name for token in targets)

    def _visit(parent: Any, prefix: str = "") -> None:
        nonlocal injected_count
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear) and _match(full_name):
                adapter_module = AdapterLinear(child, config)
                _replace_module(parent, child_name, adapter_module)
                injected_count += 1
                continue
            _visit(child, full_name)

    _visit(model)
    if injected_count == 0:
        raise ValueError(
            "Adapter injection matched 0 Linear modules. "
            f"Requested target_blocks={config.target_blocks!r}."
        )
    return injected_count


def mark_only_adapter_as_trainable(model: Any, train_bias: bool = False) -> None:
    for name, parameter in model.named_parameters():
        if ".down." in name or ".up." in name:
            parameter.requires_grad = True
        elif train_bias and name.endswith("bias"):
            parameter.requires_grad = True
        else:
            parameter.requires_grad = False


def count_trainable_parameters(model: Any) -> dict[str, float]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
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
