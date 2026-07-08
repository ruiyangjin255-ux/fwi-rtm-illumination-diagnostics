from __future__ import annotations

def freeze_module(module):
    for parameter in module.parameters():
        parameter.requires_grad = False


def unfreeze_module(module):
    for parameter in module.parameters():
        parameter.requires_grad = True


def count_parameters(module) -> tuple[int, int, float]:
    total_params = 0
    trainable_params = 0
    for parameter in module.parameters():
        count = int(parameter.numel())
        total_params += count
        if bool(getattr(parameter, "requires_grad", False)):
            trainable_params += count
    trainable_ratio = float(trainable_params / total_params) if total_params > 0 else 0.0
    return total_params, trainable_params, trainable_ratio


def print_parameter_report(model, title: str = "model", metadata: dict[str, object] | None = None):
    total_params, trainable_params, trainable_ratio = count_parameters(model)
    lines = [
        f"[{title}]",
        f"total parameters: {total_params}",
        f"trainable parameters: {trainable_params}",
        f"trainable ratio: {trainable_ratio:.6f}",
    ]
    if metadata:
        for key, value in metadata.items():
            lines.append(f"{key}: {value}")
    lines.append("top-level trainable parameters:")
    for name, child in model.named_children():
        child_total, child_trainable, child_ratio = count_parameters(child)
        lines.append(f"- {name}: trainable={child_trainable} total={child_total} ratio={child_ratio:.6f}")
    report = "\n".join(lines)
    print(report)
    return {
        "title": title,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_ratio": trainable_ratio,
        "report": report,
    }


def set_trainable_by_transfer_mode(model, transfer_mode: str):
    mode = str(transfer_mode).lower()
    if mode not in {"scratch", "frozen", "full", "adapter", "lora"}:
        raise ValueError(f"unsupported transfer_mode: {transfer_mode}")
    if mode in {"scratch", "full"}:
        unfreeze_module(model)
        return model
    freeze_module(model)
    for name, parameter in model.named_parameters():
        lowered = name.lower()
        top_level = lowered.split(".", 1)[0]
        if top_level in {"bridge", "aggregator", "decoder", "head", "regressor", "velocity_decoder"}:
            parameter.requires_grad = True
        elif mode == "adapter" and "adapter" in lowered:
            parameter.requires_grad = True
        elif mode == "lora" and ("lora_a" in lowered or "lora_b" in lowered):
            parameter.requires_grad = True
    return model
