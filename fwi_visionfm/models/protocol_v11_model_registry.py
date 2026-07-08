from __future__ import annotations

from typing import Any

from fwi_visionfm.models.protocol_v11_backbones import ProtocolV11CNNModel, ProtocolV11FeatureDecoderModel, ProtocolV11VisionModel


METHOD_SPECS: list[dict[str, Any]] = [
    {"method_id": "M1", "method_key": "cnn_baseline", "method_name": "CNN baseline", "bridge": "raw_envelope_spectrum3", "pretraining_source": "none", "kind": "cnn"},
    {"method_id": "M2", "method_key": "random_vit", "method_name": "random ViT", "bridge": "raw_envelope_spectrum3", "pretraining_source": "none", "kind": "vision", "transfer_mode": "scratch", "pretrained": False},
    {"method_id": "M3", "method_key": "dinov2_frozen", "method_name": "DINOv2 frozen", "bridge": "raw_envelope_spectrum3", "pretraining_source": "natural_image_dinov2", "kind": "vision", "transfer_mode": "frozen", "pretrained": True},
    {"method_id": "M4", "method_key": "dinov2_lora", "method_name": "DINOv2-LoRA", "bridge": "raw_envelope_spectrum3", "pretraining_source": "natural_image_dinov2", "kind": "vision", "transfer_mode": "lora", "pretrained": True},
    {"method_id": "M5", "method_key": "spectrogram_dinov2_lora", "method_name": "spectrogram-DINOv2-LoRA", "bridge": "spectrogram_multiband", "pretraining_source": "natural_image_dinov2", "kind": "vision", "transfer_mode": "lora", "pretrained": True},
    {"method_id": "M6", "method_key": "ncs2d_frozen", "method_name": "NCS2D frozen", "bridge": "raw_envelope_spectrum3", "pretraining_source": "seismic_ncs2d", "kind": "ncs"},
]


def get_method_spec(method_id_or_key: str) -> dict[str, Any]:
    for spec in METHOD_SPECS:
        if method_id_or_key in {spec["method_id"], spec["method_key"]}:
            return dict(spec)
    raise KeyError(method_id_or_key)


def build_protocol_v11_model(spec: dict[str, Any], config: dict[str, Any], *, vmin: float, vmax: float):
    kwargs = {"output_shape": tuple(config["velocity_shape"]), "vmin": vmin, "vmax": vmax, "base_channels": int(config["decoder_base_channels"])}
    if spec["kind"] == "cnn":
        return ProtocolV11CNNModel(image_size=int(config["image_size"]), **kwargs)
    if spec["kind"] == "vision":
        return ProtocolV11VisionModel(bridge=spec["bridge"], transfer_mode=spec["transfer_mode"], pretrained=bool(spec["pretrained"]), config=config, vmin=vmin, vmax=vmax)
    if spec["kind"] == "ncs":
        return ProtocolV11FeatureDecoderModel(**kwargs)
    raise ValueError(spec["kind"])


def count_parameters(model: Any) -> dict[str, int | float]:
    from torch.nn.parameter import UninitializedParameter

    module = getattr(model, "module", model)
    initialized = [p for p in module.parameters() if not isinstance(p, UninitializedParameter)]
    total = sum(p.numel() for p in initialized)
    trainable = sum(p.numel() for p in initialized if p.requires_grad)
    return {"total_parameters": int(total), "trainable_parameters": int(trainable), "trainable_ratio": float(trainable / total) if total else 0.0}
