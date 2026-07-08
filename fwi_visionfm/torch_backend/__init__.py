from __future__ import annotations

from typing import Any

from fwi_visionfm.optional_deps import missing_dependencies


def require_torch_backend() -> Any:
    if missing_dependencies("torch"):
        raise RuntimeError(
            "PyTorch backend is unavailable. Install PyTorch first, then rerun this experiment. "
            "Suggested CPU install: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
        )
    import torch

    return torch


def torch_backend_status() -> dict[str, object]:
    missing = missing_dependencies("torch")
    return {"available": not missing, "missing": missing}


from fwi_visionfm.torch_backend.data import NPZTorchDataset, build_torch_dataloader
from fwi_visionfm.torch_backend.model import (
    BoundedVelocityDecoder,
    CrossShotAggregator,
    FrozenFoundationFWI,
    FwiVisionFmTorchBaseline,
    FWITorchBaseline,
    PseudoVisionImageBridge,
    SeismicToVisionTorchBridge,
    ShotEncoderCNN,
    SimpleVisionBackbone,
    VelocityDecoderCNN,
    build_pseudo_vision_images,
    build_torch_shot_images,
)
from fwi_visionfm.torch_backend.train import (
    evaluate,
    run_openfwi_small_experiment,
    run_openfwi_scale_study,
    run_torch_ablation_experiment,
    run_torch_cpu_experiment,
    run_torch_npz_experiment,
    run_torch_smoke_experiment,
    save_checkpoint,
    set_torch_seed,
    train_one_epoch,
)

make_torch_dataloader = build_torch_dataloader

__all__ = [
    "BoundedVelocityDecoder",
    "CrossShotAggregator",
    "FrozenFoundationFWI",
    "NPZTorchDataset",
    "FWITorchBaseline",
    "FwiVisionFmTorchBaseline",
    "PseudoVisionImageBridge",
    "SeismicToVisionTorchBridge",
    "ShotEncoderCNN",
    "SimpleVisionBackbone",
    "VelocityDecoderCNN",
    "build_pseudo_vision_images",
    "build_torch_dataloader",
    "build_torch_shot_images",
    "evaluate",
    "make_torch_dataloader",
    "require_torch_backend",
    "run_openfwi_small_experiment",
    "run_openfwi_scale_study",
    "run_torch_ablation_experiment",
    "run_torch_cpu_experiment",
    "run_torch_npz_experiment",
    "run_torch_smoke_experiment",
    "save_checkpoint",
    "set_torch_seed",
    "torch_backend_status",
    "train_one_epoch",
]
