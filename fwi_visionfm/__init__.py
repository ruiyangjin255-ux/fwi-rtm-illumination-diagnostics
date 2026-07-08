"""Minimal FWI-VisionFM research scaffold."""

from fwi_visionfm.config import BridgeConfig, DataConfig, LossConfig, ModelConfig, RunConfig
from fwi_visionfm.calibration import apply_linear_calibration, fit_linear_calibration, train_linear_calibration
from fwi_visionfm.data_conversion import convert_array_dataset_to_npz, convert_openfwi_files_to_npz
from fwi_visionfm.datasets import (
    FWIBatch,
    FWISample,
    NPZSampleDataset,
    discover_npz_samples,
    load_npz_sample,
    make_synthetic_sample,
    split_sample_paths,
)
from fwi_visionfm.experiment import (
    ExperimentConfig,
    load_experiment_config,
    run_experiment_from_config,
    save_experiment_config,
)
from fwi_visionfm.foundation_models import available_foundation_backbones, create_foundation_backbone
from fwi_visionfm.foundation_train import generate_foundation_synthetic_npz, run_foundation_npz_experiment
from fwi_visionfm.models import (
    BottleneckAdapter,
    FoundationModelFWI,
    FWIModel,
    FWIVisionFMModel,
    LoRALinear,
    SeismicToVisionBridge,
    VisionBackboneWrapper,
    attach_adapters_to_vit,
    build_vision_backbone,
    count_parameters,
    freeze_module,
    print_parameter_report,
    replace_linear_with_lora,
    set_trainable_by_transfer_mode,
    unfreeze_module,
)
from fwi_visionfm.optional_deps import check_optional_dependencies
from fwi_visionfm.physics import acoustic_data_misfit, physics_consistency_loss
from fwi_visionfm.torch_backend import require_torch_backend, run_torch_npz_experiment, run_torch_smoke_experiment, torch_backend_status
from fwi_visionfm.train import run_npz_experiment, run_smoke_experiment

__all__ = [
    "BridgeConfig",
    "FoundationModelFWI",
    "FWIModel",
    "DataConfig",
    "FWIBatch",
    "FWISample",
    "FWIVisionFMModel",
    "ExperimentConfig",
    "LossConfig",
    "ModelConfig",
    "NPZSampleDataset",
    "RunConfig",
    "acoustic_data_misfit",
    "apply_linear_calibration",
    "attach_adapters_to_vit",
    "available_foundation_backbones",
    "BottleneckAdapter",
    "check_optional_dependencies",
    "count_parameters",
    "convert_array_dataset_to_npz",
    "convert_openfwi_files_to_npz",
    "create_foundation_backbone",
    "discover_npz_samples",
    "load_npz_sample",
    "load_experiment_config",
    "fit_linear_calibration",
    "freeze_module",
    "generate_foundation_synthetic_npz",
    "train_linear_calibration",
    "LoRALinear",
    "make_synthetic_sample",
    "physics_consistency_loss",
    "print_parameter_report",
    "require_torch_backend",
    "replace_linear_with_lora",
    "run_torch_npz_experiment",
    "run_torch_smoke_experiment",
    "run_smoke_experiment",
    "run_npz_experiment",
    "run_experiment_from_config",
    "run_foundation_npz_experiment",
    "save_experiment_config",
    "set_trainable_by_transfer_mode",
    "split_sample_paths",
    "torch_backend_status",
    "SeismicToVisionBridge",
    "unfreeze_module",
    "VisionBackboneWrapper",
    "build_vision_backbone",
]
