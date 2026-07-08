from __future__ import annotations

import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SUPPORTED_VARIANTS = {"ncs_2d", "ncs_2p5d", "ncs_3d"}

DEFAULT_REPO_PATHS = [
    Path(r"D:\ryjin\fwi_visionfm\external\ncs_models"),
    Path(r"D:\ryjin\fwi_visionfm\external\seismic_foundation_models"),
    Path(r"D:\ryjin\NCS_models"),
    Path(r"D:\ryjin\ncs_models"),
    Path("external/NCS_models"),
    Path("external/ncs_models"),
    Path("external/seismic_foundation_models"),
    Path("third_party/NCS_models"),
]

DEFAULT_WEIGHT_PATHS = [
    Path(r"D:\ryjin\fwi_visionfm\weights\ncs"),
    Path(r"D:\ryjin\fwi_visionfm\weights\seismic_fm"),
    Path(r"D:\ryjin\ncs_weights"),
    Path(r"D:\ryjin\NCS_weights"),
    Path("weights/ncs"),
    Path("weights/seismic_fm"),
    Path("external/ncs_weights"),
]


def _resolve_existing_path(path: str | Path | None, *, expect_file: bool | None = None) -> Path | None:
    if not path:
        return None
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    if not candidate.exists():
        return None
    if expect_file is True and not candidate.is_file():
        return None
    if expect_file is False and not candidate.is_dir():
        return None
    return candidate.resolve()


def _expected_repo_paths(repo_path: str | Path | None = None) -> list[Path]:
    values: list[Path] = []
    if repo_path:
        values.append(Path(repo_path).expanduser())
    env = os.environ.get("FWI_VISIONFM_NCS_REPO")
    if env:
        values.append(Path(env).expanduser())
    values.extend(DEFAULT_REPO_PATHS)
    return [path if path.is_absolute() else (Path.cwd() / path).resolve() for path in values]


def _expected_weight_paths(weights_path: str | Path | None = None) -> list[Path]:
    values: list[Path] = []
    if weights_path:
        values.append(Path(weights_path).expanduser())
    env = os.environ.get("FWI_VISIONFM_NCS_WEIGHTS")
    if env:
        values.append(Path(env).expanduser())
    values.extend(DEFAULT_WEIGHT_PATHS)
    return [path if path.is_absolute() else (Path.cwd() / path).resolve() for path in values]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _weight_dir_from_path(path: str | Path | None) -> Path | None:
    resolved = _resolve_existing_path(path)
    if resolved is None:
        return None
    return resolved if resolved.is_dir() else resolved.parent


def _candidate_config_dir(variant: str, weights_path: str | Path | None) -> Path | None:
    resolved = _weight_dir_from_path(weights_path)
    if resolved is None:
        return None
    if (resolved / "config.json").exists():
        return resolved
    variant_map = {
        "ncs_2d": "NCS-v1-2d-base",
        "ncs_2p5d": "NCS-v1-2.5d-base",
        "ncs_3d": "NCS-v1-3d-base",
    }
    candidate = resolved / variant_map.get(variant, variant)
    if (candidate / "config.json").exists():
        return candidate
    return resolved


def _detect_variant_weights(variant: str, weights_path: str | Path | None = None) -> dict[str, Any]:
    expected = _expected_weight_paths(weights_path)
    for candidate_root in expected:
        variant_dir = _candidate_config_dir(variant, candidate_root)
        if variant_dir is not None and (variant_dir / "config.json").exists():
            return {
                "available": True,
                "path": str(variant_dir),
                "expected_weight_paths": [str(path) for path in expected],
            }
    return detect_ncs_weights(weights_path)


def detect_ncs_repo(repo_path: str | Path | None = None) -> dict[str, Any]:
    expected = _expected_repo_paths(repo_path)
    for candidate in expected:
        resolved = _resolve_existing_path(candidate, expect_file=False)
        if resolved is not None:
            return {
                "available": True,
                "path": str(resolved),
                "expected_repo_paths": [str(path) for path in expected],
            }
    return {
        "available": False,
        "path": "",
        "reason": "NCS repo not found",
        "expected_repo_paths": [str(path) for path in expected],
    }


def detect_ncs_weights(weights_path: str | Path | None = None) -> dict[str, Any]:
    expected = _expected_weight_paths(weights_path)
    suffixes = {".pt", ".pth", ".ckpt", ".bin", ".safetensors"}
    for candidate in expected:
        resolved = _resolve_existing_path(candidate)
        if resolved is None:
            continue
        if resolved.is_file():
            return {
                "available": True,
                "path": str(resolved),
                "expected_weight_paths": [str(path) for path in expected],
            }
        if resolved.is_dir():
            if (resolved / "config.json").exists():
                return {
                    "available": True,
                    "path": str(resolved),
                    "expected_weight_paths": [str(path) for path in expected],
                }
            for path in sorted(resolved.rglob("*")):
                if path.is_file() and path.suffix.lower() in suffixes:
                    return {
                        "available": True,
                        "path": str(path.resolve()),
                        "expected_weight_paths": [str(p) for p in expected],
                    }
    return {
        "available": False,
        "path": "",
        "reason": "NCS weights not found",
        "expected_weight_paths": [str(path) for path in expected],
    }


def get_ncs_status(
    variant: str = "ncs_2d",
    *,
    repo_path: str | Path | None = None,
    weights_path: str | Path | None = None,
) -> dict[str, Any]:
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"unsupported NCS variant: {variant}")
    repo = detect_ncs_repo(repo_path)
    weights = _detect_variant_weights(variant, weights_path)
    available = bool(repo.get("available")) and bool(weights.get("available"))
    reason = ""
    if not repo.get("available"):
        reason = "repo_missing"
    elif not weights.get("available"):
        reason = "weights_missing"
    return {
        "available": available,
        "repo_found": bool(repo.get("available")),
        "weights_found": bool(weights.get("available")),
        "variant": variant,
        "reason": reason,
        "repo_path": repo.get("path", ""),
        "weights_path": weights.get("path", ""),
        "expected_repo_paths": repo.get("expected_repo_paths", []),
        "expected_weight_paths": weights.get("expected_weight_paths", []),
    }


class DummyNCSModel:
    def __init__(self, *, variant: str, device: str = "cpu") -> None:
        self.variant = variant
        self.device = device

    def encode(self, tokens: Any) -> Any:
        array = np.asarray(tokens, dtype=np.float32)
        if array.ndim == 2:
            return array.mean(axis=0, keepdims=False)
        if array.ndim == 3:
            return array.mean(axis=1)
        return array.reshape(-1).astype(np.float32)


@dataclass
class NCS2DBackboneAdapter:
    model: Any
    config_dir: Path
    feature_mode: str = "mean_patch"
    device: str = "cpu"

    def __post_init__(self) -> None:
        cfg = _read_json(self.config_dir / "config.json")
        self.hidden_size = int(cfg.get("hidden_size", 768))
        self.patch_size = int(cfg.get("patch_size", 16))
        image_size = cfg.get("image_size", 224)
        self.input_size = int(image_size[0] if isinstance(image_size, list) else image_size)
        self.metadata = {
            "model_name": "ncs_2d",
            "load_backend": "transformers",
            "feature_mode": self.feature_mode,
            "hidden_size": self.hidden_size,
            "patch_size": self.patch_size,
            "input_size": self.input_size,
            "is_real_feature": True,
        }

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        feature_mode: str = "mean_patch",
        device: str = "cpu",
    ) -> "NCS2DBackboneAdapter":
        if feature_mode not in {"mean_patch", "cls"}:
            raise ValueError(f"unsupported feature_mode for NCS 2D adapter: {feature_mode}")
        config_dir = _candidate_config_dir("ncs_2d", path)
        if config_dir is None or not (config_dir / "config.json").exists():
            raise RuntimeError(f"Failed to load NCS 2D transformers adapter: config.json not found under {path}")
        try:
            from transformers import AutoModel, ViTModel
        except Exception as exc:
            raise RuntimeError(f"Failed to load NCS 2D transformers adapter: {type(exc).__name__}: {exc}") from exc
        model = None
        last_error = None
        try:
            model = ViTModel.from_pretrained(str(config_dir), add_pooling_layer=False)
        except Exception as exc:
            last_error = exc
        if model is None:
            try:
                model = AutoModel.from_pretrained(str(config_dir))
            except Exception as exc:
                last_error = exc
        if model is None:
            raise RuntimeError(f"Failed to load NCS 2D transformers adapter: {type(last_error).__name__}: {last_error}") from last_error
        model.eval()
        if hasattr(model, "to"):
            model = model.to(device)
        return cls(model=model, config_dir=config_dir, feature_mode=feature_mode, device=device)

    def _prepare_inputs(self, pixel_values: Any) -> Any:
        from fwi_visionfm.torch_backend import require_torch_backend

        torch = require_torch_backend()
        tensor = torch.as_tensor(np.asarray(pixel_values, dtype=np.float32), dtype=torch.float32, device=self.device)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            raise ValueError(f"NCS2DBackboneAdapter expects [B,3,H,W], got {tuple(tensor.shape)}")
        if tensor.shape[1] != 3:
            raise ValueError(f"NCS2DBackboneAdapter expects 3 channels, got {tensor.shape[1]}")
        if tensor.shape[-2] != self.input_size or tensor.shape[-1] != self.input_size:
            tensor = torch.nn.functional.interpolate(
                tensor,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )
        return tensor

    def encode(self, pixel_values: Any) -> np.ndarray:
        from fwi_visionfm.torch_backend import require_torch_backend

        torch = require_torch_backend()
        tensor = self._prepare_inputs(pixel_values)
        with torch.no_grad():
            outputs = self.model(pixel_values=tensor)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None and isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            hidden = outputs[0]
        if hidden is None:
            raise RuntimeError("NCS 2D adapter forward did not return last_hidden_state")
        if self.feature_mode == "cls":
            feature = hidden[:, 0]
        else:
            feature = hidden[:, 1:].mean(dim=1) if hidden.shape[1] > 1 else hidden.mean(dim=1)
        return feature.detach().cpu().numpy().astype(np.float32)

    def encode_tokens(self, pixel_values: Any) -> np.ndarray:
        from fwi_visionfm.torch_backend import require_torch_backend

        torch = require_torch_backend()
        tensor = self._prepare_inputs(pixel_values)
        with torch.no_grad():
            outputs = self.model(pixel_values=tensor)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None and isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            hidden = outputs[0]
        if hidden is None:
            raise RuntimeError("NCS 2D adapter forward did not return last_hidden_state")
        if hidden.ndim != 3:
            raise RuntimeError(f"NCS 2D adapter expected [B,N,D] tokens, got {tuple(hidden.shape)}")
        patch_tokens = hidden[:, 1:] if hidden.shape[1] > 1 else hidden
        return patch_tokens.detach().cpu().numpy().astype(np.float32)


def _repo_src_paths(repo_root: Path) -> list[Path]:
    return [repo_root, repo_root / "src"]


def _attempt_load_real_ncs(variant: str, repo_root: Path, device: str) -> tuple[Any | None, str]:
    inserted: list[str] = []
    try:
        for repo_path in _repo_src_paths(repo_root):
            repo_str = str(repo_path)
            if repo_path.exists() and repo_str not in sys.path:
                sys.path.insert(0, repo_str)
                inserted.append(repo_str)
        for module_name in ("ncs", "ncs_models", "model", "models", "NCS.inference.pipeline"):
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            for attr in ("load_model", "build_model", "create_model", "_load_model"):
                builder = getattr(module, attr, None)
                if callable(builder):
                    try:
                        model = builder(variant=variant, device=device)
                    except TypeError:
                        try:
                            model = builder(variant)
                        except Exception as exc:
                            return None, f"{type(exc).__name__}: {exc}"
                    except Exception as exc:
                        return None, f"{type(exc).__name__}: {exc}"
                    return model, ""
        return None, "No compatible load_model/build_model/create_model API was found"
    finally:
        for repo_str in inserted:
            try:
                sys.path.remove(repo_str)
            except ValueError:
                pass


def _attempt_load_ncs_2p5d_repo_model(repo_root: Path, weights_dir: Path, device: str) -> tuple[Any | None, dict[str, Any]]:
    inserted: list[str] = []
    try:
        for repo_path in _repo_src_paths(repo_root):
            repo_str = str(repo_path)
            if repo_path.exists() and repo_str not in sys.path:
                sys.path.insert(0, repo_str)
                inserted.append(repo_str)
        try:
            importlib.import_module("NCS.models.vit25d")
        except Exception as exc:
            return None, {"pending_reason": f"vit25d registration unavailable: {type(exc).__name__}: {exc}"}
        try:
            from NCS.models.vit25d import ViT25DModel
        except Exception as exc:
            return None, {"pending_reason": f"ViT25DModel import failed: {type(exc).__name__}: {exc}"}
        try:
            model = ViT25DModel.from_pretrained(str(weights_dir))
        except Exception as exc:
            return None, {"pending_reason": f"ViT25DModel.from_pretrained failed: {type(exc).__name__}: {exc}"}
        if hasattr(model, "eval"):
            model.eval()
        if hasattr(model, "to"):
            model = model.to(device)
        return model, {
            "builder_name": "NCS.models.vit25d.ViT25DModel.from_pretrained",
            "backend": "repo_builder",
            "pseudo_2p5d_from_shot_gather": True,
        }
    finally:
        for repo_str in inserted:
            try:
                sys.path.remove(repo_str)
            except ValueError:
                pass


def inspect_ncs_2p5d_adapter(
    *,
    repo_root: str | Path,
    weights_path: str | Path,
    status_report_path: str | Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    repo_dir = _resolve_existing_path(repo_root, expect_file=False)
    weights_dir = _candidate_config_dir("ncs_2p5d", weights_path)
    payload: dict[str, Any] = {
        "variant": "ncs_2p5d",
        "repo_found": repo_dir is not None,
        "weights_found": weights_dir is not None and (weights_dir / "config.json").exists(),
        "is_real_feature": False,
        "status": "WEIGHTS_PRESENT_ADAPTER_PENDING",
        "pending_reason": "",
        "weights_path": str(weights_dir) if weights_dir is not None else "",
        "repo_path": str(repo_dir) if repo_dir is not None else "",
        "model": None,
    }
    if repo_dir is None:
        payload["status"] = "SKIPPED_NCS_UNAVAILABLE"
        payload["pending_reason"] = "repo missing"
    elif weights_dir is None or not (weights_dir / "config.json").exists():
        payload["status"] = "SKIPPED_NCS_UNAVAILABLE"
        payload["pending_reason"] = "weights missing or config.json absent"
    else:
        model, meta = _attempt_load_ncs_2p5d_repo_model(repo_dir, weights_dir, device)
        if model is None:
            payload["pending_reason"] = str(meta.get("pending_reason", "adapter pending"))
        else:
            payload["status"] = "AVAILABLE"
            payload["is_real_feature"] = True
            payload["model"] = model
            payload.update(meta)
    if status_report_path is not None:
        report_path = Path(status_report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {key: value for key, value in payload.items() if key != "model"}
        report_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def load_ncs_model(
    variant: str = "ncs_2d",
    *,
    repo_path: str | Path | None = None,
    weights_path: str | Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"unsupported NCS variant: {variant}")
    status = get_ncs_status(variant, repo_path=repo_path, weights_path=weights_path)
    if not status["available"]:
        return {
            "variant": variant,
            "status": "SKIPPED_NCS_UNAVAILABLE",
            "repo": detect_ncs_repo(repo_path),
            "weights": _detect_variant_weights(variant, weights_path),
            "model": None,
            "ncs_status": status,
        }
    if variant == "ncs_2d":
        weights_dir = _candidate_config_dir("ncs_2d", weights_path or status["weights_path"])
        try:
            adapter = NCS2DBackboneAdapter.from_pretrained(weights_dir or status["weights_path"], feature_mode="mean_patch", device=device)
        except Exception as exc:
            return {
                "variant": variant,
                "status": "CHECKPOINT_LOAD_ERROR",
                "repo": detect_ncs_repo(repo_path),
                "weights": detect_ncs_weights(weights_path),
                "model": None,
                "ncs_status": {**status, "reason": f"{type(exc).__name__}: {exc}"},
            }
        return {
            "variant": variant,
            "status": "READY",
            "repo": detect_ncs_repo(repo_path),
            "weights": _detect_variant_weights(variant, weights_path),
            "model": adapter,
            "ncs_status": {**status, "reason": "", "adapter_backend": "transformers"},
            "device": device,
            "metadata": dict(adapter.metadata),
        }
    if variant == "ncs_2p5d":
        repo_dir = _resolve_existing_path(repo_path or status["repo_path"], expect_file=False)
        weights_dir = _candidate_config_dir("ncs_2p5d", weights_path or status["weights_path"])
        inspect = inspect_ncs_2p5d_adapter(repo_root=repo_dir or "", weights_path=weights_dir or "", device=device)
        return {
            "variant": variant,
            "status": "READY" if inspect["status"] == "AVAILABLE" else "WEIGHTS_PRESENT_ADAPTER_PENDING",
            "repo": detect_ncs_repo(repo_path),
            "weights": _detect_variant_weights(variant, weights_path),
            "model": inspect.get("model"),
            "ncs_status": {**status, "reason": inspect.get("pending_reason", ""), "adapter_backend": inspect.get("backend", "repo_builder_pending")},
            "device": device,
            "metadata": {key: value for key, value in inspect.items() if key not in {"model"}},
        }
    repo_root = Path(str(status["repo_path"]))
    model, reason = _attempt_load_real_ncs(variant, repo_root, device)
    if model is None:
        return {
            "variant": variant,
            "status": "SKIPPED_NCS_API_MISMATCH",
            "repo": detect_ncs_repo(repo_path),
            "weights": _detect_variant_weights(variant, weights_path),
            "model": None,
            "ncs_status": {**status, "reason": reason or "api_mismatch"},
        }
    return {
        "variant": variant,
        "status": "READY",
        "repo": detect_ncs_repo(repo_path),
            "weights": _detect_variant_weights(variant, weights_path),
        "model": model,
        "ncs_status": status,
        "device": device,
    }
