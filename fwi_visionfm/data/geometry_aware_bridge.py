from __future__ import annotations

from typing import Any

import numpy as np

from fwi_visionfm.data.bridge_registry import build_bridge
from fwi_visionfm.models.geometry_embedding import GeometryEmbedding
from fwi_visionfm.torch_backend import require_torch_backend


class GeometryAwareBridgeWrapper:
    def __init__(self, bridge_name: str, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.bridge = build_bridge(bridge_name, self.config)
        geometry_cfg = dict(self.config.get("geometry", {}))
        self.geometry_config = {
            "enabled": bool(geometry_cfg.get("enabled", False)),
            "mode": geometry_cfg.get("mode", "sinusoidal"),
            "fusion": geometry_cfg.get("fusion", "concat"),
            "use_source": bool(geometry_cfg.get("use_source", True)),
            "use_receiver": bool(geometry_cfg.get("use_receiver", True)),
            "use_time": bool(geometry_cfg.get("use_time", True)),
            "use_offset": bool(geometry_cfg.get("use_offset", True)),
            "projection_to_3ch": bool(geometry_cfg.get("projection_to_3ch", False)),
            "embed_dim": int(geometry_cfg.get("embed_dim", 8)),
        }
        self.geometry = GeometryEmbedding(
            embed_dim=self.geometry_config["embed_dim"],
            mode=self.geometry_config["mode"],
            use_source=self.geometry_config["use_source"],
            use_receiver=self.geometry_config["use_receiver"],
            use_time=self.geometry_config["use_time"],
            use_offset=self.geometry_config["use_offset"],
        )
        self._projection_cache: dict[int, Any] = {}

    def _as_torch(self, image: Any) -> tuple[Any, bool]:
        torch = require_torch_backend()
        if isinstance(image, np.ndarray):
            return torch.as_tensor(image, dtype=torch.float32), True
        return image.float(), False

    def _project_to_3ch(self, image: Any) -> Any:
        torch = require_torch_backend()
        channels = int(image.shape[1])
        if channels == 3:
            return image
        projector = self._projection_cache.get(channels)
        if projector is None:
            nn = torch.nn
            projector = nn.Conv2d(channels, 3, kernel_size=1, bias=False)
            self._projection_cache[channels] = projector
        return projector.to(image.device)(image)

    def forward(self, records: Any, geometry: dict[str, Any] | None = None) -> dict[str, Any]:
        torch = require_torch_backend()
        base = self.bridge.forward(records)
        if not self.geometry_config["enabled"]:
            return base
        image, was_numpy = self._as_torch(base["image"])
        if image.ndim == 3:
            image = image.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        batch, _, height, width = image.shape
        record_tensor = torch.as_tensor(records, dtype=torch.float32) if isinstance(records, np.ndarray) else records
        shots = int(record_tensor.shape[1] if record_tensor.ndim == 4 else record_tensor.shape[0])
        geometry = geometry or {}
        geometry_feature = self.geometry(
            source_index=geometry.get("source_index"),
            receiver_index=geometry.get("receiver_index"),
            time_index=geometry.get("time_index"),
            offset=geometry.get("offset"),
            frequency_band_index=geometry.get("frequency_band_index"),
            target_hw=(height, width),
            batch_size=batch,
            shots=shots,
        )
        fusion = self.geometry_config["fusion"]
        if fusion == "add":
            if geometry_feature.shape[1] != image.shape[1]:
                if geometry_feature.shape[1] % image.shape[1] == 0:
                    reducer = geometry_feature.view(batch, image.shape[1], -1, height, width).mean(dim=2)
                else:
                    reducer = geometry_feature[:, : image.shape[1]]
            else:
                reducer = geometry_feature
            fused = image + reducer
        elif fusion == "concat":
            fused = torch.cat([image, geometry_feature], dim=1)
        else:
            raise ValueError(f"unsupported geometry fusion mode: {fusion}")
        if self.geometry_config["projection_to_3ch"]:
            fused = self._project_to_3ch(fused)
        metadata = dict(base["metadata"])
        metadata["geometry_config"] = dict(self.geometry_config)
        metadata["geometry_shape"] = list(geometry_feature.shape)
        out_image = fused[0] if squeeze else fused
        if was_numpy:
            out_image = out_image.detach().cpu().numpy().astype(np.float32)
        return {"image": out_image, "bridge_name": base["bridge_name"], "metadata": metadata}
