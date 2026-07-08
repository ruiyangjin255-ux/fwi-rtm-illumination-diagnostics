from __future__ import annotations

import math
from typing import Any

from fwi_visionfm.torch_backend import require_torch_backend


def _ensure_batch_vector(x: Any | None, *, batch_size: int, length: int, device: Any) -> Any:
    torch = require_torch_backend()
    if x is None:
        base = torch.linspace(0.0, 1.0, length, device=device, dtype=torch.float32)
        return base.view(1, length).repeat(batch_size, 1)
    if x.ndim != 2:
        raise ValueError(f"expected [B, L] tensor, got {tuple(x.shape)}")
    x = x.to(device=device, dtype=torch.float32)
    if x.shape[0] == 1 and batch_size > 1:
        x = x.repeat(batch_size, 1)
    if x.shape[1] != length:
        x = torch.nn.functional.interpolate(x.unsqueeze(1), size=length, mode="linear", align_corners=False).squeeze(1)
    return x


def _ensure_batch_matrix(x: Any | None, *, batch_size: int, shots: int, length: int, device: Any) -> Any:
    torch = require_torch_backend()
    if x is None:
        base = torch.linspace(0.0, 1.0, length, device=device, dtype=torch.float32)
        return base.view(1, 1, length).repeat(batch_size, shots, 1)
    if x.ndim != 3:
        raise ValueError(f"expected [B, S, L] tensor, got {tuple(x.shape)}")
    x = x.to(device=device, dtype=torch.float32)
    if x.shape[0] == 1 and batch_size > 1:
        x = x.repeat(batch_size, 1, 1)
    if x.shape[1] == 1 and shots > 1:
        x = x.repeat(1, shots, 1)
    if x.shape[2] != length:
        x = torch.nn.functional.interpolate(x.reshape(batch_size * x.shape[1], 1, x.shape[2]), size=length, mode="linear", align_corners=False)
        x = x.reshape(batch_size, -1, length)
    return x


class GeometryEmbedding:
    def __init__(
        self,
        *,
        embed_dim: int = 8,
        mode: str = "sinusoidal",
        use_source: bool = True,
        use_receiver: bool = True,
        use_time: bool = True,
        use_offset: bool = True,
    ) -> None:
        torch = require_torch_backend()
        nn = torch.nn
        self.embed_dim = int(embed_dim)
        self.mode = str(mode)
        self.use_source = bool(use_source)
        self.use_receiver = bool(use_receiver)
        self.use_time = bool(use_time)
        self.use_offset = bool(use_offset)
        if self.mode not in {"sinusoidal", "learnable"}:
            raise ValueError(f"unsupported geometry embedding mode: {self.mode}")
        if self.mode == "learnable":
            self.source_proj = nn.Sequential(nn.Linear(1, self.embed_dim), nn.GELU(), nn.Linear(self.embed_dim, self.embed_dim))
            self.receiver_proj = nn.Sequential(nn.Linear(1, self.embed_dim), nn.GELU(), nn.Linear(self.embed_dim, self.embed_dim))
            self.time_proj = nn.Sequential(nn.Linear(1, self.embed_dim), nn.GELU(), nn.Linear(self.embed_dim, self.embed_dim))
            self.offset_proj = nn.Sequential(nn.Linear(1, self.embed_dim), nn.GELU(), nn.Linear(self.embed_dim, self.embed_dim))
        else:
            self.source_proj = None
            self.receiver_proj = None
            self.time_proj = None
            self.offset_proj = None

    @property
    def output_channels(self) -> int:
        count = sum([self.use_source, self.use_receiver, self.use_time, self.use_offset])
        return int(count) * self.embed_dim

    def _sinusoidal_embed(self, values: Any) -> Any:
        torch = require_torch_backend()
        half_dim = max(1, self.embed_dim // 2)
        scale = torch.exp(torch.arange(half_dim, device=values.device, dtype=values.dtype) * (-math.log(10000.0) / max(1, half_dim - 1)))
        angles = values.unsqueeze(-1) * scale
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if emb.shape[-1] < self.embed_dim:
            emb = torch.nn.functional.pad(emb, (0, self.embed_dim - emb.shape[-1]))
        return emb[..., : self.embed_dim]

    def _embed(self, values: Any, kind: str) -> Any:
        if self.mode == "sinusoidal":
            return self._sinusoidal_embed(values)
        projector = getattr(self, f"{kind}_proj")
        return projector(values.unsqueeze(-1))

    def forward(
        self,
        *,
        source_index: Any | None = None,
        receiver_index: Any | None = None,
        time_index: Any | None = None,
        offset: Any | None = None,
        frequency_band_index: Any | None = None,
        target_hw: tuple[int, int] | None = None,
        batch_size: int | None = None,
        shots: int | None = None,
    ) -> Any:
        torch = require_torch_backend()
        del frequency_band_index
        if batch_size is None:
            for item in (source_index, receiver_index, time_index, offset):
                if item is not None:
                    batch_size = int(item.shape[0])
                    break
        if batch_size is None:
            raise ValueError("batch_size could not be inferred from geometry inputs")
        if shots is None:
            for item in (source_index, receiver_index, offset):
                if item is not None and item.ndim >= 2:
                    shots = int(item.shape[1])
                    break
        shots = int(shots or 1)
        device = None
        for item in (source_index, receiver_index, time_index, offset):
            if item is not None:
                device = item.device
                break
        device = device or torch.device("cpu")

        target_h = int(target_hw[0]) if target_hw is not None else int(time_index.shape[-1] if time_index is not None else 64)
        target_w = int(target_hw[1]) if target_hw is not None else int(receiver_index.shape[-1] if receiver_index is not None else 64)
        time_tensor = _ensure_batch_vector(time_index, batch_size=batch_size, length=target_h, device=device)
        receiver_tensor = _ensure_batch_matrix(receiver_index, batch_size=batch_size, shots=shots, length=target_w, device=device)
        source_tensor = _ensure_batch_vector(source_index, batch_size=batch_size, length=shots, device=device)
        offset_tensor = _ensure_batch_matrix(offset, batch_size=batch_size, shots=shots, length=target_w, device=device)
        fallback_geometry = source_index is None and receiver_index is None and offset is None

        features = []
        if self.use_source and (source_index is not None or fallback_geometry):
            source_feature = self._embed(source_tensor, "source").mean(dim=1)
            features.append(source_feature.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, target_h, target_w))
        if self.use_receiver and (receiver_index is not None or fallback_geometry):
            receiver_feature = self._embed(receiver_tensor, "receiver").mean(dim=1).permute(0, 2, 1)
            features.append(receiver_feature.unsqueeze(2).expand(-1, -1, target_h, -1))
        if self.use_time:
            time_feature = self._embed(time_tensor, "time").permute(0, 2, 1)
            features.append(time_feature.unsqueeze(-1).expand(-1, -1, -1, target_w))
        if self.use_offset and (offset is not None or fallback_geometry):
            offset_feature = self._embed(offset_tensor, "offset").mean(dim=1).permute(0, 2, 1)
            features.append(offset_feature.unsqueeze(2).expand(-1, -1, target_h, -1))
        output = torch.cat(features, dim=1)
        if target_hw is not None and tuple(output.shape[-2:]) != tuple(target_hw):
            output = torch.nn.functional.interpolate(output, size=target_hw, mode="bilinear", align_corners=False)
        return output

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)
