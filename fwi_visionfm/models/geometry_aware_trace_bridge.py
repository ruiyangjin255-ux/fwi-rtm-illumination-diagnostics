from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fwi_visionfm.torch_backend import require_torch_backend


def _expand_mask(mask: Any, target_ndim: int) -> Any:
    torch = require_torch_backend()
    value = mask
    while value.ndim < target_ndim:
        value = value.unsqueeze(-1)
    return value.to(dtype=torch.float32)


class TraceNeighborhoodEncoder:
    def __init__(
        self,
        *,
        time_kernel_size: int = 5,
        hidden_dim: int = 32,
        output_dim: int = 64,
        padding_mode: str = "replicate",
        trace_context_radius: int = 2,
    ) -> None:
        torch = require_torch_backend()
        nn = torch.nn
        if padding_mode not in {"replicate", "zero"}:
            raise ValueError(f"unsupported padding_mode: {padding_mode}")
        self.trace_context_radius = int(trace_context_radius)
        self.padding_mode = str(padding_mode)
        kernel = max(1, int(time_kernel_size))
        self.module = nn.Sequential(
            nn.Conv2d(1, int(hidden_dim), kernel_size=(2 * self.trace_context_radius + 1, kernel), padding=(0, kernel // 2)),
            nn.GELU(),
            nn.Conv2d(int(hidden_dim), int(output_dim), kernel_size=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def _pad(self, neighborhood: Any) -> Any:
        torch = require_torch_backend()
        if self.padding_mode == "zero":
            return neighborhood
        if neighborhood.shape[-2] == 0:
            return neighborhood
        return torch.nn.functional.pad(neighborhood, (0, 0, 0, 0, 0, 0, 0, 0))

    def build_neighborhood(self, records: Any) -> Any:
        torch = require_torch_backend()
        if records.ndim != 4:
            raise ValueError(f"expected records [B,S,R,T], got {tuple(records.shape)}")
        batch, shots, receivers, samples = records.shape
        radius = self.trace_context_radius
        if self.padding_mode == "replicate":
            padded = torch.nn.functional.pad(records, (0, 0, radius, radius), mode="replicate")
        else:
            padded = torch.nn.functional.pad(records, (0, 0, radius, radius), mode="constant", value=0.0)
        windows = []
        for index in range(receivers):
            windows.append(padded[:, :, index : index + 2 * radius + 1, :].unsqueeze(3))
        return torch.cat(windows, dim=3).reshape(batch, shots, receivers, 2 * radius + 1, samples)

    def __call__(self, records: Any) -> Any:
        torch = require_torch_backend()
        neighborhood = self.build_neighborhood(records)
        batch, shots, receivers, window, samples = neighborhood.shape
        encoded = self.module(neighborhood.reshape(batch * shots * receivers, 1, window, samples))
        return encoded.reshape(batch, shots, receivers, -1)


class ShotGlobalContextEncoder:
    def __init__(self, *, input_dim: int, output_dim: int | None = None) -> None:
        torch = require_torch_backend()
        nn = torch.nn
        output = int(output_dim or input_dim)
        self.module = nn.Sequential(
            nn.Linear(int(input_dim) * 2, output),
            nn.LayerNorm(output),
            nn.GELU(),
        )

    def __call__(self, trace_context: Any, *, valid_trace_mask: Any | None = None, valid_shot_mask: Any | None = None) -> Any:
        torch = require_torch_backend()
        if trace_context.ndim != 4:
            raise ValueError(f"expected trace_context [B,S,R,C], got {tuple(trace_context.shape)}")
        batch, shots, receivers, channels = trace_context.shape
        if valid_trace_mask is None:
            valid_trace_mask = torch.ones((batch, shots, receivers), dtype=torch.float32, device=trace_context.device)
        if valid_shot_mask is None:
            valid_shot_mask = torch.ones((batch, shots), dtype=torch.float32, device=trace_context.device)
        trace_mask = _expand_mask(valid_trace_mask.to(trace_context.device), trace_context.ndim)
        masked = trace_context * trace_mask
        denom = trace_mask.sum(dim=2).clamp_min(1.0)
        mean_context = masked.sum(dim=2) / denom
        max_fill = torch.full_like(trace_context, -1.0e9)
        max_context = torch.where(trace_mask > 0, trace_context, max_fill).max(dim=2).values
        max_context = torch.where((trace_mask.sum(dim=2) > 0), max_context, torch.zeros_like(max_context))
        merged = torch.cat([mean_context, max_context], dim=-1)
        shot_context = self.module(merged)
        shot_mask = _expand_mask(valid_shot_mask.to(trace_context.device), shot_context.ndim)
        return shot_context * shot_mask


class GeometryConditionEncoder:
    def __init__(self, *, condition_dim: int, token_dim: int, hidden_dim: int = 128) -> None:
        torch = require_torch_backend()
        nn = torch.nn
        self.condition_dim = int(condition_dim)
        self.token_dim = int(token_dim)
        self.module = nn.Sequential(
            nn.Linear(self.condition_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.token_dim),
        )

    def __call__(self, condition: Any) -> Any:
        if condition.shape[-1] != self.condition_dim:
            raise ValueError(f"expected trailing condition dim {self.condition_dim}, got {condition.shape[-1]}")
        return self.module(condition)


class PhysicsAwareTokenFusion:
    def __init__(self, *, token_dim: int, condition_dim: int) -> None:
        torch = require_torch_backend()
        nn = torch.nn
        self.layer_norm = nn.LayerNorm(int(token_dim))
        self.gate = nn.Linear(int(condition_dim), int(token_dim))
        self.gamma = nn.Linear(int(condition_dim), int(token_dim))
        self.beta = nn.Linear(int(condition_dim), int(token_dim))

    def __call__(self, patch_tokens: Any, condition_embedding: Any) -> Any:
        torch = require_torch_backend()
        if patch_tokens.shape != condition_embedding.shape:
            raise ValueError(f"token/condition shape mismatch: {tuple(patch_tokens.shape)} vs {tuple(condition_embedding.shape)}")
        normalized = self.layer_norm(patch_tokens)
        gate = torch.sigmoid(self.gate(condition_embedding))
        gamma = self.gamma(condition_embedding)
        beta = self.beta(condition_embedding)
        return patch_tokens + gate * (gamma * normalized + beta)


@dataclass
class GeometryBridgeMetadata:
    geometry_provenance: str
    trace_context_radius: int
    use_shot_global_context: bool
    use_multiscale_context: bool


class GeometryAwareTraceBridge:
    def __init__(
        self,
        *,
        token_dim: int,
        trace_context_radius: int = 2,
        geometry_condition_dim: int = 10,
        trace_context_dim: int = 64,
        shot_context_dim: int = 64,
        use_trace_context: bool = True,
        use_shot_global_context: bool = True,
        use_multiscale_context: bool = False,
        padding_mode: str = "replicate",
        geometry_provenance: str = "CANONICAL_RECONSTRUCTED",
    ) -> None:
        self.token_dim = int(token_dim)
        self.trace_context_radius = int(trace_context_radius)
        self.use_trace_context = bool(use_trace_context)
        self.use_shot_global_context = bool(use_shot_global_context)
        self.use_multiscale_context = bool(use_multiscale_context)
        self.geometry_provenance = str(geometry_provenance)
        self.trace_encoder = TraceNeighborhoodEncoder(
            trace_context_radius=self.trace_context_radius,
            output_dim=int(trace_context_dim),
            padding_mode=padding_mode,
        )
        self.shot_encoder = ShotGlobalContextEncoder(input_dim=int(trace_context_dim), output_dim=int(shot_context_dim))
        extra_dim = 0
        if self.use_trace_context:
            extra_dim += int(trace_context_dim)
        if self.use_shot_global_context:
            extra_dim += int(shot_context_dim)
        if self.use_multiscale_context:
            extra_dim += 3
        self.condition_encoder = GeometryConditionEncoder(
            condition_dim=int(geometry_condition_dim) + extra_dim,
            token_dim=self.token_dim,
        )
        self.fusion = PhysicsAwareTokenFusion(token_dim=self.token_dim, condition_dim=self.token_dim)

    def build_trace_condition(
        self,
        *,
        g_trace: Any,
        g_time: Any,
        trace_context: Any | None = None,
        shot_context: Any | None = None,
        multiscale_feature: Any | None = None,
    ) -> Any:
        parts = [g_trace, g_time]
        if self.use_trace_context:
            if trace_context is None:
                raise ValueError("trace_context is required when use_trace_context=True")
            parts.append(trace_context)
        if self.use_shot_global_context:
            if shot_context is None:
                raise ValueError("shot_context is required when use_shot_global_context=True")
            parts.append(shot_context)
        if self.use_multiscale_context:
            if multiscale_feature is None:
                raise ValueError("multiscale_feature is required when use_multiscale_context=True")
            parts.append(multiscale_feature)
        return require_torch_backend().cat(parts, dim=-1)

    def forward(
        self,
        *,
        patch_tokens: Any,
        records: Any,
        g_trace: Any,
        g_time: Any,
        valid_trace_mask: Any | None = None,
        valid_shot_mask: Any | None = None,
        multiscale_feature: Any | None = None,
    ) -> dict[str, Any]:
        torch = require_torch_backend()
        if patch_tokens.ndim != 5:
            raise ValueError(f"expected patch_tokens [B,S,R,N,D], got {tuple(patch_tokens.shape)}")
        batch, shots, receivers, patches, token_dim = patch_tokens.shape
        if token_dim != self.token_dim:
            raise ValueError(f"token_dim mismatch: {token_dim} vs expected {self.token_dim}")
        trace_context = self.trace_encoder(records)
        shot_context = self.shot_encoder(trace_context, valid_trace_mask=valid_trace_mask, valid_shot_mask=valid_shot_mask)
        trace_context_expanded = trace_context.unsqueeze(3).expand(batch, shots, receivers, patches, trace_context.shape[-1])
        shot_context_expanded = shot_context.unsqueeze(2).unsqueeze(3).expand(batch, shots, receivers, patches, shot_context.shape[-1])
        condition = self.build_trace_condition(
            g_trace=g_trace,
            g_time=g_time,
            trace_context=trace_context_expanded,
            shot_context=shot_context_expanded,
            multiscale_feature=multiscale_feature,
        )
        condition_embedding = self.condition_encoder(condition)
        fused = self.fusion(patch_tokens, condition_embedding)
        metadata = GeometryBridgeMetadata(
            geometry_provenance=self.geometry_provenance,
            trace_context_radius=self.trace_context_radius,
            use_shot_global_context=self.use_shot_global_context,
            use_multiscale_context=self.use_multiscale_context,
        )
        return {
            "tokens": fused,
            "condition_embedding": condition_embedding,
            "trace_context": trace_context,
            "shot_context": shot_context,
            "metadata": metadata.__dict__,
        }
