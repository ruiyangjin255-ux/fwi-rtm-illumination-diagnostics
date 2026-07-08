from __future__ import annotations

from typing import Any

from fwi_visionfm.torch_backend import require_torch_backend


def _flatten_scores(x: Any) -> Any:
    if x.ndim == 3:
        return x
    if x.ndim == 5:
        return x.flatten(start_dim=2)
    raise ValueError(f"shot_features must be [B,S,D] or [B,S,C,H,W], got {tuple(x.shape)}")


def _aggregate(x: Any, weights: Any) -> Any:
    while weights.ndim < x.ndim:
        weights = weights.unsqueeze(-1)
    return (x * weights).sum(dim=1)


class MeanShotAggregator:
    def forward(self, shot_features: Any, source_geometry: Any | None = None) -> dict[str, Any]:
        del source_geometry
        torch = require_torch_backend()
        x = shot_features.float()
        shots = x.shape[1]
        weights = torch.full((x.shape[0], shots), 1.0 / float(shots), dtype=x.dtype, device=x.device)
        return {"aggregated_feature": x.mean(dim=1), "attention_weights": weights}

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.forward(*args, **kwargs)


class AttentionShotAggregator:
    def __init__(self) -> None:
        torch = require_torch_backend()
        nn = torch.nn
        self.scorer = nn.LazyLinear(1)

    def score(self, pooled: Any, source_geometry: Any | None = None) -> Any:
        del source_geometry
        return self.scorer(pooled).squeeze(-1)

    def forward(self, shot_features: Any, source_geometry: Any | None = None) -> dict[str, Any]:
        torch = require_torch_backend()
        x = shot_features.float()
        pooled = _flatten_scores(x)
        scores = self.score(pooled, source_geometry=source_geometry)
        weights = torch.softmax(scores, dim=1)
        return {"aggregated_feature": _aggregate(x, weights), "attention_weights": weights}

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.forward(*args, **kwargs)


class SourceAwareAttentionAggregator(AttentionShotAggregator):
    def __init__(self) -> None:
        super().__init__()
        torch = require_torch_backend()
        nn = torch.nn
        self.geometry_scorer = nn.LazyLinear(1)

    def score(self, pooled: Any, source_geometry: Any | None = None) -> Any:
        base = self.scorer(pooled).squeeze(-1)
        if source_geometry is None:
            return base
        geometry_score = self.geometry_scorer(source_geometry.float()).squeeze(-1)
        return base + geometry_score


def build_aggregator(name: str) -> Any:
    aggregator_name = str(name)
    if aggregator_name == "mean":
        return MeanShotAggregator()
    if aggregator_name == "attention":
        return AttentionShotAggregator()
    if aggregator_name == "source_aware_attention":
        return SourceAwareAttentionAggregator()
    raise ValueError(f"unsupported aggregator: {aggregator_name}")
