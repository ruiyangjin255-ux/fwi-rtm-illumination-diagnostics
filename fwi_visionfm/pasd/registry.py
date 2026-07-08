"""Reproducible B1--B4 PASD-FWI ablation registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PASDVariant:
    """One scientifically interpretable ablation variant."""

    name: str
    aggregator: Literal["mean", "geometry_attention"]
    bridge_mode: Literal["raw", "hybrid"]
    decoder_mode: Literal["plain", "decoupled"]
    criterion: Literal["l1", "background_edge"]
    description: str


VARIANTS = {
    "B1_raw_unet": PASDVariant(
        "B1_raw_unet", "mean", "raw", "plain", "l1", "Raw gather, mean pooling, ordinary velocity decoder and L1 loss."
    ),
    "B2_hybrid_unet": PASDVariant(
        "B2_hybrid_unet", "mean", "hybrid", "plain", "l1", "Hybrid attributes with the same decoder/loss as B1."
    ),
    "B3_raw_bed": PASDVariant(
        "B3_raw_bed", "mean", "raw", "decoupled", "background_edge", "Raw input with background-edge decoder and edge-aware loss."
    ),
    "B4_pasd_fwi": PASDVariant(
        "B4_pasd_fwi", "geometry_attention", "hybrid", "decoupled", "background_edge", "Full PASD-FWI: hybrid bridge, geometry attention, decoupled decoder and loss."
    ),
    "B4_no_geometry_attention": PASDVariant(
        "B4_no_geometry_attention", "mean", "hybrid", "decoupled", "background_edge", "PASD without geometry-aware attention: hybrid bridge and background-edge decoder/loss with mean pooling."
    ),
}


def get_variant(name: str) -> PASDVariant:
    try:
        return VARIANTS[name]
    except KeyError as exc:
        choices = ", ".join(VARIANTS)
        raise KeyError(f"Unknown PASD variant '{name}'. Choose one of: {choices}") from exc
