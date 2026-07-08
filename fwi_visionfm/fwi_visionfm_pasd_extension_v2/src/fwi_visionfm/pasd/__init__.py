"""PASD-FWI extension: physics-aligned input representation and structure-aware FWI decoding."""

from .bridge import BridgeOutput, HybridAttributeBridge
from .losses import BackgroundEdgeLoss, VelocityL1Loss
from .model import PASDFWI, PASDOutput
from .registry import PASDVariant, get_variant

__all__ = [
    "BackgroundEdgeLoss",
    "BridgeOutput",
    "HybridAttributeBridge",
    "PASDFWI",
    "PASDOutput",
    "PASDVariant",
    "VelocityL1Loss",
    "get_variant",
]
