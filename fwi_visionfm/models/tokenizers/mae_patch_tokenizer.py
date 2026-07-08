from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fwi_visionfm.data.bridge_registry import build_bridge
from fwi_visionfm.datasets import load_npz_sample


def sample_to_bridge_image(sample_path: str | Path, bridge: str, *, output_size: int = 64) -> dict[str, Any]:
    sample = load_npz_sample(sample_path)
    bridge_obj = build_bridge(bridge if bridge != "shot_2p5d_view" else "raw_envelope_spectrum3", {"output_size": [int(output_size), int(output_size)]})
    image = bridge_obj.forward(sample.records)["image"]
    return {
        "image": np.asarray(image, dtype=np.float32),
        "metadata": {
            "bridge": bridge,
            "resolved_bridge": bridge_obj.name,
            "input_shape": list(sample.records.shape),
            "output_shape": list(np.asarray(image).shape),
            "metric_space": "physical_velocity",
        },
        "velocity": np.asarray(sample.velocity, dtype=np.float32),
        "source_positions": np.asarray(sample.source_positions, dtype=np.float32),
    }


def batch_bridge_images(sample_paths: list[str | Path], bridge: str, *, output_size: int = 64) -> dict[str, Any]:
    images = []
    velocities = []
    sample_ids = []
    metadata = None
    for path in sample_paths:
        payload = sample_to_bridge_image(path, bridge, output_size=output_size)
        images.append(payload["image"])
        velocities.append(payload["velocity"])
        sample_ids.append(f"{Path(path).name}:0")
        metadata = payload["metadata"]
    return {
        "images": np.stack(images, axis=0).astype(np.float32),
        "velocity": np.stack(velocities, axis=0).astype(np.float32),
        "sample_ids": sample_ids,
        "metadata": metadata or {},
    }
