from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fwi_visionfm.bridges.envelope_bridge import envelope_bridge
from fwi_visionfm.bridges.raw_bridge import raw_bridge
from fwi_visionfm.bridges.raw_offset_bridge import raw_offset_bridge
from fwi_visionfm.bridges.spectrogram_bridge import spectrogram_bridge


BRIDGES = {
    "raw_bridge": raw_bridge,
    "raw_offset_bridge": raw_offset_bridge,
    "envelope_bridge": envelope_bridge,
    "spectrogram_bridge": spectrogram_bridge,
}


def create_bridge(name: str):
    if name not in BRIDGES:
        raise ValueError(f"unsupported bridge: {name}")
    return BRIDGES[name]


def bridge_smoke() -> dict:
    records = np.ones((1, 2, 6, 8), dtype=np.float32)
    source_positions = np.array([[0.2, 0.8]], dtype=np.float32)
    rows = []
    for name, bridge in BRIDGES.items():
        try:
            output = bridge(records, source_positions)
            rows.append(
                {
                    "bridge_name": name,
                    "input_shape": list(records.shape),
                    "output_shape": list(output.shape),
                    "status": "ok",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "bridge_name": name,
                    "input_shape": list(records.shape),
                    "output_shape": [],
                    "status": str(exc),
                }
            )
    return {"rows": rows, "count": len(rows)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bridge registry smoke command.")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.smoke:
        raise SystemExit("错误: 当前 registry 只支持 --smoke。")
    payload = bridge_smoke()
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
