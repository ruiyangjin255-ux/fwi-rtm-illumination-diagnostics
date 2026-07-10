from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from admit_fwi.models.external_benchmark_loader import prepare_external_crop


DEFAULT_PATHS = {
    "marmousi": Path(r"D:\data\marmousi\marmousi.npy"),
    "sigsbee2a": Path(r"D:\data\sigsbee\sigsbee2a.npy"),
    "bp2004": Path(r"D:\data\bp2004\bp2004.npy"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local external benchmark crops without downloading data.")
    parser.add_argument("--config", type=Path, default=Path("configs/admit_external_models.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("data/admit_models"))
    args = parser.parse_args()
    for name, path in DEFAULT_PATHS.items():
        manifest = prepare_external_crop(name=name, source_path=path, output_root=args.output_root, crop=None, downsample=1)
        print(f"{name}: {manifest['status']} ({path})")


if __name__ == "__main__":
    main()
