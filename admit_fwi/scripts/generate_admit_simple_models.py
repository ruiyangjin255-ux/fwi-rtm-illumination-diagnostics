from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from admit_fwi.models.synthetic_model_bank import save_synthetic_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic diagnostic models for ADMIT-FWI.")
    parser.add_argument("--models", nargs="+", default=["simple_layered", "simple_fault"])
    parser.add_argument("--output-root", type=Path, default=Path("data/admit_models"))
    args = parser.parse_args()
    for name in args.models:
        manifest = save_synthetic_model(name, args.output_root)
        print(f"{name}: {manifest['status']} -> {args.output_root / name}")


if __name__ == "__main__":
    main()
