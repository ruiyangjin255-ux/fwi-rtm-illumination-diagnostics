from __future__ import annotations

import argparse
from pathlib import Path


def write_continued_mae_plan(output_dir: str | Path) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "continued_mae_plan.md"
    lines = [
        "# Protocol V8 continued MAE plan",
        "",
        "## Goal",
        "Compare natural-image initialization and seismic continued MAE pretraining under matched decoder-only conditions.",
        "",
        "## Matched Settings",
        "A. random ViT/MAE without seismic pretraining",
        "B. random ViT/MAE with seismic MAE pretraining",
        "C. natural-image initialized ViT/DINOv2/MAE without seismic continued pretraining",
        "D. natural-image initialized ViT/DINOv2/MAE with seismic continued MAE pretraining",
        "",
        "## Interpretation Rules",
        "- D > C: seismic continued pretraining is valuable beyond natural-image initialization.",
        "- D ≈ B: seismic-domain pretraining matters more than natural-image initialization.",
        "- B > D: natural-image prior may be harmful in this FWI setting.",
        "- Structure metrics still need boundary auxiliary or geometry-aware bridge support.",
        "",
        "## Constraints",
        "- CPU-limited",
        "- decoder-only training",
        "- frozen feature cache",
        "- not benchmark-level proof",
        "",
        "This continued MAE plan is a controlled follow-up design, not an existing benchmark result.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Protocol V8 continued MAE comparison plan.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    print(f"Wrote {write_continued_mae_plan(parse_args().output_dir)}")


if __name__ == "__main__":
    main()

