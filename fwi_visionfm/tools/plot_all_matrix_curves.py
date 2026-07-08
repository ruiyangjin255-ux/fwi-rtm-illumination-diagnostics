from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from fwi_visionfm.plot_training_curves import plot_training_curves


@dataclass(frozen=True)
class MatrixHistory:
    matrix_dir: Path
    model_name: str
    history_path: Path
    output_path: Path


HISTORY_SPECS = (
    ("torch_cnn_baseline", "torch_training_history.csv"),
    ("dummy_dinov2_frozen", "foundation_training_history.csv"),
    ("dummy_dinov2_lora", "foundation_training_history.csv"),
)


def discover_matrix_histories(root: str | Path) -> list[MatrixHistory]:
    root = Path(root)
    histories: list[MatrixHistory] = []
    for matrix_dir in sorted(root.glob("matrix_*_subset500")):
        if not matrix_dir.is_dir():
            continue
        for model_name, history_name in HISTORY_SPECS:
            history_path = matrix_dir / model_name / history_name
            if history_path.exists():
                histories.append(
                    MatrixHistory(
                        matrix_dir=matrix_dir,
                        model_name=model_name,
                        history_path=history_path,
                        output_path=matrix_dir / f"{model_name}_loss.png",
                    )
                )
    return histories


def plot_all_matrix_curves(root: str | Path, *, allow_duplicate_openmp: bool = True) -> list[Path]:
    outputs: list[Path] = []
    for item in discover_matrix_histories(root):
        outputs.append(
            plot_training_curves(
                item.history_path,
                item.output_path,
                allow_duplicate_openmp=allow_duplicate_openmp,
            )
        )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot all completed subset500 matrix curves.")
    parser.add_argument("--root", default=r"D:\ryjin\fwi_visionfm\outputs", type=Path)
    parser.add_argument("--allow-duplicate-openmp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = plot_all_matrix_curves(
        args.root,
        allow_duplicate_openmp=bool(args.allow_duplicate_openmp),
    )
    print(f"写出曲线数量: {len(outputs)}")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
