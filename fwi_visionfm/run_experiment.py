from __future__ import annotations

import argparse
from pathlib import Path

from fwi_visionfm.experiment import load_experiment_config, run_experiment_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按 JSON 配置运行 FWI-VisionFM 实验。")
    parser.add_argument("--config", type=Path, required=True, help="实验配置 JSON 文件。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experiment_from_config(load_experiment_config(args.config))
    print(f"状态: {result['状态']}")
    print(f"运行目录: {result['运行目录']}")
    print(f"摘要文件: {result['摘要文件']}")


if __name__ == "__main__":
    main()
