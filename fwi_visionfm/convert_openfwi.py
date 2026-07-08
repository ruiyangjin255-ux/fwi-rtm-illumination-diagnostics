from __future__ import annotations

import argparse
from pathlib import Path

from fwi_visionfm.data_conversion import convert_openfwi_files_to_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 OpenFWI 风格数组文件转换为 FWI-VisionFM 本地 .npz 样本格式。")
    parser.add_argument("--records", type=Path, required=True, help="地震记录数组，支持 .npy 或 .npz。")
    parser.add_argument("--velocity", type=Path, required=True, help="速度模型数组，支持 .npy 或 .npz。")
    parser.add_argument("--output-dir", type=Path, required=True, help="输出逐样本 .npz 的目录。")
    parser.add_argument("--records-key", default=None, help=".npz records 文件中的数组 key。")
    parser.add_argument("--velocity-key", default=None, help=".npz velocity 文件中的数组 key。")
    parser.add_argument("--source-positions", type=Path, default=None, help="可选炮点位置数组，支持 .npy 或 .npz。")
    parser.add_argument("--source-positions-key", default=None, help=".npz source_positions 文件中的数组 key。")
    parser.add_argument("--dataset-name", default="openfwi", help="写入 manifest.json 的数据集名称。")
    parser.add_argument("--family", default="", help="数据 family，例如 flatvel_a。")
    parser.add_argument("--split-name", default="", help="数据 split 名称，例如 train、val、test、tiny。")
    parser.add_argument("--subset-name", default="", help="数据子集名称，例如 flatvel_a_tiny16。")
    parser.add_argument(
        "--records-layout",
        choices=("samples_shots_receivers_time", "samples_shots_time_receivers"),
        default="samples_shots_time_receivers",
        help="输入 records 的 layout。OpenFWI 常见格式可用 samples_shots_time_receivers。",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="只转换前 N 个样本。")
    parser.add_argument("--sample-stride", type=int, default=1, help="每隔 stride 抽样一个样本。")
    parser.add_argument("--dry-run", action="store_true", help="只打印形状和推断结果，不写文件。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = convert_openfwi_files_to_npz(
        records_path=args.records,
        velocity_path=args.velocity,
        output_dir=args.output_dir,
        records_key=args.records_key,
        velocity_key=args.velocity_key,
        dataset_name=args.dataset_name,
        family=args.family,
        split_name=args.split_name,
        subset_name=args.subset_name,
        source_positions_path=args.source_positions,
        source_positions_key=args.source_positions_key,
        records_layout=args.records_layout,
        max_samples=args.max_samples,
        sample_stride=args.sample_stride,
        dry_run=args.dry_run,
    )
    print(f"写出样本数: {manifest['sample_count']}")
    print(f"records 原始形状: {manifest['records_original_shape']}")
    print(f"velocity 原始形状: {manifest['velocity_original_shape']}")
    print(f"family: {manifest['family']}")
    print(f"split_name: {manifest['split_name']}")
    print(f"subset_name: {manifest['subset_name']}")
    print(f"records layout: {manifest['records_layout']}")
    print(f"输出记录形状: {manifest['output_records_shape']}")
    print(f"输出速度形状: {manifest['output_velocity_shape']}")
    print(f"max_samples: {args.max_samples}")
    print(f"output_dir: {args.output_dir}")
    if args.dry_run:
        print("dry-run: 未写出任何文件")
        return
    print(f"manifest: {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
