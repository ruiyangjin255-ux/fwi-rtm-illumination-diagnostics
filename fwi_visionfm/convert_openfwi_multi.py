from __future__ import annotations

import argparse
from pathlib import Path

from fwi_visionfm.data_conversion import convert_openfwi_file_groups_to_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将多组 OpenFWI records/model 文件顺序配对并转换为逐样本 .npz。")
    parser.add_argument("--records", nargs="+", required=True, type=Path)
    parser.add_argument("--velocity", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-name", default="openfwi")
    parser.add_argument("--family", default="")
    parser.add_argument("--split-name", default="")
    parser.add_argument("--subset-name", default="")
    parser.add_argument(
        "--records-layout",
        choices=("samples_shots_receivers_time", "samples_shots_time_receivers"),
        default="samples_shots_time_receivers",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = convert_openfwi_file_groups_to_npz(
        records_paths=args.records,
        velocity_paths=args.velocity,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        family=args.family,
        split_name=args.split_name,
        subset_name=args.subset_name,
        records_layout=args.records_layout,
        max_samples=args.max_samples,
        sample_stride=args.sample_stride,
        dry_run=args.dry_run,
    )
    print(f"写出样本数: {manifest['sample_count']}")
    print(f"records 文件数: {len(manifest['source_records_paths'])}")
    print(f"velocity 文件数: {len(manifest['source_velocity_paths'])}")
    print(f"输出记录形状: {manifest['output_records_shape']}")
    print(f"输出速度形状: {manifest['output_velocity_shape']}")
    print(f"subset_name: {manifest['subset_name']}")
    if args.dry_run:
        print("dry-run: 未写出任何文件")
        return
    print(f"manifest: {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
