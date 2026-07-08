from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_EXPECTED_RECORDS_SHAPE = (500, 5, 1000, 70)
DEFAULT_EXPECTED_VELOCITY_SHAPE = (500, 1, 70, 70)

FAMILY_SPECS = {
    "FlatVel_A": {
        "dir_tokens": ("flatvel_a", "flatvel-a", "flatvel"),
        "records_names": ("data1.npy",),
        "velocity_names": ("model1.npy",),
    },
    "CurveVel_A": {
        "dir_tokens": ("curvevel_a", "curvevel-a", "curvevel"),
        "records_names": ("data1.npy",),
        "velocity_names": ("model1.npy",),
    },
    "FlatFault_A": {
        "dir_tokens": ("flatfault_a", "flatfault-a", "flatfault"),
        "records_names": ("data1.npy", "seis2_1_0.npy"),
        "velocity_names": ("model1.npy", "vel2_1_0.npy"),
    },
    "CurveFault_A": {
        "dir_tokens": ("curvefault_a", "curvefault-a", "curvefault"),
        "records_names": ("data1.npy", "seis2_1_0.npy"),
        "velocity_names": ("model1.npy", "vel2_1_0.npy"),
    },
}


def _find_family_dir(root: Path, tokens: tuple[str, ...]) -> Path | None:
    candidates = [path for path in root.iterdir() if path.is_dir()]
    for token in tokens:
        for path in candidates:
            normalized = path.name.lower().replace(" ", "")
            if normalized == token:
                return path
    for token in tokens:
        for path in candidates:
            if token in path.name.lower().replace(" ", ""):
                return path
    return None


def _find_named_file(base: Path | None, names: tuple[str, ...]) -> Path | None:
    if base is None:
        return None
    for name in names:
        matches = sorted(base.rglob(name))
        if matches:
            return matches[0]
    return None


def _shape_dtype(path: Path | None) -> tuple[list[int], str]:
    if path is None or not path.exists():
        return [], ""
    array = np.load(path, mmap_mode="r")
    return [int(v) for v in array.shape], str(array.dtype)


def _status(
    *,
    records_exists: bool,
    velocity_exists: bool,
    records_shape: list[int],
    velocity_shape: list[int],
    expected_records_shape: tuple[int, ...] | None,
    expected_velocity_shape: tuple[int, ...] | None,
) -> str:
    if not records_exists or not velocity_exists:
        return "missing"
    if expected_records_shape is not None and tuple(records_shape) != expected_records_shape:
        return "invalid_shape"
    if expected_velocity_shape is not None and tuple(velocity_shape) != expected_velocity_shape:
        return "invalid_shape"
    return "ok"


def inspect_openfwi_first_files(
    *,
    root: str | Path,
    output_md: str | Path,
    output_json: str | Path,
    expected_records_shape: tuple[int, ...] | None = DEFAULT_EXPECTED_RECORDS_SHAPE,
    expected_velocity_shape: tuple[int, ...] | None = DEFAULT_EXPECTED_VELOCITY_SHAPE,
) -> list[dict[str, Any]]:
    root = Path(root)
    records: list[dict[str, Any]] = []
    for family, spec in FAMILY_SPECS.items():
        family_dir = _find_family_dir(root, spec["dir_tokens"])
        records_path = _find_named_file(family_dir, spec["records_names"])
        velocity_path = _find_named_file(family_dir, spec["velocity_names"])
        records_shape, records_dtype = _shape_dtype(records_path)
        velocity_shape, velocity_dtype = _shape_dtype(velocity_path)
        record = {
            "family": family,
            "records_path": "" if records_path is None else str(records_path),
            "velocity_path": "" if velocity_path is None else str(velocity_path),
            "records_exists": bool(records_path and records_path.exists()),
            "velocity_exists": bool(velocity_path and velocity_path.exists()),
            "records_shape": records_shape,
            "velocity_shape": velocity_shape,
            "records_dtype": records_dtype,
            "velocity_dtype": velocity_dtype,
            "status": _status(
                records_exists=bool(records_path and records_path.exists()),
                velocity_exists=bool(velocity_path and velocity_path.exists()),
                records_shape=records_shape,
                velocity_shape=velocity_shape,
                expected_records_shape=expected_records_shape,
                expected_velocity_shape=expected_velocity_shape,
            ),
        }
        records.append(record)

    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# OpenFWI First Files Summary",
        "",
        "| family | status | records_exists | velocity_exists | records_shape | velocity_shape | records_dtype | velocity_dtype | records_path | velocity_path |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        lines.append(
            f"| {record['family']} | {record['status']} | {record['records_exists']} | {record['velocity_exists']} | "
            f"{tuple(record['records_shape']) if record['records_shape'] else ''} | "
            f"{tuple(record['velocity_shape']) if record['velocity_shape'] else ''} | "
            f"{record['records_dtype']} | {record['velocity_dtype']} | {record['records_path']} | {record['velocity_path']} |"
        )
    output_md = Path(output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect OpenFWI first data/model files.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = inspect_openfwi_first_files(
        root=args.root,
        output_md=args.output_md,
        output_json=args.output_json,
    )
    print(f"写出摘要: {args.output_md}")
    print(f"写出 JSON: {args.output_json}")
    for record in records:
        print(
            f"{record['family']}: {record['status']} "
            f"records={record['records_shape']} velocity={record['velocity_shape']}"
        )


if __name__ == "__main__":
    main()
