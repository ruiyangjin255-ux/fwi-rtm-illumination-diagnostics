from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any


def inspect_dataset_tree(root: str | Path, output: str | Path) -> list[dict[str, Any]]:
    root = Path(root)
    rows: list[dict[str, Any]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        row: dict[str, Any] = {
            "path": str(path),
            "name": path.name,
            "suffix": path.suffix.lower(),
            "size": path.stat().st_size,
        }
        if path.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(path, "r") as archive:
                    names = archive.namelist()
                    row["zip_files"] = names[:200]
                    row["zip_file_count"] = len(names)
            except Exception as exc:
                row["zip_error"] = str(exc)
        rows.append(row)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = inspect_dataset_tree(args.root, args.output)
    print("files:", len(rows))
    print("saved:", args.output)


if __name__ == "__main__":
    main()
