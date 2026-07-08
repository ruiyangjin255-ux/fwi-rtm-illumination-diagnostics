from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm


FIGSHARE_API = "https://api.figshare.com/v2/articles/{article_id}"


def safe_name(name: str) -> str:
    bad = '<>:"/\\|?*'
    for ch in bad:
        name = name.replace(ch, "_")
    return name


def download_file(url: str, output_path: Path, chunk_size: int = 1024 * 1024) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with output_path.open("wb") as handle, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=output_path.name,
        ) as progress:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)
                    progress.update(len(chunk))


def export_figshare_article_metadata(
    *,
    article_id: str,
    output_dir: str | Path,
    download: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    response = requests.get(FIGSHARE_API.format(article_id=article_id), timeout=60)
    response.raise_for_status()
    metadata = response.json()

    metadata_path = output_dir / "figshare_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    files = metadata.get("files", [])
    file_table = [
        {
            "id": item.get("id"),
            "name": item.get("name"),
            "size": item.get("size"),
            "download_url": item.get("download_url"),
            "computed_md5": item.get("computed_md5"),
        }
        for item in files
    ]
    files_path = output_dir / "figshare_files.json"
    files_path.write_text(json.dumps(file_table, indent=2, ensure_ascii=False), encoding="utf-8")

    if download:
        for item in file_table:
            download_url = item.get("download_url")
            if not download_url:
                continue
            name = safe_name(str(item["name"]))
            target = output_dir / "files" / name
            expected_size = int(item.get("size") or 0)
            if target.exists() and (expected_size == 0 or target.stat().st_size == expected_size):
                continue
            download_file(download_url, target)

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--article-id", default="30702569")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = export_figshare_article_metadata(
        article_id=args.article_id,
        output_dir=args.output_dir,
        download=args.download,
    )
    files = metadata.get("files", [])
    print("Title:", metadata.get("title"))
    print("DOI:", metadata.get("doi"))
    print("Published:", metadata.get("published_date"))
    print("Number of files:", len(files))
    for item in files:
        print(f"- {item.get('name')} | size={item.get('size')} | id={item.get('id')}")
    print("Saved:", Path(args.output_dir) / "figshare_metadata.json")
    print("Saved:", Path(args.output_dir) / "figshare_files.json")


if __name__ == "__main__":
    main()
