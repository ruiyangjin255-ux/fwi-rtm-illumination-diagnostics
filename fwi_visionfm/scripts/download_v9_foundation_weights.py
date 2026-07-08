from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _safe_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _weight_files_exist(path: Path) -> bool:
    names = {"config.json", "model.safetensors", "pytorch_model.bin", "pytorch_model.bin.index.json"}
    if not path.exists():
        return False
    files = {item.name for item in path.rglob("*") if item.is_file()}
    return "config.json" in files and any(name in files for name in ("model.safetensors", "pytorch_model.bin", "pytorch_model.bin.index.json"))


def _write_markdown(path: Path, entries: list[dict[str, Any]]) -> None:
    lines = [
        "# Protocol V9 Foundation Weight Download Status",
        "",
        "## Entries",
    ]
    for entry in entries:
        lines.append(f"- {entry['name']}: {entry['status']} | {entry.get('message', '')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_local_yaml(path: Path, *, repo_root: Path) -> Path:
    def unix(p: Path) -> str:
        return p.resolve().as_posix()

    lines = [
        "ncs:",
        f"  repo: \"{unix(repo_root / 'external' / 'ncs_models')}\"",
        f"  ncs_2d: \"{unix(repo_root / 'weights' / 'ncs' / 'NCS-v1-2d-base')}\"",
        f"  ncs_2p5d: \"{unix(repo_root / 'weights' / 'ncs' / 'NCS-v1-2.5d-base')}\"",
        "",
        "mae:",
        f"  vit_mae_base: \"{unix(repo_root / 'weights' / 'mae' / 'vit-mae-base')}\"",
        f"  vit_mae_large: \"{unix(repo_root / 'weights' / 'mae' / 'vit-mae-large')}\"",
        "",
        "requirements:",
        "  preferred_device: \"cpu\"",
        "  allow_fallback: true",
        "  no_benchmark_claim: true",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _git_available() -> bool:
    return shutil.which("git") is not None


def _clone_or_update_repo(url: str, dest: Path) -> tuple[str, str]:
    if dest.exists() and (dest / ".git").exists():
        if not _git_available():
            return "ALREADY_EXISTS", "git unavailable; existing repo kept"
        try:
            subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"], check=False, capture_output=True, text=True, timeout=120)
            return "ALREADY_EXISTS", "existing repo found"
        except Exception as exc:
            return "DOWNLOAD_FAILED", f"git pull failed: {type(exc).__name__}: {exc}"
    if not _git_available():
        return "DOWNLOAD_FAILED", "git unavailable"
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(["git", "clone", url, str(dest)], check=False, capture_output=True, text=True, timeout=300)
        if completed.returncode == 0:
            return "DOWNLOADED", "git clone ok"
        return "DOWNLOAD_FAILED", (completed.stderr or completed.stdout or "git clone failed").strip()
    except Exception as exc:
        return "DOWNLOAD_FAILED", f"git clone failed: {type(exc).__name__}: {exc}"


def _download_hf(repo_id: str, local_dir: Path) -> tuple[str, str]:
    if _weight_files_exist(local_dir):
        return "ALREADY_EXISTS", "weight files already present"
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        return "DOWNLOAD_FAILED_HF_UNAVAILABLE", "huggingface_hub unavailable"
    try:
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo_id, local_dir=str(local_dir), local_dir_use_symlinks=False, resume_download=True)
    except Exception as exc:
        message = str(exc)
        if "401" in message or "403" in message or "404" in message or "Repository Not Found" in message:
            return "DOWNLOAD_FAILED_REPO_NOT_FOUND_OR_AUTH", message
        return "DOWNLOAD_FAILED", f"{type(exc).__name__}: {exc}"
    if _weight_files_exist(local_dir):
        return "DOWNLOADED", "snapshot_download ok"
    return "INVALID_LOCAL_CACHE", "download completed but expected files are missing"


def download_v9_foundation_weights(
    *,
    repo_root: str | Path,
    output_dir: str | Path,
    download_ncs: bool = True,
    download_mae: bool = True,
    allow_fail: bool = True,
) -> dict[str, Any]:
    root = Path(repo_root)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    targets = {
        "ncs_repo": root / "external" / "ncs_models",
        "ncs_2d": root / "weights" / "ncs" / "NCS-v1-2d-base",
        "ncs_2p5d": root / "weights" / "ncs" / "NCS-v1-2.5d-base",
        "vit_mae_base": root / "weights" / "mae" / "vit-mae-base",
        "vit_mae_large": root / "weights" / "mae" / "vit-mae-large",
    }
    for target in targets.values():
        target.parent.mkdir(parents=True, exist_ok=True)
    (root / "configs").mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    git_ok = _git_available()
    hf_ok = _safe_import("huggingface_hub")
    entries.append({"name": "git", "status": "AVAILABLE" if git_ok else "DOWNLOAD_FAILED", "message": "git available" if git_ok else "git unavailable"})
    entries.append({"name": "huggingface_hub", "status": "AVAILABLE" if hf_ok else "DOWNLOAD_FAILED_HF_UNAVAILABLE", "message": "huggingface_hub available" if hf_ok else "huggingface_hub unavailable"})

    if download_ncs:
        status, message = _clone_or_update_repo("https://github.com/NorskRegnesentral/NCS_models", targets["ncs_repo"])
        entries.append({"name": "ncs_repo", "status": status, "path": str(targets["ncs_repo"]), "message": message})
        for name, repo_id in (
            ("ncs_2d", "NorskRegnesentralSTI/NCS-v1-2d-base"),
            ("ncs_2p5d", "NorskRegnesentralSTI/NCS-v1-2.5d-base"),
        ):
            status, message = _download_hf(repo_id, targets[name])
            entries.append({"name": name, "status": status, "path": str(targets[name]), "repo_id": repo_id, "message": message})
    else:
        entries.extend(
            [
                {"name": "ncs_repo", "status": "SKIPPED", "path": str(targets["ncs_repo"]), "message": "download_ncs=false"},
                {"name": "ncs_2d", "status": "SKIPPED", "path": str(targets["ncs_2d"]), "message": "download_ncs=false"},
                {"name": "ncs_2p5d", "status": "SKIPPED", "path": str(targets["ncs_2p5d"]), "message": "download_ncs=false"},
            ]
        )

    if download_mae:
        for name, repo_id in (
            ("vit_mae_base", "facebook/vit-mae-base"),
            ("vit_mae_large", "facebook/vit-mae-large"),
        ):
            status, message = _download_hf(repo_id, targets[name])
            entries.append({"name": name, "status": status, "path": str(targets[name]), "repo_id": repo_id, "message": message})
    else:
        entries.extend(
            [
                {"name": "vit_mae_base", "status": "SKIPPED", "path": str(targets["vit_mae_base"]), "message": "download_mae=false"},
                {"name": "vit_mae_large", "status": "SKIPPED", "path": str(targets["vit_mae_large"]), "message": "download_mae=false"},
            ]
        )

    yaml_path = _write_local_yaml(root / "configs" / "local_foundation_models.yaml", repo_root=root)
    payload = {
        "repo_root": str(root),
        "entries": entries,
        "local_foundation_models_yaml": str(yaml_path),
        "allow_fail": bool(allow_fail),
    }
    (out / "download_status.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(out / "download_status.md", entries)
    if not allow_fail and any(entry["status"].startswith("DOWNLOAD_FAILED") for entry in entries):
        raise RuntimeError("one or more downloads failed")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Protocol V9 foundation model code and weights.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download-ncs", type=str, default="true")
    parser.add_argument("--download-mae", type=str, default="true")
    parser.add_argument("--allow-fail", type=str, default="true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = download_v9_foundation_weights(
        repo_root=args.repo_root,
        output_dir=args.output_dir,
        download_ncs=str(args.download_ncs).lower() not in {"0", "false", "no"},
        download_mae=str(args.download_mae).lower() not in {"0", "false", "no"},
        allow_fail=str(args.allow_fail).lower() not in {"0", "false", "no"},
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

