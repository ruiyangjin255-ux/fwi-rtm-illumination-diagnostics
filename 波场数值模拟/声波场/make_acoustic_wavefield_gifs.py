from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(r"D:\ryjin")
BIN_DIR = ROOT / "fd2d_pml" / "bin"
EXE = BIN_DIR / "mod_gif.exe"
PARTABLE = BIN_DIR / "partable.dat"
OUT_DIR = ROOT / "paper_figures_source" / "acoustic" / "wavefield_gifs"
FRAME_ROOT = OUT_DIR / "frames"


MODELS = [
    {
        "key": "uniform",
        "nx": 401,
        "nz": 401,
        "dx": 10,
        "dz": 10,
        "dt": 0.001,
        "nt": 1501,
        "frame_dt": 0.1,
        "shot": 200,
        "zs": 200,
        "vel": "../vel/junyun_p_0401x0301.bin",
        "figsize": (5.8, 5.2),
    },
    {
        "key": "layer",
        "nx": 401,
        "nz": 401,
        "dx": 10,
        "dz": 10,
        "dt": 0.001,
        "nt": 2001,
        "frame_dt": 0.25,
        "shot": 200,
        "zs": 1,
        "vel": "../vel/cenzhuang_p_0401x0401.bin",
        "figsize": (5.8, 5.2),
    },
    {
        "key": "graben",
        "nx": 401,
        "nz": 401,
        "dx": 10,
        "dz": 10,
        "dt": 0.001,
        "nt": 2501,
        "frame_dt": 0.25,
        "shot": 200,
        "zs": 1,
        "vel": "../vel/diqian_p_0401x0401.bin",
        "figsize": (5.8, 5.2),
    },
    {
        "key": "seg",
        "nx": 676,
        "nz": 230,
        "dx": 10,
        "dz": 10,
        "dt": 0.001,
        "nt": 2501,
        "frame_dt": 0.25,
        "shot": 340,
        "zs": 1,
        "vel": "../vel/seg676x230.bin",
        "figsize": (8.2, 3.55),
    },
]


def write_partable(model: dict) -> None:
    text = f"""nx = {model['nx']}
nz = {model['nz']}
dx = {model['dx']}
dz = {model['dz']}
spunit = 1
dt = {model['dt']}
nt = {model['nt']}
f0 = 20.0
ts = 0.05
shotbeg = {model['shot']}
shotend = {model['shot']}
shotintvl = 20
zs = {model['zs']}
ksnp = 1
tsnp = {model['frame_dt']}
ksg = 0
velnm = {model['vel']}
datnm = ../data
"""
    PARTABLE.write_text(text, encoding="ascii")


def normalize_amp(data: np.ndarray) -> np.ndarray:
    limit = np.percentile(np.abs(data), 99.5)
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0
    out = np.clip(data / limit * 0.5, -0.5, 0.5)
    out[np.abs(out) < 0.012] = 0
    return out


def read_snapshot(path: Path, nz: int, nx: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size != nz * nx:
        raise ValueError(f"{path} has {raw.size} values, expected {nz * nx}")
    return raw.reshape((nx, nz)).T


def render_frame(data: np.ndarray, model: dict, t_sec: float, out_path: Path) -> None:
    nx, nz = model["nx"], model["nz"]
    xmax = (nx - 1) * model["dx"] / 1000.0
    zmax = (nz - 1) * model["dz"] / 1000.0

    fig, ax = plt.subplots(figsize=model["figsize"], dpi=160)
    im = ax.imshow(
        normalize_amp(data),
        cmap="seismic",
        vmin=-0.5,
        vmax=0.5,
        extent=[0, xmax, zmax, 0],
        interpolation="bilinear",
        aspect="equal",
    )
    ax.set_xlabel("Distance (km)", fontsize=11, labelpad=8)
    ax.set_ylabel("Depth (km)", fontsize=11)
    ax.tick_params(labelsize=10, direction="in", top=True, right=True, length=4, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label("Amplitude", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_ticks([-0.5, 0, 0.5])
    ax.text(
        0.5,
        -0.27,
        f"t={t_sec:.2f}s",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def make_gif(model: dict) -> Path:
    key = model["key"]
    frame_dir = FRAME_ROOT / key
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old in frame_dir.glob("*.png"):
        old.unlink()

    snapshot_files = sorted(BIN_DIR.glob(f"gif_snapshot_{model['shot']:04d}_*.bin"))
    if not snapshot_files:
        raise RuntimeError(f"No generated snapshots found for {key}")

    pngs: list[Path] = []
    for snap in snapshot_files:
        it = int(snap.stem.split("_")[-1])
        t_sec = it * model["dt"]
        data = read_snapshot(snap, model["nz"], model["nx"])
        png_name = f"{key}_t{t_sec:05.2f}s".replace(".", "p") + ".png"
        png_path = frame_dir / png_name
        render_frame(data, model, t_sec, png_path)
        pngs.append(png_path)

    first_t = int(snapshot_files[0].stem.split("_")[-1]) * model["dt"]
    last_t = int(snapshot_files[-1].stem.split("_")[-1]) * model["dt"]
    frame_tag = f"{model['frame_dt']:.2f}".replace(".", "p")
    last_tag = f"{last_t:.2f}".replace(".", "p")
    gif_path = OUT_DIR / f"{key}_wavefield_{frame_tag}s_to_{last_tag}s.gif"
    frames = [Image.open(p).convert("P", palette=Image.Palette.ADAPTIVE) for p in pngs]
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=260,
        loop=0,
        optimize=False,
        disposal=2,
    )
    for frame in frames:
        frame.close()
    return gif_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=[m["key"] for m in MODELS])
    args = parser.parse_args()
    selected = {name.lower() for name in args.models}

    if not EXE.exists():
        raise FileNotFoundError(f"Missing executable: {EXE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAME_ROOT.mkdir(parents=True, exist_ok=True)

    backup = PARTABLE.with_suffix(".dat.bak_gif_codex")
    shutil.copy2(PARTABLE, backup)
    made: list[Path] = []
    try:
        for model in MODELS:
            if model["key"] not in selected:
                continue
            for old in BIN_DIR.glob("gif_snapshot_*.bin"):
                old.unlink()
            write_partable(model)
            print(f"Running {model['key']} acoustic model...")
            subprocess.run([str(EXE)], cwd=BIN_DIR, check=True)
            gif = make_gif(model)
            made.append(gif)
            print(f"Saved {gif}")
    finally:
        shutil.copy2(backup, PARTABLE)

    print("GIF outputs:")
    for path in made:
        print(path)


if __name__ == "__main__":
    main()
