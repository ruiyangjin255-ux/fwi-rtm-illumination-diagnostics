from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable


ROOT = Path(__file__).resolve().parents[1]
VEL_FILE = ROOT / "fd2d_pml" / "vel" / "junyun_p_0401x0301.bin"
OUT_DIR = ROOT / "figure"

NX = 401
NZ = 401
DX_KM = 0.01
DZ_KM = 0.01


def read_velocity(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.float32)
    expected = NX * NZ
    if data.size != expected:
        raise ValueError(f"{path.name} has {data.size} floats, expected {expected}.")
    return data.reshape((NX, NZ)).T / 1000.0


def add_right_cax(ax, pad=0.02, width=0.025):
    divider = make_axes_locatable(ax)
    return divider.append_axes("right", size=f"{width * 100}%", pad=pad)


def plot_uniform_model():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    velocity = read_velocity(VEL_FILE)

    x_max = (NX - 1) * DX_KM
    z_max = (NZ - 1) * DZ_KM
    sx = x_max / 2.0
    sz = z_max / 2.0

    mpl.rcParams["font.family"] = "Arial"
    fig, ax = plt.subplots(figsize=(6.8, 5.8), facecolor="white")

    norm = mpl.colors.Normalize(vmin=1.8, vmax=3.0)
    im = ax.imshow(
        velocity,
        cmap="rainbow",
        norm=norm,
        extent=[0, x_max, z_max, 0],
        origin="upper",
        aspect="equal",
    )

    ax.plot(
        sx,
        sz,
        marker="*",
        markersize=16,
        markerfacecolor="red",
        markeredgecolor="black",
        markeredgewidth=0.8,
        linestyle="None",
        label="Source",
    )

    ax.set_xlabel("Distance (km)", fontsize=15)
    ax.set_ylabel("Depth (km)", fontsize=15)
    ax.xaxis.set_major_locator(MultipleLocator(0.8))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="major", labelsize=12, width=1.5)
    ax.tick_params(which="minor", width=1.0)
    ax.legend(loc="upper right", fontsize=11, frameon=True)

    cax = add_right_cax(ax)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Velocity (km/s)", fontsize=15)
    cbar.ax.tick_params(labelsize=12)

    png = OUT_DIR / "uniform_model_center_source_star.png"
    pdf = OUT_DIR / "uniform_model_center_source_star.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    print(f"Saved {png}")
    print(f"Saved {pdf}")


if __name__ == "__main__":
    plot_uniform_model()
