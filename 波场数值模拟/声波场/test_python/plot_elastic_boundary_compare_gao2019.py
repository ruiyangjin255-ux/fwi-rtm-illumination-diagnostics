from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable


ROOT = Path("D:/ryjin")
OUT_DIR = ROOT / "figure"

DOMAIN_M = 4000.0
DX = 10.0
DZ = 10.0
DT = 0.001
T_SNAP = 0.75
F0 = 12.0
VP0 = 3200.0
VS0 = 1700.0
RHO0 = 2200.0
NPML = 35
PML_ORDER = 2
RCOEF = 1.0e-6
CJK_FONT = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf")


def ricker_wavelet(nt: int) -> np.ndarray:
    t = np.arange(nt) * DT
    t0 = 1.0 / F0
    arg = (np.pi * F0 * (t - t0)) ** 2
    return (1.0 - 2.0 * arg) * np.exp(-arg)


def pml_sigma_1d(n: int, npml: int, d: float, vmax: float) -> np.ndarray:
    sigma = np.zeros(n, dtype=np.float64)
    sigma_max = -((PML_ORDER + 1) * vmax * np.log(RCOEF)) / (2.0 * npml * d)
    for i in range(npml):
        left_r = (npml - i) / npml
        right_r = (i + 1) / npml
        sigma[i] = sigma_max * left_r ** PML_ORDER
        sigma[n - npml + i] = sigma_max * right_r ** PML_ORDER
    return sigma


def pml_ab(sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b = np.exp(-sigma * DT)
    a = np.full_like(sigma, DT)
    mask = sigma > 1.0e-12
    a[mask] = (1.0 - b[mask]) / sigma[mask]
    return a, b


def gaussian_source(radius: int = 2, sigma: float = 0.9) -> np.ndarray:
    x = np.arange(-radius, radius + 1)
    z = np.arange(-radius, radius + 1)
    xx, zz = np.meshgrid(x, z)
    kernel = np.exp(-(xx * xx + zz * zz) / (2.0 * sigma * sigma))
    return kernel / np.sum(kernel)


def build_sigma(nz: int, nx: int, mode: str) -> dict[str, np.ndarray]:
    if mode == "none":
        sx_c = np.zeros(nx)
        sz_c = np.zeros(nz)
    elif mode == "sponge":
        # A deliberately simple damping layer for comparison with true split-field PML.
        sx_c = pml_sigma_1d(nx, NPML, DX, VP0) * 0.28
        sz_c = pml_sigma_1d(nz, NPML, DZ, VP0) * 0.28
    elif mode == "pml":
        sx_c = pml_sigma_1d(nx, NPML, DX, VP0)
        sz_c = pml_sigma_1d(nz, NPML, DZ, VP0)
    else:
        raise ValueError(f"Unknown boundary mode: {mode}")

    sx_h = np.zeros(nx + 1)
    sx_h[1:nx] = 0.5 * (sx_c[:-1] + sx_c[1:])
    sx_h[0] = sx_c[0]
    sx_h[nx] = sx_c[-1]

    sz_h = np.zeros(nz + 1)
    sz_h[1:nz] = 0.5 * (sz_c[:-1] + sz_c[1:])
    sz_h[0] = sz_c[0]
    sz_h[nz] = sz_c[-1]

    return {
        "sx_c": np.tile(sx_c[None, :], (nz, 1)),
        "sz_c": np.tile(sz_c[:, None], (1, nx)),
        "sx_vx": np.tile(sx_h[None, :], (nz, 1)),
        "sz_vx": np.tile(sz_c[:, None], (1, nx + 1)),
        "sx_vz": np.tile(sx_c[None, :], (nz + 1, 1)),
        "sz_vz": np.tile(sz_h[:, None], (1, nx)),
        "sx_txz": np.tile(sx_h[None, :], (nz + 1, 1)),
        "sz_txz": np.tile(sz_h[:, None], (1, nx + 1)),
    }


def simulate(boundary_mode: str) -> np.ndarray:
    nx0 = int(round(DOMAIN_M / DX)) + 1
    nz0 = int(round(DOMAIN_M / DZ)) + 1
    if boundary_mode == "none":
        pad = 0
    else:
        pad = NPML
    nx = nx0 + 2 * pad
    nz = nz0 + 2 * pad
    nt = int(round(T_SNAP / DT))

    vp = VP0 * np.ones((nz, nx), dtype=np.float64)
    vs = VS0 * np.ones((nz, nx), dtype=np.float64)
    rho = RHO0 * np.ones((nz, nx), dtype=np.float64)
    mu = rho * vs * vs
    lam = rho * vp * vp - 2.0 * mu

    sig = build_sigma(nz, nx, boundary_mode)
    a_x_c, b_x_c = pml_ab(sig["sx_c"])
    a_z_c, b_z_c = pml_ab(sig["sz_c"])
    a_x_vx, b_x_vx = pml_ab(sig["sx_vx"])
    a_z_vx, b_z_vx = pml_ab(sig["sz_vx"])
    a_x_vz, b_x_vz = pml_ab(sig["sx_vz"])
    a_z_vz, b_z_vz = pml_ab(sig["sz_vz"])
    a_x_txz, b_x_txz = pml_ab(sig["sx_txz"])
    a_z_txz, b_z_txz = pml_ab(sig["sz_txz"])

    txx_x = np.zeros((nz, nx), dtype=np.float64)
    txx_z = np.zeros((nz, nx), dtype=np.float64)
    tzz_x = np.zeros((nz, nx), dtype=np.float64)
    tzz_z = np.zeros((nz, nx), dtype=np.float64)
    vx_x = np.zeros((nz, nx + 1), dtype=np.float64)
    vx_z = np.zeros((nz, nx + 1), dtype=np.float64)
    vz_x = np.zeros((nz + 1, nx), dtype=np.float64)
    vz_z = np.zeros((nz + 1, nx), dtype=np.float64)
    txz_x = np.zeros((nz + 1, nx + 1), dtype=np.float64)
    txz_z = np.zeros((nz + 1, nx + 1), dtype=np.float64)

    wavelet = ricker_wavelet(nt)
    kernel = gaussian_source()
    radius = kernel.shape[0] // 2
    src_x = pad + nx0 // 2
    src_z = pad + nz0 // 2

    c1 = 9.0 / 8.0
    c2 = -1.0 / 24.0

    for it in range(nt):
        txx = txx_x + txx_z
        tzz = tzz_x + tzz_z
        txz = txz_x + txz_z

        jv = slice(2, nz - 2)
        iv = slice(2, nx - 2)

        dtxx_dx = (
            c1 * (txx[jv, iv] - txx[jv, 1 : nx - 3])
            + c2 * (txx[jv, 3 : nx - 1] - txx[jv, 0 : nx - 4])
        ) / DX
        dtxz_dz = (
            c1 * (txz[3 : nz - 1, iv] - txz[2 : nz - 2, iv])
            + c2 * (txz[4:nz, iv] - txz[1 : nz - 3, iv])
        ) / DZ
        rho_vx = 0.5 * (rho[jv, iv] + rho[jv, 1 : nx - 3])
        vx_x[jv, iv] = b_x_vx[jv, iv] * vx_x[jv, iv] + a_x_vx[jv, iv] * (dtxx_dx / rho_vx)
        vx_z[jv, iv] = b_z_vx[jv, iv] * vx_z[jv, iv] + a_z_vx[jv, iv] * (dtxz_dz / rho_vx)

        dtxz_dx = (
            c1 * (txz[jv, 3 : nx - 1] - txz[jv, 2 : nx - 2])
            + c2 * (txz[jv, 4:nx] - txz[jv, 1 : nx - 3])
        ) / DX
        dtzz_dz = (
            c1 * (tzz[jv, iv] - tzz[1 : nz - 3, iv])
            + c2 * (tzz[3 : nz - 1, iv] - tzz[0 : nz - 4, iv])
        ) / DZ
        rho_vz = 0.5 * (rho[jv, iv] + rho[1 : nz - 3, iv])
        vz_x[jv, iv] = b_x_vz[jv, iv] * vz_x[jv, iv] + a_x_vz[jv, iv] * (dtxz_dx / rho_vz)
        vz_z[jv, iv] = b_z_vz[jv, iv] * vz_z[jv, iv] + a_z_vz[jv, iv] * (dtzz_dz / rho_vz)

        vx = vx_x + vx_z
        vz = vz_x + vz_z

        if it < round(2.0 / F0 / DT):
            z_slice = slice(src_z - radius, src_z + radius + 1)
            x_slice = slice(src_x - radius, src_x + radius + 1)
            amp = wavelet[it] * kernel
            # Explosion source: stress source, suited to boundary-effect comparison.
            txx_x[z_slice, x_slice] += 0.5 * amp
            txx_z[z_slice, x_slice] += 0.5 * amp
            tzz_x[z_slice, x_slice] += 0.5 * amp
            tzz_z[z_slice, x_slice] += 0.5 * amp

        js = slice(2, nz - 2)
        is_ = slice(2, nx - 2)
        dvx_dx = (
            c1 * (vx[js, 3 : nx - 1] - vx[js, is_])
            + c2 * (vx[js, 4:nx] - vx[js, 1 : nx - 3])
        ) / DX
        dvz_dz = (
            c1 * (vz[3 : nz - 1, is_] - vz[js, is_])
            + c2 * (vz[4:nz, is_] - vz[1 : nz - 3, is_])
        ) / DZ

        txx_x[js, is_] = b_x_c[js, is_] * txx_x[js, is_] + a_x_c[js, is_] * (
            (lam[js, is_] + 2.0 * mu[js, is_]) * dvx_dx
        )
        txx_z[js, is_] = b_z_c[js, is_] * txx_z[js, is_] + a_z_c[js, is_] * (
            lam[js, is_] * dvz_dz
        )
        tzz_x[js, is_] = b_x_c[js, is_] * tzz_x[js, is_] + a_x_c[js, is_] * (
            lam[js, is_] * dvx_dx
        )
        tzz_z[js, is_] = b_z_c[js, is_] * tzz_z[js, is_] + a_z_c[js, is_] * (
            (lam[js, is_] + 2.0 * mu[js, is_]) * dvz_dz
        )

        dvx_dz = (
            c1 * (vx[js, is_] - vx[1 : nz - 3, is_])
            + c2 * (vx[3 : nz - 1, is_] - vx[0 : nz - 4, is_])
        ) / DZ
        dvz_dx = (
            c1 * (vz[js, is_] - vz[js, 1 : nx - 3])
            + c2 * (vz[js, 3 : nx - 1] - vz[js, 0 : nx - 4])
        ) / DX
        mu_txz = 0.25 * (
            mu[js, is_]
            + mu[1 : nz - 3, is_]
            + mu[js, 1 : nx - 3]
            + mu[1 : nz - 3, 1 : nx - 3]
        )
        txz_x[js, is_] = b_x_txz[js, is_] * txz_x[js, is_] + a_x_txz[js, is_] * (mu_txz * dvz_dx)
        txz_z[js, is_] = b_z_txz[js, is_] * txz_z[js, is_] + a_z_txz[js, is_] * (mu_txz * dvx_dz)

    vx_c = 0.5 * ((vx_x + vx_z)[:, :nx] + (vx_x + vx_z)[:, 1 : nx + 1])
    vz_c = 0.5 * ((vz_x + vz_z)[:nz, :] + (vz_x + vz_z)[1 : nz + 1, :])
    field = vx_c + vz_c
    if pad > 0:
        field = field[pad : pad + nz0, pad : pad + nx0]
    return field


def add_right_cax(ax):
    divider = make_axes_locatable(ax)
    return divider.append_axes("right", size="3%", pad=0.04)


def setup_axis(ax):
    ax.set_xlabel("Distance (km)", fontsize=10, labelpad=3)
    ax.set_ylabel("Depth (km)", fontsize=10, labelpad=3)
    ax.xaxis.set_major_locator(MultipleLocator(0.8))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="major", labelsize=9, width=1.0)
    ax.tick_params(which="minor", width=0.7)


def plot_compare():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = [
        ("none", "a.未加入吸收边界"),
        ("sponge", "b.海绵吸收边界"),
        ("pml", "c.PML吸收边界"),
    ]
    fields = []
    for mode, _ in cases:
        print(f"Simulating elastic wavefield: {mode}")
        fields.append(simulate(mode))

    scale = max(np.max(np.abs(field)) for field in fields)
    fields = [field / scale for field in fields]

    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9), facecolor="white")
    extent = [0.0, DOMAIN_M / 1000.0, DOMAIN_M / 1000.0, 0.0]
    for ax, field, (_, label) in zip(axes, fields, cases):
        im = ax.imshow(
            field,
            cmap="seismic",
            vmin=-0.50,
            vmax=0.50,
            extent=extent,
            origin="upper",
            aspect="equal",
        )
        setup_axis(ax)
        ax.text(
            0.5,
            -0.20,
            label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9.5,
            fontweight="bold",
            fontproperties=CJK_FONT,
        )
        cbar = fig.colorbar(im, cax=add_right_cax(ax))
        cbar.set_label("Amplitude", fontsize=9)
        cbar.ax.tick_params(labelsize=8.5)

    fig.subplots_adjust(wspace=0.35, bottom=0.18)
    png = OUT_DIR / "elastic_boundary_compare_gao2019_0p75s.png"
    pdf = OUT_DIR / "elastic_boundary_compare_gao2019_0p75s.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    print(f"Saved {png}")
    print(f"Saved {pdf}")


if __name__ == "__main__":
    plot_compare()
