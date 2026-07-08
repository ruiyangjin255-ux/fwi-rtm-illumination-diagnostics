from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper_figures_source" / "vti" / "qs_split_431"
DATA_DIR = ROOT / "saved_data" / "vti_qs_split_431"

L = 1200.0
DX = 3.0
DZ = 3.0
DT = 1.0e-4
TMAX = 0.20
F0 = 18.0
PML = 35

# Strong Thomsen anisotropy for an illustrative homogeneous VTI test.
VP0 = 2590.0
VS0 = 1435.0
RHO = 2680.0
EPSILON = 0.25
DELTA = -0.18
GAMMA = 0.80


def ricker(nt: int) -> np.ndarray:
    t = np.arange(nt) * DT
    t0 = 1.0 / F0
    a = (np.pi * F0 * (t - t0)) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def sponge(nz: int, nx: int) -> np.ndarray:
    damp = np.ones((nz, nx), dtype=np.float64)
    for i in range(PML):
        x = (PML - i) / PML
        value = np.exp(-0.012 * x * x)
        damp[:, i] *= value
        damp[:, -i - 1] *= value
        damp[i, :] *= value
        damp[-i - 1, :] *= value
    return damp


def d_dx(a: np.ndarray) -> np.ndarray:
    d = np.zeros_like(a)
    d[:, 2:-2] = (
        -a[:, 4:] + 8 * a[:, 3:-1] - 8 * a[:, 1:-3] + a[:, :-4]
    ) / (12 * DX)
    return d


def d_dz(a: np.ndarray) -> np.ndarray:
    d = np.zeros_like(a)
    d[2:-2, :] = (
        -a[4:, :] + 8 * a[3:-1, :] - 8 * a[1:-3, :] + a[:-4, :]
    ) / (12 * DZ)
    return d


def stiffness_from_thomsen():
    c33 = RHO * VP0**2
    c44 = RHO * VS0**2
    c11 = c33 * (1.0 + 2.0 * EPSILON)
    tmp = 2.0 * DELTA * c33 * (c33 - c44) + (c33 - c44) ** 2
    c13 = np.sqrt(max(tmp, 0.0)) - c44
    c66 = c44 * (1.0 + 2.0 * GAMMA)
    return c11, c13, c33, c44, c66


def simulate():
    nx0 = int(round(L / DX)) + 1
    nz0 = int(round(L / DZ)) + 1
    nx = nx0 + 2 * PML
    nz = nz0 + 2 * PML
    nt = int(round(TMAX / DT)) + 1
    src = ricker(nt)
    damp = sponge(nz, nx)
    c11, c13, c33, c44, c66 = stiffness_from_thomsen()

    u0 = np.zeros((nz, nx), dtype=np.float64)
    u1 = np.zeros_like(u0)
    w0 = np.zeros_like(u0)
    w1 = np.zeros_like(u0)
    y0 = np.zeros_like(u0)
    y1 = np.zeros_like(u0)

    sx = PML + nx0 // 2
    sz = PML + nz0 // 2
    rec_z = PML + int(round(420.0 / DZ))
    rec_cols = np.arange(PML, PML + nx0)

    rec_u = np.zeros((nt, nx0), dtype=np.float32)
    rec_w = np.zeros((nt, nx0), dtype=np.float32)
    rec_y = np.zeros((nt, nx0), dtype=np.float32)
    snap_step = int(round(0.2 / DT))
    snap = {}
    yy, xx = np.mgrid[-5:6, -5:6]
    src_mask = np.exp(-(xx * xx + yy * yy) / (2.0 * 2.0**2))
    src_mask = src_mask / np.sum(src_mask)
    src_rows = slice(sz - 5, sz + 6)
    src_cols = slice(sx - 5, sx + 6)

    for it in range(nt):
        exx = d_dx(u1)
        ezz = d_dz(w1)
        exz = d_dz(u1) + d_dx(w1)
        sxx = c11 * exx + c13 * ezz
        szz = c13 * exx + c33 * ezz
        sxz = c44 * exz

        ax = (d_dx(sxx) + d_dz(sxz)) / RHO
        az = (d_dx(sxz) + d_dz(szz)) / RHO

        syx = c66 * d_dx(y1)
        syz = c44 * d_dz(y1)
        ay = (d_dx(syx) + d_dz(syz)) / RHO

        u2 = 2.0 * u1 - u0 + DT * DT * ax
        w2 = 2.0 * w1 - w0 + DT * DT * az
        y2 = 2.0 * y1 - y0 + DT * DT * ay

        amp = src[it] * 2.0e5
        # Explosive source excites qP/qSV; y-force excites the decoupled qSH branch.
        u2[src_rows, src_cols] += 0.15 * amp * src_mask / RHO
        w2[src_rows, src_cols] += 0.15 * amp * src_mask / RHO
        y2[src_rows, src_cols] += amp * src_mask / RHO

        u2 *= damp
        w2 *= damp
        y2 *= damp
        u1 *= damp
        w1 *= damp
        y1 *= damp

        rec_u[it, :] = (u2[rec_z, rec_cols] - u1[rec_z, rec_cols]) / DT
        rec_w[it, :] = (w2[rec_z, rec_cols] - w1[rec_z, rec_cols]) / DT
        rec_y[it, :] = (y2[rec_z, rec_cols] - y1[rec_z, rec_cols]) / DT

        if it == snap_step:
            snap["vx"] = ((u2 - u1) / DT)[PML:PML + nz0, PML:PML + nx0].astype(np.float32)
            snap["vz"] = ((w2 - w1) / DT)[PML:PML + nz0, PML:PML + nx0].astype(np.float32)
            snap["vy"] = ((y2 - y1) / DT)[PML:PML + nz0, PML:PML + nx0].astype(np.float32)

        u0, u1 = u1, u2
        w0, w1 = w1, w2
        y0, y1 = y1, y2

    return {
        "rec_vx": rec_u,
        "rec_vz": rec_w,
        "rec_vy": rec_y,
        "snap": snap,
        "cij": np.array(stiffness_from_thomsen()),
    }


def clip(data: np.ndarray, q: float = 99.4) -> float:
    c = float(np.percentile(np.abs(data), q))
    return c if c > 0 else 1.0


def style_axis(ax, is_record=False):
    ax.tick_params(which="major", labelsize=11, width=1.0, direction="in", top=True, right=True)
    ax.tick_params(which="minor", width=0.7, direction="in", top=True, right=True)
    if is_record:
        ax.xaxis.set_major_locator(MultipleLocator(300))
        ax.xaxis.set_minor_locator(MultipleLocator(100))
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    else:
        ax.xaxis.set_major_locator(MultipleLocator(300))
        ax.xaxis.set_minor_locator(MultipleLocator(100))
        ax.yaxis.set_major_locator(MultipleLocator(300))
        ax.yaxis.set_minor_locator(MultipleLocator(100))


def plot_wavefields(data):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["axes.unicode_minus"] = False
    x = np.linspace(0, L, data["snap"]["vx"].shape[1])
    z = np.linspace(0, L, data["snap"]["vx"].shape[0])
    extent = [0, L, L, 0]
    fields = [
        ("Vx", data["snap"]["vx"]),
        ("Vz", data["snap"]["vz"]),
        ("Vy (SH)", data["snap"]["vy"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.3), facecolor="white")
    for ax, (title, field) in zip(axes, fields):
        c = clip(field)
        ax.imshow(field, cmap="gray", vmin=-c, vmax=c, extent=extent, origin="upper", aspect="equal")
        ax.set_title(title, fontsize=15, fontweight="bold", pad=6)
        ax.set_xlabel("X / m", fontsize=13)
        ax.set_ylabel("Z / m", fontsize=13)
        style_axis(ax)

    # Labels are placed to identify the three visually separated branches.
    axes[0].annotate("qP", xy=(900, 260), xytext=(980, 130),
                     arrowprops=dict(arrowstyle="->", lw=1.2), fontsize=13)
    axes[2].annotate("qS1", xy=(900, 300), xytext=(965, 190),
                     arrowprops=dict(arrowstyle="->", lw=1.2), fontsize=13)
    axes[1].annotate("qS2", xy=(600, 840), xytext=(705, 720),
                     arrowprops=dict(arrowstyle="->", lw=1.2), fontsize=13)

    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.14, top=0.86, wspace=0.24)
    fig.savefig(OUT_DIR / "vti_qs_split_wavefield_0p2s.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT_DIR / "vti_qs_split_wavefield_0p2s.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def plot_records(data):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t = np.arange(data["rec_vx"].shape[0]) * DT
    x = np.linspace(0, L, data["rec_vx"].shape[1])
    extent = [0, L, t[-1], t[0]]
    records = [
        ("Vx record", data["rec_vx"]),
        ("Vz record", data["rec_vz"]),
        ("Vy record (SH)", data["rec_vy"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2), facecolor="white")
    for ax, (title, rec) in zip(axes, records):
        gain = (t + 0.01)[:, None] ** 0.7
        disp = rec * gain
        c = clip(disp, 99.1)
        ax.imshow(disp, cmap="gray", vmin=-c, vmax=c, extent=extent, origin="upper", aspect="auto")
        ax.set_title(title, fontsize=15, fontweight="bold", pad=6)
        ax.set_xlabel("X / m", fontsize=13)
        ax.set_ylabel("t / s", fontsize=13)
        style_axis(ax, is_record=True)

    axes[0].annotate("qP", xy=(250, 0.155), xytext=(145, 0.128),
                     arrowprops=dict(arrowstyle="->", lw=1.1), fontsize=12)
    axes[2].annotate("qS1", xy=(300, 0.145), xytext=(410, 0.113),
                     arrowprops=dict(arrowstyle="->", lw=1.1), fontsize=12)
    axes[1].annotate("qS2", xy=(600, 0.155), xytext=(690, 0.135),
                     arrowprops=dict(arrowstyle="->", lw=1.1), fontsize=12)

    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.14, top=0.86, wspace=0.24)
    fig.savefig(OUT_DIR / "vti_qs_split_record_0p2s.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT_DIR / "vti_qs_split_record_0p2s.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_file = DATA_DIR / "vti_qs_split_431.npz"
    if data_file.exists():
        print(f"Loading cached data: {data_file}")
        raw = np.load(data_file)
        data = {
            "rec_vx": raw["rec_vx"],
            "rec_vz": raw["rec_vz"],
            "rec_vy": raw["rec_vy"],
            "snap": {"vx": raw["snap_vx"], "vz": raw["snap_vz"], "vy": raw["snap_vy"]},
            "cij": raw["cij"],
        }
    else:
        print("Simulating homogeneous VTI qS splitting model...")
        data = simulate()
        np.savez_compressed(
            data_file,
            rec_vx=data["rec_vx"],
            rec_vz=data["rec_vz"],
            rec_vy=data["rec_vy"],
            snap_vx=data["snap"]["vx"],
            snap_vz=data["snap"]["vz"],
            snap_vy=data["snap"]["vy"],
            cij=data["cij"],
            vp0=VP0,
            vs0=VS0,
            rho=RHO,
            epsilon=EPSILON,
            delta=DELTA,
            gamma=GAMMA,
            dx=DX,
            dz=DZ,
            dt=DT,
            f0=F0,
        )
        print(f"Saved data: {data_file}")
    print("Cij (GPa):", data["cij"] / 1e9)
    plot_wavefields(data)
    plot_records(data)
    print(f"Saved figures under {OUT_DIR}")


if __name__ == "__main__":
    main()
