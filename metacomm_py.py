# metacomm_py.py
# Python interface to metacomm C library using ctypes + NumPy

import os
import sys
import platform
import numpy as np
import ctypes as ct
from typing import Optional, Tuple
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import math

def export_metacomm_pdf(pdf_path: str,
                        e: np.ndarray,
                        P_final: np.ndarray,
                        rows: int,
                        cols: int,
                        species_labels=None,
                        presence_threshold: float | None = 1e-6):

    e2d = e.reshape(rows, cols)

    cmap = plt.cm.viridis.copy()
    cmap.set_bad((0.9, 0.9, 0.9, 1.0))

    n_sp, n_pch = P_final.shape
    assert n_pch == rows * cols

    if species_labels is None:
        species_labels = [str(i + 1) for i in range(n_sp)]

    # Presence/absence 与 richness
    thr = 1e-6 if presence_threshold is None else float(presence_threshold)
    pres = (P_final > thr).astype(float)
    richness = pres.sum(axis=0).reshape(rows, cols)


    e_var = float(np.var(e2d))
    unique_vals, counts = np.unique(e2d, return_counts=True)
    k_unique = int(len(unique_vals))

    shdi = None
    lsi = None
    if k_unique < 10:
        p = counts / counts.sum()
        shdi = float(-np.sum(p * np.log(p + 1e-12)))

        E = 0
        E += int(np.sum(e2d[:, :-1] != e2d[:, 1:]))
        E += int(np.sum(e2d[:-1, :] != e2d[1:, :]))
        # 外边界周长（与图幅大小有关）：上+下+左+右
        E += (rows * 2 + cols * 2)
        A = rows * cols
        lsi = float(E / (2.0 * math.sqrt(math.pi * A)))

    pool_total = n_sp
    observed_present = int(np.sum(pres.sum(axis=1) > 0))

    with PdfPages(pdf_path) as pdf:
        # Page 1: Summary
        plt.figure(figsize=(7, 6))
        plt.axis("off")
        lines = [
            f"Grid: {rows} x {cols} (cells = {rows*cols})",
            f"Species pool (total richness): {pool_total}",
            f"Observed species present (> {thr:g}): {observed_present}",
            f"Var(e): {e_var:.6g}",
            f"Unique values of e: {k_unique}",
        ]
        if shdi is not None and lsi is not None:
            lines += [f"SHDI: {shdi:.4f}", f"LSI: {lsi:.4f}"]
        plt.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace")
        pdf.savefig()
        plt.close()

        # Page 2: e map
        plt.figure(figsize=(6, 5))
        plt.title("Environmental layer (e)")
        plt.imshow(e2d, origin="upper", cmap=cmap, vmin=0.0, vmax=1.0)
        plt.colorbar(label="e")
        plt.tight_layout()
        pdf.savefig();
        plt.close()


        for i in range(n_sp):
            plt.figure(figsize=(6, 5))
            if presence_threshold is None:
                plt.title(f"Species {species_labels[i]} occupancy")
                plt.imshow(P_final[i].reshape(rows, cols), origin="upper",vmin=0.0, vmax=1.0)
                plt.colorbar(label="occupancy")
            else:
                plt.title(f"Species {species_labels[i]} occupancy")
                plt.imshow(P_final[i].reshape(rows, cols), origin="upper",vmin=0.0, vmax=1.0)
                plt.colorbar(label="occupancy")
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        # Final page: richness
        plt.figure(figsize=(6, 5))
        plt.title("Species richness")
        plt.imshow(richness, origin="upper", cmap=cmap)
        plt.colorbar(label="richness (#species)")
        plt.tight_layout()
        pdf.savefig();
        plt.close()

    print(f"PDF saved to: {pdf_path}")

# --------- Load shared library ---------
HERE = 'E:\\TheoEcoFramework\\CProj_TheoEco\\python'
LIB_NAMES = {
    'Linux': 'libmetacomm.so',
    'Darwin': 'libmetacomm.dylib',
    'Windows': 'metacomm.dll',
}
lib_name = LIB_NAMES.get(platform.system())
if lib_name is None:
    raise RuntimeError(f"Unsupported platform: {platform.system()}")

lib_path = os.path.join(HERE, lib_name)
lib = ct.CDLL(lib_path)

# --------- C types ---------
class CModel(ct.Structure):
    pass  # opaque

c_double_p = ct.POINTER(ct.c_double)

lib.mc_create_from_arrays.argtypes = [
    ct.c_int, ct.c_int, ct.c_int,   # rows, cols, n_sp
    c_double_p, c_double_p,         # e, rho (size rows*cols)
    c_double_p, c_double_p, c_double_p,  # mu, sigma2, c
    c_double_p, c_double_p,              # d, m
    c_double_p,                          # H (n_sp*n_sp)
    c_double_p,                          # P0 (n_sp*n_pch) or NULL
]
lib.mc_create_from_arrays.restype = ct.POINTER(CModel)

lib.mc_run.argtypes = [ct.POINTER(CModel), ct.c_int, ct.c_double, ct.c_double]
lib.mc_run.restype  = None

lib.mc_get_P.argtypes = [ct.POINTER(CModel), c_double_p]
lib.mc_get_P.restype  = None

lib.mc_free.argtypes = [ct.POINTER(CModel)]
lib.mc_free.restype  = None

# --------- Helper: numpy -> c pointer ---------
def _as_c_double_ptr(a: np.ndarray) -> c_double_p:
    assert a.dtype == np.float64 and a.flags['C_CONTIGUOUS']
    return a.ctypes.data_as(c_double_p)

# --------- High-level API ---------
def run_simulation(
        rows: int,
        cols: int,
        e: np.ndarray,
        rho: np.ndarray,
        mu: np.ndarray,
        sigma2: np.ndarray,
        c: np.ndarray,
        d: np.ndarray,
        m: np.ndarray,
        H: np.ndarray,
        P0: Optional[np.ndarray],
        steps: int = 10000,
        dt: float = 1,
        S: float = 1.0,
) -> np.ndarray:
    """Run the metacommunity simulation.

    Returns final occupancy matrix with shape (n_sp, rows*cols).
    """
    n_sp = int(mu.shape[0])
    n_pch = rows * cols

    # validate shapes/dtypes
    def _prep(x, shape):
        x = np.asarray(x, dtype=np.float64)
        if x.size != int(np.prod(shape)):
            raise ValueError(f"Expected {shape}, got {x.shape}")
        return np.ascontiguousarray(x.reshape(-1))

    e   = _prep(e,   (n_pch,))
    rho = _prep(rho, (n_pch,))
    mu = _prep(mu, (n_sp,))
    sigma2 = _prep(sigma2, (n_sp,))
    c  = _prep(c,  (n_sp,))
    d  = _prep(d,  (n_sp,))
    m  = _prep(m,  (n_sp,))

    H = np.asarray(H, dtype=np.float64)
    if H.shape != (n_sp, n_sp):
        raise ValueError(f"H must be ({n_sp},{n_sp}), got {H.shape}")
    H = np.ascontiguousarray(H.reshape(-1))

    if P0 is not None:
        P0 = _prep(P0, (n_sp, n_pch))
        p0_ptr = _as_c_double_ptr(P0)
    else:
        p0_ptr = None

    M = lib.mc_create_from_arrays(
        ct.c_int(rows), ct.c_int(cols), ct.c_int(n_sp),
        _as_c_double_ptr(e), _as_c_double_ptr(rho),
        _as_c_double_ptr(mu), _as_c_double_ptr(sigma2), _as_c_double_ptr(c),
        _as_c_double_ptr(d),  _as_c_double_ptr(m),
        _as_c_double_ptr(H),
        p0_ptr,
    )
    if not M:
        raise RuntimeError("mc_create_from_arrays failed (check MAX_* or shapes)")

    try:
        lib.mc_run(M, ct.c_int(steps), ct.c_double(dt), ct.c_double(S))
        out = np.empty((n_sp * n_pch,), dtype=np.float64)
        lib.mc_get_P(M, _as_c_double_ptr(out))
        return out.reshape(n_sp, n_pch)
    finally:
        lib.mc_free(M)



def make_rho_from_e(e: np.ndarray,
                    method: str = "inverse",     # "inverse" | "slope" | "constant"
                    rho_range=(0.0, 5.0),
                    constant_value: float = 1.0) -> np.ndarray:

    e = np.asarray(e, dtype=np.float64)

    # 先把 e 规范到 [0,1]
    emin, emax = np.nanmin(e), np.nanmax(e)
    if emax > emin:
        e01 = (e - emin) / (emax - emin)
    else:
        e01 = np.zeros_like(e)

    if method == "inverse":

        base = 1.0 - e01
    elif method == "slope":

        gx, gy = np.gradient(e.astype(np.float64))
        slope = np.hypot(gx, gy)
        smin, sptp = np.nanmin(slope), np.ptp(slope)
        base = (slope - smin) / (sptp + 1e-12)
    elif method == "constant":
        base = np.full_like(e, constant_value, dtype=np.float64)
    else:
        raise ValueError(f"Unknown method: {method}")

    rmin, rmax = rho_range
    bmin, bptp = np.nanmin(base), np.ptp(base)
    if bptp == 0:
        rho = np.full_like(base, rmin)
    else:
        rho = rmin + (base - bmin) / bptp * (rmax - rmin)
    return rho
# --------- Example usage & plotting ---------
if __name__ == "__main__":

    os.chdir('E:\\TheoEcoFramework')
    species_config_path = 'config_species_1.xlsx'
    spatial_config_path = 'config_spatial_2.xlsx'


    species_config = pd.read_excel(species_config_path,  index_col=0)
    species_config = species_config.loc[:, sorted(species_config.columns, key=lambda x: int(x))]


    import matplotlib.pyplot as plt

    rows, cols = 128, 128
    n_sp = 4
    n_pch = rows * cols

    spatial_df = pd.read_excel(spatial_config_path, sheet_name='E', index_col=0)
    spatial_df = spatial_df.apply(pd.to_numeric, errors='coerce')  # 强制数值
    assert spatial_df.shape == (rows, cols), f"spatial_config expect ({rows},{cols}), but get {spatial_df.shape}"
    e = spatial_df.to_numpy(dtype=np.float64)


    mask = np.isfinite(e)
    e_model = np.where(mask, e, -1.0)

    resistance_config_path = 'config_resistance_2.xlsx'

    _xls = pd.ExcelFile(resistance_config_path)
    _sheet = None
    for cand in ('rho', 'R', 'resistance', 'RHO', 'RESISTANCE'):
        if cand in _xls.sheet_names:
            _sheet = cand
            break
    if _sheet is None:
        _sheet = _xls.sheet_names[0]

    rho_df = pd.read_excel(resistance_config_path, sheet_name=_sheet, index_col=0)
    rho_df = rho_df.apply(pd.to_numeric, errors='coerce')
    assert rho_df.shape == (rows, cols), f"resistance_config expect ({rows},{cols}), but get {rho_df.shape}"

    rho_raw = rho_df.to_numpy(dtype=np.float64)

    rho_in = np.where(mask, rho_raw, np.nan)
    inside = mask & np.isfinite(rho_raw)
    if inside.sum() == 0:
        raise ValueError("Resistance invalid, check config_resistance_2.xlsx。")
    fill_val = float(np.nanmedian(rho_raw[inside]))
    rho_in[mask & ~np.isfinite(rho_in)] = fill_val

    DO_MINMAX_TO_0_5 = False
    if DO_MINMAX_TO_0_5:
        rmin = float(np.nanmin(rho_in[mask]))
        rmax = float(np.nanmax(rho_in[mask]))
        if rmax > rmin:
            rho_in = (rho_in - rmin) / (rmax - rmin) * 5.0
        else:
            rho_in = np.zeros_like(rho_in)

    rho = np.where(mask, rho_in, 1e6)

    rng = np.random.default_rng(42)
    P0 = rng.uniform(0.0, 1e-3, size=(n_sp, rows * cols))
    P0[:, ~mask.ravel()] = 0.0

    print(f"e ∈ [{np.nanmin(e_model):.3f}, {np.nanmax(e_model):.3f}] | "
          f"rho(in-mask) ∈ [{np.nanmin(rho[mask]):.3f}, {np.nanmax(rho[mask]):.3f}]")

    print(f"e ∈ [{e.min():.3f}, {e.max():.3f}] | rho ∈ [{rho.min():.3f}, {rho.max():.3f}]")

    # Environment & resistance -- DEFAULT
    # yy, xx = np.mgrid[0:rows, 0:cols]
    # e = (xx / (cols - 1)).astype(np.float64)  # 0..1 eastward gradient
    # rho = np.ones((rows, cols), dtype=np.float64) * 1.0

    # Species pool: mu spaced along [0,1], narrow-ish niches
    mu = species_config.loc["mu"].to_numpy(dtype=np.float64)
    sigma2 = species_config.loc["sigma2"].to_numpy(dtype=np.float64)

    # DEFAULT SETTINGS
    #c = np.full(n_sp, 0.4)
    #d = np.full(n_sp, 0.03)
    #m = np.full(n_sp, 0.05)

    # LOAD FROM CONFIGURATION


    c = species_config.loc["c"].to_numpy(dtype=np.float64)
    d = species_config.loc["d"].to_numpy(dtype=np.float64)
    m = species_config.loc["m"].to_numpy(dtype=np.float64)

    # Asymmetric H example: random in [0,1), zero diag
    rng = np.random.default_rng(42)
    H = rng.random((n_sp, n_sp))
    np.fill_diagonal(H, 0.0)


    P_final = run_simulation(
        rows, cols,
        e_model.reshape(-1), rho.reshape(-1),
        mu, sigma2, c, d, m,
        H,
        P0,
        steps=1000
        , dt=0.1, S=1.0,
    )


    labels = [str(c) for c in species_config.columns]
    export_metacomm_pdf("metacomm_report.pdf",
                        e.reshape(-1),
                        P_final,
                        rows, cols,
                        species_labels=labels,
                        presence_threshold=0.1)

    print("Done.")
