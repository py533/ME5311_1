# analysis.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:
    from scipy.signal import welch
except Exception:  # pragma: no cover
    welch = None


def compute_mean_field(data: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Compute time-mean field over selected frames."""
    return np.mean(np.asarray(data[idx], dtype=np.float64), axis=0)


def compute_fluctuation(data: np.ndarray, mean_field: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Compute fluctuation snapshots u' = u - mean(u)."""
    snaps = np.asarray(data[idx], dtype=np.float64)
    return snaps - mean_field[None, ...]


def snapshot_energy(fluct_snaps: np.ndarray) -> np.ndarray:
    """Compute energy proxy per snapshot: mean(|u'|^2)."""
    ux = fluct_snaps[..., 0]
    uy = fluct_snaps[..., 1]
    return np.mean(ux * ux + uy * uy, axis=(1, 2))


def build_data_matrix(fluct_snaps: np.ndarray, dtype=np.float32) -> np.ndarray:
    """Flatten vector field snapshots into matrix X (M, F).

    Order: [ux(:), uy(:)] concatenated.
    """
    M, ny, nx, nc = fluct_snaps.shape
    assert nc == 2
    ux = fluct_snaps[..., 0].reshape(M, ny * nx)
    uy = fluct_snaps[..., 1].reshape(M, ny * nx)
    X = np.concatenate([ux, uy], axis=1).astype(dtype, copy=False)
    return X


@dataclass
class PODResult:
    modes: np.ndarray       # (K, F) spatial modes (orthonormal)
    coeffs: np.ndarray      # (M, K) temporal coefficients a_r(t)
    svals: np.ndarray       # (K,) singular values
    energy: np.ndarray      # (K,) normalized modal energy (fraction)
    cum_energy: np.ndarray  # (K,) cumulative energy


def compute_pod(X: np.ndarray, num_modes: int = 50) -> PODResult:
    """Compute POD via economy SVD on snapshot matrix X (M, F).

    X = U S V^T
    spatial modes = rows of V^T -> V (F,K); we store modes as (K,F).
    coefficients = U S -> (M,K)
    """
    X = np.asarray(X, dtype=np.float64)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    K = min(int(num_modes), Vt.shape[0])

    modes = Vt[:K, :]                # (K,F)
    coeffs = (U[:, :K] * S[:K])      # (M,K)

    lam = S[:K] ** 2
    energy = lam / np.sum(S ** 2)
    cum_energy = np.cumsum(energy)

    return PODResult(
        modes=modes,
        coeffs=coeffs,
        svals=S[:K],
        energy=energy,
        cum_energy=cum_energy,
    )


def mode_vector_to_field(mode_vec: np.ndarray, grid_n: int = 64) -> np.ndarray:
    """Convert a flattened mode vector (F,) back to (N,N,2) field."""
    F = mode_vec.size
    half = F // 2
    ux = mode_vec[:half].reshape(grid_n, grid_n)
    uy = mode_vec[half:].reshape(grid_n, grid_n)
    return np.stack([ux, uy], axis=-1)


def reconstruct_from_modes(a: np.ndarray, modes: np.ndarray) -> np.ndarray:
    """Reconstruct flattened field vec from coefficients and modes.

    a: (K,), modes: (K,F)
    return: (F,)
    """
    return np.sum(a[:, None] * modes, axis=0)


def normalized_reconstruction_error(true_field: np.ndarray, recon_field: np.ndarray) -> float:
    """Compute relative L2 error ||u-u_hat|| / ||u|| for one snapshot field."""
    true_field = np.asarray(true_field, dtype=np.float64)
    recon_field = np.asarray(recon_field, dtype=np.float64)
    denom = float(np.linalg.norm(true_field.ravel()))
    if denom <= 0 or not np.isfinite(denom):
        return float("nan")
    num = float(np.linalg.norm((true_field - recon_field).ravel()))
    return num / denom


def temporal_psd_from_signal(x: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute one-sided PSD using rFFT (no windowing)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    x = x - np.mean(x)
    X = np.fft.rfft(x)
    psd = (np.abs(X) ** 2) / n
    freqs = np.fft.rfftfreq(n, d=dt)
    return freqs, psd


def dominant_frequency_from_psd(
    freqs: np.ndarray,
    psd: np.ndarray,
    fmin: float = 1e-12,
) -> tuple[float, float]:
    """Return dominant frequency and its PSD value above fmin."""
    freqs = np.asarray(freqs, dtype=np.float64)
    psd = np.asarray(psd, dtype=np.float64)
    mask = np.isfinite(freqs) & np.isfinite(psd) & (freqs >= float(fmin))
    if not np.any(mask):
        return float("nan"), float("nan")
    i_local = int(np.argmax(psd[mask]))
    f_dom = float(freqs[mask][i_local])
    p_dom = float(psd[mask][i_local])
    return f_dom, p_dom


def mean_energy_spectrum_2d(
    data: np.ndarray,
    idx: np.ndarray,
    dx: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute time-averaged 2D energy spectrum E(kx,ky) (SHIFTED).

    E2D = < |FFT(ux)|^2 + |FFT(uy)|^2 >_t
    Returns:
      E2D_shifted: (N,N)
      kx_shifted:  (N,)
      ky_shifted:  (N,)
    """
    snaps = np.asarray(data[idx], dtype=np.float64)  # (M,N,N,2)
    ux = snaps[..., 0]
    uy = snaps[..., 1]

    Fux = np.fft.fft2(ux, axes=(1, 2))
    Fuy = np.fft.fft2(uy, axes=(1, 2))

    E2D = np.mean(np.abs(Fux) ** 2 + np.abs(Fuy) ** 2, axis=0)
    E2D = np.fft.fftshift(E2D)

    N = E2D.shape[0]
    k = np.fft.fftfreq(N, d=dx)  # cycles per unit
    k_shift = np.fft.fftshift(k)
    return E2D, k_shift, k_shift


def find_discrete_peak_from_2d(
    E2D_shifted: np.ndarray,
    kx_1d: np.ndarray,
    ky_1d: np.ndarray,
    exclude_center: int = 2,
) -> dict:
    """Find the strongest discrete peak away from DC in shifted 2D spectrum."""
    S = np.asarray(E2D_shifted, dtype=np.float64).copy()
    ny, nx = S.shape
    cy, cx = ny // 2, nx // 2

    r = int(exclude_center)
    S[cy - r : cy + r + 1, cx - r : cx + r + 1] = -np.inf

    iy, ix = np.unravel_index(int(np.argmax(S)), S.shape)
    peak_energy = float(E2D_shifted[iy, ix])
    peak_kx = float(kx_1d[ix])
    peak_ky = float(ky_1d[iy])
    peak_kmag = float(np.hypot(peak_kx, peak_ky))

    return {
        "peak_ix": int(ix),
        "peak_iy": int(iy),
        "peak_kx": peak_kx,
        "peak_ky": peak_ky,
        "peak_kmag": peak_kmag,
        "peak_energy": peak_energy,
    }


def isotropic_spectrum_from_2d(
    E2D_shifted: np.ndarray,
    kx_1d: np.ndarray,
    ky_1d: np.ndarray,
    n_bins: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Radially average E2D_shifted into E(k).

    Returns:
      k_centers, E_k_mean, k_peak (excluding k=0)
    """
    E = np.asarray(E2D_shifted, dtype=np.float64)
    N = E.shape[0]
    if n_bins is None:
        n_bins = N // 2

    KX, KY = np.meshgrid(kx_1d, ky_1d)
    Kr = np.sqrt(KX**2 + KY**2)

    kmax = float(np.max(Kr))
    edges = np.linspace(0.0, kmax, int(n_bins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    E_k = np.zeros_like(centers)
    counts = np.zeros_like(centers)

    flatK = Kr.ravel()
    flatE = E.ravel()
    bin_id = np.searchsorted(edges, flatK, side="right") - 1
    valid = (bin_id >= 0) & (bin_id < len(centers)) & np.isfinite(flatE)

    for b, val in zip(bin_id[valid], flatE[valid]):
        E_k[b] += val
        counts[b] += 1.0

    counts[counts == 0] = np.nan
    E_k = E_k / counts

    # peak excluding k ~ 0
    mask = np.isfinite(E_k) & (centers > 0)
    if np.any(mask):
        i_peak = int(np.nanargmax(E_k[mask]))
        k_peak = float(centers[mask][i_peak])
    else:
        k_peak = float("nan")
    return centers, E_k, k_peak


def segmented_isotropic_spectra(
    data: np.ndarray,
    idx: np.ndarray,
    dx: float,
    n_segments: int = 3,
) -> dict:
    """Compute isotropic spectra for segments and their mean, plus stability diagnostics."""
    idx = np.asarray(idx, dtype=int)
    M = len(idx)
    n_segments = max(1, int(n_segments))
    seg_edges = np.linspace(0, M, n_segments + 1, dtype=int)

    # Use total spectrum bins as common grid
    E2D_total, kx, ky = mean_energy_spectrum_2d(data, idx, dx=dx)
    k_centers, E_mean, k_peak_mean = isotropic_spectrum_from_2d(E2D_total, kx, ky)

    E_segs = []
    k_peaks = []

    for s in range(n_segments):
        a = int(seg_edges[s])
        b = int(seg_edges[s + 1])
        seg_idx = idx[a:b]
        if len(seg_idx) < 2:
            continue

        E2D_seg, kx_s, ky_s = mean_energy_spectrum_2d(data, seg_idx, dx=dx)
        k_c, E_k, k_peak = isotropic_spectrum_from_2d(E2D_seg, kx_s, ky_s, n_bins=len(k_centers))

        # Interpolate E_k onto common k_centers if needed
        if k_c.shape != k_centers.shape or np.any(np.abs(k_c - k_centers) > 1e-12):
            E_k_interp = np.interp(k_centers, k_c, np.nan_to_num(E_k, nan=0.0))
            E_k = E_k_interp

        E_segs.append(E_k)
        k_peaks.append(k_peak)

    E_segs = np.asarray(E_segs, dtype=np.float64) if len(E_segs) > 0 else np.zeros((0, len(k_centers)))
    k_peaks = np.asarray(k_peaks, dtype=np.float64) if len(k_peaks) > 0 else np.zeros((0,))

    return {
        "kx_1d": kx,
        "ky_1d": ky,
        "k_bin_centers": k_centers,
        "E_k_mean": E_mean,
        "E_k_segments": E_segs,
        "k_peaks_segments": k_peaks,
        "k_peak_mean": k_peak_mean,
        "E2D_total": E2D_total,
    }


def spectrum_axis_cuts(E2D_shifted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract axis cuts through DC from shifted 2D spectrum."""
    E = np.asarray(E2D_shifted, dtype=np.float64)
    cy = E.shape[0] // 2
    cx = E.shape[1] // 2
    cut_kx = E[cy, :]
    cut_ky = E[:, cx]
    return cut_kx, cut_ky


def anisotropy_ratio_from_cuts(
    kx_1d: np.ndarray,
    ky_1d: np.ndarray,
    cut_kx: np.ndarray,
    cut_ky: np.ndarray,
    exclude_dc: int = 2,
) -> float:
    """Compute a simple anisotropy metric: integral energy on kx-axis over ky-axis.

    Uses numpy.trapezoid for numerical integration (compatible with newer NumPy).
    """
    kx_1d = np.asarray(kx_1d, dtype=np.float64)
    ky_1d = np.asarray(ky_1d, dtype=np.float64)
    cut_kx = np.asarray(cut_kx, dtype=np.float64)
    cut_ky = np.asarray(cut_ky, dtype=np.float64)

    cx = len(kx_1d) // 2
    cy = len(ky_1d) // 2

    # Exclude a centered DC neighborhood in index space.
    r = int(max(0, exclude_dc))
    mask_x = np.ones_like(cut_kx, dtype=bool)
    mask_y = np.ones_like(cut_ky, dtype=bool)
    mask_x[max(0, cx - r) : min(len(mask_x), cx + r + 1)] = False
    mask_y[max(0, cy - r) : min(len(mask_y), cy + r + 1)] = False

    kx_use = np.asarray(kx_1d[mask_x], dtype=np.float64)
    ky_use = np.asarray(ky_1d[mask_y], dtype=np.float64)
    ex_use = np.maximum(np.asarray(cut_kx[mask_x], dtype=np.float64), 0.0)
    ey_use = np.maximum(np.asarray(cut_ky[mask_y], dtype=np.float64), 0.0)

    # Fold +/-k onto |k| and average duplicated bins to avoid non-monotonic x.
    def _fold_abs(k: np.ndarray, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        kab = np.round(np.abs(k), decimals=12)
        u, inv = np.unique(kab, return_inverse=True)
        e_sum = np.bincount(inv, weights=e, minlength=len(u)).astype(np.float64)
        cnt = np.bincount(inv, minlength=len(u)).astype(np.float64)
        cnt[cnt == 0] = np.nan
        e_mean = e_sum / cnt
        good = np.isfinite(u) & np.isfinite(e_mean)
        return u[good], e_mean[good]

    x_x, y_x = _fold_abs(kx_use, ex_use)
    x_y, y_y = _fold_abs(ky_use, ey_use)

    if x_x.size < 2 or x_y.size < 2:
        return float("nan")

    Ex = float(np.trapezoid(y_x, x=x_x))
    Ey = float(np.trapezoid(y_y, x=x_y))
    if not np.isfinite(Ex) or not np.isfinite(Ey) or Ey <= 0:
        return float("nan")
    return Ex / Ey


def anisotropy_peak_ratio_from_cuts(
    kx_1d: np.ndarray,
    ky_1d: np.ndarray,
    cut_kx: np.ndarray,
    cut_ky: np.ndarray,
    exclude_dc: int = 2,
) -> float:
    """Peak-based anisotropy ratio: max(E_kx) / max(E_ky) after DC exclusion."""
    kx_1d = np.asarray(kx_1d, dtype=np.float64)
    ky_1d = np.asarray(ky_1d, dtype=np.float64)
    cut_kx = np.asarray(cut_kx, dtype=np.float64)
    cut_ky = np.asarray(cut_ky, dtype=np.float64)

    cx = len(kx_1d) // 2
    cy = len(ky_1d) // 2
    r = int(max(0, exclude_dc))

    mask_x = np.ones_like(cut_kx, dtype=bool)
    mask_y = np.ones_like(cut_ky, dtype=bool)
    mask_x[max(0, cx - r): min(len(mask_x), cx + r + 1)] = False
    mask_y[max(0, cy - r): min(len(mask_y), cy + r + 1)] = False

    ex = np.asarray(cut_kx[mask_x], dtype=np.float64)
    ey = np.asarray(cut_ky[mask_y], dtype=np.float64)
    ex = ex[np.isfinite(ex) & (ex > 0)]
    ey = ey[np.isfinite(ey) & (ey > 0)]
    if ex.size == 0 or ey.size == 0:
        return float("nan")
    mx = float(np.max(ex))
    my = float(np.max(ey))
    if not np.isfinite(mx) or not np.isfinite(my) or my <= 0:
        return float("nan")
    return mx / my


def fit_power_law_slope(
    k: np.ndarray,
    E: np.ndarray,
    kmin: float,
    kmax: float,
) -> float:
    """Fit slope in log-log: log(E)=a+s*log(k), return s."""
    k = np.asarray(k, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    m = np.isfinite(k) & np.isfinite(E) & (k > 0) & (E > 0) & (k >= kmin) & (k <= kmax)
    if np.count_nonzero(m) < 3:
        return float("nan")
    x = np.log(k[m])
    y = np.log(E[m])
    s, _ = np.polyfit(x, y, 1)
    return float(s)


def two_segment_slopes_from_spectrum(
    k: np.ndarray,
    E: np.ndarray,
) -> dict:
    """Estimate two log-log slope segments by splitting valid k-range at geometric midpoint."""
    k = np.asarray(k, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    m = np.isfinite(k) & np.isfinite(E) & (k > 0) & (E > 0)
    if np.count_nonzero(m) < 8:
        return {
            "slope_seg1": float("nan"), "kmin_seg1": float("nan"), "kmax_seg1": float("nan"),
            "slope_seg2": float("nan"), "kmin_seg2": float("nan"), "kmax_seg2": float("nan"),
        }

    kv = np.asarray(k[m], dtype=np.float64)
    Ev = np.asarray(E[m], dtype=np.float64)
    order = np.argsort(kv)
    kv = kv[order]
    Ev = Ev[order]

    kmin = float(kv[0])
    kmax = float(kv[-1])
    ksplit = float(np.sqrt(kmin * kmax))

    m1 = kv <= ksplit
    m2 = kv >= ksplit
    if np.count_nonzero(m1) < 3 or np.count_nonzero(m2) < 3:
        imid = len(kv) // 2
        m1 = np.zeros_like(kv, dtype=bool)
        m2 = np.zeros_like(kv, dtype=bool)
        m1[:imid] = True
        m2[imid:] = True

    kmin1 = float(kv[m1][0])
    kmax1 = float(kv[m1][-1])
    kmin2 = float(kv[m2][0])
    kmax2 = float(kv[m2][-1])

    return {
        "slope_seg1": fit_power_law_slope(kv, Ev, kmin1, kmax1),
        "kmin_seg1": kmin1,
        "kmax_seg1": kmax1,
        "slope_seg2": fit_power_law_slope(kv, Ev, kmin2, kmax2),
        "kmin_seg2": kmin2,
        "kmax_seg2": kmax2,
    }


def peak_background_median_from_2d(
    E2D_shifted: np.ndarray,
    peak_ix: int,
    peak_iy: int,
    inner_radius: float = 1.5,
    outer_radius: float = 4.5,
) -> float:
    """Background estimate around peak via annulus median."""
    E = np.asarray(E2D_shifted, dtype=np.float64)
    ny, nx = E.shape
    yy, xx = np.indices((ny, nx))
    rr = np.sqrt((xx - float(peak_ix)) ** 2 + (yy - float(peak_iy)) ** 2)
    mask = (rr >= float(inner_radius)) & (rr <= float(outer_radius)) & np.isfinite(E)
    vals = E[mask]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return float("nan")
    return float(np.median(vals))


def temporal_psd_welch_from_signal(
    x: np.ndarray,
    dt: float,
    nperseg: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute one-sided PSD using Welch; fallback to rFFT PSD if SciPy is unavailable."""
    x = np.asarray(x, dtype=np.float64)
    fs = 1.0 / float(dt)
    if welch is None:
        return temporal_psd_from_signal(x, dt=dt)

    if nperseg is None:
        nperseg = min(256, max(32, x.size // 4))
    nperseg = int(max(8, min(nperseg, x.size)))
    freqs, psd = welch(x, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2, detrend="constant")
    return np.asarray(freqs, dtype=np.float64), np.asarray(psd, dtype=np.float64)
