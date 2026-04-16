# main.py
from __future__ import annotations

import numpy as np

from config import CFG
from load_data import load_vector_field, make_frame_indices
import analysis as A
import plot as P


def main() -> None:
    CFG.figs_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Load ----------
    data = load_vector_field(CFG.data_path, grid_n=CFG.grid_n, n_components=CFG.n_components, mmap=True)
    n_frames = int(data.shape[0])

    # ---------- Indices ----------
    idx_svd = make_frame_indices(n_frames, stride=CFG.stride, max_frames=CFG.max_frames_svd)
    idx_fft = make_frame_indices(n_frames, stride=CFG.stride, max_frames=CFG.max_frames_fft)

    # ---------- Mean & fluctuation for POD ----------
    mean_field = A.compute_mean_field(data, idx_svd)
    fluct = A.compute_fluctuation(data, mean_field, idx_svd)  # (M,N,N,2)

    # Pick a representative snapshot for "True vs Recon"
    e = A.snapshot_energy(fluct)
    pick_local = int(np.argmax(e))
    fluct_pick = fluct[pick_local]  # (N,N,2)

    # ---------- POD ----------
    X = A.build_data_matrix(fluct, dtype=np.float32)  # (M,F)
    pod_res = A.compute_pod(X, num_modes=CFG.num_modes_svd)

    # Energy ranks for compactness evidence
    r90 = int(np.searchsorted(pod_res.cum_energy, 0.90) + 1)
    r95 = int(np.searchsorted(pod_res.cum_energy, 0.95) + 1)
    r99 = int(np.searchsorted(pod_res.cum_energy, 0.99) + 1)

    # Prepare POD dict for plot
    pod_dict = {
        "energy": pod_res.energy,
        "energy_cum": pod_res.cum_energy,
    }

    # Mode fields (1–4) to dict list with |mode|
    pod_modes = []
    for i in range(CFG.num_modes_show):
        field = A.mode_vector_to_field(pod_res.modes[i], grid_n=CFG.grid_n)
        umag = np.sqrt(field[..., 0] ** 2 + field[..., 1] ** 2)
        pod_modes.append({"mode_idx": i, "umag": umag})

    # ---------- Reconstruction using modes 1–4 at selected snapshot ----------
    a_pick = pod_res.coeffs[pick_local, :CFG.num_modes_show]

    # IMPORTANT: recon_vec is 1D in the SAME ordering as modes / data_matrix
    recon_vec = A.reconstruct_from_modes(a_pick, pod_res.modes[:CFG.num_modes_show, :])

    # FIX: do NOT reshape directly; use the same inverse mapping as modes
    recon_pick = A.mode_vector_to_field(recon_vec, grid_n=CFG.grid_n)  # (N,N,2)
    recon_relerr = A.normalized_reconstruction_error(fluct_pick, recon_pick)
    recon_rmse = float(np.sqrt(np.mean((np.asarray(fluct_pick, dtype=np.float64) - np.asarray(recon_pick, dtype=np.float64)) ** 2)))

    # Pass magnitude explicitly to avoid any mismatch inside plot
    true_umag = np.sqrt(fluct_pick[..., 0] ** 2 + fluct_pick[..., 1] ** 2)
    recon_umag = np.sqrt(recon_pick[..., 0] ** 2 + recon_pick[..., 1] ** 2)
    recon_dict = {
        "true": fluct_pick,
        "recon": recon_pick,
        "true_umag": true_umag,
        "recon_umag": recon_umag,
    }

    # ---------- 2D spectrum + isotropic + segments ----------
    seg = A.segmented_isotropic_spectra(data, idx_fft, dx=CFG.dx, n_segments=CFG.n_segments)
    E2D = seg["E2D_total"]
    kx = seg["kx_1d"]
    ky = seg["ky_1d"]

    peak = A.find_discrete_peak_from_2d(E2D, kx, ky, exclude_center=2)

    k_centers = seg["k_bin_centers"]
    E_mean = seg["E_k_mean"]
    E_segs = seg["E_k_segments"]
    k_peaks = seg["k_peaks_segments"]

    k_peak_mean = float(seg.get("k_peak_mean", np.nan))
    finite_kp = np.asarray(k_peaks, dtype=float)
    finite_kp = finite_kp[np.isfinite(finite_kp) & (finite_kp > 0)]
    k_peak_std = float(np.std(finite_kp)) if finite_kp.size else float("nan")
    k_peak_cov = float(k_peak_std / np.mean(finite_kp)) if finite_kp.size else float("nan")

    # Axis cuts + anisotropy ratio
    axis_cut_exclude_dc = 2
    cut_kx, cut_ky = A.spectrum_axis_cuts(E2D)
    ratio = A.anisotropy_ratio_from_cuts(kx, ky, cut_kx, cut_ky, exclude_dc=axis_cut_exclude_dc)

    spec_dict = {
        "E2D": E2D,
        "kx_1d": kx,
        "ky_1d": ky,
        "cut_kx": cut_kx,
        "cut_ky": cut_ky,
    }
    radial_dict = {
        "k_bin_centers": k_centers,
        "E_k_mean": E_mean,
        "E_k_segments": E_segs,
        "k_peaks_segments": k_peaks,
    }
    aniso_dict = {"ratio_kx_over_ky": ratio, "exclude_dc": axis_cut_exclude_dc}

    # ---------- Temporal PSD of POD coefficients a1..a4 ----------
    dt_eff = CFG.dt * CFG.stride
    freqs_list = []
    psd_list = []
    for i in range(CFG.num_modes_show):
        freqs, psd = A.temporal_psd_from_signal(pod_res.coeffs[:, i], dt=dt_eff)
        freqs_list.append(freqs)
        psd_list.append(psd)
    freqs_a = freqs_list[0]
    psd_arr = np.vstack(psd_list)  # (4, nf)

    temporal_psd = {"freqs": freqs_a, "psd": psd_arr, "n_modes": CFG.num_modes_show}

    # Dominant temporal frequencies per leading mode
    dom_freqs = []
    for i in range(CFG.num_modes_show):
        fi, _ = A.dominant_frequency_from_psd(freqs_list[i], psd_list[i], fmin=1e-6)
        dom_freqs.append(fi)

    # Welch PSD for metrics table (recommended for robust peak frequency)
    f_peak_modes = []
    t_star_modes = []
    for i in range(CFG.num_modes_show):
        fw, pw = A.temporal_psd_welch_from_signal(pod_res.coeffs[:, i], dt=dt_eff)
        fi_w, _ = A.dominant_frequency_from_psd(fw, pw, fmin=1e-6)
        f_peak_modes.append(float(fi_w))
        t_star_modes.append(float(1.0 / fi_w) if np.isfinite(fi_w) and fi_w > 0 else float("nan"))

    # Project-level metrics for figure annotation and report
    peak_kmag = float(peak.get("peak_kmag", np.nan))
    lambda_star = float(1.0 / peak_kmag) if np.isfinite(peak_kmag) and peak_kmag > 0 else float("nan")
    metrics = {
        "r90": r90,
        "r95": r95,
        "recon_relerr": recon_relerr,
        "peak_kmag": peak_kmag,
        "lambda_star": lambda_star,
        "k_peak_mean": k_peak_mean,
        "k_peak_std": k_peak_std,
        "k_peak_cov": k_peak_cov,
        "anisotropy_ratio": ratio,
        "dom_freqs": np.asarray(dom_freqs, dtype=float),
        "dt_eff": dt_eff,
    }

    # ---------- Plot one figure ----------
    out_path = CFG.figs_dir / "q1_q4_summary_B.png"
    P.plot_one_figure_B(
        pod=pod_dict,
        spec=spec_dict,
        peak=peak,
        radial=radial_dict,
        out_path=out_path,
        title="ME5311 Project 1: Consolidated Results (Q1–Q4)",
        pod_modes=pod_modes,
        recon=recon_dict,
        temporal_psd=temporal_psd,
        aniso=aniso_dict,
        k_window=CFG.k_window,
        metrics=metrics,
    )

    # ---------- Save concise text summary for report writing ----------
    summary_path = CFG.figs_dir / "analysis_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("ME5311 Project 1 — Quantitative Summary (Q1–Q4)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Frames used for POD: {len(idx_svd)} (stride={CFG.stride})\n")
        f.write(f"Frames used for spectra: {len(idx_fft)} (stride={CFG.stride})\n")
        f.write("\n[Q1] POD dominant structures\n")
        f.write(f"  Modes for 90% energy: r90 = {r90}\n")
        f.write(f"  Modes for 95% energy: r95 = {r95}\n")
        f.write(f"  Relative reconstruction error (modes 1–{CFG.num_modes_show}): {recon_relerr:.4f}\n")
        f.write("\n[Q2/Q3] Spatial scales and forced periodic signature\n")
        f.write(f"  2D peak wavenumber magnitude |k*|: {peak_kmag:.6g}\n")
        f.write(f"  Characteristic wavelength λ* = 1/|k*|: {lambda_star:.6g}\n")
        f.write(f"  Segment-mean isotropic-peak k: {k_peak_mean:.6g}\n")
        f.write(f"  Segment peak std: {k_peak_std:.6g}\n")
        f.write(f"  Segment peak CoV: {k_peak_cov:.4f}\n")
        f.write("\n[Q4] Spatio-temporal anisotropy and temporal scales\n")
        f.write(f"  Anisotropy ratio Ex/Ey (axis cuts): {ratio:.6g}\n")
        for i, fi in enumerate(dom_freqs, start=1):
            f.write(f"  Dominant frequency of POD mode {i}: {fi:.6g}\n")

    print("Saved:", out_path.resolve())
    print("Saved summary:", summary_path.resolve())


if __name__ == "__main__":
    main()
