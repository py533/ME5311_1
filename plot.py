# plot.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LogLocator, LogFormatterMathtext


def _smooth_1d(y: np.ndarray, win: int = 11) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    win = int(max(3, win))
    if win % 2 == 0:
        win += 1
    if y.size < win:
        return y
    pad = win // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=float) / float(win)
    return np.convolve(ypad, kernel, mode="valid")


def _fold_abs_curve(k: np.ndarray, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fold signed axis spectrum onto |k| and average duplicated bins."""
    k = np.asarray(k, dtype=float)
    e = np.asarray(e, dtype=float)
    mask = np.isfinite(k) & np.isfinite(e)
    k = k[mask]
    e = e[mask]
    if k.size == 0:
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)

    k_abs = np.round(np.abs(k), decimals=12)
    u, inv = np.unique(k_abs, return_inverse=True)
    e_sum = np.bincount(inv, weights=e, minlength=len(u)).astype(float)
    cnt = np.bincount(inv, minlength=len(u)).astype(float)
    cnt[cnt == 0] = np.nan
    e_mean = e_sum / cnt

    good = np.isfinite(u) & np.isfinite(e_mean)
    return u[good], e_mean[good]


def _align_1d_pair(
    x: np.ndarray,
    y: np.ndarray,
    x_name: str,
    y_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Align two 1-D arrays safely and emit warning if shape is inconsistent."""
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if x_arr.size == y_arr.size:
        return x_arr, y_arr

    n = min(x_arr.size, y_arr.size)
    print(
        f"[plot warning] {x_name} (n={x_arr.size}) and {y_name} (n={y_arr.size}) "
        f"size mismatch; truncating to n={n}."
    )
    if n <= 0:
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)
    return x_arr[:n], y_arr[:n]


def _imshow(ax, fig, img, title: str, cmap: str = "viridis",
           vmin=None, vmax=None, cbar: bool = True):
    im = ax.imshow(img, origin="lower", aspect="equal",
                   cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=6, fontweight="bold", pad=2)
    ax.tick_params(direction="in", length=1.8, width=0.5, labelsize=5)
    if cbar:
        cb = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.025)
        cb.ax.tick_params(labelsize=5)
    return im


def _panel_label(ax, label: str, position: str = "top-left") -> None:
    if position == "top-right":
        x, ha = 0.98, "right"
    else:
        x, ha = 0.02, "left"
    ax.text(
        x,
        0.98,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha=ha,
        va="top",
        bbox=dict(boxstyle="round,pad=0.12", fc="white", alpha=0.88, lw=0.25),
    )


def plot_one_figure_B(
    pod: dict,
    spec: dict,
    peak: dict,
    radial: dict,
    out_path: str | Path,
    title: str = "ME5311 Project 1: Consolidated Results (Q1–Q4)",
    pod_modes: list[dict] | None = None,
    recon: dict | None = None,
    temporal_psd: dict | None = None,
    aniso: dict | None = None,
    k_window: float | None = None,
    metrics: dict | None = None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Publication style (Science/Nature-like readability) ----
    style_rc = {
        "font.family": "Arial",
        "font.size": 7,
        "axes.linewidth": 0.5,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "axes.titleweight": "bold",
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "legend.fontsize": 5,
        "legend.title_fontsize": 5,
        "legend.frameon": True,
        "legend.fancybox": True,
        "legend.framealpha": 0.8,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.4,
        "legend.borderpad": 0.3,
        "legend.labelspacing": 0.25,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.22,
        "figure.dpi": 200,
        "savefig.dpi": 600,
        "savefig.format": "png",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.default": "regular",
    }
    _style_ctx = matplotlib.rc_context(rc=style_rc)
    _style_ctx.__enter__()

    cmap_all = "cividis"  # print-friendly, perceptually uniform
    k_plot_scale = 1.0

    # ========= Main layout: 1 row x 2 cols (left column equalized to 4 panels) =========
    fig = plt.figure(figsize=(7.1, 8.9))
    gs_main = fig.add_gridspec(
        1, 2,
        width_ratios=[1.0, 1.0],
        left=0.08,
        right=0.985,
        top=0.965,
        bottom=0.07,
        wspace=0.28,
    )

    gs_left = gs_main[0, 0].subgridspec(4, 1, hspace=0.34)
    gs_right = gs_main[0, 1].subgridspec(3, 1, hspace=0.28)

    # =========================================================
    # Row 1 Col 1: (Q1a) POD energy (truncate but show early modes clearly)
    # =========================================================
    ax_energy = fig.add_subplot(gs_left[0, 0])
    energy_cum = np.asarray(pod.get("energy_cum", []), dtype=float)
    energy = np.asarray(pod.get("energy", []), dtype=float)
    r = np.arange(1, len(energy_cum) + 1)

    # Show only first modes for readability
    r_show = int(min(35, max(20, len(r))))
    r_show = min(r_show, len(r))

    if len(r) > 0:
        rr = r[:r_show]
        cc = energy_cum[:r_show]
        ax_energy.plot(rr, cc, "o-", ms=2.8, lw=1.2, color="tab:blue", label="Cumulative energy")

        # add dashed reference lines
        for y, lab, col in [(0.90, "90%", "#ff4d4f"), (0.95, "95%", "#ff1f3d")]:
            ax_energy.axhline(y, ls="--", lw=0.9, alpha=0.95, color=col)
            ax_energy.text(rr[-1], y + 0.005, lab, fontsize=6, ha="right", va="bottom", color=col)


    ax_energy.set_title("A. Cumulative POD Energy", fontsize=8, fontweight="bold", pad=3)
    ax_energy.set_xlabel("Mode index", fontsize=7)
    ax_energy.set_ylabel("Energy fraction", fontsize=7)
    if len(r) > 0:
        ax_energy.set_xlim(1, r_show)
    ax_energy.set_ylim(0, 1.03)
    ax_energy.grid(True)
    ax_energy.tick_params(direction="in", length=2, width=0.5, labelsize=6)
    ax_energy.legend(loc="upper right", bbox_to_anchor=(0.98, 0.84))

    # =========================================================
    # Right block (rows 1-2): merged panel B (uniform 3x2 image grid)
    # row1-2: modes 1-4, row3: true vs recon
    # =========================================================
    ax_modes_outer = fig.add_subplot(gs_right[0:2, 0])
    ax_modes_outer.axis("off")
    ax_modes_outer.set_title(
        r"B. POD Modes and Reconstruction ($|\phi_r|$)",
        loc="center",
        fontsize=8,
        fontweight="semibold",
        pad=7,
    )

    gs_bc = gs_right[0:2, 0].subgridspec(3, 2, wspace=0.12, hspace=0.20)

    modes = pod_modes if pod_modes is not None else []
    # Optional: make mode color scale consistent across 1-4 (same vmin/vmax)
    if len(modes) >= 4:
        vmax_mode = max(float(np.max(m["umag"])) for m in modes[:4])
        vmin_mode = 0.0
    else:
        vmax_mode, vmin_mode = None, None

    mode_axes = []
    im_mode = None
    cb_modes = None
    for i in range(4):
        axm = fig.add_subplot(gs_bc[i // 2, i % 2])
        mode_axes.append(axm)
        if i < len(modes):
            umag = np.asarray(modes[i]["umag"], dtype=float)
            im_mode = _imshow(axm, fig, umag, f"POD Mode {i+1}", cmap=cmap_all, vmin=vmin_mode, vmax=vmax_mode, cbar=False)
        else:
            axm.text(0.5, 0.5, "N/A", ha="center", va="center", transform=axm.transAxes)
            axm.set_title(f"POD Mode {i+1}", fontsize=7, fontweight="bold", pad=2)
            axm.set_axis_off()
        axm.set_xlabel("x", fontsize=6)
        if i % 2 == 0:
            axm.set_ylabel("y", fontsize=6)

    if im_mode is not None:
        cb_modes = fig.colorbar(im_mode, ax=mode_axes, fraction=0.020, pad=0.02)
        cb_modes.ax.tick_params(labelsize=5)
        cb_modes.outline.set_linewidth(0.6)

    # True vs Recon in the 3rd row with the same panel size/style
    ax_true = fig.add_subplot(gs_bc[2, 0])
    ax_rec = fig.add_subplot(gs_bc[2, 1])

    cb_tr = None
    if recon is not None and ("true_umag" in recon) and ("recon_umag" in recon):
        true_umag = np.asarray(recon["true_umag"], dtype=float)
        recon_umag = np.asarray(recon["recon_umag"], dtype=float)

        # Consistent color scale between True and Recon
        vmax_tr = float(max(np.max(true_umag), np.max(recon_umag)))
        vmin_tr = 0.0

        im_tr = _imshow(ax_true, fig, recon_umag, "Reconstructed |u'| (Modes 1-4)", cmap=cmap_all, vmin=vmin_tr, vmax=vmax_tr, cbar=False)
        _imshow(ax_rec, fig, true_umag, "Reference |u'|", cmap=cmap_all, vmin=vmin_tr, vmax=vmax_tr, cbar=False)
        ax_true.set_xlabel("x", fontsize=6)
        ax_true.set_ylabel("y", fontsize=6)
        ax_rec.set_xlabel("x", fontsize=6)
        cb_tr = fig.colorbar(im_tr, ax=[ax_true, ax_rec], fraction=0.025, pad=0.03)
        cb_tr.ax.tick_params(labelsize=5)
        cb_tr.outline.set_linewidth(0.6)
    else:
        ax_true.axis("off")
        ax_rec.axis("off")

    # =========================================================
    # Left column rows 3-4: (Q4) Temporal PSD and anisotropy cuts
    # =========================================================
    # (Q4f) Temporal PSD of POD coefficients
    ax_psd = fig.add_subplot(gs_left[2, 0])
    if temporal_psd is not None:
        freqs = np.asarray(temporal_psd["freqs"], dtype=float)
        psd_arr = np.asarray(temporal_psd["psd"], dtype=float)
        n_modes = int(temporal_psd.get("n_modes", min(4, psd_arr.shape[0])))

        # show up to 0.04 as requested
        fmax = 0.04
        valid = (freqs > 0) & (freqs <= fmax)

        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        for i in range(min(4, n_modes, psd_arr.shape[0])):
            y = np.maximum(psd_arr[i], 1e-30)
            y_s = _smooth_1d(y, win=11)
            yv = y_s[valid]
            fv = freqs[valid]
            yv = np.asarray(yv, dtype=float)
            if yv.size == 0:
                continue
            y_pos = yv[np.isfinite(yv) & (yv > 0)]
            if y_pos.size == 0:
                continue
            yv_psd = np.clip(yv, 1e-20, None)
            ax_psd.semilogy(fv, yv_psd, lw=1.0, color=colors[i], alpha=0.9, label=f"Mode {i+1}")

        ax_psd.set_title(r"D. Temporal PSD of POD Coefficients",
                         fontsize=8, fontweight="bold", pad=3)
        ax_psd.set_xlabel("Frequency $f$", fontsize=7)
        ax_psd.set_ylabel("PSD", fontsize=7)
        ax_psd.yaxis.set_major_locator(LogLocator(base=10.0))
        ax_psd.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        ax_psd.grid(True, which="both")
        ax_psd.tick_params(direction="in", length=2, width=0.5, labelsize=6)
        ax_psd.legend(loc="upper right")

        if np.any(valid):
            ax_psd.set_xlim(0.0, fmax)
    else:
        ax_psd.text(0.5, 0.5, "Temporal PSD: N/A", ha="center", va="center", transform=ax_psd.transAxes)
        ax_psd.set_axis_off()

    # (Q4e) anisotropy axis-cuts
    ax_cut = fig.add_subplot(gs_left[3, 0])
    cut_kx = np.asarray(spec.get("cut_kx", []), dtype=float)
    cut_ky = np.asarray(spec.get("cut_ky", []), dtype=float)
    if len(cut_kx) and len(cut_ky):
        kx_axis = np.asarray(spec.get("kx_1d", []), dtype=float)
        ky_axis = np.asarray(spec.get("ky_1d", []), dtype=float)
        kx_axis, cut_kx = _align_1d_pair(kx_axis, cut_kx, "kx_1d", "cut_kx")
        ky_axis, cut_ky = _align_1d_pair(ky_axis, cut_ky, "ky_1d", "cut_ky")

        m1 = np.isfinite(kx_axis) & np.isfinite(cut_kx) & (cut_kx > 0)
        m2 = np.isfinite(ky_axis) & np.isfinite(cut_ky) & (cut_ky > 0)
        if np.any(m1):
            ax_cut.semilogy(
                k_plot_scale * kx_axis[m1],
                cut_kx[m1],
                lw=1.4,
                ls="--",
                color="royalblue",
                alpha=0.95,
                zorder=3,
                label="E(kx, ky=0)",
            )
        if np.any(m2):
            ax_cut.semilogy(
                k_plot_scale * ky_axis[m2],
                cut_ky[m2],
                lw=1.2,
                ls="--",
                color="darkorange",
                alpha=0.85,
                zorder=2,
                label="E(kx=0, ky)",
            )

        x_candidates = []
        if np.any(m1):
            x_candidates.append(np.max(np.abs(k_plot_scale * kx_axis[m1])))
        if np.any(m2):
            x_candidates.append(np.max(np.abs(k_plot_scale * ky_axis[m2])))
        if x_candidates:
            x_lim = float(max(x_candidates))
            if np.isfinite(x_lim) and x_lim > 0:
                ax_cut.set_xlim(-x_lim, x_lim)

        if aniso is not None:
            ratio_box = float(aniso.get("ratio_kx_over_ky", np.nan))
            if np.isfinite(ratio_box):
                ax_cut.text(
                    0.98,
                    0.16,
                    rf"$E_x/E_y={ratio_box:.2f}$",
                    transform=ax_cut.transAxes,
                    fontsize=6,
                    ha="right",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85, lw=0.35, ec="0.5"),
                )

        ax_cut.set_title("E. One-Dimensional Spectral Axis Cuts", fontsize=8, fontweight="bold", pad=3)
        ax_cut.set_xlabel("Wavenumber k", fontsize=7)
        ax_cut.set_ylabel("Energy", fontsize=7)
        ax_cut.xaxis.set_major_locator(MaxNLocator(nbins=7))
        ax_cut.grid(True)
        ax_cut.tick_params(direction="in", length=2, width=0.5, labelsize=6)
        ax_cut.legend(loc="upper right")
    else:
        ax_cut.text(0.5, 0.5, "Axis cuts: N/A", ha="center", va="center", transform=ax_cut.transAxes)
        ax_cut.set_axis_off()

    # =========================================================
    # Right column row 3: (Q2/Q3) 2D spectrum (viridis + peak text only)
    # =========================================================
    ax_2d = fig.add_subplot(gs_right[2, 0])
    E2D = np.asarray(spec["E2D"], dtype=float)
    kx_1d = np.asarray(spec["kx_1d"], dtype=float)
    ky_1d = np.asarray(spec["ky_1d"], dtype=float)

    if E2D.ndim != 2:
        raise ValueError(f"spec['E2D'] must be 2D, got shape={E2D.shape}")
    ny_eff = min(E2D.shape[0], ky_1d.size)
    nx_eff = min(E2D.shape[1], kx_1d.size)
    if (E2D.shape[0] != ky_1d.size) or (E2D.shape[1] != kx_1d.size):
        print(
            f"[plot warning] E2D shape {E2D.shape} is inconsistent with "
            f"(len(ky), len(kx))=({ky_1d.size}, {kx_1d.size}); "
            f"cropping to ({ny_eff}, {nx_eff})."
        )
        E2D = E2D[:ny_eff, :nx_eff]
        kx_1d = kx_1d[:nx_eff]
        ky_1d = ky_1d[:ny_eff]

    eps = 1e-12
    E_log = np.log10(E2D + eps)

    if k_window is not None:
        kw = float(k_window)
        kw = max(1e-12, min(kw, float(min(np.max(np.abs(kx_1d)), np.max(np.abs(ky_1d))))))
        kx_mask = np.abs(kx_1d) <= kw
        ky_mask = np.abs(ky_1d) <= kw
        view = E_log[np.ix_(ky_mask, kx_mask)]
        extent = [
            float(k_plot_scale * kx_1d[kx_mask][0]),
            float(k_plot_scale * kx_1d[kx_mask][-1]),
            float(k_plot_scale * ky_1d[ky_mask][0]),
            float(k_plot_scale * ky_1d[ky_mask][-1]),
        ]
        ttl = f"Mean 2D Energy Spectrum (log10, |k|≤{k_plot_scale * kw:.2g})"
    else:
        view = E_log
        extent = [
            float(k_plot_scale * kx_1d[0]),
            float(k_plot_scale * kx_1d[-1]),
            float(k_plot_scale * ky_1d[0]),
            float(k_plot_scale * ky_1d[-1]),
        ]
        ttl = "Mean 2D Energy Spectrum (log10)"

    im2 = ax_2d.imshow(view, origin="lower", aspect="equal",
                       cmap=cmap_all, extent=extent, interpolation="nearest")
    cb2 = fig.colorbar(im2, ax=ax_2d, fraction=0.04, pad=0.02)
    cb2.ax.tick_params(labelsize=5)
    cb2.outline.set_linewidth(0.6)

    # slight shift of panel F toward the left/center
    p_f = ax_2d.get_position()
    ax_2d.set_position([p_f.x0 - 0.014, p_f.y0 + 0.010, p_f.width, p_f.height])
    p_cb = cb2.ax.get_position()
    cb2.ax.set_position([p_cb.x0 - 0.014, p_cb.y0 + 0.010, p_cb.width, p_cb.height])

    ax_2d.set_title(f"F. {ttl}", fontsize=8, fontweight="bold", pad=3)
    ax_2d.set_xlabel(r"Wavenumber $k_x$", fontsize=7)
    ax_2d.set_ylabel(r"Wavenumber $k_y$", fontsize=7, labelpad=6)
    ax_2d.tick_params(direction="in", length=2, width=0.5, labelsize=6)
    ax_2d.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax_2d.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax_2d.yaxis.get_offset_text().set_size(6)


    # =========================================================
    # Left column row 2: (Q2/Q3) isotropic spectrum (prettier, less clutter)
    # =========================================================
    ax_iso = fig.add_subplot(gs_left[1, 0])
    k_centers = np.asarray(radial["k_bin_centers"], dtype=float)
    E_mean = np.asarray(radial["E_k_mean"], dtype=float)
    k_centers, E_mean = _align_1d_pair(k_centers, E_mean, "k_bin_centers", "E_k_mean")
    E_segs = np.asarray(radial.get("E_k_segments", np.zeros((0, len(k_centers)))), dtype=float)

    valid = np.isfinite(k_centers) & (k_centers > 0) & np.isfinite(E_mean) & (E_mean > 0)
    if np.any(valid):
        k_valid = np.asarray(k_centers[valid], dtype=float)
        E_valid = np.asarray(E_mean[valid], dtype=float)
        k_valid_disp = k_plot_scale * k_valid
        ax_iso.semilogy(k_valid_disp, E_valid, "o-", ms=3, lw=1.8, alpha=0.98)

        # 3 log-spaced segments on k>0: boundaries k1, k2
        kmin = float(np.min(k_valid))
        kmax = float(np.max(k_valid))
        if np.isfinite(kmin) and np.isfinite(kmax) and (kmax > kmin):
            lkmin = float(np.log(kmin))
            lkmax = float(np.log(kmax))
            lk1 = lkmin + (lkmax - lkmin) / 3.0
            lk2 = lkmin + 2.0 * (lkmax - lkmin) / 3.0
            k1 = float(np.exp(lk1))
            k2 = float(np.exp(lk2))

            def _fit_slope(k_arr: np.ndarray, e_arr: np.ndarray, ka: float, kb: float) -> float:
                m = (k_arr >= ka) & (k_arr <= kb) & np.isfinite(k_arr) & np.isfinite(e_arr) & (k_arr > 0) & (e_arr > 0)
                if np.count_nonzero(m) < 3:
                    return float("nan")
                x = np.log(k_arr[m])
                y = np.log(e_arr[m])
                s, _ = np.polyfit(x, y, 1)
                return float(s)


        ax_iso.set_title("C. Isotropic Energy Spectrum", fontsize=8, fontweight="bold", pad=3)
        ax_iso.set_xlabel(r"Wavenumber $k$", fontsize=7)
        ax_iso.set_ylabel(r"$E(|k|)$", fontsize=7)
        x_lo = float(np.nanmin(k_valid_disp))
        x_hi = float(np.nanmax(k_valid_disp))
        if np.isfinite(x_lo) and np.isfinite(x_hi) and x_hi > x_lo:
            ax_iso.set_xlim(x_lo, x_hi)
        ax_iso.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax_iso.grid(True, which="both")
        ax_iso.tick_params(direction="in", length=2, width=0.5, labelsize=6)
    else:
        ax_iso.text(0.5, 0.5, "E(k): N/A", ha="center", va="center", transform=ax_iso.transAxes)
        ax_iso.set_axis_off()

    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    _style_ctx.__exit__(None, None, None)

