# config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CFG:
    # ---------- Paths ----------
    project_root: Path = Path(__file__).resolve().parent
    data_path: Path = project_root / "data" / "vector_64.npy"
    figs_dir: Path = project_root / "outputs"

    # ---------- Data ----------
    grid_n: int = 64
    n_components: int = 2
    dt: float = 0.2           # original time step in the dataset
    dx: float = 1.0           # grid spacing (unit length per grid)
    # ---------- Subsampling ----------
    stride: int = 5           # use every 'stride' frames
    max_frames_svd: int = 3000
    max_frames_fft: int = 3000

    # ---------- POD ----------
    num_modes_svd: int = 100
    num_modes_show: int = 4

    # ---------- Segmented spectra ----------
    n_segments: int = 3

    # ---------- 2D spectrum view window (optional) ----------
    k_window: float | None = None
