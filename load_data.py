# load_data.py
from __future__ import annotations

from pathlib import Path
import numpy as np


def load_vector_field(
    npy_path: str | Path,
    grid_n: int = 64,
    n_components: int = 2,
    mmap: bool = True,
) -> np.ndarray:
    """Load vector field data from .npy.

    Expected shape: (nt, grid_n, grid_n, n_components).
    """
    npy_path = Path(npy_path)
    if not npy_path.exists():
        raise FileNotFoundError(f"Data file not found: {npy_path}")

    arr = np.load(npy_path, mmap_mode="r" if mmap else None)
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D array, got shape={arr.shape}")

    nt, ny, nx, nc = arr.shape
    if ny != grid_n or nx != grid_n or nc != n_components:
        raise ValueError(
            f"Unexpected shape {arr.shape}. Expected (nt,{grid_n},{grid_n},{n_components})."
        )
    return arr


def make_frame_indices(
    n_frames: int,
    stride: int = 1,
    max_frames: int | None = None,
) -> np.ndarray:
    """Create frame indices with optional stride and cap."""
    stride = max(1, int(stride))
    idx = np.arange(0, int(n_frames), stride, dtype=int)
    if max_frames is not None and len(idx) > int(max_frames):
        idx = idx[: int(max_frames)]
    return idx
