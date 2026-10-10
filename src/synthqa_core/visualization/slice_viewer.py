"""Anatomical slice viewing with mm extents and optional overlays."""
import numpy as np
import matplotlib.pyplot as plt

_PLANES = {"axial": (0, 1, 2), "coronal": (1, 0, 2), "sagittal": (2, 0, 1)}

def get_slice(volume, plane="axial", index=None, spacing_xyz=(1., 1., 3.)):
    """Return (slice2d, extent_mm). plane fixes the first axis index."""
    fix, row_ax, col_ax = _PLANES[plane]
    n = volume.shape[fix]
    i = n // 2 if index is None else index
    sl = [slice(None)] * 3
    sl[fix] = i
    arr = volume[tuple(sl)]
    sp = np.asarray(spacing_xyz, float)  # (X, Y, Z)
    axis_mm = {0: sp[2], 1: sp[1], 2: sp[0]}
    if fix == 0:   arr2 = arr                      # (Y, X)
    elif fix == 1: arr2 = arr                      # (Z, X)
    else:          arr2 = arr                      # (Z, Y)
    rows_mm = axis_mm[row_ax] * arr2.shape[0]
    cols_mm = axis_mm[col_ax] * arr2.shape[1]
    return arr2, [0, cols_mm, rows_mm, 0]

def plot_slice(volume, plane="axial", index=None, spacing_xyz=(1., 1., 3.),
               cmap="bone", vmin=None, vmax=None, overlay=None, overlay_cmap="inferno",
               overlay_alpha=0.55, overlay_floor=None, title="", cbar_label=None):
    arr, ext = get_slice(volume, plane, index, spacing_xyz)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, extent=ext)
    if overlay is not None:
        oarr, _ = get_slice(overlay, plane, index, spacing_xyz)
        if overlay_floor is not None:
            oarr = np.ma.masked_where(oarr < overlay_floor, oarr)
        im = ax.imshow(oarr, cmap=overlay_cmap, alpha=overlay_alpha, extent=ext)
        if cbar_label:
            fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_title(title)
    ax.set_xlabel("mm"); ax.set_ylabel("mm")
    return fig, ax

def compare_slices(volumes, titles, plane="axial", index=None,
                   spacing_xyz=(1., 1., 3.), cmap="bone", vmin=None, vmax=None):
    fig, axes = plt.subplots(1, len(volumes), figsize=(6*len(volumes), 6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, vol, t in zip(axes, volumes, titles):
        arr, ext = get_slice(vol, plane, index, spacing_xyz)
        ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, extent=ext)
        ax.set_title(t)
    return fig, axes