"""WEPL and ΔWEPL heatmaps in mm extents."""
import numpy as np
import matplotlib.pyplot as plt

def plot_wepl(wepl2d, extent=None, cmap="viridis", title="", cbar_label="WEPL (mm)"):
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(wepl2d, cmap=cmap, extent=extent)
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_title(title); ax.set_xlabel("mm"); ax.set_ylabel("mm")
    return fig, ax

def plot_delta_wepl(d2d, extent=None, vmax=None, cmap="bwr", title="",
                    cbar_label="ΔWEPL (mm)"):
    vmax = vmax or np.nanmax(np.abs(d2d)) or 1.0
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(d2d, cmap=cmap, vmin=-vmax, vmax=vmax, extent=extent)
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_title(title); ax.set_xlabel("mm"); ax.set_ylabel("mm")
    return fig, ax