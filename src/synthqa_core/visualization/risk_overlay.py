"""Risk-map and risk-class overlays on anatomy."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

RISK_CMAP = LinearSegmentedColormap.from_list("risk", ["black", "red", "yellow", "white"])

def overlay_risk(ct2d, risk2d, extent=None, ct_range=(-1024, 2000), risk_floor=0.01,
                 alpha=0.6, contours=(), legend_handles=(), title=""):
    """contours: iterable of (array2d, level, color)."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(ct2d, cmap="bone", vmin=ct_range[0], vmax=ct_range[1], extent=extent)
    masked = np.ma.masked_where(risk2d < risk_floor, risk2d)
    im = ax.imshow(masked, cmap=RISK_CMAP, alpha=alpha, extent=extent)
    fig.colorbar(im, ax=ax, label="Risk (Gy·mm)")
    for arr, lvl, col in contours:
        ax.contour(arr, levels=[lvl], colors=col, linewidths=1.5, extent=extent)
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right")
    ax.set_title(title); ax.set_xlabel("mm"); ax.set_ylabel("mm")
    return fig, ax

def overlay_classes(ct2d, cls2d, extent=None, ct_range=(-1024, 2000),
                    colors=(None, "yellow", "red"), alpha=0.45, title=""):
    """cls2d: 0 low / 1 medium / 2 high risk."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(ct2d, cmap="bone", vmin=ct_range[0], vmax=ct_range[1], extent=extent)
    rgb = np.zeros((*cls2d.shape, 4))
    for k, col in enumerate(colors):
        if col is None:
            continue
        m = cls2d == k
        rgb[m] = plt.cm.colors.to_rgba(col)
    rgb[..., 3] = np.where(cls2d > 0, alpha, 0.0)
    ax.imshow(rgb, extent=extent)
    handles = [Line2D([], [], color=c, lw=6, label=l)
               for c, l in [(colors[1], "Medium risk"), (colors[2], "High risk")]]
    ax.legend(handles=handles, loc="upper right")
    ax.set_title(title); ax.set_xlabel("mm"); ax.set_ylabel("mm")
    return fig, ax