"""Gamma-index analysis (global dose normalization), vectorized shifted-array search."""
import numpy as np


def _axis_spacing(spacing_xyz):
    """spacing_xyz is ordered (X, Y, Z); NumPy arrays are ordered (Z, Y, X)."""
    return np.array([spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]], float)


def gamma_3d(ref, evl, dose_crit=0.03, dist_crit_mm=3.0, spacing_xyz=(1., 1., 3.),
             dose_threshold=0.10):
    """
    3D gamma index with global dose normalization (dose_crit as fraction of ref max).

    Evaluation is restricted to voxels with ref >= dose_threshold * ref.max().
    The search over distance-to-agreement offsets is vectorized as a shifted-array
    minimum inside a bounding box of the evaluation mask (padded by the DTA radius).

    Parameters
    ----------
    ref, evl : ndarray (Z, Y, X), identical shapes
    dose_crit : float, dose criterion as fraction of ref.max() (e.g. 0.03 = 3%)
    dist_crit_mm : float, distance criterion in mm (e.g. 3.0)
    spacing_xyz : tuple (sx, sy, sz) in mm
    dose_threshold : float, evaluation mask threshold as fraction of ref.max()

    Returns
    -------
    dict with keys:
        gamma       : ndarray (Z, Y, X), gamma values (NaN outside evaluation mask)
        pass_rate   : float, fraction of evaluated voxels with gamma <= 1
        n_evaluated : int, number of evaluated voxels
    """
    sp = _axis_spacing(spacing_xyz)
    dmax = ref.max()
    nan_out = {"gamma": np.full(ref.shape, np.nan), "pass_rate": np.nan, "n_evaluated": 0}
    if dmax <= 0:
        return nan_out

    mask = ref >= dose_threshold * dmax
    if not mask.any():
        return nan_out

    # DTA search radius in voxels per axis (Z, Y, X order)
    radii = np.ceil(dist_crit_mm / sp).astype(int)

    # Bounding box of the evaluation mask, padded by the search radius
    idx = np.argwhere(mask)
    lo = np.clip(idx.min(0) - radii, 0, ref.shape)
    hi = np.clip(idx.max(0) + radii + 1, 0, ref.shape)
    sub_ref = ref[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    sub_shape = sub_ref.shape

    # Pad the evaluation volume so every shifted window is fully in-bounds.
    # Original index i maps to padded index i + radii, therefore the sub-volume
    # origin inside the PADDED array is lo + radii (this was the bug: using `lo`
    # made negative offsets wrap around and produce empty windows).
    pad_eval = np.pad(evl, [(radii[i], radii[i]) for i in range(3)],
                      mode='constant', constant_values=0.0)
    r0 = lo + radii

    gamma = np.full(sub_shape, np.inf)
    dd_crit = dose_crit * dmax

    for dz in range(-radii[0], radii[0] + 1):
        for dy in range(-radii[1], radii[1] + 1):
            for dx in range(-radii[2], radii[2] + 1):
                dist = np.sqrt((dz * sp[0]) ** 2 +
                               (dy * sp[1]) ** 2 +
                               (dx * sp[2]) ** 2)
                if dist > dist_crit_mm:
                    continue
                w = pad_eval[r0[0] + dz: r0[0] + dz + sub_shape[0],
                             r0[1] + dy: r0[1] + dy + sub_shape[1],
                             r0[2] + dx: r0[2] + dx + sub_shape[2]]
                cand = np.sqrt((dist / dist_crit_mm) ** 2 +
                               ((w - sub_ref) / dd_crit) ** 2)
                np.minimum(gamma, cand, out=gamma)

    full = np.full(ref.shape, np.nan)
    full[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = gamma
    sub_mask = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

    return {"gamma": full,
            "pass_rate": float(np.mean(gamma[sub_mask] <= 1.0)),
            "n_evaluated": int(sub_mask.sum())}


def gamma_1d(ref, evl, dx_mm, dose_crit=0.03, dist_crit_mm=3.0, dose_threshold=0.10):
    """
    1D gamma index for depth-dose profiles.

    Returns dict with gamma array (NaN outside evaluation mask) and pass_rate.
    """
    dmax = ref.max()
    if dmax <= 0:
        return {"gamma": np.full(ref.shape, np.nan), "pass_rate": np.nan}
    mask = ref >= dose_threshold * dmax
    k = int(np.ceil(dist_crit_mm / dx_mm))
    gamma = np.full(ref.shape, np.nan)
    dd_crit = dose_crit * dmax

    for i in np.where(mask)[0]:
        best = np.inf
        lo_j = max(0, i - k)
        hi_j = min(len(ref), i + k + 1)
        for j in range(lo_j, hi_j):
            dist = abs(i - j) * dx_mm
            cand = np.sqrt((dist / dist_crit_mm) ** 2 +
                           ((evl[j] - ref[i]) / dd_crit) ** 2)
            if cand < best:
                best = cand
        gamma[i] = best

    return {"gamma": gamma, "pass_rate": float(np.mean(gamma[mask] <= 1.0))}