"""Finite WEPL-to-depth mapping and interval-gradient diagnostics."""
import numpy as np

def inverse_monotone_profile(wepl_mm, depth_mm, target_wepl_mm):
    """Return depth where a non-decreasing WEPL profile reaches target WEPL.

    Repeated WEPL coordinates are collapsed by retaining the first depth, so
    zero-SPR/out-of-image plateaus do not violate np.interp's x requirement.
    Returns NaN when the target lies outside the sampled WEPL range.
    """
    w = np.asarray(wepl_mm, dtype=float)
    z = np.asarray(depth_mm, dtype=float)
    valid = np.isfinite(w) & np.isfinite(z)
    w, z = w[valid], z[valid]
    if len(w) < 2 or np.any(np.diff(w) < -1e-9):
        return np.nan
    unique_w, first = np.unique(w, return_index=True)
    if len(unique_w) < 2 or target_wepl_mm < unique_w[0] or target_wepl_mm > unique_w[-1]:
        return np.nan
    return float(np.interp(target_wepl_mm, unique_w, z[first]))


def interval_spr_diagnostics(paired, z0_mm, z1_mm):
    """Path-length-weighted SPR summaries over a physical ray interval."""
    ray_len = float(paired["ray_length_mm"])
    lo, hi = sorted((float(z0_mm), float(z1_mm)))
    edges = np.asarray(paired["t_edges"], dtype=float) * ray_len
    ref = np.asarray(paired["spr_ref_segments"], dtype=float)
    sct = np.asarray(paired["spr_sct_segments"], dtype=float)
    lengths = np.maximum(0.0, np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo))
    keep = lengths > 1e-9
    if not np.any(keep):
        return {"interval_length_mm": 0.0, "mean_spr_ref_interval": np.nan,
                "mean_spr_sct_interval": np.nan, "spr_ref_min_interval": np.nan,
                "spr_ref_max_interval": np.nan, "spr_sct_min_interval": np.nan,
                "spr_sct_max_interval": np.nan, "spr_ref_std_interval": np.nan,
                "spr_sct_std_interval": np.nan, "spr_ref_cv_interval": np.nan,
                "spr_sct_cv_interval": np.nan, "spr_ref_variation_interval": np.nan,
                "spr_sct_variation_interval": np.nan}
    rr, ss, ww = ref[keep], sct[keep], lengths[keep]
    mean_r = float(np.average(rr, weights=ww))
    mean_s = float(np.average(ss, weights=ww))
    def weighted_std(a, mean):
        return float(np.sqrt(np.average((a - mean) ** 2, weights=ww)))
    std_r, std_s = weighted_std(rr, mean_r), weighted_std(ss, mean_s)
    def variation(a, mean):
        return float((np.max(a) - np.min(a)) / mean) if mean > 0 else np.nan
    return {"interval_length_mm": float(ww.sum()),
            "mean_spr_ref_interval": mean_r, "mean_spr_sct_interval": mean_s,
            "spr_ref_min_interval": float(np.min(rr)), "spr_ref_max_interval": float(np.max(rr)),
            "spr_sct_min_interval": float(np.min(ss)), "spr_sct_max_interval": float(np.max(ss)),
            "spr_ref_std_interval": std_r, "spr_sct_std_interval": std_s,
            "spr_ref_cv_interval": float(std_r / mean_r) if mean_r > 0 else np.nan,
            "spr_sct_cv_interval": float(std_s / mean_s) if mean_s > 0 else np.nan,
            "spr_ref_variation_interval": variation(rr, mean_r),
            "spr_sct_variation_interval": variation(ss, mean_s)}

