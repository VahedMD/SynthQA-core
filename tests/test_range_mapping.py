import numpy as np
from synthqa_core.surrogates.range_mapping import inverse_monotone_profile, interval_spr_diagnostics


def test_inverse_monotone_profile_handles_plateaus_and_interpolates():
    z = np.array([0., 1., 2., 3., 4.])
    w = np.array([0., 1., 1., 3., 4.])
    assert inverse_monotone_profile(w, z, 2.) == 2.0


def test_inverse_monotone_profile_rejects_unreachable_and_nonmonotone():
    assert np.isnan(inverse_monotone_profile([0, 1, 2], [0, 1, 2], 3))
    assert np.isnan(inverse_monotone_profile([0, 2, 1], [0, 1, 2], 1))


def test_interval_spr_diagnostics_path_length_weighted():
    paired = {
        "ray_length_mm": 10., "t_edges": np.array([0., .5, 1.]),
        "spr_ref_segments": np.array([1., .2]),
        "spr_sct_segments": np.array([1.2, .4]),
    }
    d = interval_spr_diagnostics(paired, 2., 8.)
    assert d["interval_length_mm"] == 6.
    assert np.isclose(d["mean_spr_ref_interval"], .6)
    assert np.isclose(d["mean_spr_sct_interval"], .8)
    assert np.isclose(d["spr_ref_variation_interval"], (1. - .2) / .6)
    assert np.isclose(d["spr_ref_std_interval"], np.sqrt(.16))
    assert np.isclose(d["spr_ref_cv_interval"], np.sqrt(.16) / .6)
    assert np.isfinite(d["spr_sct_cv_interval"])
