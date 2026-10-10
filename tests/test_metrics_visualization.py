import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from synthqa_core.metrics.gamma_index import gamma_3d, gamma_1d
from synthqa_core.metrics.dose_difference import summary, relative_difference_global
from synthqa_core.metrics.range_metrics import distal_edge, distal_edge_map, range_shift_map
from synthqa_core.visualization import slice_viewer, wepl_heatmap, risk_overlay

def test_gamma_identity_and_shift():
    z = np.linspace(0, 100, 60)
    peak = np.exp(-0.5*((z-60)/3)**2)
    ref = np.tile(peak, (20, 20, 1)).T.reshape(60, 20, 20)
    g_id = gamma_3d(ref, ref.copy(), spacing_xyz=(2., 2., 2.))
    assert g_id["pass_rate"] == 1.0
    shifted = np.roll(ref, 3, axis=0)          # 6 mm shift > 3 mm criterion
    g_sh = gamma_3d(ref, shifted, spacing_xyz=(2., 2., 2.))
    assert g_sh["pass_rate"] < 0.5

def test_gamma_1d():
    z = np.arange(100, dtype=float)
    ref = np.exp(-0.5*((z-60)/3)**2)
    g = gamma_1d(ref, ref.copy(), dx_mm=1.0)
    assert g["pass_rate"] == 1.0

def test_dose_difference():
    ref = np.ones((4, 4, 4)) * 2.0
    evl = ref + 0.2
    s = summary(ref, evl)
    assert np.isclose(s["mean_abs_diff_Gy"], 0.2)
    assert np.isclose(s["mean_rel_diff_pct"], 10.0)

def test_range_metrics():
    depth = np.arange(200) * 1.0
    prof = np.exp(-0.5*((depth-120)/4)**2)
    r80 = distal_edge(depth, prof, 0.8)
    assert 120 < r80 < 128
    dose = np.tile(prof, (5, 6, 1)).T          # (200, 6, 5)
    emap = distal_edge_map(dose, 1.0, axis=0)
    assert np.allclose(emap, r80)
    dose2 = np.roll(dose, 4, axis=0)
    smap = range_shift_map(dose, dose2, 1.0, axis=0)
    assert np.nanmean(smap) > 3.0              # ~4 mm deeper

def test_visualization_smoke():
    vol = np.random.rand(30, 40, 50)
    fig, ax = slice_viewer.plot_slice(vol, plane="coronal", spacing_xyz=(1., 1., 3.))
    assert ax is not None
    fig2, ax2 = wepl_heatmap.plot_delta_wepl(np.random.rand(30, 40) - 0.5)
    assert ax2 is not None
    fig3, ax3 = risk_overlay.overlay_classes(np.random.rand(30, 40)*1000,
                                             np.random.randint(0, 3, (30, 40)))
    assert ax3 is not None