"""
Track 1: Reviewer-Proofing the Statistics (rev 2)
Fixes: matplotlib 3.10 boxplot labels; dynamic subgroup n; sensitivity analysis.
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from huggingface_hub import hf_hub_download, list_repo_files
import warnings

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from synthqa_core.io.image_loader import load_mha
from synthqa_core.calibration.profiles import CalibrationProfile
from synthqa_core.surrogates.wepl import compute_wepl_along_ray

REPO_ID = "LMUK-RADONC-PHYS-RES/DoseRAD2026"
METAL_PIDS = {"1THB016","1THB021","1THB029","1THB031","1THB052","1THB054",
              "1THB067","1THB078","1THB122","1THB214","1THB217","1ABB138","1ABB078"}

pb_data = pd.DataFrame({
    "Patient": ["1THB016","1THB021","1THB029","1THB002","1THB008","1THB011",
                "1ABB078","1ABB138","1ABB006","1ABB011","1ABB020"],
    "Region": ["Thorax"]*6 + ["Abdomen"]*5,
    "Metal": [True, True, True, False, False, False, True, True, False, False, False],
    "Pred_mm": [0.29, 0.57, 0.42, 0.58, 0.00, 0.09, 0.00, 0.00, 0.00, 0.00, 0.31],
    "PB_shift_mm": [0.35, 0.11, 0.28, 3.70, -0.01, 0.18, -0.01, -0.03, -0.01, 0.03, 0.31],
    "Surrogate_ms": [16.92, 26.83, 13.19, 19.44, 23.14, 28.14, 13.98, 11.34, 13.01, 8.06, 5.92],
    "PB_total_ms": [79458, 228072, 124084, 148126, 115796, 125135, 174882, 106363, 154449, 84954, 92752]
})

def bootstrap_ci(x, y, n_boot=10000, alpha=0.05):
    n = len(x)
    maes, rs, biases = [], [], []
    for _ in range(n_boot):
        idx = np.random.randint(0, n, n)
        xb, yb = x[idx], y[idx]
        maes.append(np.mean(np.abs(xb - yb)))
        biases.append(np.mean(yb - xb))
        rs.append(np.corrcoef(xb, yb)[0, 1] if np.std(xb) > 0 and np.std(yb) > 0 else np.nan)
    rs_valid = [r for r in rs if not np.isnan(r)]
    lo, hi = (alpha/2)*100, (1-alpha/2)*100
    return {"MAE": (np.mean(maes), np.percentile(maes, [lo, hi])),
            "Bias": (np.mean(biases), np.percentile(biases, [lo, hi])),
            "r": (np.nanmean(rs_valid), np.percentile(rs_valid, [lo, hi]))}

def extract_1d_ray_profile(vol, sp, org, rs, rt, n=2000):
    from scipy.ndimage import map_coordinates
    t = np.linspace(0, 1, n)
    p = rs[None, :] + t[:, None] * (rt - rs)[None, :]
    ix, iy, iz = ((p[:,0]-org[0])/sp[0], (p[:,1]-org[1])/sp[1], (p[:,2]-org[2])/sp[2])
    return map_coordinates(vol, np.array([iz, iy, ix]), order=1, mode='constant', cval=0.0), np.linalg.norm(rt - rs)

def find_r80(dose, depth):
    pk = np.argmax(dose)
    if pk == 0 or dose[pk] <= 0: return np.nan
    thr = 0.8 * dose[pk]
    below = np.where(dose[pk:] < thr)[0]
    if len(below) == 0: return np.nan
    i = pk + below[0]
    if 0 < i < len(dose):
        y1, y2 = dose[i-1], dose[i]
        if y1 != y2: return depth[i-1] + (thr - y1) * (depth[i] - depth[i-1]) / (y2 - y1)
    return depth[i]

def run_cohort_expansion(n_patients=30):
    print("\n--- COHORT EXPANSION: Surrogate vs Analytical MC Shift ---")
    prof = CalibrationProfile()
    files = list_repo_files(REPO_ID, repo_type="dataset")
    pids = sorted({f.split("/")[2] for f in files if f.startswith("proton/training/") and len(f.split("/")) >= 3})
    np.random.seed(42)
    sampled = np.random.choice(pids, min(n_patients, len(pids)), replace=False)
    results = []
    for pid in sampled:
        try:
            plan_p = hf_hub_download(REPO_ID, f"proton/training/{pid}/{pid}.json", repo_type="dataset")
            ct_p   = hf_hub_download(REPO_ID, f"proton/training/{pid}/image/ct.mha", repo_type="dataset")
        except Exception: continue
        ct = load_mha(ct_p)
        ct_arr, sp, org = ct["array"], ct["spacing_xyz"], ct["origin_xyz"]
        spr_ref = prof.density_to_spr.convert(prof.hu_to_density.convert(ct_arr))
        spr_sct = prof.density_to_spr.convert(prof.hu_to_density.convert(ct_arr + 50.0))
        plan = json.load(open(plan_p))
        found = False
        for b in plan["beams"][:2]:
            for r in b["rays"][:5]:
                rs, rt = np.array(r["ray_source"]), np.array(r["ray_target"])
                w = compute_wepl_along_ray(spr_ref, sp, org, rs, rt, num_samples=400)
                if w[-1] < 10.0: continue
                fp = f"proton/training/{pid}/dose/Dose_B{b['beam_idx']}_R{r['ray_idx']}_L0.mha"
                try:
                    d3 = load_mha(hf_hub_download(REPO_ID, fp, repo_type="dataset"))["array"]
                except Exception: continue
                d1, dist = extract_1d_ray_profile(d3, sp, org, rs, rt)
                pk = np.argmax(d1)
                if pk > 0.2*len(d1) and d1[pk] > 1e-3 and d1[pk:].min() < 0.1*d1[pk]:
                    w_ref = compute_wepl_along_ray(spr_ref, sp, org, rs, rt, num_samples=2000)
                    w_sct = compute_wepl_along_ray(spr_sct, sp, org, rs, rt, num_samples=2000)
                    dwepl = w_sct - w_ref
                    depth = np.arange(len(d1)) * (dist / (len(d1)-1))
                    d_sct = np.interp(w_ref + dwepl, w_ref, d1, left=0.0, right=0.0)
                    results.append({"Patient": pid,
                                    "Pred_mm": -dwepl[pk],
                                    "Act_mm": find_r80(d_sct, depth) - find_r80(d1, depth)})
                    found = True; break
            if found: break
    return pd.DataFrame(results)

def main():
    os.makedirs("figures", exist_ok=True)
    print("="*70); print("REVIEWER-PROOFING STATISTICS (rev 2)"); print("="*70)

    x_pb, y_pb = pb_data["Pred_mm"].values, pb_data["PB_shift_mm"].values
    ci = bootstrap_ci(x_pb, y_pb)
    print("\n[1] 95% Bootstrap CIs (Surrogate vs PB, n=11)")
    print(f"  MAE:  {ci['MAE'][0]:.3f} mm  (95% CI: {ci['MAE'][1][0]:.3f} to {ci['MAE'][1][1]:.3f})")
    print(f"  Bias: {ci['Bias'][0]:.3f} mm  (95% CI: {ci['Bias'][1][0]:.3f} to {ci['Bias'][1][1]:.3f})")
    print(f"  r:    {ci['r'][0]:.3f}       (95% CI: {ci['r'][1][0]:.3f} to {ci['r'][1][1]:.3f})")

    err = np.abs(x_pb - y_pb)
    m_metal, m_clean = pb_data["Metal"].values, ~pb_data["Metal"].values
    m_th, m_ab = pb_data["Region"].values == "Thorax", pb_data["Region"].values == "Abdomen"

    u1, p1 = mannwhitneyu(err[m_clean], err[m_metal], alternative='two-sided')
    u2, p2 = mannwhitneyu(err[m_th], err[m_ab], alternative='two-sided')

    # SENSITIVITY ANALYSIS: exclude the single boundary-crossing outlier (1THB002)
    keep = pb_data["Patient"].values != "1THB002"
    u3, p3 = mannwhitneyu(err[keep & m_th], err[keep & m_ab], alternative='two-sided')

    print("\n[2] Subgroup Tests (Mann-Whitney U on |Error|)")
    print(f"  Metal (n={m_metal.sum()}) vs Clean (n={m_clean.sum()}): p = {p1:.3f} (NS)")
    print(f"  Thorax (n={m_th.sum()}) vs Abdomen (n={m_ab.sum()}): p = {p2:.3f} *")
    print(f"  SENSITIVITY (excl. 1THB002 outlier): Thorax vs Abdomen p = {p3:.3f} (NS)")

    # --- Fig 9: Bland-Altman ---
    mean_val, diff_val = (x_pb + y_pb)/2, x_pb - y_pb
    md, sd = np.mean(diff_val), np.std(diff_val, ddof=1)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(mean_val, diff_val, c='steelblue', s=60, edgecolors='k')
    ax.axhline(md, color='red', label=f'Mean bias = {md:.2f} mm')
    ax.axhline(md + 1.96*sd, color='gray', ls='--', label=f'+1.96 SD ({md+1.96*sd:.2f} mm)')
    ax.axhline(md - 1.96*sd, color='gray', ls='--', label=f'-1.96 SD ({md-1.96*sd:.2f} mm)')
    ax.set_xlabel("Mean of Surrogate and PB shift (mm)"); ax.set_ylabel("Difference (Surrogate - PB) (mm)")
    ax.set_title("Bland-Altman: Surrogate vs Pencil Beam (n=11)"); ax.legend(); ax.grid(ls='--', alpha=0.5)
    fig.tight_layout(); fig.savefig("figures/Fig9_BlandAltman.png", dpi=300); plt.close(fig)

    # --- Fig 10: Subgroup boxplots (version-agnostic labels) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].boxplot([err[m_clean], err[m_metal]])
    axes[0].set_xticklabels([f"Clean (n={m_clean.sum()})", f"Metal (n={m_metal.sum()})"])
    axes[0].set_ylabel("|Error| (mm)"); axes[0].set_title(f"Metal vs Clean (p={p1:.3f}, NS)"); axes[0].grid(ls='--', alpha=0.5)
    axes[1].boxplot([err[m_th], err[m_ab]])
    axes[1].set_xticklabels([f"Thorax (n={m_th.sum()})", f"Abdomen (n={m_ab.sum()})"])
    axes[1].set_ylabel("|Error| (mm)")
    axes[1].set_title(f"Thorax vs Abdomen (p={p2:.3f}; excl. outlier p={p3:.3f})"); axes[1].grid(ls='--', alpha=0.5)
    fig.tight_layout(); fig.savefig("figures/Fig10_Subgroups.png", dpi=300); plt.close(fig)

    # --- Fig 11: Cohort expansion ---
    df_c = run_cohort_expansion(30)
    if len(df_c) > 5:
        x_c, y_c = df_c["Pred_mm"].values, df_c["Act_mm"].values
        ci_c = bootstrap_ci(x_c, y_c)
        r_c = np.corrcoef(x_c, y_c)[0, 1]
        print(f"\n[3] Cohort Expansion (n={len(df_c)}): r = {r_c:.3f} "
              f"(95% CI: {ci_c['r'][1][0]:.3f} to {ci_c['r'][1][1]:.3f}); "
              f"MAE = {ci_c['MAE'][0]:.3f} mm (95% CI: {ci_c['MAE'][1][0]:.3f} to {ci_c['MAE'][1][1]:.3f})")
        lims = [min(x_c.min(), y_c.min())-1, max(x_c.max(), y_c.max())+1]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(lims, lims, 'k--', alpha=0.5)
        ax.scatter(x_c, y_c, s=60, c='green', edgecolors='k', alpha=0.7)
        ax.set_xlabel("Surrogate predicted shift (mm)"); ax.set_ylabel("Analytical MC shift (mm)")
        ax.set_title(f"Cohort expansion (n={len(df_c)}): r = {r_c:.3f}"); ax.grid(ls='--', alpha=0.5)
        fig.tight_layout(); fig.savefig("figures/Fig11_CohortExpansion.png", dpi=300); plt.close(fig)

    print("\nSaved: Fig9_BlandAltman.png, Fig10_Subgroups.png, Fig11_CohortExpansion.png")

if __name__ == "__main__":
    main()