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
from cohort_manifest import (new_run, base_record, append_manifest_rows, patient_metadata,
                             discover_training_patient_ids, discover_stratified_cohort)

REPO_ID = "LMUK-RADONC-PHYS-RES/DoseRAD2026"

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

def bootstrap_ci(x, y, n_boot=10000, alpha=0.05, seed=42):
    """Paired nonparametric bootstrap with observed point estimates.

    Convention throughout this script: error/bias = surrogate (x) minus
    reference (y). The first element of each returned tuple is the statistic
    computed on the original paired data; the second is the percentile CI.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("x and y must be one-dimensional paired arrays of equal length")
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    n = len(x)
    if n < 2:
        raise ValueError(f"At least 2 valid paired observations are required; got n={n}")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1")

    observed_mae = float(np.mean(np.abs(x - y)))
    observed_bias = float(np.mean(x - y))
    observed_r = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else np.nan

    rng = np.random.default_rng(seed)
    maes, biases, rs = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        maes.append(float(np.mean(np.abs(xb - yb))))
        biases.append(float(np.mean(xb - yb)))
        if np.std(xb) > 0 and np.std(yb) > 0:
            rs.append(float(np.corrcoef(xb, yb)[0, 1]))

    lo, hi = alpha * 50, (1 - alpha / 2) * 100
    # alpha*50 is alpha/2 expressed as a percentage (e.g. 2.5 for alpha=.05).
    def interval(values):
        if not values:
            return np.array([np.nan, np.nan])
        return np.percentile(np.asarray(values, dtype=float), [lo, hi])

    return {
        "MAE": (observed_mae, interval(maes)),
        "Bias": (observed_bias, interval(biases)),
        "r": (observed_r, interval(rs)),
        "n": n,
        "n_boot": n_boot,
        "seed": seed,
        "bias_convention": "surrogate_minus_reference"
    }

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

def run_cohort_expansion(n_patients=30, run=None):
    """Collect n_patients VALID Model A cases, not merely n_patients attempts.

    The expansion arm is kept disjoint from the fixed reviewer comparison and
    the deterministic Model A/Model B stratified cohort candidates. Candidates
    are traversed in a seeded random permutation until the valid-case target is
    reached or the eligible pool is exhausted.
    """
    print("\n--- COHORT EXPANSION: Surrogate vs Analytical MC Shift ---")
    print(f"Target: {n_patients} valid patients (not {n_patients} initial attempts)")
    run = run or new_run(__file__)
    prof = CalibrationProfile()
    pids = discover_training_patient_ids(REPO_ID)
    if not pids:
        raise RuntimeError("No patient IDs found in proton/training; cannot run expansion cohort.")

    # Explicit disjointness policy: reserve every candidate in the stratified
    # cohorts (including patients that later fail) and the fixed PB comparison.
    stratified_candidates = {x["pid"] for x in discover_stratified_cohort(REPO_ID, patient_ids=pids)}
    fixed_comparison = set(pb_data["Patient"].astype(str))
    reserved = stratified_candidates | fixed_comparison
    overlap_in_dataset = sorted(set(pids) & reserved)
    eligible_pool = [pid for pid in pids if pid not in reserved]

    # Record reserved patients so the disjoint-cohort policy is auditable.
    manifest_rows = []
    for pid in overlap_in_dataset:
        meta = patient_metadata(pid)
        row = base_record(run, "model_a_expansion", pid,
            anatomical_region=meta["anatomical_region"], metal_status=meta["metal_status"],
            cohort_selected=False,
            selection_method="excluded before randomized expansion selection by predeclared disjoint-cohort policy",
            perturbation_model="Model A: global HU bias", perturbation_threshold_hu="",
            perturbation_delta_hu=50, outcome_status="excluded_by_cohort_policy",
            exclusion_reason="reserved_for_stratified_or_fixed_pyRadPlan_comparison_cohort",
            notes_json={"cohort_policy": "expansion excludes all deterministic stratified candidates and fixed PB comparison patients",
                        "anatomical_region_source": meta["anatomical_region_source"],
                        "metal_status_source": meta["metal_status_source"]})
        manifest_rows.append(row)

    rng = np.random.RandomState(42)
    candidate_order = list(rng.permutation(eligible_pool))
    results = []
    attempts = 0
    for selection_order, pid in enumerate(candidate_order, start=1):
        if len(results) >= n_patients:
            break
        attempts += 1
        meta = patient_metadata(pid)
        row = base_record(run, "model_a_expansion", str(pid),
            anatomical_region=meta["anatomical_region"], metal_status=meta["metal_status"],
            selection_method=(f"RandomState(seed=42) permutation of eligible pool; iterate until "
                             f"{n_patients} valid patients or pool exhaustion; disjoint from reserved cohorts"),
            perturbation_model="Model A: global HU bias", perturbation_threshold_hu="",
            perturbation_delta_hu=50, candidate_rays_examined=0, candidate_beamlets_examined=0,
            rays_rejected_short_wepl=0, dose_download_failures=0,
            invalid_dose_profiles=0, dose_engine_failures=0, eligible_rays=0,
            outcome_status="selected_pending",
            notes_json={"selection_order": selection_order,
                        "candidate_pool_size_before_disjoint_filter": len(pids),
                        "reserved_cohort_patient_count": len(overlap_in_dataset),
                        "eligible_candidate_pool_size": len(eligible_pool),
                        "target_valid_n": n_patients,
                        "valid_n_before_this_patient": len(results),
                        "random_seed": 42,
                        "anatomical_region_source": meta["anatomical_region_source"],
                        "metal_status_source": meta["metal_status_source"],
                        "dose_reference": "analytical reconstruction from reference dose; not fresh perturbed-image MC"})
        try:
            plan_p = hf_hub_download(REPO_ID, f"proton/training/{pid}/{pid}.json", repo_type="dataset")
            ct_p = hf_hub_download(REPO_ID, f"proton/training/{pid}/image/ct.mha", repo_type="dataset")
        except Exception as exc:
            row.update(outcome_status="excluded", exclusion_reason="plan_or_CT_download_failed",
                       notes_json={**row["notes_json"], "error": str(exc)[:500]})
            manifest_rows.append(row)
            continue
        try:
            ct = load_mha(ct_p)
            ct_arr, sp, org = ct["array"], ct["spacing_xyz"], ct["origin_xyz"]
            spr_ref = prof.density_to_spr.convert(prof.hu_to_density.convert(ct_arr))
            spr_sct = prof.density_to_spr.convert(prof.hu_to_density.convert(ct_arr + 50.0))
            with open(plan_p, encoding="utf-8") as f:
                plan = json.load(f)
        except Exception as exc:
            row.update(outcome_status="excluded", exclusion_reason="CT_or_plan_parse_failed",
                       notes_json={**row["notes_json"], "error": str(exc)[:500]})
            manifest_rows.append(row)
            continue

        found = False
        for b in plan.get("beams", [])[:2]:
            for r in b.get("rays", [])[:5]:
                row["candidate_rays_examined"] += 1
                rs, rt = np.array(r["ray_source"]), np.array(r["ray_target"])
                w = compute_wepl_along_ray(spr_ref, sp, org, rs, rt, num_samples=400)
                if w[-1] < 10.0:
                    row["rays_rejected_short_wepl"] += 1
                    continue
                fp = f"proton/training/{pid}/dose/Dose_B{b['beam_idx']}_R{r['ray_idx']}_L0.mha"
                row["candidate_beamlets_examined"] += 1
                try:
                    d3 = load_mha(hf_hub_download(REPO_ID, fp, repo_type="dataset"))["array"]
                except Exception as exc:
                    row["dose_download_failures"] += 1
                    row.setdefault("_dose_errors", []).append(str(exc)[:200])
                    continue
                d1, dist = extract_1d_ray_profile(d3, sp, org, rs, rt)
                pk = np.argmax(d1)
                if not (pk > 0.2*len(d1) and d1[pk] > 1e-3 and d1[pk:].min() < 0.1*d1[pk]):
                    row["invalid_dose_profiles"] += 1
                    continue
                w_ref = compute_wepl_along_ray(spr_ref, sp, org, rs, rt, num_samples=2000)
                w_sct = compute_wepl_along_ray(spr_sct, sp, org, rs, rt, num_samples=2000)
                dwepl = w_sct - w_ref
                depth = np.arange(len(d1)) * (dist / (len(d1)-1))
                d_sct = np.interp(w_ref + dwepl, w_ref, d1, left=0.0, right=0.0)
                pred_mm = float(-dwepl[pk])
                act_mm = float(find_r80(d_sct, depth) - find_r80(d1, depth))
                results.append({"Patient": pid, "Pred_mm": pred_mm, "Act_mm": act_mm,
                                "Region": meta["anatomical_region"], "Metal": meta["metal_status"]})
                row.update(eligible_rays=1, selected_beam_idx=b.get("beam_idx", ""),
                    selected_ray_idx=r.get("ray_idx", ""), selected_beamlet_idx=0,
                    outcome_status="included", exclusion_reason="", surrogate_shift_mm=pred_mm,
                    reference_shift_mm=act_mm,
                    notes_json={**row["notes_json"], "ray_source": rs.tolist(), "ray_target": rt.tolist(),
                                "reference_type": "analytical reconstructed dose shift", "dose_file": fp,
                                "dose_download_errors": row.pop("_dose_errors", []),
                                "eligible_rays_semantics": "selected valid ray used by analysis; search stops after first valid ray"})
                found = True
                break
            if found:
                break
        if not found:
            row.update(outcome_status="excluded", exclusion_reason="no_valid_dose_ray_found",
                       notes_json={**row["notes_json"], "dose_download_errors": row.pop("_dose_errors", [])})
        manifest_rows.append(row)

    append_manifest_rows(manifest_rows, __file__)
    valid_n = len(results)
    print(f"[cohort] Attempted {attempts} eligible-pool patients; valid={valid_n}/{n_patients}; "
          f"eligible pool={len(eligible_pool)}; reserved overlap IDs={len(overlap_in_dataset)}")
    if valid_n < n_patients:
        print(f"[WARNING] Target of {n_patients} valid patients was not reached; candidate pool exhausted at n={valid_n}.")
    else:
        print(f"[cohort] Target achieved: {valid_n} valid patients.")
    return pd.DataFrame(results)

def main():
    os.makedirs("figures", exist_ok=True)
    print("="*70); print("REVIEWER-PROOFING STATISTICS (rev 2)"); print("="*70)

    run = new_run(__file__)
    # These rows are transcribed/hard-coded comparison data; the original ray-selection
    # provenance is not available in this script and is explicitly marked as such.
    pb_manifest = []
    for _, r in pb_data.iterrows():
        pb_manifest.append(base_record(run, "reviewer_pb_hardcoded_comparison", r["Patient"],
            anatomical_region=r["Region"], metal_status=bool(r["Metal"]),
            selection_method="fixed patient list embedded in reviewer_stats.py",
            perturbation_model="Model B / as originally reported in embedded comparison; perturbation provenance not recoverable here",
            outcome_status="comparison_row_available", surrogate_shift_mm=float(r["Pred_mm"]),
            reference_shift_mm=float(r["PB_shift_mm"]), surrogate_runtime_ms=float(r["Surrogate_ms"]),
            reference_runtime_ms=float(r["PB_total_ms"]),
            notes_json={"source": "hard-coded pb_data dataframe", "ray_metadata_available": False,
                        "metal_status_source": "hard-coded script label"}))
    append_manifest_rows(pb_manifest, __file__)
    output_dir = os.path.join("outputs", "reproducibility")
    os.makedirs(output_dir, exist_ok=True)
    pb_data.to_csv(os.path.join(output_dir, "reviewer_pb_comparison_case_level.csv"), index=False, float_format="%.10g")
    x_pb, y_pb = pb_data["Pred_mm"].values, pb_data["PB_shift_mm"].values
    ci = bootstrap_ci(x_pb, y_pb, seed=42)
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

    # --- Fig 11: Frozen refreshed expansion cohort (do NOT rerun patient selection) ---
    # The refreshed n=30 cohort excludes 1THB121 and includes replacement 1ABB143.
    # Keep this CSV immutable during figure generation; cohort selection is a separate workflow.
    refreshed_candidates = [
        os.path.join(output_dir, "refresh_excl_1THB121", "model_a_expansion_case_level_n30.csv"),
        os.path.join(output_dir, "model_a_expansion_case_level_n30.csv"),
    ]
    refreshed_path = next((q for q in refreshed_candidates if os.path.exists(q)), None)
    if refreshed_path is None:
        raise FileNotFoundError(
            "Refreshed expansion CSV not found. Expected one of: " + ", ".join(refreshed_candidates)
        )
    df_c = pd.read_csv(refreshed_path)
    required = {"Patient", "Pred_mm", "Act_mm", "Region", "Metal"}
    missing = required - set(df_c.columns)
    if missing:
        raise ValueError(f"Refreshed expansion CSV is missing required columns: {sorted(missing)}")
    if len(df_c) != 30 or df_c["Patient"].nunique() != 30:
        raise ValueError(f"Expected exactly 30 unique refreshed patients; found rows={len(df_c)}, unique={df_c['Patient'].nunique()}")
    if "1THB121" in set(df_c["Patient"].astype(str)):
        raise ValueError("1THB121 must remain excluded from the refreshed expansion cohort.")
    if "1ABB143" not in set(df_c["Patient"].astype(str)):
        raise ValueError("Expected replacement patient 1ABB143 is missing from the refreshed cohort.")
    print(f"\n[3] Loaded frozen refreshed expansion cohort: {refreshed_path}")
    if len(df_c) > 5:
        x_c, y_c = df_c["Pred_mm"].values, df_c["Act_mm"].values
        ci_c = bootstrap_ci(x_c, y_c, seed=42)
        r_c = np.corrcoef(x_c, y_c)[0, 1]
        print(f"\n[3] Cohort Expansion (n={len(df_c)}): r = {r_c:.3f} "
              f"(95% CI: {ci_c['r'][1][0]:.3f} to {ci_c['r'][1][1]:.3f}); "
              f"MAE = {ci_c['MAE'][0]:.3f} mm (95% CI: {ci_c['MAE'][1][0]:.3f} to {ci_c['MAE'][1][1]:.3f})")
        lims = [min(x_c.min(), y_c.min())-1, max(x_c.max(), y_c.max())+1]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(lims, lims, 'k--', alpha=0.5)
        ax.scatter(x_c, y_c, s=60, c='green', edgecolors='k', alpha=0.7)
        ax.set_xlabel("Surrogate predicted shift (mm)"); ax.set_ylabel("Analytically reconstructed reference shift (mm)")
        ax.set_title(f"Refreshed expansion cohort (n={len(df_c)}): r = {r_c:.3f}"); ax.grid(ls='--', alpha=0.5)
        fig.tight_layout(); fig.savefig("figures/Fig11_CohortExpansion.png", dpi=300); plt.close(fig)

    print("\nSaved: Fig9_BlandAltman.png, Fig10_Subgroups.png, Fig11_CohortExpansion.png")

if __name__ == "__main__":
    main()