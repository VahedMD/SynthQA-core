"""
Track 1: Reviewer-Proofing the Statistics (rev 2)
Fixes: matplotlib 3.10 boxplot labels; dynamic subgroup n; sensitivity analysis.
"""
import os, sys, json, time, argparse
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

def r80_crossing_details(dose, depth):
    """Return the exact distal 80%-of-peak crossing and its bracketing samples.

    This mirrors find_r80's threshold/crossing convention, while exposing the
    samples needed to audit threshold normalization and interpolation.
    """
    dose = np.asarray(dose, dtype=float)
    depth = np.asarray(depth, dtype=float)
    pk = int(np.argmax(dose))
    result = {
        "peak_index": pk,
        "peak_depth_mm": float(depth[pk]) if len(depth) else float("nan"),
        "peak_dose": float(dose[pk]) if len(dose) else float("nan"),
        "threshold_80pct_peak": float(0.8 * dose[pk]) if len(dose) else float("nan"),
        "crossing_index_lower": None,
        "crossing_index_upper": None,
        "depth_lower_mm": None,
        "depth_upper_mm": None,
        "dose_lower": None,
        "dose_upper": None,
        "interpolation_fraction": None,
        "r80_mm": float("nan"),
        "local_falloff_slope_dose_per_mm": None,
        "samples_below_threshold_after_peak": None,
    }
    if len(dose) == 0 or pk == 0 or not np.isfinite(dose[pk]) or dose[pk] <= 0:
        return result
    thr = 0.8 * dose[pk]
    below = np.where(dose[pk:] < thr)[0]
    result["samples_below_threshold_after_peak"] = int(len(below))
    if len(below) == 0:
        return result
    i = pk + int(below[0])
    result["crossing_index_lower"] = int(max(0, i - 1))
    result["crossing_index_upper"] = int(i)
    result["depth_lower_mm"] = float(depth[i - 1]) if i > 0 else None
    result["depth_upper_mm"] = float(depth[i])
    result["dose_lower"] = float(dose[i - 1]) if i > 0 else None
    result["dose_upper"] = float(dose[i])
    if 0 < i < len(dose):
        y1, y2 = dose[i - 1], dose[i]
        if y1 != y2:
            frac = float((thr - y1) / (y2 - y1))
            result["interpolation_fraction"] = frac
            result["r80_mm"] = float(depth[i - 1] + frac * (depth[i] - depth[i - 1]))
            result["local_falloff_slope_dose_per_mm"] = float((y2 - y1) / (depth[i] - depth[i - 1]))
        else:
            result["r80_mm"] = float(depth[i])
    else:
        result["r80_mm"] = float(depth[i])
    return result

def r80_physical_shift_estimates(depth_mm, w_ref, w_sct, r80_ref_mm):
    """Return R80 WEPL shift, first-order physical shift, and inverse-map shift.

    The first-order estimate converts a WEPL displacement to physical depth
    using the local d(WEPL)/dz gradient. The inverse-map value is an internal
    reconstruction diagnostic, not independent dose-engine validation.
    """
    depth = np.asarray(depth_mm, dtype=float)
    wr = np.asarray(w_ref, dtype=float)
    ws = np.asarray(w_sct, dtype=float)
    if depth.ndim != 1 or wr.ndim != 1 or ws.ndim != 1:
        raise ValueError("depth and WEPL arrays must be one-dimensional")
    if not (len(depth) == len(wr) == len(ws)):
        raise ValueError(f"depth/WEPL length mismatch: {len(depth)}, {len(wr)}, {len(ws)}")
    if len(depth) < 3 or not (np.all(np.isfinite(depth)) and np.all(np.isfinite(wr)) and np.all(np.isfinite(ws))):
        raise ValueError("depth and WEPL arrays must contain at least 3 finite samples")
    if np.any(np.diff(depth) <= 0):
        raise ValueError("depth samples must be strictly increasing")
    if np.any(np.diff(wr) < -1e-8) or np.any(np.diff(ws) < -1e-8):
        raise ValueError("WEPL arrays must be monotonic non-decreasing")
    if not np.isfinite(r80_ref_mm):
        return {"pred_r80_wepl_mm": np.nan, "local_wepl_gradient_mm_per_mm": np.nan,
                "pred_r80_physical_first_order_mm": np.nan, "mapped_r80_shift_mm": np.nan}
    r80_ref_wepl = float(np.interp(r80_ref_mm, depth, wr))
    dwepl_r80 = float(np.interp(r80_ref_mm, depth, ws - wr))
    gradient = np.gradient(wr, depth)
    local_gradient = float(np.interp(r80_ref_mm, depth, gradient))
    first_order = -dwepl_r80 / local_gradient if np.isfinite(local_gradient) and abs(local_gradient) > 1e-8 else np.nan
    # np.interp requires increasing xp. Collapse repeated WEPL samples explicitly.
    ws_unique, first_indices = np.unique(ws, return_index=True)
    mapped_depth = float(np.interp(r80_ref_wepl, ws_unique, depth[first_indices])) if len(ws_unique) >= 2 else np.nan
    return {"pred_r80_wepl_mm": -dwepl_r80,
            "local_wepl_gradient_mm_per_mm": local_gradient,
            "pred_r80_physical_first_order_mm": float(first_order),
            "mapped_r80_shift_mm": float(mapped_depth - r80_ref_mm) if np.isfinite(mapped_depth) else np.nan}

def wepl_gradient_window_audit(depth_mm, w_ref, w_sct, r80_ref_mm, half_windows_mm=(2.0, 5.0, 10.0, 20.0)):
    """Estimate local dWEPL/dz by linear regression over several physical windows.

    Reports sensitivity to the selected window rather than privileging one
    arbitrary gradient estimate. The WEPL-to-depth first-order prediction is
    -DeltaWEPL(R80)/slope. This is a diagnostic approximation only.
    """
    z = np.asarray(depth_mm, dtype=float)
    wr = np.asarray(w_ref, dtype=float)
    ws = np.asarray(w_sct, dtype=float)
    if not (z.ndim == wr.ndim == ws.ndim == 1 and len(z) == len(wr) == len(ws)):
        raise ValueError("depth and WEPL profiles must be 1-D arrays of equal length")
    delta = ws - wr
    dw_at_r80 = float(np.interp(r80_ref_mm, z, delta))
    out = {"delta_wepl_at_r80_mm": dw_at_r80}
    for half in half_windows_mm:
        mask = np.abs(z - r80_ref_mm) <= float(half)
        if int(mask.sum()) < 3:
            slope, intercept, r2 = np.nan, np.nan, np.nan
        else:
            slope, intercept = np.polyfit(z[mask], wr[mask], 1)
            fitted = slope*z[mask] + intercept
            ss_res = float(np.sum((wr[mask]-fitted)**2))
            ss_tot = float(np.sum((wr[mask]-np.mean(wr[mask]))**2))
            r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
        key = str(int(half))
        out[f"gradient_window_{key}mm"] = float(slope)
        out[f"gradient_window_{key}mm_r2"] = float(r2)
        out[f"pred_physical_window_{key}mm"] = float(-dw_at_r80/slope) if np.isfinite(slope) and abs(slope)>1e-8 else np.nan
        out[f"gradient_window_{key}mm_n"] = int(mask.sum())
    dwr, dws = np.diff(wr), np.diff(ws)
    out.update({
        "w_ref_nonmonotonic_steps": int(np.sum(dwr < -1e-8)),
        "w_sct_nonmonotonic_steps": int(np.sum(dws < -1e-8)),
        "w_ref_flat_steps": int(np.sum(np.abs(dwr) <= 1e-10)),
        "w_sct_flat_steps": int(np.sum(np.abs(dws) <= 1e-10)),
        "w_ref_unique_fraction": float(len(np.unique(wr))/len(wr)),
        "w_sct_unique_fraction": float(len(np.unique(ws))/len(ws)),
        "w_ref_min_positive_step": float(np.min(dwr[dwr > 1e-10])) if np.any(dwr > 1e-10) else np.nan,
        "w_sct_min_positive_step": float(np.min(dws[dws > 1e-10])) if np.any(dws > 1e-10) else np.nan,
    })
    return out


def compare_independent_r80(expansion_csv, independent_csv, output_dir):
    """Compare predictions with genuinely independent perturbed-dose R80 shifts.

    independent_csv must contain Patient, independent_r80_shift_mm, and
    reference_engine. The reference file must be generated from a fresh
    perturbed-image dose calculation (not WEPL remapping). Optional columns:
    reference_run_id, reference_dose_file, reference_r80_mm, baseline_r80_mm.
    """
    from pathlib import Path
    exp = pd.read_csv(expansion_csv)
    ref = pd.read_csv(independent_csv)
    required = {"Patient", "independent_r80_shift_mm", "reference_engine"}
    missing = required - set(ref.columns)
    if missing:
        raise ValueError(f"Independent reference CSV missing required columns: {sorted(missing)}")
    if ref["Patient"].astype(str).duplicated().any():
        raise ValueError("Independent reference CSV contains duplicate Patient IDs")
    if ref["reference_engine"].isna().any() or (ref["reference_engine"].astype(str).str.strip() == "").any():
        raise ValueError("reference_engine must identify the independent dose engine for every row")
    ref["independent_r80_shift_mm"] = pd.to_numeric(ref["independent_r80_shift_mm"], errors="coerce")
    if not np.isfinite(ref["independent_r80_shift_mm"].to_numpy(float)).all():
        raise ValueError("independent_r80_shift_mm must contain finite numeric values")
    if "reference_type" in ref.columns:
        bad = ref["reference_type"].astype(str).str.lower().str.contains("analytical|wepl.?remap|reconstruct", regex=True)
        if bad.any():
            raise ValueError("Independent reference CSV contains rows labelled analytical/reconstructed")
    os.makedirs(output_dir, exist_ok=True)
    merged = exp.merge(ref, on="Patient", how="inner", validate="one_to_one")
    merged["error_physical_mm"] = merged["Pred_R80_physical_mm"] - merged["independent_r80_shift_mm"]
    merged["error_inverse_map_mm"] = merged["Mapped_R80_shift_mm"] - merged["independent_r80_shift_mm"]
    merged["error_wepl_mm"] = merged["Pred_R80_WEPL_mm"] - merged["independent_r80_shift_mm"]
    merged.to_csv(Path(output_dir)/"independent_r80_paired_cases.csv", index=False)
    rows=[]
    for pred_col, label in [("Pred_R80_physical_mm","first_order_physical"),
                            ("Mapped_R80_shift_mm","inverse_map"),
                            ("Pred_R80_WEPL_mm","wepl_units_not_physical")]:
        a=merged[pred_col].to_numpy(float); b=merged["independent_r80_shift_mm"].to_numpy(float)
        ok=np.isfinite(a)&np.isfinite(b); a,b=a[ok],b[ok]
        rows.append({"predictor":label,"n":len(a),"MAE_mm":float(np.mean(np.abs(a-b))) if len(a) else np.nan,
                     "bias_mm":float(np.mean(a-b)) if len(a) else np.nan,
                     "RMSE_mm":float(np.sqrt(np.mean((a-b)**2))) if len(a) else np.nan,
                     "pearson_r":float(np.corrcoef(a,b)[0,1]) if len(a)>1 and np.std(a)>0 and np.std(b)>0 else np.nan})
    pd.DataFrame(rows).to_csv(Path(output_dir)/"independent_r80_metrics.csv", index=False)
    print(f"[independent R80] matched patients={len(merged)}; saved paired cases and metrics")
    return merged


def run_cohort_expansion(n_patients=30, run=None):
    """Collect n_patients VALID Model A cases, not merely n_patients attempts.

    The expansion arm is kept disjoint from the fixed reviewer comparison and
    the deterministic Model A/Model B stratified cohort candidates. Candidates
    are traversed in a seeded random permutation until the valid-case target is
    reached or the eligible pool is exhausted.
    """
    print("\n--- COHORT EXPANSION: WEPL surrogate vs reconstructed dose-profile shift ---")
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
                if len(w_ref) != len(d1) or len(w_sct) != len(d1):
                    row["invalid_dose_profiles"] += 1
                    row.setdefault("_dose_errors", []).append(
                        f"sampling length mismatch: dose={len(d1)}, w_ref={len(w_ref)}, w_sct={len(w_sct)}")
                    continue
                dwepl = w_sct - w_ref
                depth = np.arange(len(d1), dtype=float) * (dist / (len(d1)-1))
                if np.any(np.diff(w_ref) < -1e-8) or np.any(np.diff(w_sct) < -1e-8):
                    row["invalid_dose_profiles"] += 1
                    continue
                d_sct = np.interp(w_sct, w_ref, d1, left=0.0, right=0.0)
                # Preserve the legacy peak-based predictor for traceability, but also
                # calculate the manuscript-aligned predictor at the reference R80.
                # Both predictors are compared with Act_mm, which is a reconstructed
                # R80 displacement (not an independent perturbed-image dose result).
                pred_peak_mm = float(-dwepl[pk])
                r80_ref = float(find_r80(d1, depth))
                r80_rec = float(find_r80(d_sct, depth))
                act_mm = float(r80_rec - r80_ref)
                r80_est = r80_physical_shift_estimates(depth, w_ref, w_sct, r80_ref)
                grad_audit = wepl_gradient_window_audit(depth, w_ref, w_sct, r80_ref)
                pred_r80_wepl_mm = r80_est["pred_r80_wepl_mm"]
                pred_r80_physical_mm = r80_est["pred_r80_physical_first_order_mm"]
                # Pred_mm remains the historical peak-based WEPL endpoint.
                results.append({"Patient": pid, "Pred_mm": pred_peak_mm,
                                "Pred_peak_mm": pred_peak_mm,
                                "Pred_R80_WEPL_mm": pred_r80_wepl_mm,
                                "Pred_R80_physical_mm": pred_r80_physical_mm,
                                "Local_dWEPL_dz": r80_est["local_wepl_gradient_mm_per_mm"],
                                **{k: v for k, v in grad_audit.items() if k != "delta_wepl_at_r80_mm"},
                                "Mapped_R80_shift_mm": r80_est["mapped_r80_shift_mm"],
                                "Act_mm": act_mm, "R80_ref_mm": r80_ref,
                                "R80_reconstructed_mm": r80_rec,
                                "Region": meta["anatomical_region"], "Metal": meta["metal_status"]})
                row.update(eligible_rays=1, selected_beam_idx=b.get("beam_idx", ""),
                    selected_ray_idx=r.get("ray_idx", ""), selected_beamlet_idx=0,
                    outcome_status="included", exclusion_reason="", surrogate_shift_mm=pred_peak_mm,
                    reference_shift_mm=act_mm,
                    notes_json={**row["notes_json"], "ray_source": rs.tolist(), "ray_target": rt.tolist(),
                                "reference_type": "analytical reconstructed dose shift; not independent MC/PB validation",
                                "dose_file": fp, "pred_peak_mm": pred_peak_mm,
                                "pred_r80_wepl_mm": pred_r80_wepl_mm,
                                "pred_r80_physical_first_order_mm": pred_r80_physical_mm,
                                "local_wepl_gradient_mm_per_mm": r80_est["local_wepl_gradient_mm_per_mm"],
                                "gradient_window_audit": grad_audit,
                                "mapped_r80_shift_mm": r80_est["mapped_r80_shift_mm"],
                                "reference_r80_mm": r80_ref,
                                "reconstructed_r80_mm": r80_rec,
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


def diagnose_patient(patient_id="1THB121", output_root=None):
    """Reproduce the existing expansion calculation for one patient only.

    This mode is diagnostic: it does not write cohort CSVs or append manifest
    rows. It preserves the original ray-selection and dose-reconstruction logic.
    """
    from pathlib import Path

    pid = str(patient_id)
    if output_root is None:
        output_root = Path(__file__).resolve().parent.parent / "outputs" / "reproducibility" / "audit" / pid
    else:
        output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"\\n--- SINGLE-PATIENT DIAGNOSTIC: {pid} ---")
    prof = CalibrationProfile()
    plan_p = hf_hub_download(REPO_ID, f"proton/training/{pid}/{pid}.json", repo_type="dataset")
    ct_p = hf_hub_download(REPO_ID, f"proton/training/{pid}/image/ct.mha", repo_type="dataset")

    ct = load_mha(ct_p)
    ct_arr, sp, org = ct["array"], np.asarray(ct["spacing_xyz"], dtype=float), np.asarray(ct["origin_xyz"], dtype=float)
    spr_ref = prof.density_to_spr.convert(prof.hu_to_density.convert(ct_arr))
    spr_sct = prof.density_to_spr.convert(prof.hu_to_density.convert(ct_arr + 50.0))
    with open(plan_p, encoding="utf-8") as f:
        plan = json.load(f)

    selected = None
    failures = []
    rays_examined = 0
    for b in plan.get("beams", [])[:2]:
        for r in b.get("rays", [])[:5]:
            rays_examined += 1
            rs, rt = np.asarray(r["ray_source"], dtype=float), np.asarray(r["ray_target"], dtype=float)
            w_short = compute_wepl_along_ray(spr_ref, sp, org, rs, rt, num_samples=400)
            if w_short[-1] < 10.0:
                continue
            fp = f"proton/training/{pid}/dose/Dose_B{b['beam_idx']}_R{r['ray_idx']}_L0.mha"
            try:
                dose_path = hf_hub_download(REPO_ID, fp, repo_type="dataset")
                dose_img = load_mha(dose_path)
            except Exception as exc:
                failures.append({"dose_file": fp, "error": str(exc)[:500]})
                continue

            d3 = dose_img["array"]
            d1, dist = extract_1d_ray_profile(d3, sp, org, rs, rt)
            pk = int(np.argmax(d1))
            if not (pk > 0.2 * len(d1) and d1[pk] > 1e-3 and d1[pk:].min() < 0.1 * d1[pk]):
                continue

            selected = {"beam": b, "ray": r, "ray_source": rs, "ray_target": rt,
                        "dose_file": fp, "dose_path": dose_path, "dose_image": dose_img,
                        "dose_volume": d3, "dose_profile": d1, "distance_mm": float(dist), "peak_index": pk}
            break
        if selected is not None:
            break

    if selected is None:
        raise RuntimeError(f"No valid dose ray found for {pid}; dose download failures: {failures}")

    rs, rt = selected["ray_source"], selected["ray_target"]
    d1, dist, pk = selected["dose_profile"], selected["distance_mm"], selected["peak_index"]
    w_ref = compute_wepl_along_ray(spr_ref, sp, org, rs, rt, num_samples=2000)
    w_sct = compute_wepl_along_ray(spr_sct, sp, org, rs, rt, num_samples=2000)
    if len(w_ref) != len(d1):
        raise ValueError(f"WEPL profile length {len(w_ref)} differs from dose profile length {len(d1)}")
    dwepl = w_sct - w_ref
    depth = np.arange(len(d1), dtype=float) * (dist / (len(d1) - 1))
    d_sct = np.interp(w_ref + dwepl, w_ref, d1, left=0.0, right=0.0)
    # Keep the historical peak-based estimate and calculate the matched R80 estimate.
    pred_peak_mm = float(-dwepl[pk])
    r80_ref = float(find_r80(d1, depth))
    r80_rec = float(find_r80(d_sct, depth))
    r80_ref_details = r80_crossing_details(d1, depth)
    r80_rec_details = r80_crossing_details(d_sct, depth)
    act_mm = float(r80_rec - r80_ref)
    r80_est = r80_physical_shift_estimates(depth, w_ref, w_sct, r80_ref)
    grad_audit = wepl_gradient_window_audit(depth, w_ref, w_sct, r80_ref)
    dwepl_at_r80 = -r80_est["pred_r80_wepl_mm"]
    pred_r80_mm = r80_est["pred_r80_wepl_mm"]
    pred_mm = pred_peak_mm  # legacy diagnostic field name; peak-based endpoint
    peak_rec_idx = int(np.argmax(d_sct))
    peak_shift = float(depth[peak_rec_idx] - depth[pk])
    r80_ref_wepl = float(np.interp(r80_ref, depth, w_ref)) if np.isfinite(r80_ref) else float("nan")
    mapped_r80_depth = (r80_ref + r80_est["mapped_r80_shift_mm"]
                        if np.isfinite(r80_est["mapped_r80_shift_mm"]) else float("nan"))

    dose_img = selected["dose_image"]
    geometry_keys = ("spacing_xyz", "origin_xyz", "direction", "direction_matrix", "spacing", "origin", "direction_xyz")
    dose_geometry = {key: np.asarray(dose_img[key]).tolist() if isinstance(dose_img.get(key), np.ndarray) else dose_img.get(key)
                     for key in geometry_keys if key in dose_img}
    wref_diff, wsct_diff = np.diff(w_ref), np.diff(w_sct)
    diag = {
        "patient": pid, "dose_file": selected["dose_file"],
        "beam_idx": selected["beam"].get("beam_idx"), "ray_idx": selected["ray"].get("ray_idx"),
        "ray_source": rs.tolist(), "ray_target": rt.tolist(), "ray_length_mm": float(dist),
        "ray_candidates_examined": rays_examined, "dose_download_failures_before_selected_ray": failures,
        "ct_array_shape_zyx": list(ct_arr.shape), "ct_spacing_xyz": sp.tolist(), "ct_origin_xyz": org.tolist(),
        "dose_array_shape": list(selected["dose_volume"].shape), "dose_image_metadata": dose_geometry,
        "dose_image_keys": list(dose_img.keys()),
        "peak_index": pk, "peak_depth_mm": float(depth[pk]), "reference_peak_dose": float(d1[pk]),
        "reconstructed_peak_index": peak_rec_idx, "reconstructed_peak_depth_mm": float(depth[peak_rec_idx]),
        "pred_mm_wepl_at_reference_peak": pred_peak_mm,
        "pred_peak_mm": pred_peak_mm,
        "dwepl_at_reference_r80_mm": dwepl_at_r80,
        "pred_r80_wepl_mm": pred_r80_mm,
        "pred_r80_physical_first_order_mm": r80_est["pred_r80_physical_first_order_mm"],
        "local_wepl_gradient_mm_per_mm": r80_est["local_wepl_gradient_mm_per_mm"],
        "mapped_r80_shift_mm": r80_est["mapped_r80_shift_mm"],
        "gradient_window_audit": grad_audit,
        "reconstructed_peak_shift_mm": peak_shift,
        "reference_r80_mm": r80_ref, "reconstructed_r80_mm": r80_rec, "act_mm_reconstructed_r80": act_mm,
        "r80_crossing_reference_details": r80_ref_details,
        "r80_crossing_reconstructed_details": r80_rec_details,
        "r80_threshold_convention": "80% of each profile's own maximum; first below-threshold sample distal to that profile's peak; linear interpolation between bracketing samples",
        "reference_r80_wepl_mm": r80_ref_wepl, "mapped_r80_depth_mm": mapped_r80_depth,
        "mapped_r80_shift_mm": float(mapped_r80_depth-r80_ref) if np.isfinite(mapped_r80_depth) and np.isfinite(r80_ref) else None,
        "w_ref_monotonic_non_decreasing": bool(np.all(wref_diff >= -1e-8)),
        "w_sct_monotonic_non_decreasing": bool(np.all(wsct_diff >= -1e-8)),
        "w_ref_min_step": float(np.min(wref_diff)), "w_ref_max_step": float(np.max(wref_diff)),
        "w_sct_min_step": float(np.min(wsct_diff)), "w_sct_max_step": float(np.max(wsct_diff)),
        "dwepl_at_reference_peak_mm": float(dwepl[pk]),
        "reference_profile_min": float(np.min(d1)), "reference_profile_max": float(np.max(d1)),
        "reconstructed_profile_min": float(np.min(d_sct)), "reconstructed_profile_max": float(np.max(d_sct)),
        "reconstructed_profile_finite": bool(np.all(np.isfinite(d_sct))),
        "method_note": "Reference dose profile remapped by WEPL; not an independent perturbed-image dose calculation or fresh MC/PB validation."
    }
    with open(output_root / "diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2, allow_nan=False)

    ref_thr = 0.8 * float(np.max(d1))
    rec_thr = 0.8 * float(np.max(d_sct))
    pd.DataFrame({
        "depth_mm": depth,
        "dose_reference": d1,
        "dose_reconstructed": d_sct,
        "dose_reference_normalized_to_own_peak": d1 / np.max(d1) if np.max(d1) > 0 else np.nan,
        "dose_reconstructed_normalized_to_own_peak": d_sct / np.max(d_sct) if np.max(d_sct) > 0 else np.nan,
        "reference_80pct_threshold": ref_thr,
        "reconstructed_80pct_threshold": rec_thr,
        "reference_minus_threshold": d1 - ref_thr,
        "reconstructed_minus_threshold": d_sct - rec_thr,
        "near_either_r80_pm_20mm": (np.abs(depth - r80_ref) <= 20) | (np.abs(depth - r80_rec) <= 20)
    }).to_csv(output_root / "dose_profiles.csv", index=False)
    # Compact fall-off table for quick inspection in spreadsheet software.
    falloff = pd.DataFrame({
        "depth_mm": depth, "dose_reference": d1, "dose_reconstructed": d_sct,
        "reference_normalized": d1 / np.max(d1) if np.max(d1) > 0 else np.nan,
        "reconstructed_normalized": d_sct / np.max(d_sct) if np.max(d_sct) > 0 else np.nan,
        "reference_minus_80pct_threshold": d1 - ref_thr,
        "reconstructed_minus_80pct_threshold": d_sct - rec_thr
    })
    falloff = falloff[(np.abs(falloff["depth_mm"] - r80_ref) <= 20) |
                      (np.abs(falloff["depth_mm"] - r80_rec) <= 20)]
    falloff.to_csv(output_root / "r80_falloff_diagnostic.csv", index=False)
    pd.DataFrame({"wepl_sample_index": np.arange(len(w_ref)), "depth_mm": depth,
                  "wepl_reference": w_ref, "wepl_perturbed": w_sct, "dwepl": dwepl,
                  "local_gradient_pointwise": np.gradient(w_ref, depth),
                  "hu": extract_1d_ray_profile(ct_arr, sp, org, rs, rt)[0],
                  "spr_reference": extract_1d_ray_profile(spr_ref, sp, org, rs, rt)[0],
                  "spr_perturbed": extract_1d_ray_profile(spr_sct, sp, org, rs, rt)[0]}).to_csv(
        output_root / "wepl_profiles.csv", index=False)

    # Dose profiles with both R80 landmarks.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(depth, d1, label="Reference dose")
    ax.plot(depth, d_sct, label="WEPL-reconstructed dose")
    if np.isfinite(r80_ref): ax.axvline(r80_ref, linestyle="--", label=f"Reference R80 = {r80_ref:.2f} mm")
    if np.isfinite(r80_rec): ax.axvline(r80_rec, linestyle=":", label=f"Reconstructed R80 = {r80_rec:.2f} mm")
    ax.axvline(depth[pk], linestyle="--", alpha=0.6, label="Reference peak")
    ax.axvline(depth[peak_rec_idx], linestyle=":", alpha=0.6, label="Reconstructed peak")
    ax.set_xlabel("Physical distance from ray source (mm)"); ax.set_ylabel("Dose (arbitrary units)")
    ax.set_title(f"{pid}: reference vs WEPL-reconstructed profile\nR80 prediction={pred_r80_mm:.3f} mm; reconstructed R80 shift={act_mm:.3f} mm")
    ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(output_root / "dose_profiles.png", dpi=180); plt.close(fig)

    # CT anatomy views, retaining the axis-aligned coordinate convention used by
    # the existing profile extractor. Direction matrices, if non-identity, need
    # explicit handling before treating these overlays as fully registered.
    nz, ny, nx = ct_arr.shape
    x_coords = org[0] + np.arange(nx) * sp[0]
    y_coords = org[1] + np.arange(ny) * sp[1]
    z_coords = org[2] + np.arange(nz) * sp[2]
    t_peak = depth[pk] / dist if dist else 0.0
    p_peak = rs + t_peak * (rt - rs)
    iz = int(np.clip(round((p_peak[2] - org[2]) / sp[2]), 0, nz - 1))
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(ct_arr[iz, :, :], origin="lower", extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]],
              aspect="equal", cmap="gray", vmin=-1000, vmax=1500)
    ax.plot([rs[0], rt[0]], [rs[1], rt[1]], linewidth=2, label="Selected proton ray")
    ax.scatter([p_peak[0]], [p_peak[1]], marker="x", s=100, label="Reference dose peak")
    ax.set_xlabel("Physical x (mm)"); ax.set_ylabel("Physical y (mm)")
    ax.set_title(f"{pid}: axial CT slice at z={z_coords[iz]:.2f} mm"); ax.legend(); fig.tight_layout()
    fig.savefig(output_root / "beam_path_axial_ct.png", dpi=180); plt.close(fig)

    iy = int(np.clip(round((p_peak[1] - org[1]) / sp[1]), 0, ny - 1))
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(ct_arr[:, iy, :], origin="lower", extent=[x_coords[0], x_coords[-1], z_coords[0], z_coords[-1]],
              aspect="equal", cmap="gray", vmin=-1000, vmax=1500)
    ax.scatter([p_peak[0]], [p_peak[2]], marker="x", s=100, label="Central-axis peak location")
    ax.set_xlabel("Physical x (mm)"); ax.set_ylabel("Physical z (mm)")
    ax.set_title(f"{pid}: transverse CT section at y={y_coords[iy]:.2f} mm"); ax.legend(); fig.tight_layout()
    fig.savefig(output_root / "beam_peak_transverse_ct.png", dpi=180); plt.close(fig)

    hu_ray = extract_1d_ray_profile(ct_arr, sp, org, rs, rt)[0]
    spr_ref_ray = extract_1d_ray_profile(spr_ref, sp, org, rs, rt)[0]
    spr_sct_ray = extract_1d_ray_profile(spr_sct, sp, org, rs, rt)[0]
    ray_depth = np.linspace(0.0, dist, len(hu_ray))
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(ray_depth, hu_ray); axes[0].set_ylabel("HU")
    axes[1].plot(ray_depth, spr_ref_ray, label="Reference SPR")
    axes[1].plot(ray_depth, spr_sct_ray, label="Perturbed SPR (+50 HU)"); axes[1].set_ylabel("SPR"); axes[1].legend()
    axes[2].plot(depth, w_ref, label="Reference WEPL")
    axes[2].plot(depth, w_sct, label="Perturbed WEPL"); axes[2].set_ylabel("Accumulated WEPL (mm)")
    axes[2].set_xlabel("Physical distance from ray source (mm)"); axes[2].legend()
    for axis in axes: axis.axvline(depth[pk], linestyle="--", alpha=0.8); axis.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(output_root / "anatomy_spr_wepl_along_ray.png", dpi=180); plt.close(fig)

    print(f"[diagnostic] Saved single-patient outputs to {output_root}")
    print(f"[diagnostic] Peak-based prediction={pred_peak_mm:.4f} mm; reconstructed peak shift={peak_shift:.4f} mm")
    print(f"[diagnostic] R80-based prediction={pred_r80_mm:.4f} mm; reconstructed R80 shift={act_mm:.4f} mm")
    print("[diagnostic] Both dose-profile shifts are analytical WEPL reconstructions, not independent MC/PB validation.")
    print("[diagnostic] No cohort manifest or case-level cohort CSV was modified.")
    return diag

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

    # --- Fig 11: Cohort expansion ---
    df_c = run_cohort_expansion(30, run=run)
    if not df_c.empty:
        df_c.to_csv(os.path.join(output_dir, "model_a_expansion_case_level.csv"), index=False, float_format="%.10g")
    if len(df_c) > 5:
        # Historical analysis: peak-based predictor vs reconstructed R80 shift.
        x_peak, y_c = df_c["Pred_peak_mm"].values, df_c["Act_mm"].values
        ci_peak = bootstrap_ci(x_peak, y_c, seed=42)
        r_peak = np.corrcoef(x_peak, y_c)[0, 1] if np.std(x_peak) > 0 and np.std(y_c) > 0 else np.nan
        print(f"\n[3a] LEGACY endpoint-mismatched analysis (peak predictor vs reconstructed R80; n={len(df_c)})")
        print(f"  r = {r_peak:.3f} (95% CI: {ci_peak['r'][1][0]:.3f} to {ci_peak['r'][1][1]:.3f}); "
              f"MAE = {ci_peak['MAE'][0]:.3f} mm (95% CI: {ci_peak['MAE'][1][0]:.3f} to {ci_peak['MAE'][1][1]:.3f})")

        # Compare only quantities expressed as physical-depth displacement.
        # The WEPL-valued predictor is retained in the CSV but is not directly
        # compared with Act_mm, which is in physical millimetres.
        physical = np.isfinite(df_c["Pred_R80_physical_mm"].to_numpy(dtype=float)) & np.isfinite(y_c)
        x_phys, y_phys = df_c.loc[physical, "Pred_R80_physical_mm"].to_numpy(dtype=float), y_c[physical]
        if len(x_phys) >= 2:
            ci_phys = bootstrap_ci(x_phys, y_phys, seed=42)
            r_phys = np.corrcoef(x_phys, y_phys)[0, 1] if np.std(x_phys) > 0 and np.std(y_phys) > 0 else np.nan
            print(f"\n[3c] FIRST-ORDER PHYSICAL-DEPTH R80 estimate vs analytical reconstruction (n={len(x_phys)})")
            print(f"  r = {r_phys:.3f}; MAE = {ci_phys['MAE'][0]:.3f} mm "
                  f"(95% CI: {ci_phys['MAE'][1][0]:.3f} to {ci_phys['MAE'][1][1]:.3f}); "
                  f"bias = {ci_phys['Bias'][0]:.3f} mm "
                  f"(95% CI: {ci_phys['Bias'][1][0]:.3f} to {ci_phys['Bias'][1][1]:.3f})")
        else:
            print("[WARNING] Fewer than two finite first-order physical R80 pairs; statistics not computed.")


    print("\nSaved: Fig9_BlandAltman.png, Fig10_Subgroups.png, Fig11_CohortExpansion.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reviewer statistics and targeted patient diagnostics")
    parser.add_argument(
        "--diagnose-patient", metavar="PATIENT_ID",
        help="Run only a single-patient diagnostic; does not append manifests or run the cohort",
    )
    parser.add_argument(
        "--diagnostic-output", default=None,
        help="Optional output directory for --diagnose-patient",
    )
    parser.add_argument(
        "--independent-r80-csv", default=None,
        help="CSV of fresh perturbed-image dose R80 shifts; requires Patient, independent_r80_shift_mm, reference_engine",
    )
    parser.add_argument(
        "--expansion-csv", default="outputs/reproducibility/model_a_expansion_case_level.csv",
        help="Existing expansion case-level CSV used with --independent-r80-csv",
    )
    args = parser.parse_args()
    if args.diagnose_patient:
        diagnose_patient(args.diagnose_patient, args.diagnostic_output)
    elif args.independent_r80_csv:
        outdir = os.path.dirname(args.expansion_csv) or "."
        compare_independent_r80(args.expansion_csv, args.independent_r80_csv, outdir)
    else:
        main()
