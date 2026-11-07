"""
Stratified DoseRAD2026 Benchmark (v10 - Proximal Bone Tracking & Clean Output)
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import matplotlib.lines as mlines
from huggingface_hub import hf_hub_download, list_repo_files
import matplotlib.pyplot as plt

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from synthqa_core.io.image_loader import load_mha
from synthqa_core.calibration.profiles import CalibrationProfile
from synthqa_core.surrogates.wepl import compute_wepl_along_ray
from cohort_manifest import new_run, base_record, append_manifest_rows, discover_stratified_cohort

REPO_ID = "LMUK-RADONC-PHYS-RES/DoseRAD2026"

MODE         = "bone_only"   
BONE_THRESH  = 300.0         
BONE_SHIFT   = -200.0        

def apply_sct_error(ct_arr):
    sct = ct_arr.copy()
    sct[ct_arr > BONE_THRESH] += BONE_SHIFT
    return sct

def range_mm(E_mev):
    return 0.022 * E_mev ** 1.77

def discover_patients():
    """Use the shared deterministic stratified-cohort definition."""
    return discover_stratified_cohort(REPO_ID)

def extract_1d_ray_profile(vol, sp, org, rs, rt, num_samples=2000):
    from scipy.ndimage import map_coordinates
    t = np.linspace(0, 1, num_samples)
    p = rs[None, :] + t[:, None] * (rt - rs)[None, :]
    ix, iy, iz = (p[:,0]-org[0])/sp[0], (p[:,1]-org[1])/sp[1], (p[:,2]-org[2])/sp[2]
    prof = map_coordinates(vol, np.array([iz, iy, ix]), order=1, mode='constant', cval=0.0)
    dist = np.linalg.norm(rt - rs)
    return prof, dist, dist/(num_samples-1)

def find_r80(dose, depth):
    pk = np.argmax(dose)
    if pk == 0 or dose[pk] <= 0: return np.nan
    thr = 0.8 * dose[pk]
    below = np.where(dose[pk:] < thr)[0]
    if len(below) == 0: return np.nan
    i = pk + below[0]
    if 0 < i < len(dose):
        y1, y2 = dose[i-1], dose[i]
        if y1 != y2:
            return depth[i-1] + (thr - y1) * (depth[i] - depth[i-1]) / (y2 - y1)
    return depth[i]

def run():
    run_meta = new_run(__file__)
    prof = CalibrationProfile()
    results, timings, manifest_rows = [], [], []
    plt.figure(figsize=(10, 7))

    patient_configs = discover_patients()
    for selection_order, cfg in enumerate(patient_configs, start=1):
        pid = cfg["pid"]
        row = base_record(run_meta, "model_b_stratified_bone_water", pid,
            anatomical_region=cfg["region"], metal_status=bool(cfg["metal"]),
            selection_method="deterministic sorted patient IDs by prefix and metal-status quotas; first n per stratum",
            perturbation_model="Model B: high-HU reduction", perturbation_threshold_hu=BONE_THRESH,
            perturbation_delta_hu=BONE_SHIFT, candidate_rays_examined=0, candidate_beamlets_examined=0,
            rays_rejected_short_wepl=0, dose_download_failures=0, invalid_dose_profiles=0,
            dose_engine_failures=0, eligible_rays=0, outcome_status="selected_pending",
            notes_json={"selection_order": selection_order,
                        "target_stratum_counts": {"thorax_metal": 3, "thorax_clean": 3, "abdomen_metal": 2, "abdomen_clean": 4},
                        "mode": MODE})
        print(f"\n{cfg['region']} {pid} (Metal={cfg['metal']})...", end=" ")
        try:
            plan_p = hf_hub_download(REPO_ID, f"proton/training/{pid}/{pid}.json", repo_type="dataset")
            ct_p   = hf_hub_download(REPO_ID, f"proton/training/{pid}/image/ct.mha", repo_type="dataset")
        except Exception as exc:
            row.update(outcome_status="excluded", exclusion_reason="plan_or_CT_download_failed",
                       notes_json={**row["notes_json"], "error": str(exc)[:500]})
            manifest_rows.append(row)
            print("[missing]"); continue

        try:
            ct = load_mha(ct_p)
        except Exception as exc:
            row.update(outcome_status="excluded", exclusion_reason="CT_parse_failed",
                       notes_json={**row["notes_json"], "error": str(exc)[:500]})
            manifest_rows.append(row)
            print("[invalid CT]"); continue
        ct_arr, sp, org = ct["array"], ct["spacing_xyz"], ct["origin_xyz"]
        spr_ref = prof.density_to_spr.convert(prof.hu_to_density.convert(ct_arr))
        spr_sct = prof.density_to_spr.convert(prof.hu_to_density.convert(apply_sct_error(ct_arr)))

        try:
            with open(plan_p, encoding="utf-8") as f:
                plan = json.load(f)
        except Exception as exc:
            row.update(outcome_status="excluded", exclusion_reason="plan_json_parse_failed",
                       notes_json={**row["notes_json"], "error": str(exc)[:500]})
            manifest_rows.append(row)
            print("[invalid plan]"); continue

        # Pre-selection: Find rays that have *any* bone in the path to prioritize downloads
        bone_candidates = []
        water_candidates = []
        for b in plan["beams"][:4]:
            for r in b["rays"]:
                row["candidate_rays_examined"] += 1
                rs, rt = np.array(r["ray_source"]), np.array(r["ray_target"])
                w = compute_wepl_along_ray(spr_ref, sp, org, rs, rt, num_samples=400)
                if w[-1] < 10.0:
                    row["rays_rejected_short_wepl"] += 1
                    continue
                target_wepl = 0.9 * w[-1]
                dist = np.linalg.norm(rt - rs)
                ct1, _, _ = extract_1d_ray_profile(ct_arr, sp, org, rs, rt, num_samples=400)
                bone_mm = dist * np.mean(ct1 > BONE_THRESH)
                for bl in r["beamlets"]:
                    row["candidate_beamlets_examined"] += 1
                    e_fit = abs(range_mm(bl["energy"]) - target_wepl)
                    cand = (e_fit, b, r, bl, rs, rt, bone_mm)
                    if bone_mm > 1.0: bone_candidates.append(cand)
                    else: water_candidates.append(cand)
                        
        bone_candidates.sort(key=lambda c: c[0])
        water_candidates.sort(key=lambda c: c[0])
        candidates = bone_candidates[:3] + water_candidates[:2]
        
        if not candidates:
            row.update(outcome_status="excluded", exclusion_reason="no_candidate_rays_after_wepl_filter")
            manifest_rows.append(row)
            print("SKIPPED"); continue

        found = False
        for _, b, r, bl, rs, rt, _ in candidates:
            b_idx, r_idx, l_idx = b.get("beam_idx",0), r.get("ray_idx",0), bl.get("beamlet_idx",0)
            fp = f"proton/training/{pid}/dose/Dose_B{b_idx}_R{r_idx}_L{l_idx}.mha"
            try:
                d3 = load_mha(hf_hub_download(REPO_ID, fp, repo_type="dataset"))["array"]
            except Exception:
                row["dose_download_failures"] += 1
                continue
            d1, dist, step = extract_1d_ray_profile(d3, sp, org, rs, rt, num_samples=2000)
            pk = np.argmax(d1)
            if pk > 0.2*len(d1) and d1[pk] > 1e-3 and d1[pk:].min() < 0.1*d1[pk]:
                found = True
                row.update(selected_beam_idx=b_idx, selected_ray_idx=r_idx, selected_beamlet_idx=l_idx,
                           eligible_rays=1, outcome_status="included", exclusion_reason="",
                           notes_json={**row["notes_json"], "selected_dose_file": fp,
                                       "selected_energy_mev": bl.get("energy"),
                                       "ray_source": rs.tolist(), "ray_target": rt.tolist(),
                                       "eligible_rays_semantics": "selected valid ray; search stops after first valid profile"})
                print(f"(B{b_idx}R{r_idx}L{l_idx})", end=" ")
                break
            else:
                row["invalid_dose_profiles"] += 1
        if not found:
            row.update(outcome_status="excluded", exclusion_reason="no_valid_dose_profile_in_candidate_set")
            manifest_rows.append(row)
            print("SKIPPED"); continue

        # STRICT PROXIMAL BONE CHECK: Only count bone *before* the Bragg peak
        ct1_full, _, _ = extract_1d_ray_profile(ct_arr, sp, org, rs, rt, num_samples=2000)
        bone_mm_proximal = step * np.sum(ct1_full[:pk] > BONE_THRESH)

        t0 = time.perf_counter()
        w_ref = compute_wepl_along_ray(spr_ref, sp, org, rs, rt, num_samples=2000)
        w_sct = compute_wepl_along_ray(spr_sct, sp, org, rs, rt, num_samples=2000)
        dwepl = w_sct - w_ref
        timings.append((time.perf_counter()-t0)*1000)

        depth = np.arange(len(d1)) * step
        grad = np.gradient(d1, step)
        risk = np.abs(dwepl) * np.abs(grad)

        d_sct = np.interp(w_ref + dwepl, w_ref, d1, left=0.0, right=0.0)
        err = np.abs(d_sct - d1)

        pred_shift = -dwepl[pk]
        act_shift  = find_r80(d_sct, depth) - find_r80(d1, depth)

        gate = np.abs(grad) > 0.10*np.abs(grad).max()
        if gate.sum() < 15: gate = np.abs(grad) > 0.02*np.abs(grad).max()
        
        # Handle nan gracefully for flat water rays
        if gate.sum() > 5 and np.std(risk[gate]) > 0:
            r_g = np.corrcoef(risk[gate], err[gate])[0, 1]
        else:
            r_g = np.nan

        results.append({"Patient": pid, "Region": cfg["region"], "Metal": cfg["metal"],
                        "Bone_mm": bone_mm_proximal, "r_gated": r_g,
                        "Pred_mm": pred_shift, "Act_mm": act_shift, "Time_ms": timings[-1]})
        row.update(surrogate_shift_mm=float(pred_shift), reference_shift_mm=float(act_shift),
                   surrogate_runtime_ms=float(timings[-1]),
                   notes_json={**row["notes_json"], "proximal_bone_path_mm": float(bone_mm_proximal),
                               "gated_risk_error_correlation": None if np.isnan(r_g) else float(r_g),
                               "bone_water_subgroup": "bone" if bone_mm_proximal >= 1.0 else "water"})
        manifest_rows.append(row)
        
        c = 'red' if cfg['metal'] else 'blue'
        m = 'x' if cfg['region'] == 'Thorax' else 'o'
        plt.scatter(pred_shift, act_shift, s=80, c=c, marker=m, alpha=0.9)
        print(f"Prox_Bone={bone_mm_proximal:.1f}mm | pred={pred_shift:+.2f} act={act_shift:+.2f} mm")

    append_manifest_rows(manifest_rows, __file__)
    df = pd.DataFrame(results)
    out_dir = os.path.join("outputs", "reproducibility")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "model_b_bone_water_case_level.csv"), index=False, float_format="%.10g")
    dv = df.dropna(subset=["Act_mm"])
    
    # Split into Bone Rays and Water Rays for statistics
    posbone = dv[dv["Bone_mm"] >= 1.0]
    nobone = dv[dv["Bone_mm"] < 1.0]

    r_corr = np.corrcoef(dv["Pred_mm"], dv["Act_mm"])[0, 1] if len(dv) > 1 else np.nan
    mae  = np.mean(np.abs(dv["Act_mm"] - dv["Pred_mm"]))
    bias = np.mean(dv["Act_mm"] - dv["Pred_mm"])

    print("\n" + "="*70)
    print(f"BONE-ONLY ERROR MODEL ({BONE_SHIFT:+.0f} HU for HU>{BONE_THRESH:.0f})")
    
    if len(posbone) > 1:
        r_pos = np.corrcoef(posbone["Pred_mm"], posbone["Act_mm"])[0, 1]
        mae_pos = np.mean(np.abs(posbone['Act_mm'] - posbone['Pred_mm']))
        print(f"SENSITIVITY (Bone Rays, n={len(posbone)}): r = {r_pos:.3f} | MAE = {mae_pos:.2f} mm")
    else:
        print(f"SENSITIVITY (Bone Rays, n={len(posbone)}): Insufficient data.")
        
    print(f"SPECIFICITY (Water Rays, n={len(nobone)}): mean |pred| = {np.mean(np.abs(nobone['Pred_mm'])):.2f} mm, mean |act| = {np.mean(np.abs(nobone['Act_mm'])):.2f} mm")
    print(f"OVERALL AGREEMENT: r = {r_corr:.3f} | MAE = {mae:.2f} mm | bias = {bias:+.2f} mm | n = {len(dv)}")
    print("="*70)
    
    # Clean table formatting
    df_print = df.copy()
    df_print['r_gated'] = df_print['r_gated'].apply(lambda x: f"{x:.3f}" if not np.isnan(x) else "N/A (Flat)")
    df_print['Pred_mm'] = df_print['Pred_mm'].apply(lambda x: f"{x:.2f}")
    df_print['Act_mm'] = df_print['Act_mm'].apply(lambda x: f"{x:.2f}")
    
    print(df_print.to_markdown(index=False))

    lims = [min(dv["Pred_mm"].min(), dv["Act_mm"].min())-1, max(dv["Pred_mm"].max(), dv["Act_mm"].max())+1]
    lims = [min(lims[0], -1), max(lims[1], 1)]
    plt.plot(lims, lims, 'k--', alpha=0.5)
    
    text_str = f"Overall: r={r_corr:.2f}, MAE={mae:.2f}mm\n"
    if len(posbone) > 1: text_str += f"Bone Rays (n={len(posbone)}): r={r_pos:.2f}\n"
    text_str += f"Water Rays (n={len(nobone)}): Specificity verified"
    
    plt.text(0.05, 0.92, text_str, transform=plt.gca().transAxes, fontsize=11,
             bbox=dict(facecolor='white', alpha=0.8))
    plt.legend(handles=[
        mlines.Line2D([], [], color='blue', marker='x', ls='None', ms=10, label='Thorax (Clean)'),
        mlines.Line2D([], [], color='red',  marker='x', ls='None', ms=10, label='Thorax (Metal)'),
        mlines.Line2D([], [], color='blue', marker='o', ls='None', ms=10, label='Abdomen (Clean)'),
        mlines.Line2D([], [], color='red',  marker='o', ls='None', ms=10, label='Abdomen (Metal)')],
        loc='lower right')
    plt.xlabel("Surrogate Predicted R80 Shift (mm)"); plt.ylabel("Actual MC R80 Shift (mm)")
    plt.title(f"Distal Edge Displacement — bone-only sCT error ({BONE_SHIFT:+.0f} HU)")
    plt.xlim(lims); plt.ylim(lims); plt.grid(ls='--', alpha=0.5)
    plt.tight_layout(); plt.savefig("figures/Fig7_BoneOnly_Scatter.png", dpi=300)
    print("\nSaved figures/Fig7_BoneOnly_Scatter.png")
    print(f"Median ray time: {np.median(timings):.2f} ms")

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    run()