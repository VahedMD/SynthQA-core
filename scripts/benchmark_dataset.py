"""
Stratified DoseRAD2026 Benchmark 
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

def range_mm(E_mev):          # CSDA-like range-energy fit in water
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
        row = base_record(run_meta, "model_a_stratified", pid,
            anatomical_region=cfg["region"], metal_status=bool(cfg["metal"]),
            selection_method="deterministic sorted patient IDs by prefix and metal-status quotas; first n per stratum",
            perturbation_model="Model A: global HU bias", perturbation_threshold_hu="",
            perturbation_delta_hu=50, candidate_rays_examined=0, candidate_beamlets_examined=0,
            rays_rejected_short_wepl=0, dose_download_failures=0,
            invalid_dose_profiles=0, dose_engine_failures=0, eligible_rays=0, outcome_status="selected_pending",
            notes_json={"selection_order": selection_order, "target_stratum_counts":
                        {"thorax_metal": 3, "thorax_clean": 3, "abdomen_metal": 2, "abdomen_clean": 4},
                        "dose_reference": "analytical reconstruction from reference dose; not fresh perturbed-image MC"})
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
        spr_sct = prof.density_to_spr.convert(prof.hu_to_density.convert(ct_arr + 50.0))


        try:
            with open(plan_p, encoding="utf-8") as f:
                plan = json.load(f)
        except Exception as exc:
            row.update(outcome_status="excluded", exclusion_reason="plan_json_parse_failed",
                       notes_json={**row["notes_json"], "error": str(exc)[:500]})
            manifest_rows.append(row)
            print("[invalid plan]"); continue

        # --- Energy-based beamlet pre-selection (no blind downloads) ---
        candidates = []
        for b in plan["beams"][:4]:
            for r in b["rays"][:6]:
                row["candidate_rays_examined"] += 1
                rs, rt = np.array(r["ray_source"]), np.array(r["ray_target"])
                w = compute_wepl_along_ray(spr_ref, sp, org, rs, rt, num_samples=400)
                if w[-1] < 10.0:
                    row["rays_rejected_short_wepl"] += 1
                    continue
                target_wepl = 0.9 * w[-1]                      # peak deep but inside profile
                for bl in r["beamlets"]:
                    row["candidate_beamlets_examined"] += 1
                    candidates.append((abs(range_mm(bl["energy"]) - target_wepl), b, r, bl, rs, rt))
        candidates.sort(key=lambda c: c[0])
        found = False
        for _, b, r, bl, rs, rt in candidates[:4]:              # max 4 downloads per patient
            b_idx = b.get("beam_idx", 0)
            r_idx = r.get("ray_idx", 0)
            l_idx = bl.get("beamlet_idx", 0)
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
        if not found:
            row.update(outcome_status="excluded", exclusion_reason="no_valid_dose_profile_in_top_4_candidates")
            manifest_rows.append(row)
            print("SKIPPED"); continue

        t0 = time.perf_counter()
        # FIX: Explicitly match the 2000 samples used in the dose profile extraction
        w_ref = compute_wepl_along_ray(spr_ref, sp, org, rs, rt, num_samples=2000)
        w_sct = compute_wepl_along_ray(spr_sct, sp, org, rs, rt, num_samples=2000)
        dwepl = w_sct - w_ref
        timings.append((time.perf_counter()-t0)*1000)

        depth = np.arange(len(d1)) * step
        grad = np.gradient(d1, step)
        risk = np.abs(dwepl) * np.abs(grad)

        # CORRECT physics: higher SPR -> peak moves PROXIMAL -> sample reference at W + dW
        d_sct = np.interp(w_ref + dwepl, w_ref, d1, left=0.0, right=0.0)
        err = np.abs(d_sct - d1)

        pk = np.argmax(d1)
        pred_shift = -dwepl[pk]                 # depth convention: negative = proximal
        act_shift  = find_r80(d_sct, depth) - find_r80(d1, depth)

        gate = np.abs(grad) > 0.10*np.abs(grad).max()
        if gate.sum() < 15: gate = np.abs(grad) > 0.02*np.abs(grad).max()
        r_g = np.corrcoef(risk[gate], err[gate])[0, 1] if gate.sum() > 5 else np.nan

        results.append({"Patient": pid, "Region": cfg["region"], "Metal": cfg["metal"],
                        "r_gated": r_g, "Pred_mm": pred_shift, "Act_mm": act_shift,
                        "Time_ms": timings[-1]})
        row.update(surrogate_shift_mm=float(pred_shift), reference_shift_mm=float(act_shift),
                   surrogate_runtime_ms=float(timings[-1]),
                   notes_json={**row["notes_json"], "gated_risk_error_correlation": None if np.isnan(r_g) else float(r_g),
                               "gate_rule": "|gradient| > 0.10*max(|gradient|), fallback 0.02 if fewer than 15 voxels",
                               "eligible_rays_semantics": "one selected valid ray; search stops after first valid dose profile"})
        manifest_rows.append(row)
        c = 'red' if cfg['metal'] else 'blue'
        m = 'x' if cfg['region'] == 'Thorax' else 'o'
        plt.scatter(pred_shift, act_shift, s=80, c=c, marker=m, alpha=0.9)
        print(f"pred={pred_shift:+.2f} act={act_shift:+.2f} mm")

    append_manifest_rows(manifest_rows, __file__)
    df = pd.DataFrame(results)
    out_dir = os.path.join("outputs", "reproducibility")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "model_a_stratified_case_level.csv"), index=False, float_format="%.10g")
    dv = df.dropna(subset=["Act_mm"])                       # nan-safe statistics
    r_corr = np.corrcoef(dv["Pred_mm"], dv["Act_mm"])[0, 1]
    mae = np.mean(np.abs(dv["Act_mm"] - dv["Pred_mm"]))
    bias = np.mean(dv["Act_mm"] - dv["Pred_mm"])

    print("\n" + "="*70)
    print(f"RANGE-SHIFT AGREEMENT: r = {r_corr:.3f} | MAE = {mae:.2f} mm | bias = {bias:+.2f} mm | n = {len(dv)}")
    print("="*70)
    print(df.to_markdown(index=False, floatfmt=["s","s","s",".3f",".2f",".2f",".2f"]))

    lims = [min(dv["Pred_mm"].min(), dv["Act_mm"].min())-1, max(dv["Pred_mm"].max(), dv["Act_mm"].max())+1]
    plt.plot(lims, lims, 'k--', alpha=0.5)
    plt.text(0.05, 0.92, f"r = {r_corr:.3f}\nMAE = {mae:.2f} mm\nbias = {bias:+.2f} mm",
             transform=plt.gca().transAxes, fontsize=11,
             bbox=dict(facecolor='white', alpha=0.8))
    plt.legend(handles=[
        mlines.Line2D([], [], color='blue', marker='x', ls='None', ms=10, label='Thorax (Clean)'),
        mlines.Line2D([], [], color='red',  marker='x', ls='None', ms=10, label='Thorax (Metal)'),
        mlines.Line2D([], [], color='blue', marker='o', ls='None', ms=10, label='Abdomen (Clean)'),
        mlines.Line2D([], [], color='red',  marker='o', ls='None', ms=10, label='Abdomen (Metal)')],
        loc='lower right')
    plt.xlabel("Surrogate Predicted R80 Shift (mm)"); plt.ylabel("Actual MC R80 Shift (mm)")
    plt.title("Distal Edge Displacement (+50 HU uniform bias)")
    plt.xlim(lims); plt.ylim(lims); plt.grid(ls='--', alpha=0.5)
    plt.tight_layout(); plt.savefig("figures/Fig6_Stratified_Scatter.png", dpi=300)
    print("Saved figures/Fig6_Stratified_Scatter.png")
    print(f"Median ray time: {np.median(timings):.2f} ms")

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    run()