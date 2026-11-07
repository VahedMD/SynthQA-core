"""
Head-to-Head Benchmark: WEPL Surrogate vs pyRadPlan (Hong Pencil Beam)
Generates Table 4: computation time + surrogate-vs-PB range-shift agreement.

Key fixes:
 - Snap beamlet energy to the nearest pyRadPlan machine energy (root cause of
   the 'KeyError: 120.4273' failure).
 - Run pyRadPlan on BOTH reference CT and perturbed sCT -> true PB range shift.
 - Wider bone-ray search (fewer skips).
 - Suppress harmless Siddon raytracer warnings; print traceback once on error.
"""
import os, sys, json, time, tempfile, warnings, traceback
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download, list_repo_files

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from synthqa_core.io.image_loader import load_mha
from synthqa_core.calibration.profiles import CalibrationProfile
from synthqa_core.surrogates.wepl import compute_wepl_along_ray
from cohort_manifest import new_run, base_record, append_manifest_rows, discover_stratified_cohort

REPO_ID = "LMUK-RADONC-PHYS-RES/DoseRAD2026"

BONE_THRESH, BONE_SHIFT = 300.0, -200.0

# DoseRAD2026 Appendix-B HU->density curve (identical to the MC ground truth)
HLUT = [(-1024,1.2e-3),(-999,1.21e-3),(-200,0.8043754),(-199,0.8183035),
        (-10,1.006579),(-9,0.9966749),(120,1.126553),(121,1.095097),
        (3000,3.027294),(4000,3.698428)]

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from pyRadPlan import IonPlan, calc_dose_forward
        from pyRadPlan.ct import ct_from_file
        from pyRadPlan.cst import StructureSet
        from pyRadPlan.stf import SteeringInformation
        from pyRadPlan.stf._beam import Beam
        from pyRadPlan.geometry import get_beam_rotation_matrix
        from pyRadPlan.machines import load_from_name
    HAS_PB = True
except ImportError:
    HAS_PB = False
    print("[WARNING] pyRadPlan not installed -> pip install pyRadPlan==0.3.5")

def range_mm(E): return 0.022 * E ** 1.77

def discover_patients():
    """Use the shared deterministic stratified-cohort definition."""
    return discover_stratified_cohort(REPO_ID)

def ray_profile(vol, sp, org, rs, rt, n=2000):
    from scipy.ndimage import map_coordinates
    t = np.linspace(0, 1, n)
    p = rs[None, :] + t[:, None] * (rt - rs)[None, :]
    ix, iy, iz = ((p[:,0]-org[0])/sp[0], (p[:,1]-org[1])/sp[1], (p[:,2]-org[2])/sp[2])
    prof = map_coordinates(vol, np.array([iz, iy, ix]), order=1, mode='constant', cval=0.0)
    return prof, np.linalg.norm(rt - rs)

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

def write_tmp_ct(arr, sp, org, direction, path):
    img = sitk.GetImageFromArray(arr.astype(np.int16))
    img.SetSpacing(sp); img.SetOrigin(org); img.SetDirection(direction)
    sitk.WriteImage(img, path)

def run_pb_dose(ct_path, b, bl, energy, machine, machine_energies):
    """One timed pyRadPlan forward calculation; returns (dose_arr, sp, org, ms)."""
    ct_pr = ct_from_file(ct_path)
    cst = StructureSet(vois=[], ct_image=ct_pr); cst.create_body_seg()
    pln = IonPlan(radiation_mode="protons", machine="Generic")
    pln.prop_dose_calc = {"dose_grid": {"resolution": ct_pr.resolution},
                          "air_offset_correction": True,
                          "geometric_lateral_cutoff": 25.0,
                          "trace_on_dose_grid": True,
                          "hlut": np.array(HLUT, dtype=float)}
    sad = machine.sad
    R = get_beam_rotation_matrix(float(b["gantry_angle"]), 0.0)
    src_bev = np.array([0.0, -sad, 0.0])
    beam = Beam.model_validate({
        "gantry_angle": float(b["gantry_angle"]), "couch_angle": 0.0,
        "radiation_mode": "protons", "machine": "Generic", "SAD": sad,
        "iso_center": np.array(bl["_rt"], dtype=float),
        "source_point_bev": src_bev, "source_point": R @ src_bev,
        "rays": [{"ray_pos_bev": np.zeros(3), "ray_pos": np.zeros(3),
                  "beamlets": [{"energy": energy}]}]})
    stf = SteeringInformation(beams=[beam])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.seterr(divide='ignore', invalid='ignore')
        t0 = time.perf_counter()
        res = calc_dose_forward(ct_pr, cst, stf, pln)
        ms = (time.perf_counter() - t0) * 1000

    d = res["physical_dose"]
    return sitk.GetArrayFromImage(d), d.GetSpacing(), d.GetOrigin(), ms

def run():
    run_meta = new_run(__file__)
    prof = CalibrationProfile()
    results, manifest_rows = [], []
    machine = load_from_name("protons", "Generic") if HAS_PB else None
    machine_energies = np.array(sorted(machine.energies), dtype=float) if HAS_PB else None

    patient_configs = discover_patients()
    for selection_order, cfg in enumerate(patient_configs, start=1):
        pid = cfg["pid"]
        row = base_record(run_meta, "model_b_pyradplan", pid,
            anatomical_region=cfg["region"], metal_status=bool(cfg["metal"]),
            selection_method="deterministic sorted patient IDs by prefix and metal-status quotas; first n per stratum",
            perturbation_model="Model B: high-HU reduction", perturbation_threshold_hu=BONE_THRESH,
            perturbation_delta_hu=BONE_SHIFT, candidate_rays_examined=0, candidate_beamlets_examined=0,
            rays_rejected_short_wepl=0, dose_download_failures=0,
            invalid_dose_profiles=0, dose_engine_failures=0, eligible_rays=0, outcome_status="selected_pending",
            notes_json={"selection_order": selection_order,
                        "target_stratum_counts": {"thorax_metal": 3, "thorax_clean": 3, "abdomen_metal": 2, "abdomen_clean": 4},
                        "dose_engine": "pyRadPlan forward calculations on reference CT and perturbed sCT"})
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
        ct_arr, sp, org, direction = ct["array"], ct["spacing_xyz"], ct["origin_xyz"], ct["direction"]
        spr_ref = prof.density_to_spr.convert(prof.hu_to_density.convert(ct_arr))
        sct_arr = ct_arr.copy(); sct_arr[ct_arr > BONE_THRESH] += BONE_SHIFT
        spr_sct = prof.density_to_spr.convert(prof.hu_to_density.convert(sct_arr))
        try:
            with open(plan_p, encoding="utf-8") as f:
                plan = json.load(f)
        except Exception as exc:
            row.update(outcome_status="excluded", exclusion_reason="plan_json_parse_failed",
                       notes_json={**row["notes_json"], "error": str(exc)[:500]})
            manifest_rows.append(row)
            print("[invalid plan]"); continue

        # --- candidate rays: prefer proximal bone, snap energy ---
        cands = []
        for b in plan["beams"][:4]:
            for r in b["rays"]:
                row["candidate_rays_examined"] += 1
                rs, rt = np.array(r["ray_source"]), np.array(r["ray_target"])
                w = compute_wepl_along_ray(spr_ref, sp, org, rs, rt, num_samples=400)
                if w[-1] < 10.0:
                    row["rays_rejected_short_wepl"] += 1
                    continue
                dist = np.linalg.norm(rt - rs)
                ct1, _ = ray_profile(ct_arr, sp, org, rs, rt, n=400)
                bone_mm = dist * np.mean(ct1 > BONE_THRESH)
                for bl in r["beamlets"]:
                    row["candidate_beamlets_examined"] += 1
                    e_fit = abs(range_mm(bl["energy"]) - 0.9 * w[-1])
                    cands.append((0.0 if bone_mm > 1.0 else 1.0, e_fit, b, r, bl, rs, rt))
        cands.sort(key=lambda c: (c[0], c[1]))

        # --- write perturbed sCT once per patient ---
        tmp_sct = os.path.join(tempfile.gettempdir(), f"{pid}_sct.mha")
        write_tmp_ct(sct_arr, sp, org, direction, tmp_sct)

        done = False
        for _, _, b, r, bl, rs, rt in cands[:3]:
            energy = float(machine_energies[np.argmin(np.abs(machine_energies - bl["energy"]))]
                           if HAS_PB else bl["energy"])
            blx = dict(bl); blx["_rt"] = rt

            # --- 1) pyRadPlan on reference CT (timed) ---
            try:
                pb_ref_arr, pb_sp, pb_org, ms_ref = run_pb_dose(ct_p, b, blx, energy, machine, machine_energies)
            except Exception as e:
                row["dose_engine_failures"] += 1
                print(f"\n  [PB ERROR ref] {e}"); traceback.print_exc(); continue

            d1, dist = ray_profile(pb_ref_arr, pb_sp, pb_org, rs, rt)
            pk = np.argmax(d1)
            if not (pk > 0.2 * len(d1) and d1[pk] > 1e-3 and d1[pk:].min() < 0.1 * d1[pk]):
                row["invalid_dose_profiles"] += 1
                continue  # shallow/invalid peak -> next candidate

            # --- 2) pyRadPlan on perturbed sCT (timed) ---
            try:
                pb_sct_arr, _, _, ms_sct = run_pb_dose(tmp_sct, b, blx, energy, machine, machine_energies)
            except Exception as e:
                row["dose_engine_failures"] += 1
                print(f"\n  [PB ERROR sct] {e}"); continue

            depth = np.arange(len(d1)) * (dist / (len(d1) - 1))
            d2, _ = ray_profile(pb_sct_arr, pb_sp, pb_org, rs, rt)
            pb_shift = find_r80(d2, depth) - find_r80(d1, depth)

            # --- 3) WEPL surrogate (timed) ---
            t0 = time.perf_counter()
            w_ref = compute_wepl_along_ray(spr_ref, sp, org, rs, rt, num_samples=2000)
            w_sct = compute_wepl_along_ray(spr_sct, sp, org, rs, rt, num_samples=2000)
            ms_sur = (time.perf_counter() - t0) * 1000
            pred_shift = -(w_sct - w_ref)[pk]

            results.append({"Patient": pid, "Region": cfg["region"], "Metal": cfg["metal"],
                            "E_MeV": energy, "Surrogate_ms": ms_sur,
                            "PB_ref_ms": ms_ref, "PB_sct_ms": ms_sct,
                            "PB_total_ms": ms_ref + ms_sct,
                            "Pred_mm": pred_shift, "PB_shift_mm": pb_shift})
            row.update(selected_beam_idx=b.get("beam_idx", 0), selected_ray_idx=r.get("ray_idx", 0),
                selected_beamlet_idx=bl.get("beamlet_idx", 0), selected_energy_mev=energy,
                eligible_rays=1, outcome_status="included", exclusion_reason="",
                surrogate_shift_mm=float(pred_shift), reference_shift_mm=float(pb_shift),
                surrogate_runtime_ms=float(ms_sur), reference_runtime_ms=float(ms_ref + ms_sct),
                notes_json={**row["notes_json"], "ray_source": rs.tolist(), "ray_target": rt.tolist(),
                            "energy_before_machine_snap_mev": bl.get("energy"),
                            "energy_snapped_to_machine": bool(HAS_PB and energy != float(bl.get("energy", energy))),
                            "eligible_rays_semantics": "one selected valid ray; search stops after first valid pyRadPlan dose pair"})
            print(f"(B{b.get('beam_idx',0)}R{r.get('ray_idx',0)}L{bl.get('beamlet_idx',0)}, "
                  f"E={energy:.1f}) pred={pred_shift:+.2f} PB={pb_shift:+.2f} mm | "
                  f"sur={ms_sur:.2f} ms vs PB={ms_ref+ms_sct:.0f} ms")
            done = True
            break
        if not done:
            row.update(outcome_status="excluded", exclusion_reason="no_valid_pyRadPlan_ray_after_top_3_candidates")
            print("SKIPPED")
        manifest_rows.append(row)

    append_manifest_rows(manifest_rows, __file__)
    df = pd.DataFrame(results)
    out_dir = os.path.join("outputs", "reproducibility")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "model_b_pyradplan_case_level.csv"), index=False, float_format="%.10g")
    print("\n" + "=" * 78)
    print("TABLE 4: WEPL SURROGATE vs pyRadPlan (Hong PB) — TIME & AGREEMENT")
    print("=" * 78)
    if len(df):
        dv = df.dropna(subset=["PB_shift_mm"])
        agree = np.mean(np.abs(dv["Pred_mm"] - dv["PB_shift_mm"]))
        speed = np.median(dv["PB_total_ms"]) / np.median(dv["Surrogate_ms"])
        print(f"Surrogate median: {np.median(dv['Surrogate_ms']):.2f} ms | "
              f"PB median (ref+sCT): {np.median(dv['PB_total_ms']):.0f} ms | "
              f"SPEEDUP: {speed:.0f}x")
        print(f"Surrogate-vs-PB shift agreement: MAE = {agree:.2f} mm (n={len(dv)})")
    print(df.to_markdown(index=False,
          floatfmt=["s","s","s",".1f",".2f",".1f",".1f",".1f",".2f",".2f"]))

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    if len(df):
        dv = df.dropna(subset=["PB_shift_mm"])
        lims = [min(dv["Pred_mm"].min(), dv["PB_shift_mm"].min()) - 1,
                max(dv["Pred_mm"].max(), dv["PB_shift_mm"].max()) + 1]
        ax[0].plot(lims, lims, 'k--', alpha=0.5)
        ax[0].scatter(dv["Pred_mm"], dv["PB_shift_mm"], s=70, c='steelblue', edgecolors='k')
        ax[0].set_xlabel("Surrogate predicted shift (mm)")
        ax[0].set_ylabel("pyRadPlan PB shift (mm)")
        ax[0].set_title("Surrogate vs Pencil Beam")
        ax[0].grid(ls='--', alpha=0.5)
        x = np.arange(len(dv))
        ax[1].bar(x - 0.2, dv["Surrogate_ms"], 0.4, label="WEPL surrogate", color='steelblue')
        ax[1].bar(x + 0.2, dv["PB_total_ms"], 0.4, label="pyRadPlan (ref+sCT)", color='indianred')
        ax[1].set_yscale('log')
        ax[1].set_xticks(x); ax[1].set_xticklabels(dv["Patient"], rotation=45)
        ax[1].set_ylabel("Time (ms, log scale)")
        ax[1].legend(); ax[1].grid(ls='--', alpha=0.5)
        ax[1].set_title("Computation time per beamlet")
    fig.tight_layout()
    fig.savefig("figures/Fig8_Surrogate_vs_pyRadPlan.png", dpi=300)
    print("Saved figures/Fig8_Surrogate_vs_pyRadPlan.png")

if __name__ == "__main__":
    run()