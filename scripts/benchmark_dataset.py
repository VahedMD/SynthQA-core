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

REPO_ID = "LMUK-RADONC-PHYS-RES/DoseRAD2026"
METAL_PIDS = {"1THB016","1THB021","1THB029","1THB031","1THB052","1THB054",
              "1THB067","1THB078","1THB122","1THB214","1THB217","1ABB138","1ABB078"}

def range_mm(E_mev):          # CSDA-like range-energy fit in water
    return 0.022 * E_mev ** 1.77

def discover_patients():
    files = list_repo_files(REPO_ID, repo_type="dataset")
    pids = sorted({f.split("/")[2] for f in files if f.startswith("proton/training/") and len(f.split("/")) >= 3})
    sel = lambda pre, met, n: [p for p in pids if p.startswith(pre) and (p in METAL_PIDS) == met][:n]
    return ([{"pid": p, "region": "Thorax",  "metal": True}  for p in sel("1THB", True, 3)] +
            [{"pid": p, "region": "Thorax",  "metal": False} for p in sel("1THB", False, 3)] +
            [{"pid": p, "region": "Abdomen", "metal": True}  for p in sel("1ABB", True, 2)] +
            [{"pid": p, "region": "Abdomen", "metal": False} for p in sel("1ABB", False, 4)])

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
    prof = CalibrationProfile()
    results, timings = [], []
    plt.figure(figsize=(10, 7))

    for cfg in discover_patients():
        pid = cfg["pid"]
        print(f"\n{cfg['region']} {pid} (Metal={cfg['metal']})...", end=" ")
        try:
            plan_p = hf_hub_download(REPO_ID, f"proton/training/{pid}/{pid}.json", repo_type="dataset")
            ct_p   = hf_hub_download(REPO_ID, f"proton/training/{pid}/image/ct.mha", repo_type="dataset")
        except Exception:
            print("[missing]"); continue

        ct = load_mha(ct_p)
        ct_arr, sp, org = ct["array"], ct["spacing_xyz"], ct["origin_xyz"]
        spr_ref = prof.density_to_spr.convert(prof.hu_to_density.convert(ct_arr))
        spr_sct = prof.density_to_spr.convert(prof.hu_to_density.convert(ct_arr + 50.0))


        plan = json.load(open(plan_p))

        # --- Energy-based beamlet pre-selection (no blind downloads) ---
        candidates = []
        for b in plan["beams"][:4]:
            for r in b["rays"][:6]:
                rs, rt = np.array(r["ray_source"]), np.array(r["ray_target"])
                w = compute_wepl_along_ray(spr_ref, sp, org, rs, rt, num_samples=400)
                target_wepl = 0.9 * w[-1]                      # peak deep but inside profile
                for bl in r["beamlets"]:
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
                continue
            d1, dist, step = extract_1d_ray_profile(d3, sp, org, rs, rt, num_samples=2000)
            pk = np.argmax(d1)
            if pk > 0.2*len(d1) and d1[pk] > 1e-3 and d1[pk:].min() < 0.1*d1[pk]:
                found = True
                print(f"(B{b_idx}R{r_idx}L{l_idx})", end=" ")
                break
        if not found:
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
        c = 'red' if cfg['metal'] else 'blue'
        m = 'x' if cfg['region'] == 'Thorax' else 'o'
        plt.scatter(pred_shift, act_shift, s=80, c=c, marker=m, alpha=0.9)
        print(f"pred={pred_shift:+.2f} act={act_shift:+.2f} mm")

    df = pd.DataFrame(results)
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