import numpy as np

from synthqa_core.surrogates.stratified_r80 import (
    ANALYSIS_VERSION,
    find_r80,
    evaluate_selected_beamlet,
    benchmark_result_row,
)


def _synthetic_case():
    z, y, x = np.mgrid[0:40, 0:5, 0:5]
    dose = 0.01 * np.exp(-0.5 * ((z - 25.0) / 3.0) ** 2)
    spr_ref = np.ones((40, 5, 5), dtype=float)
    spr_sct = np.full((40, 5, 5), 1.05, dtype=float)
    dose_meta = {
        "array": dose.astype(float),
        "spacing_xyz": (1.0, 1.0, 1.0),
        "origin_xyz": (0.0, 0.0, 0.0),
        "direction": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    }
    return spr_ref, spr_sct, dose_meta


def test_find_r80_distal_crossing():
    depth = np.arange(10, dtype=float)
    dose = np.array([0.0, 0.1, 0.4, 1.0, 0.9, 0.8, 0.7, 0.5, 0.2, 0.0])
    value = find_r80(dose, depth)
    assert np.isfinite(value)
    assert 4.9 <= value <= 5.1


def test_shared_r80_evaluation_is_finite_and_consistent():
    spr_ref, spr_sct, dose_meta = _synthetic_case()
    result = evaluate_selected_beamlet(
        patient_id="SYNTHETIC",
        spr_ref=spr_ref,
        spr_sct=spr_sct,
        ct_spacing_xyz=(1.0, 1.0, 1.0),
        ct_origin_xyz=(0.0, 0.0, 0.0),
        ct_direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        dose_meta=dose_meta,
        ray_source_xyz=(2.0, 2.0, 0.0),
        ray_target_xyz=(2.0, 2.0, 39.0),
        profile_samples=2000,
    )
    assert np.isfinite(result["first_order_mm"])
    assert np.isfinite(result["finite_map_mm"])
    assert abs(result["inverse_vs_profile_mm"]) < 0.03
    assert result["linearization_error_mm"] < 0.2


def test_benchmark_row_carries_exact_selection_provenance():
    spr_ref, spr_sct, dose_meta = _synthetic_case()
    metrics = evaluate_selected_beamlet(
        patient_id="SYNTHETIC", spr_ref=spr_ref, spr_sct=spr_sct,
        ct_spacing_xyz=(1.0, 1.0, 1.0), ct_origin_xyz=(0.0, 0.0, 0.0),
        ct_direction=None, dose_meta=dose_meta,
        ray_source_xyz=(2.0, 2.0, 0.0), ray_target_xyz=(2.0, 2.0, 39.0),
    )
    row = benchmark_result_row(
        patient_id="SYNTHETIC", region="Thorax", metal=False,
        selection={
            "beam_idx": 2, "ray_idx": 3, "beamlet_idx": 0, "energy_mev": 114.5,
            "dose_file": "Dose_B2_R3_L0.mha", "selection_rank": 1,
            "energy_mismatch_mm": 0.1,
            "ray_source_xyz": (2.0, 2.0, 0.0), "ray_target_xyz": (2.0, 2.0, 39.0),
            "profile_samples": 2000,
        },
        metrics=metrics, r_gated=float("nan"), runtime_ms=1.0,
    )
    assert row["AnalysisVersion"] == ANALYSIS_VERSION
    assert row["BeamIdx"] == 2 and row["RayIdx"] == 3 and row["BeamletIdx"] == 0
    assert row["DoseFile"] == "Dose_B2_R3_L0.mha"
    assert row["RaySourceXYZ"] == "[2.0,2.0,0.0]"
    assert "spr_sct_cv_interval" in row
    assert np.isfinite(row["spr_sct_cv_interval"])
