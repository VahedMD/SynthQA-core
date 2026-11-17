from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from cohort_manifest import (METAL_PIDS, STRATIFIED_QUOTAS,
                             discover_stratified_candidate_pool,
                             discover_stratified_cohort,
                             discover_stratified_reservation_pool)


def test_published_metal_artifact_counts_match_appendix_c():
    thorax = sorted(pid for pid in METAL_PIDS if pid.startswith("1THB"))
    abdomen = sorted(pid for pid in METAL_PIDS if pid.startswith("1ABB"))
    assert len(thorax) == 11
    assert abdomen == ["1ABB078", "1ABB138"]
    assert len(METAL_PIDS) == 13


def test_feasible_stratified_targets_are_3_3_2_4():
    assert STRATIFIED_QUOTAS == (("1THB", True, 3), ("1THB", False, 3),
                                 ("1ABB", True, 2), ("1ABB", False, 4))


def test_candidate_pool_includes_replacement_after_failed_first_candidate():
    ids = ["1THB016", "1THB021", "1THB029", "1THB031",
           "1THB002", "1THB008", "1THB011", "1THB013",
           "1ABB078", "1ABB138",
           "1ABB006", "1ABB011", "1ABB020", "1ABB021", "1ABB022"]
    pool = discover_stratified_candidate_pool("unused", patient_ids=ids)
    thorax_metal = [x["pid"] for x in pool if x["region"] == "Thorax" and x["metal"]]
    assert thorax_metal == ["1THB016", "1THB021", "1THB029", "1THB031"]
    assert [x["target_count"] for x in pool if x["region"] == "Thorax" and x["metal"]] == [3, 3, 3, 3]
    # The legacy cohort helper still returns only nominal quota candidates.
    nominal = discover_stratified_cohort("unused", patient_ids=ids)
    assert "1THB031" not in {x["pid"] for x in nominal}


def test_reservation_pool_includes_one_fallback_candidate():
    ids = ["1THB016", "1THB021", "1THB029", "1THB031",
           "1THB002", "1THB008", "1THB011", "1THB013",
           "1ABB078", "1ABB138",
           "1ABB006", "1ABB011", "1ABB020", "1ABB021", "1ABB022"]
    reserved = {x["pid"] for x in discover_stratified_reservation_pool("unused", ids, 1)}
    assert {"1THB016", "1THB021", "1THB029", "1THB031"}.issubset(reserved)
