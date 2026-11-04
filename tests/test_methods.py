import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from common import compute_q_star, compute_u_star, make_outer_fold
from referral import deploy, footprint_cost, multi_otsu_three_regions


def test_outer_fold_is_wafer_disjoint():
    frame = pd.DataFrame({
        "wafer": list("AAABBBCCC"), "roi_id": range(9), "z_star": 0,
        "x": 0, "y": 0, "w": 1, "h": 1, "a": 0,
    })
    train, valid, test = make_outer_fold(frame, "C", "B")
    assert set(train.wafer) == {"A"}
    assert set(valid.wafer) == {"B"}
    assert set(test.wafer) == {"C"}


def test_privileged_targets_match_equations_2_and_4():
    frame = pd.DataFrame({
        "deff": [5, 5], "dmax": [10, 10], "snr": [2, 2], "snr_ref": [4, 4],
        "rho_shadow": [0.1, 0.1], "rho_sat": [0.2, 0.2], "c_cont": [0.7, 0.7],
        "has_depth_structure": [True, False],
    })
    q = compute_q_star(frame)
    assert np.allclose(q, [0.68, 0.34])
    assert np.allclose(compute_u_star(q, [0, 1]), [0.476, 0.34])


def test_otsu_deployment_has_no_label_argument():
    scores = np.r_[np.linspace(0.05, 0.2, 20), np.linspace(0.4, 0.55, 20), np.linspace(0.8, 0.95, 20)]
    low, high = multi_otsu_three_regions(scores)
    result = deploy(scores, np.zeros(len(scores), dtype=int), (low, high))
    assert result["referred"].any()
    assert (result["direct_prediction"][scores >= high] == 2).all()


def test_footprint_union_does_not_double_count_overlap():
    frame = pd.DataFrame({
        "x0_aline": [0, 1], "x1_aline": [2, 3],
        "y0_aline": [0, 0], "y1_aline": [2, 2],
    })
    assert footprint_cost(frame, [True, True]) == 6
