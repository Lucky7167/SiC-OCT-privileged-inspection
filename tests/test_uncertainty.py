import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import uncertainty
from referral import deploy
from uncertainty import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_MONTE_CARLO_REPLICATES,
    bootstrap_intervals,
    class_composition_monte_carlo,
    operating_point_metrics,
    summarize_class_composition,
)


def _frame():
    frames = []
    for wafer in ("W1", "W2"):
        frame = pd.DataFrame({
            "wafer": wafer,
            "z_star": [2, 1, 2, 1],
            "is_candidate": True,
            "prediction": [1, 1, 1, 1],
            "defect_id": ["a", "s1", "a", "s2"],
            "score": [0.1, 0.4, 0.6, 0.9],
            "x0_aline": [0, 1, 2, 3],
            "x1_aline": [1, 2, 3, 4],
            "y0_aline": 0,
            "y1_aline": 1,
        })
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_operating_point_metrics_use_fixed_thresholds_and_footprint_budget():
    result = operating_point_metrics(_frame(), thresholds=(0.25, 0.75))
    assert np.isclose(result["auroc"], 0.25)
    assert np.isclose(result["budget_percent"], 50.0)
    assert result["direct_precision"] == 0.0
    expected_final = np.array([1, 1, 2, 2] * 2)
    expected = np.median([
        f1_score(group.z_star, expected_final[index * 4:(index + 1) * 4], labels=[0, 1, 2], average="macro", zero_division=0)
        for index, (_, group) in enumerate(_frame().groupby("wafer"))
    ])
    assert np.isclose(result["macro_f1"], expected)


def test_bootstrap_returns_wafer_and_hierarchical_intervals_for_all_metrics():
    result = bootstrap_intervals(
        _frame(), n_resamples=24, seed=4, thresholds=(0.25, 0.75)
    )
    assert set(result.resampling) == {"wafer", "hierarchical"}
    assert set(result.metric) == {"auroc", "budget_percent", "macro_f1", "direct_precision"}
    assert len(result) == 8
    assert (result.n_resamples == 24).all()
    budget = result[result.metric == "budget_percent"]
    assert np.allclose(budget.estimate, 50.0)
    assert ((budget.lower <= budget.estimate) & (budget.estimate <= budget.upper)).all()
    precision = result[result.metric == "direct_precision"]
    assert np.allclose(precision.estimate, 0.0)
    assert DEFAULT_BOOTSTRAP_REPLICATES == 4000


def test_monte_carlo_outputs_all_conditions_and_operating_diagnostics(monkeypatch):
    monkeypatch.setattr(uncertainty, "multi_otsu_three_regions", lambda values: (0.25, 0.75))
    result = class_composition_monte_carlo(
        _frame(), fractions=(1.0, 0.0), n_resamples=2, seed=3
    )
    assert len(result) == 8
    assert set(result.subsurface_percent) == {0.0, 100.0}
    expected = {
        "low_zone_miss_rate",
        "direct_high_surface_error_rate",
        "referral_share",
        "threshold_low",
        "threshold_high",
    }
    assert expected <= set(result.columns)
    complete = result[result.subsurface_fraction == 1.0]
    assert np.allclose(complete.low_zone_miss_rate, 1.0)
    assert np.allclose(complete.direct_high_surface_error_rate, 0.5)
    assert np.allclose(complete.referral_share, 0.5)
    assert np.allclose(complete.threshold_low, 0.25)
    assert np.allclose(complete.threshold_high, 0.75)
    assert DEFAULT_MONTE_CARLO_REPLICATES == 500


def test_monte_carlo_summary_has_median_and_interquartile_range(monkeypatch):
    monkeypatch.setattr(uncertainty, "multi_otsu_three_regions", lambda values: (0.25, 0.75))
    samples = class_composition_monte_carlo(_frame(), fractions=(1.0,), n_resamples=3)
    result = summarize_class_composition(samples)
    assert len(result) == 2
    assert np.allclose(result.low_zone_miss_rate_median, 1.0)
    assert np.allclose(result.direct_high_surface_error_rate_q1, 0.5)
    assert np.allclose(result.direct_high_surface_error_rate_q3, 0.5)
    assert np.allclose(result.threshold_low_median, 0.25)


def test_deployed_referral_masks_match_uncertainty_preparation():
    frame = _frame()
    first = frame[frame.wafer == "W1"]
    decision = deploy(first.score, first.prediction, (0.25, 0.75))
    assert decision["referred"].sum() == 2
    assert decision["direct_prediction"].tolist() == [1, 1, 1, 2]


def test_operating_point_and_bootstrap_use_candidates_not_positive_labels():
    frame = _frame()
    frame["is_candidate"] = [True, True, False, True] * 2
    point = operating_point_metrics(frame, thresholds=(0.25, 0.75))
    assert np.isclose(point["auroc"], 0.0)
    assert np.isclose(point["budget_percent"], 100.0 / 3.0)
    assert point["direct_precision"] == 0.0
    expected_final = np.array([1, 1, 1, 2] * 2)
    expected_macro = np.median([
        f1_score(group.z_star, expected_final[index * 4:(index + 1) * 4], labels=[0, 1, 2], average="macro", zero_division=0)
        for index, (_, group) in enumerate(frame.groupby("wafer"))
    ])
    assert np.isclose(point["macro_f1"], expected_macro)
    result = bootstrap_intervals(frame, n_resamples=8, seed=4, thresholds=(0.25, 0.75))
    for metric, estimate in result.groupby("metric"):
        assert np.allclose(estimate.estimate, point[metric])


def test_hierarchical_resampling_preserves_candidate_membership():
    frame = _frame()
    frame["is_candidate"] = [True, True, False, True] * 2
    first = frame[frame.wafer == "W1"]
    item = uncertainty._prepare_wafer(first, "score", (0.25, 0.75))
    item["score_column"] = "score"
    sampled = uncertainty._resampled_prepared([item], [0], True, np.random.default_rng(4))[0]
    assert sampled["operational"].tolist() == sampled["frame"].is_candidate.tolist()
    assert sampled["final"][~sampled["operational"]].tolist() == sampled["frame"].loc[~sampled["operational"], "prediction"].tolist()


def test_monte_carlo_estimation_and_evaluation_exclude_noncandidate_positives(monkeypatch):
    frame = _frame()
    frame["is_candidate"] = [True, True, False, True] * 2
    observed = []

    def thresholds(values):
        observed.append(np.asarray(values).tolist())
        return 0.25, 0.75

    monkeypatch.setattr(uncertainty, "multi_otsu_three_regions", thresholds)
    result = class_composition_monte_carlo(frame, fractions=(1.0,), n_resamples=1, seed=3)
    assert observed == [[0.1, 0.4, 0.9], [0.1, 0.4, 0.9]]
    assert np.allclose(result.low_zone_miss_rate, 1.0)
    assert np.allclose(result.direct_high_surface_error_rate, 0.5)
    assert np.allclose(result.referral_share, 1.0 / 3.0)
