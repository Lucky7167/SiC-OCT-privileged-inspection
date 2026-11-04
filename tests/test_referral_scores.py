import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from referral import (
    SCORE_NAMES,
    defect_coverage,
    extract_referral_scores,
    inverse_margin,
    matched_count_comparison,
    minimum_recall_budget,
    predictive_entropy,
    reject_option,
    top_k,
    training_derived_fixed_thresholds,
)


def _score_frame():
    return pd.DataFrame({
        "classifier_p_normal": [0.7, 0.2],
        "classifier_p_surface": [0.2, 0.3],
        "classifier_p_subsurface": [0.1, 0.5],
        "anomaly_score": [0.3, 0.4],
        "direct_score": [0.5, 0.6],
        "student_score": [0.7, 0.8],
        "teacher_score": [0.9, 0.95],
        "direct_p_subsurface": [0.11, 0.12],
        "student_p_subsurface": [0.21, 0.22],
    })


def test_unified_extraction_returns_all_eight_scores():
    result = extract_referral_scores(_score_frame())
    assert tuple(result.columns) == SCORE_NAMES
    expected_probabilities = np.array([[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]])
    assert np.allclose(result.entropy, predictive_entropy(expected_probabilities))
    assert np.allclose(result.inverse_margin, [0.5, 0.8])
    assert np.allclose(result.inverse_margin, inverse_margin(expected_probabilities))
    assert np.allclose(result.student, [0.7, 0.8])
    assert np.allclose(result.direct_p_subsurface, [0.11, 0.12])


def test_top_k_and_reject_option_preserve_decision_semantics():
    scores = np.array([0.2, 0.8, 0.5, 0.8])
    assert top_k(scores, 2).tolist() == [False, True, False, True]
    assert top_k(scores, 0).sum() == 0
    result = reject_option(scores, [0, 1, 0, 1], threshold=0.6)
    assert result["referred"].tolist() == [False, True, False, True]
    assert result["direct_prediction"].tolist() == [0, 1, 0, 1]


def test_training_thresholds_are_estimated_without_test_scores():
    train = np.r_[np.linspace(0.05, 0.2, 12), np.linspace(0.4, 0.55, 12), np.linspace(0.8, 0.95, 12)]
    test = np.array([0.1, 0.5, 0.9])
    result = training_derived_fixed_thresholds(train, test, [0, 1, 0])
    assert 0.2 < result["low"] < 0.4
    assert 0.4 < result["high"] < 0.8
    assert result["referred"].tolist() == [False, True, False]
    assert result["direct_prediction"].tolist() == [0, 1, 2]


def _comparison_frame():
    return pd.DataFrame({
        "wafer": ["W1"] * 6,
        "z_star": [2, 2, 1, 1, 2, 1],
        "is_candidate": True,
        "prediction": [1, 1, 1, 1, 1, 1],
        "defect_id": ["a", "a", "s1", "s2", "b", "s3"],
        "student": [0.05, 0.10, 0.20, 0.45, 0.50, 0.55],
        "entropy": [0.55, 0.50, 0.45, 0.20, 0.10, 0.05],
        "x0_aline": [0, 0, 1, 2, 3, 4],
        "x1_aline": [1, 1, 2, 3, 4, 5],
        "y0_aline": 0,
        "y1_aline": 1,
    })


def test_matched_count_comparison_uses_within_wafer_counts_and_defect_ids():
    frame = _comparison_frame()
    result = matched_count_comparison(frame, scores=frame[["student", "entropy"]])
    assert len(result) == 2
    assert result.referral_count.nunique() == 1
    student = result[result.score == "student"].iloc[0]
    assert student.subsurface_defects_total == 2
    assert student.ranking_defects_covered <= 2
    assert student.combined_defects_covered <= 2
    assert student.combined_errors_corrected <= student.microscopy_errors_total
    coverage = defect_coverage(frame, [True, True, False, False, False, False])
    assert coverage == {"covered": 1, "total": 2, "recall": 0.5}


def test_minimum_recall_budget_uses_union_footprint_and_unique_defects():
    frame = _comparison_frame()
    result = minimum_recall_budget(frame, [0.9, 0.8, 0.1, 0.2, 0.7, 0.3])
    assert result["referral_count"] == 3
    assert result["recall"] == 1.0
    assert np.isclose(result["budget_percent"], 40.0)


def test_minimum_recall_budget_excludes_noncandidate_positive_rows():
    frame = _comparison_frame()
    frame.loc[4, "is_candidate"] = False
    result = minimum_recall_budget(frame, [0.9, 0.8, 0.1, 0.2, 0.7, 0.3])
    assert result["referral_count"] == 1
    assert result["recall"] == 1.0
    assert np.isclose(result["budget_percent"], 25.0)


def test_matched_count_comparison_excludes_noncandidate_positive_rows():
    frame = _comparison_frame()
    frame.loc[4, "is_candidate"] = False
    result = matched_count_comparison(frame, scores=frame[["student", "entropy"]])
    assert len(result) == 2
    assert (result.subsurface_defects_total == 1).all()
    assert (result.microscopy_errors_total == 2).all()
