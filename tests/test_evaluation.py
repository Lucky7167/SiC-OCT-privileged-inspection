import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import evaluate as evaluation
from referral import oracle_recall_budget_curve


def _inputs(seed=0, scores=None):
    scores = scores or [0.1, 0.8, 0.9, 0.2, 0.05]
    manifest = pd.DataFrame({
        "roi_id": ["r0", "r1", "r2", "r3", "r4"],
        "wafer": ["W1"] * 5,
        "z_star": [1, 2, 2, 2, 0],
        "defect_id": ["nonsubsurface", "defect-a", "defect-a", "defect-b", "surface"],
        "is_candidate": [True, True, True, True, False],
        "x0_aline": [0, 0, 0, 2, 4],
        "x1_aline": [2, 2, 2, 3, 5],
        "y0_aline": [0] * 5,
        "y1_aline": [2] * 5,
    })
    predictions = pd.DataFrame({
        "roi_id": manifest.roi_id,
        "wafer": manifest.wafer,
        "z_star": manifest.z_star,
        "model": "student",
        "seed": seed,
        "score": scores,
        "prediction": 0,
    })
    return predictions, manifest


def _decision(scores, predictions, thresholds=None):
    size = len(scores)
    return {
        "low": 0.3,
        "high": 0.7,
        "referred": np.array([False, True, False, False])[:size],
        "direct_prediction": np.zeros(size, dtype=int),
    }


def test_evaluation_deduplicates_physical_defect_recall():
    predictions, manifest = _inputs()
    with patch("evaluate.deploy", side_effect=_decision):
        result = evaluation.evaluate(predictions, manifest)
    row = result.iloc[0]
    assert row.seed == 0
    assert row.subsurface_defects_total == 2
    assert row.subsurface_defects_retained == 1
    assert row.subsurface_recall == 0.5
    assert np.isclose(row.budget_candidate_guided_percent, 100.0 * 4 / 6)


def test_summary_pools_each_seed_and_uses_sample_standard_deviation():
    first_predictions, manifest = _inputs(0)
    second_predictions, _ = _inputs(1, [0.5, 0.2, 0.9, 0.8, 0.05])
    predictions = pd.concat([first_predictions, second_predictions], ignore_index=True)
    with patch("evaluate.deploy", side_effect=_decision):
        result = evaluation.summarize(predictions, manifest)
    pooled = result[result.summary_type == "pooled"]
    across = result[result.summary_type == "across_seed"].iloc[0]
    assert len(pooled) == 2
    assert pooled.loc[pooled.primary, "seed"].tolist() == [0]
    assert np.allclose(pooled.auroc, [1.0, 2.0 / 3.0])
    assert np.allclose(pooled.subsurface_defects_total, [2, 2])
    assert np.allclose(pooled.subsurface_defects_retained, [1, 1])
    assert np.allclose(pooled.subsurface_recall, [0.5, 0.5])
    assert across.n_seeds == 2
    assert np.isclose(across.auroc_mean, 5.0 / 6.0)
    assert np.isclose(across.auroc_sample_sd, np.std([1.0, 2.0 / 3.0], ddof=1))


def test_oracle_deduplicates_recall_without_changing_footprint_union():
    frame = pd.DataFrame({
        "z_star": [2, 2, 2],
        "defect_id": ["defect-a", "defect-a", "defect-b"],
        "x0_aline": [0, 0, 2],
        "x1_aline": [2, 2, 3],
        "y0_aline": [0, 0, 0],
        "y1_aline": [2, 2, 2],
    })
    curve = oracle_recall_budget_curve(frame, [0, 1, 2])
    assert curve.recall.tolist() == [0.5, 0.5, 1.0]
    assert np.allclose(curve.budget_percent, [100.0 * 4 / 6, 100.0 * 4 / 6, 100.0])


def test_missing_defect_id_is_an_explicit_error():
    predictions, manifest = _inputs()
    with pytest.raises(ValueError, match="defect_id"):
        evaluation.evaluate(predictions, manifest.drop(columns="defect_id"))
    with pytest.raises(ValueError, match="defect_id"):
        oracle_recall_budget_curve(manifest.drop(columns="defect_id"), [0])


def test_summary_uses_median_wafer_macro_f1():
    predictions, manifest = _inputs()
    manifest.loc[manifest.roi_id == "r2", "wafer"] = "W2"
    manifest.loc[manifest.roi_id == "r3", "wafer"] = "W3"
    predictions = predictions.copy()
    predictions.loc[predictions.roi_id == "r2", "wafer"] = "W2"
    predictions.loc[predictions.roi_id == "r3", "wafer"] = "W3"
    manifest.loc[5] = {
        "roi_id": "r5", "wafer": "W2", "z_star": 1, "defect_id": "non-w2",
        "is_candidate": True, "x0_aline": 0, "x1_aline": 2, "y0_aline": 0, "y1_aline": 2,
    }
    manifest.loc[6] = {
        "roi_id": "r6", "wafer": "W3", "z_star": 1, "defect_id": "non-w3",
        "is_candidate": True, "x0_aline": 0, "x1_aline": 2, "y0_aline": 0, "y1_aline": 2,
    }
    predictions.loc[5] = {
        "roi_id": "r5", "wafer": "W2", "z_star": 1, "model": "student",
        "seed": 0, "score": 0.05, "prediction": 0,
    }
    predictions.loc[6] = {
        "roi_id": "r6", "wafer": "W3", "z_star": 1, "model": "student",
        "seed": 0, "score": 0.05, "prediction": 0,
    }
    macro_values = {"W1": 0.95, "W2": 0.90, "W3": 0.70}

    def fake_operating_point(group):
        operational = group[group.is_candidate]
        row = {
            "threshold_low": 0.3,
            "threshold_high": 0.7,
            "referral_fraction": 1.0,
            "subsurface_defects_total": 1,
            "subsurface_defects_retained": 1,
            "subsurface_recall": 1.0,
            "budget_candidate_guided_percent": 100.0,
            "macro_f1": macro_values[group.wafer.iloc[0]],
        }
        return row, operational, np.ones(len(operational), dtype=bool)

    with patch("evaluate._student_operating_point", side_effect=fake_operating_point):
        result = evaluation.summarize(predictions, manifest)
    pooled = result[result.summary_type == "pooled"].iloc[0]
    assert np.isclose(pooled.macro_f1, 0.90)
    assert pooled.subsurface_defects_total == 3
    assert pooled.subsurface_defects_retained == 3
    assert pooled.subsurface_recall == 1.0
