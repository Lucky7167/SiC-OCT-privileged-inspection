import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import evaluate as evaluation
from common import candidate_mask


def _inputs(seed=0, scores=None):
    scores = [0.1, 0.8, 0.9, 0.05, 0.99] if scores is None else scores
    manifest = pd.DataFrame({
        "roi_id": ["r0", "r1", "r2", "r3", "r4"],
        "wafer": ["W1"] * 5,
        "z_star": [1, 2, 2, 2, 0],
        "defect_id": ["nonsubsurface", "defect-a", "defect-a", "defect-b", "normal"],
        "is_candidate": [True, True, True, False, False],
        "x0_aline": [0, 2, 4, 6, 8],
        "x1_aline": [2, 4, 6, 8, 10],
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
        "prediction": [0, 0, 0, 2, 0],
    })
    return predictions, manifest


def _decision(scores, predictions, thresholds=None):
    referred = np.zeros(len(scores), dtype=bool)
    referred[1] = True
    return {
        "low": 0.3,
        "high": 0.7,
        "referred": referred,
        "direct_prediction": np.asarray(predictions, dtype=int).copy(),
    }


def test_operational_metrics_use_only_manifest_candidates():
    predictions, manifest = _inputs()
    observed = []

    def record_decision(scores, microscopy_predictions, thresholds=None):
        observed.append(np.asarray(scores).tolist())
        return _decision(scores, microscopy_predictions, thresholds)

    with patch("evaluate.deploy", side_effect=record_decision):
        result = evaluation.evaluate(predictions, manifest)
    row = result.iloc[0]
    assert observed == [[0.1, 0.8, 0.9]]
    assert np.isclose(row.auroc, 1.0)
    assert np.isclose(row.auprc, 1.0)
    assert np.isclose(row.referral_fraction, 1.0 / 3.0)
    assert row.subsurface_defects_total == 1
    assert row.subsurface_defects_retained == 1
    assert np.isclose(row.subsurface_recall, 1.0)
    assert np.isclose(row.budget_candidate_guided_percent, 100.0 / 3.0)
    assert np.isclose(row.macro_f1, 13.0 / 30.0)


def test_missing_candidate_field_is_an_explicit_error():
    predictions, manifest = _inputs()
    with pytest.raises(ValueError, match="is_candidate"):
        evaluation.evaluate(predictions, manifest.drop(columns="is_candidate"))


@pytest.mark.parametrize("value", [None, np.nan, ""])
def test_null_candidate_value_is_an_explicit_error(value):
    predictions, manifest = _inputs()
    manifest = manifest.astype({"is_candidate": object})
    manifest.loc[0, "is_candidate"] = value
    with pytest.raises(ValueError, match="is_candidate"):
        evaluation.evaluate(predictions, manifest)


def test_summary_preserves_candidate_pool_and_seed_aggregation():
    first_predictions, manifest = _inputs()
    second_predictions, _ = _inputs(1, [0.5, 0.2, 0.9, 0.99, 0.01])
    predictions = pd.concat([first_predictions, second_predictions], ignore_index=True)
    with patch("evaluate.deploy", side_effect=_decision):
        result = evaluation.summarize(predictions, manifest)
    pooled = result[result.summary_type == "pooled"]
    across = result[result.summary_type == "across_seed"].iloc[0]
    assert len(pooled) == 2
    assert pooled.loc[pooled.primary, "seed"].tolist() == [0]
    assert np.allclose(pooled.auroc, [1.0, 0.5])
    assert np.allclose(pooled.subsurface_defects_total, [1, 1])
    assert np.allclose(pooled.subsurface_defects_retained, [1, 1])
    assert np.allclose(pooled.subsurface_recall, [1, 1])
    assert across.n_seeds == 2
    assert np.isclose(across.auroc_mean, 0.75)
    assert np.isclose(across.auroc_sample_sd, np.std([1.0, 0.5], ddof=1))


def test_summary_preserves_median_wafer_macro_f1():
    manifest = pd.DataFrame({
        "roi_id": [f"r{index}" for index in range(6)],
        "wafer": np.repeat(["W1", "W2", "W3"], 2),
        "z_star": [1, 2] * 3,
        "defect_id": ["non-a", "defect-a", "non-b", "defect-b", "non-c", "defect-c"],
        "is_candidate": True,
        "x0_aline": [0] * 6,
        "x1_aline": [2] * 6,
        "y0_aline": [0] * 6,
        "y1_aline": [2] * 6,
    })
    predictions = pd.DataFrame({
        "roi_id": manifest.roi_id,
        "wafer": manifest.wafer,
        "z_star": manifest.z_star,
        "model": "student",
        "seed": 0,
        "score": [0.1, 0.9] * 3,
        "prediction": 0,
    })
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
    assert np.isclose(pooled.subsurface_recall, 1.0)


def test_candidate_mask_accepts_only_explicit_boolean_and_binary_values():
    frame = pd.DataFrame({
        "is_candidate": pd.Series(
            [True, False, 1, 0, 1.0, 0.0, "true", "FALSE", " 1 ", "0"],
            dtype=object,
        )
    })
    assert candidate_mask(frame).tolist() == [True, False, True, False, True, False, True, False, True, False]


@pytest.mark.parametrize("value", [2, -1, 1.5, "2", "yes", "no"])
def test_candidate_mask_rejects_non_boolean_and_non_binary_values(value):
    frame = pd.DataFrame({"is_candidate": pd.Series([value], dtype=object)})
    with pytest.raises(ValueError, match="is_candidate"):
        candidate_mask(frame)
