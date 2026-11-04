import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import targets


def _frame():
    frame = pd.DataFrame({
        "wafer": [wafer for wafer in ("W1", "W2", "W3", "W4") for _ in range(3)],
        "roi_id": [f"roi_{index}" for index in range(12)],
        "z_star": [0, 1, 2] * 4,
        "x": np.arange(12),
        "y": 0,
        "w": 1,
        "h": 1,
        "a": 0,
        "deff": 10.0,
        "dmax": 10.0,
        "snr": 2.0,
        "snr_ref": 2.0,
        "rho_shadow": 0.0,
        "rho_sat": 0.0,
        "c_cont": 1.0,
        "has_depth_structure": True,
    })
    return frame


def test_final_targets_keep_cross_fitted_nll_without_in_sample_overwrite(monkeypatch):
    train = _frame()
    test = train.iloc[:2].copy()
    fits = []

    def fake_fit(inner_train, inner_valid, normalization, epochs=100):
        fits.append((inner_train.copy(), inner_valid.copy()))
        return {"train_wafers": tuple(sorted(inner_train.wafer.unique()))}

    def fake_nll(model, frame, normalization):
        return {roi_id: float(roi_id.split("_")[1]) + 1 for roi_id in frame.roi_id}

    monkeypatch.setattr(targets, "_fit_classifier", fake_fit)
    monkeypatch.setattr(targets, "_nll", fake_nll)
    result, result_test, normalization = targets.build_final_targets(train, test)
    expected = np.arange(12, dtype=float) / 11
    assert len(fits) == 4
    assert all(len(inner_train) == 9 and len(inner_valid) == 3 for inner_train, inner_valid in fits)
    assert np.allclose(result.R, expected)
    assert np.allclose(result.q_star, 1.0)
    assert np.allclose(result.U_star, 0.7 + 0.3 * expected)
    assert "R" not in train
    assert "R" not in result_test
    assert set(normalization) == {"x", "y", "w", "h", "a"}
