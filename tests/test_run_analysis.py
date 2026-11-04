import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import run_analysis


def _frame():
    frames = []
    for wafer in ("W1", "W2"):
        frames.append(pd.DataFrame({
            "wafer": wafer,
            "z_star": [2, 1, 2, 1],
            "is_candidate": True,
            "prediction": [1, 1, 1, 1],
            "defect_id": ["a", "s1", "b", "s2"],
            "score": [0.1, 0.4, 0.6, 0.9],
            "student": [0.1, 0.4, 0.6, 0.9],
            "entropy": [0.9, 0.6, 0.4, 0.1],
            "inverse_margin": [0.9, 0.6, 0.4, 0.1],
            "anomaly": [0.2, 0.3, 0.7, 0.8],
            "direct": [0.2, 0.4, 0.6, 0.8],
            "teacher": [0.1, 0.3, 0.7, 0.9],
            "direct_p_subsurface": [0.2, 0.4, 0.6, 0.8],
            "student_p_subsurface": [0.1, 0.4, 0.6, 0.9],
            "x0_aline": [0, 1, 2, 3],
            "x1_aline": [1, 2, 3, 4],
            "y0_aline": 0,
            "y1_aline": 1,
        }))
    return pd.concat(frames, ignore_index=True)


def test_fixed_threshold_analysis_uses_other_wafers_and_writes_csv(tmp_path):
    source = _frame()
    result = run_analysis.fixed_threshold_analysis(source)
    assert len(result) == 2
    assert np.allclose(result.threshold_low, 0.099609375)
    assert np.allclose(result.threshold_high, 0.599609375)
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "fixed.csv"
    source.to_csv(input_path, index=False)
    written = run_analysis.main([
        "fixed-threshold", "--input", str(input_path), "--output", str(output_path)
    ])
    assert output_path.is_file()
    pd.testing.assert_frame_equal(written, pd.read_csv(output_path))


def test_score_comparison_command_reads_explicit_csv(tmp_path):
    input_path = tmp_path / "scores.csv"
    output_path = tmp_path / "comparison.csv"
    _frame().to_csv(input_path, index=False)
    result = run_analysis.main([
        "score-comparison", "--input", str(input_path), "--output", str(output_path)
    ])
    assert len(result) == 16
    assert set(result.wafer) == {"W1", "W2"}
    assert output_path.is_file()


def test_ablation_command_outputs_results_and_configs(tmp_path):
    frame = pd.concat([
        _frame().assign(model="Direct"),
        _frame().assign(model="+score"),
    ], ignore_index=True)
    input_path = tmp_path / "ablation.csv"
    output_path = tmp_path / "results.csv"
    configs_path = tmp_path / "configs.csv"
    frame.to_csv(input_path, index=False)
    result = run_analysis.main([
        "ablation", "--input", str(input_path), "--output", str(output_path),
        "--configs-output", str(configs_path),
    ])
    assert len(result) == 4
    assert len(pd.read_csv(configs_path)) == 20
    assert output_path.is_file()


def test_bootstrap_command_forwards_resampling_options(tmp_path):
    input_path = tmp_path / "bootstrap.csv"
    output_path = tmp_path / "intervals.csv"
    _frame().to_csv(input_path, index=False)
    result = run_analysis.main([
        "bootstrap", "--input", str(input_path), "--output", str(output_path),
        "--n-resamples", "4", "--seed", "7",
    ])
    assert len(result) == 8
    assert (result.n_resamples == 4).all()
    assert output_path.is_file()


def test_composition_command_outputs_summary_and_samples(tmp_path):
    input_path = tmp_path / "composition.csv"
    output_path = tmp_path / "summary.csv"
    samples_path = tmp_path / "samples.csv"
    _frame().to_csv(input_path, index=False)
    result = run_analysis.main([
        "composition", "--input", str(input_path), "--output", str(output_path),
        "--fractions", "1", "0", "--n-resamples", "2",
        "--samples-output", str(samples_path),
    ])
    assert len(result) == 4
    assert len(pd.read_csv(samples_path)) == 8
    assert output_path.is_file()


def test_fixed_threshold_analysis_uses_only_candidate_training_scores(monkeypatch):
    source = _frame()
    source["is_candidate"] = [True, True, False, True] * 2
    observed = []

    def thresholds(values):
        observed.append(np.asarray(values).tolist())
        return 0.25, 0.75

    monkeypatch.setattr(run_analysis, "multi_otsu_three_regions", thresholds)
    result = run_analysis.fixed_threshold_analysis(source)
    assert observed == [[0.1, 0.4, 0.9], [0.1, 0.4, 0.9]]
    assert np.allclose(result.auroc, 0.0)
    assert np.allclose(result.budget_percent, 100.0 / 3.0)
