import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from candidates import (
    detect_candidates,
    eight_connected_components,
    merge_anomalous_patches,
    oct_only_en_face_candidates,
    per_wafer_kmeans,
)
from oct_features import (
    contrast_continuity,
    detect_raw_oct_candidates,
    effective_depth_um,
    q_star_intermediates,
    raw_oct_candidate_mask,
    saturation_fraction,
    shadow_fraction,
    snr_db,
)
from robot import nearest_neighbor, nearest_neighbor_2opt, route_length, two_opt
from timing import (
    acquisition_pipeline_time,
    adjudication_durations,
    summarize_adjudication,
    time_to_result,
)


def _embeddings():
    return np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [-0.1, 0.0],
        [0.0, 0.1],
        [10.0, 10.0],
        [10.1, 10.0],
    ])


def test_per_wafer_kmeans_uses_larger_cluster_as_background():
    result = per_wafer_kmeans(_embeddings(), ["W1"] * 6)
    assert result["anomalous"].tolist() == [False, False, False, False, True, True]
    assert (result["anomaly_score"][:4] < 0.11).all()
    assert (result["anomaly_score"][4:] > 10).all()


def test_diagonal_anomalies_merge_with_eight_connectivity():
    mask = np.eye(3, dtype=bool)
    labels, count = eight_connected_components(mask)
    assert count == 1
    scores = mask.astype(float) * np.array([[1, 2, 3]]).T
    rois = merge_anomalous_patches(mask, scores)
    assert len(rois) == 1
    assert rois[0]["n_patches"] == 3
    assert rois[0]["a"] == 3


def test_oct_only_en_face_detection_reuses_label_free_candidate_pipeline():
    rows = np.array([0, 0, 1, 1, 3, 3])
    cols = np.array([0, 1, 0, 1, 3, 4])
    wafers = np.array(["W1"] * 6)
    microscopy = detect_candidates(_embeddings(), rows, cols, wafers)
    oct_only = oct_only_en_face_candidates(_embeddings(), rows, cols, wafers)
    pd.testing.assert_frame_equal(microscopy["rois"], oct_only["rois"])
    assert microscopy["rois"]["a"].iloc[0] > 10


def test_raw_oct_candidates_require_three_consecutive_bscans():
    volume = np.zeros((4, 2, 2))
    volume[:3, 0, 0] = 4.0
    volume[0, 1, 1] = 100.0
    mask = raw_oct_candidate_mask(volume, np.zeros((2, 2)), np.ones((2, 2)))
    assert mask[:3, 0, 0].all()
    assert not mask[3, 0, 0]
    assert not mask[:, 1, 1].any()
    background = np.zeros((2, 4, 2, 2))
    background[1] = 2.0
    assert detect_raw_oct_candidates(volume, background).shape == volume.shape


def test_snr_shadow_and_saturation_match_manuscript_constants():
    expected_snr = 20 * np.log10(10 / 1.4826)
    assert np.isclose(snr_db([10, 12, 14], [1, 2, 3]), expected_snr)
    assert snr_db([1], [1, 2, 3]) == -np.inf
    assert shadow_fraction([1.0, 100.0], [100.0, 100.0]) == 0.5
    raw = np.array([0.97, 0.98, 1.0])
    assert np.isclose(saturation_fraction(raw, np.ones(3, dtype=bool), 1.0), 2 / 3)


def test_continuity_effective_depth_and_q_star_intermediates():
    structures = [
        [(10, 20)],
        [(12, 22)],
        [(50, 60)],
        [(51, 61)],
        [(52, 62)],
    ]
    assert np.isclose(contrast_continuity(structures), 0.6)
    assert np.isclose(effective_depth_um(25.9), 10.0)
    result = q_star_intermediates(5, 10, 2, 0.1, 0.2, 0.7)
    assert np.isclose(result["snr_term"], 2 / 6)
    assert np.isclose(result["q_star"], (0.5 + 2 / 6 + 0.9 + 0.8 + 0.7) / 5)
    absent = q_star_intermediates(5, 10, 2, 0.1, 0.2, 0.7, False)
    assert absent["depth_term"] == 0
    assert absent["snr_term"] == 0
    assert absent["continuity_term"] == 0
    assert np.isclose(absent["q_star"], 0.34)


def test_nearest_neighbor_and_two_opt_produce_valid_non_worsening_routes():
    points = np.array([
        [0.0, 0.0],
        [4.0, 0.0],
        [1.0, 2.0],
        [5.0, 3.0],
        [2.0, 5.0],
        [6.0, 6.0],
    ])
    initial = nearest_neighbor(points, start=[-1, 0])
    refined = two_opt(points, initial)
    combined = nearest_neighbor_2opt(points)
    assert initial[0] == 0
    assert sorted(refined) == list(range(len(points)))
    assert route_length(points, refined) <= route_length(points, initial)
    assert route_length(points, combined) <= route_length(points, nearest_neighbor(points))


def test_adjudication_uses_display_to_decision_intervals():
    events = pd.DataFrame({
        "wafer": ["W1", "W1", "W2"],
        "reader": ["A", "A", "B"],
        "display_time": [100.0, 200.0, 300.0],
        "decision_time": [110.0, 220.0, 315.0],
        "file_loading_seconds": [1000.0, 1000.0, 1000.0],
    })
    durations = adjudication_durations(events)
    assert durations.adjudication_seconds.tolist() == [10.0, 20.0, 15.0]
    summary = summarize_adjudication(events)
    assert summary.total_seconds.tolist() == [30.0, 15.0]
    with pytest.raises(ValueError, match="cannot precede"):
        adjudication_durations([{"display_time": 2, "decision_time": 1}])


def test_pipeline_accounts_for_concurrency_and_time_to_result():
    assert acquisition_pipeline_time(68, 5, 100, 90) == 173
    events = pd.DataFrame({
        "stage": ["microscopy", "robot_motion", "oct_acquisition", "oct_reconstruction", "data_transfer"],
        "start": [0, 68, 73, 80, 0],
        "end": [68, 73, 173, 160, 500],
    })
    acquisition = acquisition_pipeline_time(events)
    assert acquisition == 173
    assert time_to_result(acquisition, 30) == 203
