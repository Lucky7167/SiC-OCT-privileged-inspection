from __future__ import annotations
import numpy as np
import pandas as pd

from common import candidate_mask


def multi_otsu_three_regions(scores, bins=256):
    values = np.asarray(scores, float)
    histogram, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    probability = histogram / max(histogram.sum(), 1)
    centers = (edges[:-1] + edges[1:]) / 2
    omega = np.cumsum(probability); moment = np.cumsum(probability * centers)
    total = moment[-1]; best, thresholds = -np.inf, None
    for first in range(0, bins - 2):
        for second in range(first + 1, bins - 1):
            weights = (omega[first], omega[second] - omega[first], 1 - omega[second])
            if min(weights) <= 0:
                continue
            means = (
                moment[first] / weights[0],
                (moment[second] - moment[first]) / weights[1],
                (total - moment[second]) / weights[2],
            )
            objective = sum(w * (m - total) ** 2 for w, m in zip(weights, means))
            if objective > best:
                best, thresholds = objective, (centers[first], centers[second])
    if thresholds is None:
        raise ValueError("score distribution cannot be split into three non-empty regions")
    return thresholds


def deploy(scores, microscopy_predictions, thresholds=None):
    scores = np.asarray(scores, float); predictions = np.asarray(microscopy_predictions, int)
    low, high = thresholds or multi_otsu_three_regions(scores)
    referred = (scores >= low) & (scores < high)
    final = predictions.copy(); final[scores >= high] = 2
    return {"low": low, "high": high, "referred": referred, "direct_prediction": final}


def footprint_cost(frame: pd.DataFrame, selected) -> int:
    required = {"x0_aline", "x1_aline", "y0_aline", "y1_aline"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"footprint cost requires {sorted(missing)}")
    rectangles = frame.loc[np.asarray(selected, bool), list(required)]
    cells = set()
    for row in rectangles.itertuples(index=False):
        for x in range(int(row.x0_aline), int(row.x1_aline)):
            cells.update((x, y) for y in range(int(row.y0_aline), int(row.y1_aline)))
    return len(cells)


def normalized_budget(frame, selected, reference):
    denominator = footprint_cost(frame, reference)
    return 100.0 * footprint_cost(frame, selected) / denominator


def oracle_recall_budget_curve(frame, order, positive_label=2):
    if "defect_id" not in frame.columns:
        raise ValueError("oracle recall-budget evaluation requires manifest defect_id")
    if frame.defect_id.isna().any() or frame.defect_id.astype(str).str.strip().eq("").any():
        raise ValueError("manifest defect_id cannot contain missing values")
    labels = frame.z_star.to_numpy(); selected = np.zeros(len(frame), dtype=bool)
    positive = labels == positive_label
    total = frame.loc[positive, "defect_id"].nunique(); rows = []
    reference = np.ones(len(frame), dtype=bool)
    for index in order:
        selected[index] = True
        retained = frame.loc[positive & selected, "defect_id"].nunique()
        recall = retained / max(total, 1)
        rows.append((normalized_budget(frame, selected, reference), recall))
    return pd.DataFrame(rows, columns=["budget_percent", "recall"])


SCORE_NAMES = (
    "entropy",
    "inverse_margin",
    "anomaly",
    "direct",
    "student",
    "teacher",
    "direct_p_subsurface",
    "student_p_subsurface",
)
REJECT_OPTION_SCORES = frozenset({"entropy", "inverse_margin"})
PROBABILITY_COLUMNS = ("p_normal", "p_surface", "p_subsurface")


def predictive_entropy(probabilities):
    values = np.asarray(probabilities, float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("probabilities must be a two-dimensional class-probability array")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    return -np.sum(np.where(values > 0, values * np.log(values), 0.0), axis=1)


def inverse_margin(probabilities):
    values = np.asarray(probabilities, float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("probabilities must be a two-dimensional class-probability array")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    ordered = np.sort(values, axis=1)
    return 1.0 - ordered[:, -1] + ordered[:, -2]


def _score_column(frame, names, required=True):
    for name in names:
        if name in frame.columns:
            return frame[name].to_numpy(float)
    if required:
        raise ValueError(f"referral scores require one of {list(names)}")
    return None


def extract_referral_scores(frame: pd.DataFrame, probabilities=None):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    result = pd.DataFrame(index=frame.index)
    entropy = _score_column(frame, ("entropy", "predictive_entropy"), False)
    margin = _score_column(frame, ("inverse_margin",), False)
    if probabilities is None:
        classifier_columns = tuple(f"classifier_{name}" for name in PROBABILITY_COLUMNS)
        if all(name in frame.columns for name in classifier_columns):
            probabilities = frame.loc[:, classifier_columns].to_numpy(float)
        elif all(name in frame.columns for name in PROBABILITY_COLUMNS):
            probabilities = frame.loc[:, PROBABILITY_COLUMNS].to_numpy(float)
        elif entropy is None or margin is None:
            raise ValueError("entropy and inverse margin require class probabilities")
    if probabilities is not None:
        values = np.asarray(probabilities, float)
        if len(values) != len(frame):
            raise ValueError("probabilities and frame must have equal lengths")
        if entropy is None:
            entropy = predictive_entropy(values)
        if margin is None:
            margin = inverse_margin(values)
    result["entropy"] = entropy
    result["inverse_margin"] = margin
    result["anomaly"] = _score_column(frame, ("anomaly", "anomaly_score", "a"))
    result["direct"] = _score_column(frame, ("direct", "direct_score"))
    result["student"] = _score_column(frame, ("student", "student_score", "score"))
    result["teacher"] = _score_column(frame, ("teacher", "teacher_score"))
    result["direct_p_subsurface"] = _score_column(
        frame, ("direct_p_subsurface", "p_subsurface_direct")
    )
    result["student_p_subsurface"] = _score_column(
        frame, ("student_p_subsurface", "p_subsurface_student", "p_subsurface")
    )
    values = result.to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("referral scores must be finite")
    return result


def referral_scores(frame: pd.DataFrame, probabilities=None):
    return extract_referral_scores(frame, probabilities)


def score_order(scores):
    values = np.asarray(scores, float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("scores must be a finite one-dimensional array")
    return np.argsort(-values, kind="stable")


def top_k(scores, k):
    values = np.asarray(scores, float)
    if values.ndim != 1 or int(k) != k or not 0 <= int(k) <= len(values):
        raise ValueError("k must be an integer between zero and the number of scores")
    selected = np.zeros(len(values), dtype=bool)
    selected[score_order(values)[:int(k)]] = True
    return selected


def top_k_referral(scores, k):
    return top_k(scores, k)


def reject_option(scores, microscopy_predictions, threshold=None):
    values = np.asarray(scores, float)
    predictions = np.asarray(microscopy_predictions, int)
    if values.ndim != 1 or values.shape != predictions.shape:
        raise ValueError("scores and microscopy predictions must have equal one-dimensional shapes")
    if threshold is None:
        threshold = multi_otsu_three_regions(values)[1]
    referred = values >= threshold
    return {
        "threshold": float(threshold),
        "referred": referred,
        "direct_prediction": predictions.copy(),
    }


def two_threshold(scores, microscopy_predictions, thresholds=None):
    return deploy(scores, microscopy_predictions, thresholds)


def training_derived_fixed_thresholds(train_scores, test_scores, microscopy_predictions):
    thresholds = multi_otsu_three_regions(np.asarray(train_scores, float))
    return deploy(test_scores, microscopy_predictions, thresholds)


def training_derived_fixed_threshold(train_scores, test_scores, microscopy_predictions):
    return training_derived_fixed_thresholds(train_scores, test_scores, microscopy_predictions)


def _validate_defect_ids(frame):
    if "defect_id" not in frame.columns:
        raise ValueError("defect-level evaluation requires defect_id")
    missing = frame.defect_id.isna() | frame.defect_id.astype(str).str.strip().eq("")
    if missing.any():
        raise ValueError("defect_id cannot contain missing values")


def defect_coverage(frame, selected, positive_label=2):
    _validate_defect_ids(frame)
    chosen = np.asarray(selected, bool)
    if chosen.shape != (len(frame),):
        raise ValueError("selected must have one Boolean value per frame row")
    positive = frame.z_star.to_numpy() == positive_label
    total = frame.loc[positive, "defect_id"].nunique()
    covered = frame.loc[positive & chosen, "defect_id"].nunique()
    return {"covered": int(covered), "total": int(total), "recall": covered / max(total, 1)}


def minimum_recall_budget(frame, scores, target_recall=1.0, positive_label=2):
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1]")
    operational = candidate_mask(frame)
    candidates = frame.loc[operational]
    scores = np.asarray(scores, float)
    if scores.ndim != 1 or len(scores) != len(frame) or not np.isfinite(scores).all():
        raise ValueError("scores must contain one finite value per frame row")
    curve = oracle_recall_budget_curve(candidates, score_order(scores[operational]), positive_label)
    reached = curve.index[curve.recall >= target_recall]
    if len(reached) == 0:
        return {"budget_percent": np.nan, "referral_count": 0, "recall": 0.0}
    index = reached[0]
    return {
        "budget_percent": float(curve.loc[index, "budget_percent"]),
        "referral_count": int(index + 1),
        "recall": float(curve.loc[index, "recall"]),
    }


def matched_count_comparison(frame: pd.DataFrame, scores=None, reference_score="student"):
    required = {"wafer", "z_star", "prediction", "defect_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"matched-count comparison requires {sorted(missing)}")
    _validate_defect_ids(frame)
    score_frame = extract_referral_scores(frame) if scores is None else scores
    if isinstance(score_frame, pd.DataFrame):
        score_map = {name: score_frame[name].to_numpy(float) for name in score_frame.columns}
    elif isinstance(score_frame, dict):
        score_map = {name: np.asarray(value, float) for name, value in score_frame.items()}
    else:
        raise TypeError("scores must be a DataFrame or a mapping of score arrays")
    if reference_score not in score_map:
        raise ValueError(f"unknown reference score: {reference_score}")
    candidate_mask(frame)
    rows = []
    for wafer, group in frame.groupby("wafer", sort=True):
        positions = frame.index.get_indexer(group.index)
        if (positions < 0).any():
            raise ValueError("frame index must be unique")
        operational = candidate_mask(group)
        group = group.iloc[np.flatnonzero(operational)]
        positions = positions[operational]
        if group.empty:
            raise ValueError("matched-count comparison requires candidate rows in every wafer")
        reference = deploy(
            score_map[reference_score][positions], group.prediction.to_numpy(int)
        )
        k = int(reference["referred"].sum())
        for name, values in score_map.items():
            selected = top_k(values[positions], k)
            ranking = defect_coverage(group, selected)
            combined_selected = selected.copy()
            direct_high = np.zeros(len(group), dtype=bool)
            if name not in REJECT_OPTION_SCORES:
                decision = deploy(values[positions], group.prediction.to_numpy(int))
                direct_high = values[positions] >= decision["high"]
                combined_selected |= direct_high
            combined = defect_coverage(group, combined_selected)
            labels = group.z_star.to_numpy(int)
            initial = group.prediction.to_numpy(int)
            errors = initial != labels
            ranking_errors = int(np.sum(selected & errors))
            direct_corrections = int(np.sum(
                direct_high & ~selected & (labels == 2) & (initial != 2)
            ))
            rows.append({
                "wafer": wafer,
                "score": name,
                "referral_count": k,
                "ranking_defects_covered": ranking["covered"],
                "combined_defects_covered": combined["covered"],
                "subsurface_defects_total": ranking["total"],
                "direct_accepted_count": int(direct_high.sum()),
                "ranking_errors_corrected": ranking_errors,
                "combined_errors_corrected": ranking_errors + direct_corrections,
                "microscopy_errors_total": int(errors.sum()),
                "ranking_budget_percent": normalized_budget(
                    group, selected, np.ones(len(group), dtype=bool)
                ),
            })
    return pd.DataFrame(rows)
