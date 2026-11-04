from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from common import candidate_mask
from referral import deploy, footprint_cost, multi_otsu_three_regions


DEFAULT_BOOTSTRAP_REPLICATES = 4000
DEFAULT_MONTE_CARLO_REPLICATES = 500
SUBSURFACE_FRACTIONS = (1.0, 0.5, 0.25, 0.1, 0.05, 0.0)


def _required_columns(frame, score_column):
    required = {
        "wafer",
        "z_star",
        "is_candidate",
        "prediction",
        "defect_id",
        score_column,
        "x0_aline",
        "x1_aline",
        "y0_aline",
        "y1_aline",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"uncertainty analysis requires {sorted(missing)}")
    missing_ids = frame.defect_id.isna() | frame.defect_id.astype(str).str.strip().eq("")
    if missing_ids.any():
        raise ValueError("defect_id cannot contain missing values")


def _safe_auroc(labels, scores):
    labels = np.asarray(labels, int)
    if len(np.unique(labels)) < 2:
        return np.nan
    return float(roc_auc_score(labels, scores))


def _safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else np.nan


def _thresholds_for_wafer(thresholds, wafer):
    if thresholds is None:
        return None
    if isinstance(thresholds, dict):
        return thresholds[wafer]
    return thresholds


def _prepare_wafer(group, score_column, thresholds=None):
    group = group.reset_index(drop=True)
    operational = candidate_mask(group)
    decision = deploy(
        group.loc[operational, score_column].to_numpy(float),
        group.loc[operational, "prediction"].to_numpy(int),
        thresholds,
    )
    final = group.prediction.to_numpy(int).copy()
    operational_final = decision["direct_prediction"]
    operational_final[decision["referred"]] = group.loc[
        operational, "z_star"
    ].to_numpy(int)[decision["referred"]]
    final[operational] = operational_final
    referred = np.zeros(len(group), dtype=bool)
    direct_high = np.zeros(len(group), dtype=bool)
    referred[operational] = decision["referred"]
    direct_high[operational] = group.loc[operational, score_column].to_numpy(float) >= decision["high"]
    return {
        "frame": group,
        "operational": operational,
        "referred": referred,
        "direct_high": direct_high,
        "final": final,
        "low": float(decision["low"]),
        "high": float(decision["high"]),
    }


def _metrics(prepared, macro_f1=True):
    frames = [item["frame"] for item in prepared]
    if not frames:
        return {"auroc": np.nan, "budget_percent": np.nan, "macro_f1": np.nan, "direct_precision": np.nan}
    frame = pd.concat(frames, ignore_index=True)
    direct_high = np.concatenate([item["direct_high"] for item in prepared])
    labels = frame.z_star.to_numpy(int)
    positive = np.concatenate([
        item["frame"].loc[item["operational"], "z_star"].to_numpy(int) == 2
        for item in prepared
    ])
    score_values = np.concatenate([
        item["frame"].loc[item["operational"], item["score_column"]].to_numpy(float)
        for item in prepared
    ])
    denominator = 0
    numerator = 0
    for item in prepared:
        operational_frame = item["frame"].loc[item["operational"]]
        reference = np.ones(len(operational_frame), dtype=bool)
        denominator += footprint_cost(operational_frame, reference)
        numerator += footprint_cost(
            operational_frame, item["referred"][item["operational"]]
        )
    high_labels = labels[direct_high]
    if macro_f1:
        wafer_values = [
            f1_score(
                item["frame"].z_star.to_numpy(int),
                item["final"],
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
            for item in prepared
        ]
        macro = float(np.median(wafer_values))
    else:
        macro = np.nan
    return {
        "auroc": _safe_auroc(positive, score_values),
        "budget_percent": 100.0 * numerator / denominator if denominator else np.nan,
        "macro_f1": macro,
        "direct_precision": _safe_ratio(np.sum(high_labels == 2), len(high_labels)),
    }


def _point_metrics(prepared):
    return _metrics(prepared)


def operating_point_metrics(frame: pd.DataFrame, score_column="score", thresholds=None):
    _required_columns(frame, score_column)
    prepared = []
    for wafer, group in frame.groupby("wafer", sort=True):
        item = _prepare_wafer(group, score_column, _thresholds_for_wafer(thresholds, wafer))
        item["score_column"] = score_column
        prepared.append(item)
    result = _point_metrics(prepared)
    if len(prepared) == 1:
        result["threshold_low"] = prepared[0]["low"]
        result["threshold_high"] = prepared[0]["high"]
    return result


def _resampled_prepared(prepared, selected, hierarchical, rng):
    result = []
    for wafer_index in selected:
        item = prepared[wafer_index]
        if not hierarchical:
            result.append(item)
            continue
        indices = rng.integers(0, len(item["frame"]), len(item["frame"]))
        sampled = item["frame"].iloc[indices].reset_index(drop=True)
        operational = candidate_mask(sampled)
        referred = np.zeros(len(sampled), dtype=bool)
        direct_high = np.zeros(len(sampled), dtype=bool)
        final = sampled.prediction.to_numpy(int).copy()
        scores = sampled.loc[operational, item["score_column"]].to_numpy(float)
        low_scores = scores < item["low"]
        high_scores = scores >= item["high"]
        referred_scores = ~low_scores & ~high_scores
        referred[np.flatnonzero(operational)[referred_scores]] = True
        direct_high[np.flatnonzero(operational)[high_scores]] = True
        operational_final = sampled.loc[operational, "prediction"].to_numpy(int).copy()
        operational_final[high_scores] = 2
        operational_final[referred_scores] = sampled.loc[operational, "z_star"].to_numpy(int)[referred_scores]
        final[operational] = operational_final
        result.append({
            "frame": sampled,
            "operational": operational,
            "referred": referred,
            "direct_high": direct_high,
            "final": final,
            "low": item["low"],
            "high": item["high"],
            "score_column": item["score_column"],
        })
    return result


def bootstrap_intervals(
    frame: pd.DataFrame,
    n_resamples=DEFAULT_BOOTSTRAP_REPLICATES,
    confidence_level=0.95,
    seed=0,
    score_column="score",
    thresholds=None,
):
    _required_columns(frame, score_column)
    if int(n_resamples) != n_resamples or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    if frame.wafer.nunique() < 1:
        raise ValueError("bootstrap requires at least one wafer")
    prepared = []
    for wafer, group in frame.groupby("wafer", sort=True):
        item = _prepare_wafer(group, score_column, _thresholds_for_wafer(thresholds, wafer))
        item["score_column"] = score_column
        prepared.append(item)
    point = _point_metrics(prepared)
    rng = np.random.default_rng(seed)
    metrics = ("auroc", "budget_percent", "macro_f1", "direct_precision")
    samples = {mode: {metric: [] for metric in metrics} for mode in ("wafer", "hierarchical")}
    for _ in range(int(n_resamples)):
        selected = rng.integers(0, len(prepared), len(prepared))
        for mode in ("wafer", "hierarchical"):
            resampled = _resampled_prepared(prepared, selected, mode == "hierarchical", rng)
            values = _metrics(resampled)
            for metric in metrics:
                samples[mode][metric].append(values[metric])
    alpha = (1.0 - confidence_level) / 2.0
    rows = []
    for mode in ("wafer", "hierarchical"):
        for metric in metrics:
            values = np.asarray(samples[mode][metric], float)
            valid = values[np.isfinite(values)]
            lower, upper = (
                np.quantile(valid, [alpha, 1.0 - alpha]) if len(valid) else (np.nan, np.nan)
            )
            rows.append({
                "resampling": mode,
                "metric": metric,
                "estimate": point[metric],
                "lower": float(lower),
                "upper": float(upper),
                "n_resamples": int(n_resamples),
                "n_valid": int(len(valid)),
            })
    return pd.DataFrame(rows)


def wafer_bootstrap(frame, n_resamples=DEFAULT_BOOTSTRAP_REPLICATES, **kwargs):
    result = bootstrap_intervals(frame, n_resamples=n_resamples, **kwargs)
    return result[result.resampling == "wafer"].reset_index(drop=True)


def hierarchical_bootstrap(frame, n_resamples=DEFAULT_BOOTSTRAP_REPLICATES, **kwargs):
    result = bootstrap_intervals(frame, n_resamples=n_resamples, **kwargs)
    return result[result.resampling == "hierarchical"].reset_index(drop=True)


def _composition_sample(group, fraction, sample_size, rng):
    operational = group.loc[candidate_mask(group)]
    subsurface = operational[operational.z_star == 2]
    surface = operational[operational.z_star == 1]
    if sample_size is None:
        positive_count = int(round(len(subsurface) * fraction))
        positive_count = min(max(positive_count, 0), len(subsurface))
        positive_indices = rng.choice(len(subsurface), positive_count, replace=False)
        return pd.concat([
            subsurface.iloc[positive_indices],
            surface,
        ]).reset_index(drop=True)
    if int(sample_size) != sample_size or sample_size < 1:
        raise ValueError("sample_size must be a positive integer")
    sample_size = int(sample_size)
    prevalence = len(subsurface) / max(len(operational), 1)
    positive_count = min(int(round(sample_size * prevalence * fraction)), sample_size)
    negative_count = sample_size - positive_count
    positive_indices = rng.choice(
        len(subsurface), positive_count, replace=positive_count > len(subsurface)
    ) if positive_count else np.array([], dtype=int)
    negative_indices = rng.choice(
        len(surface), negative_count, replace=negative_count > len(surface)
    ) if negative_count else np.array([], dtype=int)
    return pd.concat([
        subsurface.iloc[positive_indices],
        surface.iloc[negative_indices],
    ]).reset_index(drop=True)


def _monte_carlo_row(wafer, fraction, replicate, estimation, evaluation, score_column):
    row = {
        "wafer": wafer,
        "subsurface_fraction": float(fraction),
        "subsurface_percent": 100.0 * float(fraction),
        "replicate": int(replicate),
        "threshold_low": np.nan,
        "threshold_high": np.nan,
        "low_zone_miss_count": np.nan,
        "low_zone_miss_rate": np.nan,
        "direct_high_surface_error_count": np.nan,
        "direct_high_surface_error_rate": np.nan,
        "referral_share": np.nan,
    }
    try:
        low, high = multi_otsu_three_regions(estimation[score_column].to_numpy(float))
    except ValueError:
        return row
    operational = evaluation.loc[candidate_mask(evaluation)].reset_index(drop=True)
    scores = operational[score_column].to_numpy(float)
    labels = operational.z_star.to_numpy(int)
    low_zone = scores < low
    direct_high = scores >= high
    referred = ~low_zone & ~direct_high
    positive = operational[labels == 2]
    positive_low = positive[low_zone[labels == 2]]
    total_defects = positive.defect_id.nunique()
    missed_defects = positive_low.defect_id.nunique()
    surface_count = int(np.sum(labels == 1))
    surface_errors = int(np.sum(direct_high & (labels == 1)))
    row.update({
        "threshold_low": float(low),
        "threshold_high": float(high),
        "low_zone_miss_count": int(missed_defects),
        "low_zone_miss_rate": _safe_ratio(missed_defects, total_defects),
        "direct_high_surface_error_count": surface_errors,
        "direct_high_surface_error_rate": _safe_ratio(surface_errors, surface_count),
        "referral_share": float(referred.mean()) if len(referred) else np.nan,
    })
    return row


def class_composition_monte_carlo(
    frame: pd.DataFrame,
    fractions=SUBSURFACE_FRACTIONS,
    n_resamples=DEFAULT_MONTE_CARLO_REPLICATES,
    sample_size=None,
    seed=0,
    score_column="score",
):
    _required_columns(frame, score_column)
    if int(n_resamples) != n_resamples or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    values = tuple(float(value) for value in fractions)
    if not values or any(not 0 <= value <= 1 for value in values):
        raise ValueError("fractions must contain values in [0, 1]")
    rng = np.random.default_rng(seed)
    rows = []
    for wafer, group in frame.groupby("wafer", sort=True):
        for fraction in values:
            for replicate in range(int(n_resamples)):
                estimation = _composition_sample(group, fraction, sample_size, rng)
                rows.append(_monte_carlo_row(
                    wafer, fraction, replicate, estimation, group, score_column
                ))
    return pd.DataFrame(rows)


def monte_carlo_class_composition(frame, **kwargs):
    return class_composition_monte_carlo(frame, **kwargs)


def summarize_class_composition(samples: pd.DataFrame):
    metrics = (
        "low_zone_miss_rate",
        "direct_high_surface_error_rate",
        "referral_share",
        "threshold_low",
        "threshold_high",
    )
    rows = []
    keys = ["wafer", "subsurface_fraction", "subsurface_percent"]
    for key, group in samples.groupby(keys, sort=True):
        row = dict(zip(keys, key))
        for metric in metrics:
            values = group[metric].dropna().to_numpy(float)
            row[f"{metric}_median"] = float(np.median(values)) if len(values) else np.nan
            row[f"{metric}_q1"] = float(np.quantile(values, 0.25)) if len(values) else np.nan
            row[f"{metric}_q3"] = float(np.quantile(values, 0.75)) if len(values) else np.nan
            row[f"{metric}_mean"] = float(np.mean(values)) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
