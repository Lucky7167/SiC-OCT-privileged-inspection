from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from common import (
    META_COLUMNS,
    candidate_mask,
    feature_alignment_loss,
    logit_distillation_loss,
    pairwise_rank_loss,
)
from referral import minimum_recall_budget
from uncertainty import operating_point_metrics

DEFAULT_ALPHA = 0.7
DEFAULT_TAU = 3.0
LOSS_TERMS = ("ce", "rank", "feature", "logit", "score")
LOSS_WEIGHTS = {"ce": 1.0, "rank": 1.0, "feature": 1.0, "logit": 0.5, "score": 1.0}
ALPHA_VALUES = (0.5, 0.7, 0.9)
TAU_VALUES = (1.0, 3.0, 5.0)


def _configuration(
    name,
    category,
    terms,
    alpha=DEFAULT_ALPHA,
    tau=DEFAULT_TAU,
    use_risk=True,
    metadata_columns=META_COLUMNS,
    use_microscopy=True,
    component="",
):
    terms = tuple(terms)
    metadata_columns = tuple(metadata_columns)
    if not terms or set(terms) - set(LOSS_TERMS) or len(set(terms)) != len(terms):
        raise ValueError("loss terms must be unique known terms")
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be positive")
    if set(metadata_columns) - set(META_COLUMNS) or len(set(metadata_columns)) != len(metadata_columns):
        raise ValueError("metadata columns must be unique known columns")
    weights = {term: LOSS_WEIGHTS[term] if term in terms else 0.0 for term in LOSS_TERMS}
    return {
        "name": name,
        "category": category,
        "component": component,
        "loss_terms": terms,
        "loss_weights": weights,
        "alpha": float(alpha),
        "tau": float(tau),
        "use_risk": bool(use_risk),
        "metadata_columns": metadata_columns,
        "use_microscopy": bool(use_microscopy),
    }


def incremental_distillation_configs():
    return [
        _configuration("Direct", "incremental", ("ce", "rank")),
        _configuration("+logit", "incremental", ("ce", "rank", "logit")),
        _configuration("+feature", "incremental", ("ce", "rank", "logit", "feature")),
        _configuration("+score", "incremental", LOSS_TERMS),
    ]


def leave_one_loss_configs():
    result = [_configuration("Full", "leave_one_loss", LOSS_TERMS)]
    result.extend(
        _configuration(f"w/o {term}", "leave_one_loss", tuple(x for x in LOSS_TERMS if x != term))
        for term in LOSS_TERMS
    )
    return result


def hyperparameter_sensitivity_configs(alphas=ALPHA_VALUES, taus=TAU_VALUES):
    result = []
    for alpha in alphas:
        result.append(_configuration(
            f"alpha={alpha}", "hyperparameter", LOSS_TERMS,
            alpha=float(alpha), component="alpha",
        ))
    for tau in taus:
        result.append(_configuration(
            f"tau={tau}", "hyperparameter", LOSS_TERMS,
            tau=float(tau), component="tau",
        ))
    return result


def metadata_ablation_configs():
    return [
        _configuration("Full metadata", "metadata", LOSS_TERMS),
        _configuration(
            "w/o coordinates", "metadata", LOSS_TERMS,
            metadata_columns=("w", "h", "a"),
        ),
        _configuration(
            "w/o microscopy metadata", "metadata", LOSS_TERMS,
            metadata_columns=("x", "y"),
        ),
        _configuration(
            "metadata only", "metadata", LOSS_TERMS,
            use_microscopy=False,
        ),
    ]


def all_ablation_configs():
    return (
        incremental_distillation_configs()
        + leave_one_loss_configs()
        + hyperparameter_sensitivity_configs()
        + metadata_ablation_configs()
    )


def configured_loss(config: Mapping):
    weights = config.get("loss_weights")
    if weights is None:
        terms = tuple(config.get("loss_terms", LOSS_TERMS))
        weights = {term: LOSS_WEIGHTS[term] if term in terms else 0.0 for term in LOSS_TERMS}
    unknown = set(weights) - set(LOSS_TERMS)
    if unknown:
        raise ValueError(f"unknown loss terms: {sorted(unknown)}")
    values = {term: float(weights.get(term, 0.0)) for term in LOSS_TERMS}
    if any(not np.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("loss weights must be finite and non-negative")
    if not any(values.values()):
        raise ValueError("at least one loss term must be enabled")
    tau = float(config.get("tau", DEFAULT_TAU))
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be positive")

    def loss(output, batch, class_counts):
        total = None
        terms = {
            "ce": lambda: F.cross_entropy(output["logits"], batch["z_star"]),
            "rank": lambda: pairwise_rank_loss(output["score"], batch["U_star"]),
            "feature": lambda: feature_alignment_loss(
                output["projected_feature"], batch["teacher_feature"]
            ),
            "logit": lambda: logit_distillation_loss(
                output["logits"], batch["teacher_logits"], tau
            ),
            "score": lambda: F.huber_loss(
                output["score"], batch["teacher_score"], delta=1.0
            ),
        }
        for term, weight in values.items():
            if weight:
                value = weight * terms[term]()
                total = value if total is None else total + value
        return total

    return loss


def loss_registry(configs: Iterable[Mapping] | None = None):
    configurations = incremental_distillation_configs() if configs is None else configs
    return {config["name"]: configured_loss(config) for config in configurations}


LOSS_REGISTRY = loss_registry()


def configs_to_frame(configs: Iterable[Mapping]):
    rows = []
    for config in configs:
        weights = config.get("loss_weights", {})
        rows.append({
            "name": config["name"],
            "category": config["category"],
            "component": config.get("component", ""),
            "loss_terms": ";".join(config["loss_terms"]),
            **{f"weight_{term}": float(weights.get(term, 0.0)) for term in LOSS_TERMS},
            "alpha": float(config.get("alpha", DEFAULT_ALPHA)),
            "tau": float(config.get("tau", DEFAULT_TAU)),
            "use_risk": bool(config.get("use_risk", True)),
            "metadata_columns": ";".join(config.get("metadata_columns", META_COLUMNS)),
            "use_microscopy": bool(config.get("use_microscopy", True)),
        })
    return pd.DataFrame(rows)


def write_configs(configs: Iterable[Mapping], output):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = configs_to_frame(configs)
    result.to_csv(path, index=False)
    return result


def _safe_auroc(labels, scores):
    labels = np.asarray(labels, int)
    if len(np.unique(labels)) < 2:
        return np.nan
    return float(roc_auc_score(labels, scores))


def evaluate_ablations(frame: pd.DataFrame, model_column="model", score_column="score"):
    required = {
        model_column,
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
        raise ValueError(f"ablation evaluation requires {sorted(missing)}")
    keys = [model_column]
    if "seed" in frame.columns:
        keys.append("seed")
    keys.append("wafer")
    rows = []
    for key, group in frame.groupby(keys, sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        values = dict(zip(keys, key))
        candidates = group.loc[candidate_mask(group)]
        if candidates.empty:
            raise ValueError("ablation evaluation requires candidate rows in every group")
        ranking = minimum_recall_budget(candidates, candidates[score_column].to_numpy(float))
        operating = operating_point_metrics(group, score_column=score_column)
        row = {
            "model": values[model_column],
            "wafer": values["wafer"],
            "auroc": _safe_auroc(
                candidates.z_star.to_numpy(int) == 2,
                candidates[score_column].to_numpy(float),
            ),
            "budget_percent": ranking["budget_percent"],
            "complete_recall_budget_percent": ranking["budget_percent"],
            "complete_recall_count": ranking["referral_count"],
            "operating_budget_percent": operating["budget_percent"],
            "macro_f1": operating["macro_f1"],
            "direct_precision": operating["direct_precision"],
            "threshold_low": operating["threshold_low"],
            "threshold_high": operating["threshold_high"],
        }
        if "seed" in values:
            row["seed"] = values["seed"]
        rows.append(row)
    return pd.DataFrame(rows)


def write_results(result: pd.DataFrame, output):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    return result
