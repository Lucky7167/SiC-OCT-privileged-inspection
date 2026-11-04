import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from common import candidate_mask
from referral import deploy, footprint_cost


def _merge_inputs(predictions, manifest):
    if "defect_id" not in manifest.columns:
        raise ValueError("evaluation requires manifest defect_id")
    missing = manifest.defect_id.isna() | manifest.defect_id.astype(str).str.strip().eq("")
    if missing.any():
        raise ValueError("manifest defect_id cannot contain missing values")
    candidate_mask(manifest)
    predictions = predictions.copy()
    if "seed" not in predictions.columns:
        predictions["seed"] = 0
    return predictions.merge(manifest, on=["roi_id", "wafer", "z_star"], validate="many_to_one")


def _student_operating_point(augmented):
    operational = augmented.loc[candidate_mask(augmented)].copy()
    decision = deploy(operational.score, operational.prediction)
    referred = decision["referred"]
    final = decision["direct_prediction"]
    final[referred] = operational.z_star.to_numpy()[referred]
    positive = operational.z_star.to_numpy() == 2
    total = operational.loc[positive, "defect_id"].nunique()
    retained = operational.loc[positive & (final == 2), "defect_id"].nunique()
    row = {
        "threshold_low": decision["low"],
        "threshold_high": decision["high"],
        "referral_fraction": referred.mean(),
        "subsurface_defects_total": total,
        "subsurface_defects_retained": retained,
        "subsurface_recall": retained / max(total, 1),
        "budget_candidate_guided_percent": 100.0 * footprint_cost(
            operational, referred
        ) / footprint_cost(operational, np.ones(len(operational), dtype=bool)),
    }
    final_augmented = augmented.prediction.to_numpy().copy()
    positions = augmented.index.get_indexer(operational.index)
    final_augmented[positions] = final
    row["macro_f1"] = f1_score(augmented.z_star, final_augmented, average="macro")
    return row, operational, referred


def _evaluate_merged(merged):
    rows = []
    for (model, wafer, seed), group in merged.groupby(["model", "wafer", "seed"]):
        operational = group.loc[candidate_mask(group)]
        labels = (operational.z_star == 2).astype(int)
        row = {
            "model": model,
            "wafer": wafer,
            "seed": seed,
            "auroc": roc_auc_score(labels, operational.score),
            "auprc": average_precision_score(labels, operational.score),
        }
        if model == "student":
            operating_point, _, _ = _student_operating_point(group)
            row.update(operating_point)
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate(predictions, manifest):
    return _evaluate_merged(_merge_inputs(predictions, manifest))


def _set_summary_statistics(row, pooled, metric):
    values = pooled[metric].dropna().astype(float).to_numpy()
    row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
    row[f"{metric}_sample_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan


def summarize(predictions, manifest):
    merged = _merge_inputs(predictions, manifest)
    per_wafer = _evaluate_merged(merged)
    pooled_rows = []
    metrics = [
        "auroc",
        "auprc",
        "budget_candidate_guided_percent",
        "macro_f1",
        "subsurface_defects_total",
        "subsurface_defects_retained",
        "subsurface_recall",
    ]
    for (model, seed), group in merged.groupby(["model", "seed"]):
        operational = group.loc[candidate_mask(group)]
        labels = (operational.z_star == 2).astype(int)
        row = {
            "summary_type": "pooled",
            "model": model,
            "seed": seed,
            "primary": seed == 0,
            "n_seeds": 1,
            "auroc": roc_auc_score(labels, operational.score),
            "auprc": average_precision_score(labels, operational.score),
        }
        if model == "student":
            numerator = 0
            denominator = 0
            total = 0
            retained = 0
            for _, wafer_group in group.groupby("wafer"):
                operating_point, wafer_operational, referred = _student_operating_point(wafer_group)
                numerator += footprint_cost(wafer_operational, referred)
                denominator += footprint_cost(
                    wafer_operational, np.ones(len(wafer_operational), dtype=bool)
                )
                total += int(operating_point["subsurface_defects_total"])
                retained += int(operating_point["subsurface_defects_retained"])
            row["budget_candidate_guided_percent"] = 100.0 * numerator / denominator
            row["subsurface_defects_total"] = total
            row["subsurface_defects_retained"] = retained
            row["subsurface_recall"] = retained / max(total, 1)
            wafer_metrics = per_wafer[(per_wafer.model == model) & (per_wafer.seed == seed)]
            row["macro_f1"] = wafer_metrics.macro_f1.median()
        pooled_rows.append(row)
    pooled = pd.DataFrame(pooled_rows)
    across_rows = []
    for model, group in pooled.groupby("model"):
        row = {
            "summary_type": "across_seed",
            "model": model,
            "seed": np.nan,
            "primary": False,
            "n_seeds": group.seed.nunique(),
        }
        for metric in metrics:
            if metric in group:
                _set_summary_statistics(row, group, metric)
        across_rows.append(row)
    return pd.DataFrame(pooled_rows + across_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path("outputs/lowo/predictions_all_folds.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/lowo/metrics_per_wafer.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("outputs/lowo/metrics_summary.csv"))
    args = parser.parse_args()
    predictions = pd.read_csv(args.predictions)
    manifest = pd.read_csv(args.manifest)
    result = evaluate(predictions, manifest)
    summary = summarize(predictions, manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    summary.to_csv(args.summary_output, index=False)
