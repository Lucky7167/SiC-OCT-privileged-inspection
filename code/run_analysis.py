from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ablations import all_ablation_configs, evaluate_ablations, write_configs, write_results
from common import candidate_mask
from referral import matched_count_comparison, multi_otsu_three_regions
from uncertainty import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_MONTE_CARLO_REPLICATES,
    SUBSURFACE_FRACTIONS,
    bootstrap_intervals,
    class_composition_monte_carlo,
    operating_point_metrics,
    summarize_class_composition,
)


def fixed_threshold_analysis(frame, train_frame=None, score_column="score"):
    if "wafer" not in frame.columns:
        raise ValueError("fixed-threshold analysis requires wafer")
    candidate_mask(frame)
    source = frame if train_frame is None else train_frame
    source_candidates = candidate_mask(source)
    rows = []
    for wafer, group in frame.groupby("wafer", sort=True):
        train = source.loc[(source.wafer != wafer).to_numpy() & source_candidates]
        for key in ("model", "seed", "fold"):
            if key in group.columns and key in train.columns and group[key].nunique() == 1:
                train = train[train[key] == group[key].iloc[0]]
        if train.empty:
            raise ValueError(f"no training scores available for wafer {wafer}")
        thresholds = multi_otsu_three_regions(train[score_column].to_numpy(float))
        metrics = operating_point_metrics(group, score_column=score_column, thresholds=thresholds)
        row = {"wafer": wafer, "threshold_low": thresholds[0], "threshold_high": thresholds[1], **metrics}
        for key in ("model", "seed", "fold"):
            if key in group.columns and group[key].nunique() == 1:
                row[key] = group[key].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def score_comparison_analysis(frame, reference_score="student"):
    return matched_count_comparison(frame, reference_score=reference_score)


def ablation_analysis(frame, model_column="model", score_column="score"):
    return evaluate_ablations(frame, model_column=model_column, score_column=score_column)


def bootstrap_analysis(
    frame,
    n_resamples=DEFAULT_BOOTSTRAP_REPLICATES,
    confidence_level=0.95,
    seed=0,
    score_column="score",
):
    return bootstrap_intervals(
        frame,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
        score_column=score_column,
    )


def composition_analysis(
    frame,
    fractions=SUBSURFACE_FRACTIONS,
    n_resamples=DEFAULT_MONTE_CARLO_REPLICATES,
    sample_size=None,
    seed=0,
    score_column="score",
):
    samples = class_composition_monte_carlo(
        frame,
        fractions=fractions,
        n_resamples=n_resamples,
        sample_size=sample_size,
        seed=seed,
        score_column=score_column,
    )
    return samples, summarize_class_composition(samples)


def _input_output(parser):
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-column", default="score")


def build_parser():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    fixed = commands.add_parser("fixed-threshold")
    _input_output(fixed)
    fixed.add_argument("--train-input", type=Path)

    comparison = commands.add_parser("score-comparison")
    _input_output(comparison)
    comparison.add_argument("--reference-score", default="student")

    ablation = commands.add_parser("ablation")
    _input_output(ablation)
    ablation.add_argument("--model-column", default="model")
    ablation.add_argument("--configs-output", type=Path)

    bootstrap = commands.add_parser("bootstrap")
    _input_output(bootstrap)
    bootstrap.add_argument("--n-resamples", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    bootstrap.add_argument("--confidence-level", type=float, default=0.95)
    bootstrap.add_argument("--seed", type=int, default=0)

    composition = commands.add_parser("composition")
    _input_output(composition)
    composition.add_argument("--fractions", type=float, nargs="+", default=SUBSURFACE_FRACTIONS)
    composition.add_argument("--n-resamples", type=int, default=DEFAULT_MONTE_CARLO_REPLICATES)
    composition.add_argument("--sample-size", type=int)
    composition.add_argument("--seed", type=int, default=0)
    composition.add_argument("--samples-output", type=Path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    frame = pd.read_csv(args.input)
    if args.command == "fixed-threshold":
        train = pd.read_csv(args.train_input) if args.train_input else None
        result = fixed_threshold_analysis(frame, train, args.score_column)
    elif args.command == "score-comparison":
        result = score_comparison_analysis(frame, args.reference_score)
    elif args.command == "ablation":
        result = ablation_analysis(frame, args.model_column, args.score_column)
        if args.configs_output:
            write_configs(all_ablation_configs(), args.configs_output)
    elif args.command == "bootstrap":
        result = bootstrap_analysis(
            frame,
            n_resamples=args.n_resamples,
            confidence_level=args.confidence_level,
            seed=args.seed,
            score_column=args.score_column,
        )
    else:
        samples, result = composition_analysis(
            frame,
            fractions=args.fractions,
            n_resamples=args.n_resamples,
            sample_size=args.sample_size,
            seed=args.seed,
            score_column=args.score_column,
        )
        if args.samples_output:
            write_results(samples, args.samples_output)
    write_results(result, args.output)
    return result


if __name__ == "__main__":
    main()
