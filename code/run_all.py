import argparse
import subprocess
import sys
from pathlib import Path


def _seeds(values):
    if not values or len(set(values)) != len(values) or any(value not in range(5) for value in values):
        raise argparse.ArgumentTypeError("seeds must be unique integers from 0 through 4")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--quick", action="store_true", help="One-epoch W1 smoke check, not paper reproduction")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()
    try:
        args.seeds = _seeds(args.seeds)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    root = Path(__file__).resolve().parents[1]
    if not args.train and args.predictions is None:
        parser.error("Specify --train or an explicit --predictions file; old outputs are never selected automatically")
    if not args.train and not args.predictions.is_file():
        parser.error("The explicitly selected predictions file does not exist")
    if args.train and args.output.exists() and any(args.output.iterdir()):
        parser.error("Training output must be empty to avoid mixing runs")
    subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=root, check=True)
    if args.train:
        command = [
            sys.executable,
            str(root / "code/train_lowo.py"),
            "--output",
            str(args.output),
            "--seeds",
            *map(str, args.seeds),
        ]
        if args.quick:
            command.extend([
                "--test-wafer",
                "W1",
                "--mae-epochs",
                "1",
                "--teacher-epochs",
                "1",
                "--student-epochs",
                "1",
            ])
        subprocess.run(command, cwd=root, check=True)
        predictions = args.output / "predictions_all_folds.csv"
    else:
        predictions = args.predictions
    summary_output = args.summary_output or args.output / "metrics_summary.csv"
    subprocess.run([
        sys.executable,
        str(root / "code/evaluate.py"),
        "--predictions",
        str(predictions),
        "--manifest",
        str(root / "data/manifest.csv"),
        "--output",
        str(args.output / "metrics_per_wafer.csv"),
        "--summary-output",
        str(summary_output),
    ], cwd=root, check=True)


if __name__ == "__main__":
    main()
