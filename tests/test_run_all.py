import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import run_all


def test_run_all_forwards_seed_list_and_summary_output(tmp_path, monkeypatch):
    commands = []
    output = tmp_path / "run"
    summary = tmp_path / "summary.csv"
    monkeypatch.setattr(sys, "argv", [
        "run_all.py",
        "--train",
        "--output",
        str(output),
        "--seeds",
        "0",
        "2",
        "4",
        "--summary-output",
        str(summary),
    ])
    monkeypatch.setattr(run_all.subprocess, "run", lambda command, **kwargs: commands.append(command))
    run_all.main()
    assert len(commands) == 3
    train_command = commands[1]
    seed_position = train_command.index("--seeds")
    assert train_command[seed_position + 1:seed_position + 4] == ["0", "2", "4"]
    evaluate_command = commands[2]
    assert evaluate_command[evaluate_command.index("--summary-output") + 1] == str(summary)
    assert evaluate_command[evaluate_command.index("--predictions") + 1] == str(output / "predictions_all_folds.csv")
