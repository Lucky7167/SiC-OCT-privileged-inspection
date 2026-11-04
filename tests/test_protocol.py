import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from common import CFG, make_inner_folds, make_lowo_split
from train_lowo import CheckpointSelector, _validation_metrics, median_epoch


def _frame():
    return pd.DataFrame({
        "wafer": [wafer for wafer in ("W1", "W2", "W3", "W4", "W5") for _ in range(2)],
        "roi_id": [f"roi_{index}" for index in range(10)],
        "z_star": [0, 2] * 5,
        "x": 0,
        "y": 0,
        "w": 1,
        "h": 1,
        "a": 0,
    })


def test_inner_rotation_uses_every_outer_training_wafer_once():
    outer_train, test = make_lowo_split(_frame(), "W5")
    folds = make_inner_folds(outer_train)
    assert set(outer_train.wafer) == {"W1", "W2", "W3", "W4"}
    assert set(test.wafer) == {"W5"}
    assert [validation for validation, _, _ in folds] == ["W1", "W2", "W3", "W4"]
    for validation, train, valid in folds:
        assert set(valid.wafer) == {validation}
        assert len(set(train.wafer)) == 3
        assert not set(train.wafer) & set(valid.wafer)
        assert not set(train.wafer) & set(test.wafer)
        assert not set(valid.wafer) & set(test.wafer)


def test_teacher_checkpoint_minimizes_u_loss_minus_spearman():
    selector = CheckpointSelector("min", 2)
    assert not selector.update(0.5, 1, {"weight": torch.tensor(1.0)})
    assert not selector.update(0.4, 2, {"weight": torch.tensor(2.0)})
    assert not selector.update(0.45, 3, {"weight": torch.tensor(3.0)})
    assert selector.update(0.45, 4, {"weight": torch.tensor(4.0)})
    assert selector.best_epoch == 2
    assert selector.best_state["weight"].item() == 2.0


def test_student_checkpoint_maximizes_spearman():
    selector = CheckpointSelector("max", 2)
    assert not selector.update(0.3, 1, {"weight": torch.tensor(1.0)})
    assert not selector.update(0.8, 2, {"weight": torch.tensor(2.0)})
    assert not selector.update(0.7, 3, {"weight": torch.tensor(3.0)})
    assert selector.update(0.7, 4, {"weight": torch.tensor(4.0)})
    assert selector.best_epoch == 2


def test_validation_selection_matches_protocol():
    scores = torch.tensor([0.1, 0.4, 0.3, 0.9])
    targets = torch.tensor([0.2, 0.5, 0.1, 0.8])
    batch = {"output": {"score": scores}, "U_star": targets}
    forward = lambda model, value: {"score": value["output"]["score"].to(value["U_star"].device)}
    loss = lambda output, value, counts: torch.tensor(0.0)
    teacher = _validation_metrics(torch.nn.Identity(), [batch], loss, None, forward, "teacher")
    student = _validation_metrics(torch.nn.Identity(), [batch], loss, None, forward, "student")
    expected_u_loss = torch.nn.functional.huber_loss(scores, targets, delta=1.0).item()
    expected_rho = spearmanr(scores.numpy(), targets.numpy())[0]
    assert np.isclose(teacher["L_U_val"], expected_u_loss)
    assert np.isclose(teacher["selection"], expected_u_loss - expected_rho)
    assert np.isclose(student["selection"], expected_rho)


def test_final_epoch_is_ceiling_of_inner_median():
    assert median_epoch([1, 2, 3, 4]) == 3
    assert median_epoch([2, 5, 7, 9]) == 6


def test_paper_optimization_defaults_are_centralized():
    assert CFG.seed == 0
    assert CFG.seeds == (0,)
    assert CFG.teacher_learning_rate == 1e-4
    assert CFG.student_learning_rate == 3e-4
    assert CFG.weight_decay == 1e-4
    assert CFG.teacher_epochs == 150
    assert CFG.student_epochs == 100
    assert CFG.teacher_patience == 20
    assert CFG.student_patience == 15


def test_two_seed_protocol_reuses_only_mae(tmp_path, monkeypatch):
    import train_lowo
    from common import fit_normalization

    calls = {"mae": [], "inner": [], "final": [], "cache": []}

    class FakeModel:
        def to(self, device):
            return self

    def fake_mae(frame, output, epochs, batch_size):
        calls["mae"].append((tuple(sorted(frame.wafer.unique())), output))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"mae")

    def fake_targets(train, valid, test):
        train = train.copy()
        valid = valid.copy()
        train["q_star"] = 0.5
        train["U_star"] = 0.5
        valid["q_star"] = 0.5
        valid["U_star"] = 0.5
        return train, valid, test.copy(), fit_normalization(train)

    def fake_final_targets(train, test):
        train = train.copy()
        train["q_star"] = 0.5
        train["U_star"] = 0.5
        return train, test.copy(), fit_normalization(train)

    def fake_fit(*args):
        checkpoint = args[8]
        kind = args[9]
        calls["inner"].append((checkpoint, kind))
        return train_lowo.FitResult(args[0], [{"epoch": 1, "selection": 0.0}], 1)

    def fake_fixed(*args):
        calls["final"].append(args[7])
        return train_lowo.FitResult(args[0], [{"epoch": 1, "train_loss": 0.0}], 1)

    def fake_cache(model, frame, normalization):
        calls["cache"].append(tuple(sorted(frame.wafer.unique())))
        return {}

    def fake_predict(model, frame, normalization, fold, name, seed):
        return pd.DataFrame([{
            "roi_id": frame.iloc[0].roi_id,
            "fold": fold,
            "model": name,
            "seed": seed,
            "score": 0.5,
            "prediction": 0,
            "p_normal": 1.0,
            "p_surface": 0.0,
            "p_subsurface": 0.0,
        }])

    monkeypatch.setattr(train_lowo, "train_mae", fake_mae)
    monkeypatch.setattr(train_lowo, "build_fold_targets", fake_targets)
    monkeypatch.setattr(train_lowo, "build_final_targets", fake_final_targets)
    monkeypatch.setattr(train_lowo, "Teacher", lambda path: FakeModel())
    monkeypatch.setattr(train_lowo, "Student", FakeModel)
    monkeypatch.setattr(train_lowo, "_fit", fake_fit)
    monkeypatch.setattr(train_lowo, "_fit_fixed", fake_fixed)
    monkeypatch.setattr(train_lowo, "_cache_lessons", fake_cache)
    monkeypatch.setattr(train_lowo, "_predict", fake_predict)
    result = train_lowo.run_fold(_frame(), "W5", tmp_path, seeds=(0, 1))
    assert len(calls["mae"]) == 5
    assert len(calls["inner"]) == 32
    assert sum(kind == "teacher" for _, kind in calls["inner"]) == 8
    assert len(calls["final"]) == 8
    assert len(calls["cache"]) == 10
    assert set(result.seed) == {0, 1}
    assert len(result) == 6
    for seed in (0, 1):
        summary = pd.read_json(tmp_path / "fold_W5" / f"seed_{seed}" / "epochs.json", typ="series")
        assert summary["final_epochs"]["teacher"] == 1
        assert (tmp_path / "fold_W5" / f"seed_{seed}" / "teacher_lessons.pt").is_file()
