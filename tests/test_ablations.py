import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from ablations import (
    LOSS_REGISTRY,
    all_ablation_configs,
    configs_to_frame,
    evaluate_ablations,
    hyperparameter_sensitivity_configs,
    incremental_distillation_configs,
    leave_one_loss_configs,
    metadata_ablation_configs,
)
from common import logit_distillation_loss, pairwise_rank_loss


def test_incremental_distillation_is_cumulative():
    configs = incremental_distillation_configs()
    assert [config["name"] for config in configs] == ["Direct", "+logit", "+feature", "+score"]
    assert configs[0]["loss_terms"] == ("ce", "rank")
    assert configs[1]["loss_terms"] == ("ce", "rank", "logit")
    assert configs[2]["loss_terms"] == ("ce", "rank", "logit", "feature")
    assert configs[3]["loss_terms"] == ("ce", "rank", "feature", "logit", "score")
    assert configs[-1]["loss_weights"]["logit"] == 0.5


def test_leave_one_and_hyperparameter_configs_vary_only_requested_fields():
    configs = leave_one_loss_configs()
    assert len(configs) == 6
    full = set(configs[0]["loss_terms"])
    for config in configs[1:]:
        assert len(full - set(config["loss_terms"])) == 1
        assert len(config["loss_terms"]) == len(full) - 1
    sensitivity = hyperparameter_sensitivity_configs()
    assert [config["alpha"] for config in sensitivity[:3]] == [0.5, 0.7, 0.9]
    assert [config["tau"] for config in sensitivity[3:]] == [1.0, 3.0, 5.0]
    assert all(config["loss_terms"] == configs[0]["loss_terms"] for config in sensitivity)


def test_metadata_configs_describe_input_removals():
    configs = metadata_ablation_configs()
    assert configs[0]["metadata_columns"] == ("x", "y", "w", "h", "a")
    assert configs[1]["metadata_columns"] == ("w", "h", "a")
    assert configs[2]["metadata_columns"] == ("x", "y")
    assert configs[3]["metadata_columns"] == ("x", "y", "w", "h", "a")
    assert configs[3]["use_microscopy"] is False
    table = configs_to_frame(all_ablation_configs())
    assert len(table) == 20
    assert table.name.is_unique


def test_callable_registry_applies_incremental_logit_loss():
    assert set(LOSS_REGISTRY) == {"Direct", "+logit", "+feature", "+score"}
    output = {
        "logits": torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        "score": torch.tensor([0.2, 0.8]),
    }
    batch = {
        "z_star": torch.tensor([0, 1]),
        "U_star": torch.tensor([0.1, 0.9]),
        "teacher_logits": torch.tensor([[1.5, 0.2, 0.1], [0.1, 1.2, 0.3]]),
    }
    expected = (
        F.cross_entropy(output["logits"], batch["z_star"])
        + pairwise_rank_loss(output["score"], batch["U_star"])
        + 0.5 * logit_distillation_loss(output["logits"], batch["teacher_logits"], 3.0)
    )
    actual = LOSS_REGISTRY["+logit"](output, batch, torch.tensor([1, 1, 0]))
    assert torch.allclose(actual, expected)


def test_ablation_results_include_ranking_budget_and_operating_metrics():
    frames = []
    for model in ("Direct", "+score"):
        for wafer in ("W1", "W2"):
            frames.append(pd.DataFrame({
                "model": model,
                "wafer": wafer,
                "z_star": [2, 1, 2, 1],
                "is_candidate": True,
                "prediction": [1, 1, 1, 1],
                "defect_id": ["a", "s1", "b", "s2"],
                "score": [0.1, 0.4, 0.6, 0.9],
                "x0_aline": [0, 1, 2, 3],
                "x1_aline": [1, 2, 3, 4],
                "y0_aline": 0,
                "y1_aline": 1,
            }))
    result = evaluate_ablations(pd.concat(frames, ignore_index=True))
    assert len(result) == 4
    assert np.allclose(result.auroc, 0.25)
    assert np.allclose(result.budget_percent, 100.0)
    assert np.allclose(result.operating_budget_percent, 50.0)
    assert set(result.model) == {"Direct", "+score"}


def test_ablation_metrics_exclude_noncandidate_positive_rows():
    frame = pd.DataFrame({
        "model": "Direct",
        "wafer": "W1",
        "z_star": [2, 1, 2, 1],
        "is_candidate": [True, True, False, True],
        "prediction": [1, 1, 1, 1],
        "defect_id": ["a", "s1", "excluded", "s2"],
        "score": [0.1, 0.4, 0.6, 0.9],
        "x0_aline": [0, 1, 2, 3],
        "x1_aline": [1, 2, 3, 4],
        "y0_aline": 0,
        "y1_aline": 1,
    })
    result = evaluate_ablations(frame)
    row = result.iloc[0]
    assert np.isclose(row.auroc, 0.0)
    assert np.isclose(row.budget_percent, 100.0)
    assert row.complete_recall_count == 3
    assert np.isclose(row.operating_budget_percent, 200.0 / 3.0)
