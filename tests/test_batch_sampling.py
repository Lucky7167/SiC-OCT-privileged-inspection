import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from common import CFG, SubsurfaceSampler, config_metadata, fit_normalization
from train_lowo import _student_loader, _teacher_loader


def _frame():
    return pd.DataFrame({
        "wafer": "W1",
        "roi_id": [f"roi_{index}" for index in range(13)],
        "z_star": [2, 0, 1, 0, 1, 0, 1, 0, 2, 0, 1, 0, 2],
        "x": 0,
        "y": 0,
        "w": 1,
        "h": 1,
        "a": 0,
    })


def test_every_batch_and_incomplete_last_batch_meet_subsurface_fraction():
    frame = _frame()
    sampler = SubsurfaceSampler(frame, batch_size=4)
    batches = list(sampler)
    assert len(batches) == len(sampler)
    assert [len(batch) for batch in batches] == [4, 4, 4, 4, 3]
    for batch in batches:
        positive = sum(frame.iloc[index].z_star == 2 for index in batch)
        assert positive >= int(np.ceil(CFG.subsurface_fraction * len(batch)))
    flattened = [index for batch in batches for index in batch]
    other = [index for index in flattened if frame.iloc[index].z_star != 2]
    positive = [index for index in flattened if frame.iloc[index].z_star == 2]
    assert sorted(other) == sorted(np.flatnonzero(frame.z_star.to_numpy() != 2).tolist())
    assert len(positive) > len(set(positive))


def test_batch_sampling_is_reproducible_from_torch_seed():
    sampler = SubsurfaceSampler(_frame(), batch_size=4)
    torch.manual_seed(19)
    first = list(sampler)
    torch.manual_seed(19)
    second = list(sampler)
    torch.manual_seed(20)
    third = list(sampler)
    assert first == second
    assert first != third


def test_teacher_and_student_training_loaders_use_batch_sampler():
    frame = _frame()
    normalization = fit_normalization(frame)
    teacher = _teacher_loader(frame, normalization, True)
    student = _student_loader(frame, normalization, None, True)
    assert isinstance(teacher.batch_sampler, SubsurfaceSampler)
    assert isinstance(student.batch_sampler, SubsurfaceSampler)
    assert teacher.batch_size is None
    assert student.batch_size is None
    assert teacher.batch_sampler.batch_size == CFG.teacher_batch_size
    assert student.batch_sampler.batch_size == CFG.batch_size


def test_hyperparameters_are_recorded_as_fixed_before_outer_evaluation():
    assert CFG.hyperparameters == "fixed_before_outer_evaluation"
    assert config_metadata(CFG)["hyperparameters"] == "fixed_before_outer_evaluation"
