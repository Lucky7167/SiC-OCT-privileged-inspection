import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from common import CFG, ROIDataset, compute_q_star, compute_u_star, fit_normalization
from models import MicroscopyClassifier

def _counts(frame):
    values = frame.z_star.value_counts().reindex(range(3), fill_value=0).to_numpy()
    return torch.tensor(values, dtype=torch.float32)


def _fit_classifier(train, valid, normalization, epochs=100):
    model = MicroscopyClassifier().to(CFG.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    train_loader = DataLoader(ROIDataset(train, normalization), batch_size=32, shuffle=True)
    valid_loader = DataLoader(ROIDataset(valid, normalization), batch_size=64)
    best, state, stale = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            output = model(batch["feat_local"].to(CFG.device), batch["feat_ctx"].to(CFG.device), batch["metadata"].to(CFG.device))
            loss = torch.nn.functional.cross_entropy(output["logits"], batch["z_star"].to(CFG.device))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval(); losses = []
        with torch.no_grad():
            for batch in valid_loader:
                output = model(batch["feat_local"].to(CFG.device), batch["feat_ctx"].to(CFG.device), batch["metadata"].to(CFG.device))
                losses.append(torch.nn.functional.cross_entropy(output["logits"], batch["z_star"].to(CFG.device)).item())
        value = float(np.mean(losses))
        if value < best:
            best, state, stale = value, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= 15:
                break
    model.load_state_dict(state)
    return model


@torch.no_grad()
def _nll(model, frame, normalization):
    rows = {}
    for batch in DataLoader(ROIDataset(frame, normalization), batch_size=64):
        output = model(batch["feat_local"].to(CFG.device), batch["feat_ctx"].to(CFG.device), batch["metadata"].to(CFG.device))
        labels = batch["z_star"].to(CFG.device)
        loss = -torch.log_softmax(output["logits"], -1).gather(1, labels[:, None]).squeeze(1)
        rows.update(zip(batch["roi_id"], loss.cpu().numpy()))
    return rows


def _cross_fitted_nll(frame):
    risk = {}
    wafers = sorted(map(str, frame.wafer.unique()))
    if len(wafers) < 2:
        raise ValueError("risk cross-fitting requires at least two training wafers")
    for held in wafers:
        inner_valid = frame[frame.wafer.astype(str) == held]
        inner_train = frame[frame.wafer.astype(str) != held]
        inner_norm = fit_normalization(inner_train)
        risk.update(_nll(_fit_classifier(inner_train, inner_valid, inner_norm), inner_valid, inner_norm))
    raw = np.array([risk[x] for x in frame.roi_id])
    lower, upper = raw.min(), raw.max()
    return risk, lower, max(upper - lower, np.finfo(float).eps)


def _set_training_targets(frame, risk, lower, scale):
    raw = np.array([risk[x] for x in frame.roi_id])
    frame["R"] = np.clip((raw - lower) / scale, 0, 1)
    frame["q_star"] = compute_q_star(frame)
    frame["U_star"] = compute_u_star(frame.q_star, frame.R, 0.7)


def build_final_targets(train, test):
    train, test = train.copy(), test.copy()
    risk, lower, scale = _cross_fitted_nll(train)
    _set_training_targets(train, risk, lower, scale)
    return train, test, fit_normalization(train)


def build_fold_targets(outer_train, validation, test):
    outer_train, validation, test = (x.copy() for x in (outer_train, validation, test))
    risk, lower, scale = _cross_fitted_nll(outer_train)
    normalization = fit_normalization(outer_train)

    final_model = _fit_classifier(outer_train, validation, normalization)
    risk.update(_nll(final_model, validation, normalization))
    _set_training_targets(outer_train, risk, lower, scale)
    _set_training_targets(validation, risk, lower, scale)
    return outer_train, validation, test, normalization
