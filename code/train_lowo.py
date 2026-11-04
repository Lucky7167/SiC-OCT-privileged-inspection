import argparse
import copy
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from common import (
    CFG,
    ROIDataset,
    SubsurfaceSampler,
    load_manifest,
    make_inner_folds,
    make_lowo_split,
    save_fold_metadata,
    save_run_metadata,
    set_seed,
)
from losses import direct_loss, direct_reg_loss, student_loss, teacher_loss
from models import Student, Teacher
from targets import build_final_targets, build_fold_targets
from train_mae import train as train_mae


@dataclass
class FitResult:
    model: torch.nn.Module
    history: list[dict]
    best_epoch: int


class CheckpointSelector:
    def __init__(self, mode: str, patience: int):
        if mode not in {"min", "max"}:
            raise ValueError("checkpoint mode must be min or max")
        self.mode = mode
        self.patience = patience
        self.best = float("inf") if mode == "min" else -float("inf")
        self.best_epoch = None
        self.best_state = None
        self.stale = 0

    def update(self, value: float, epoch: int, state: dict) -> bool:
        if not np.isfinite(value):
            raise FloatingPointError("non-finite checkpoint selection metric")
        improved = value < self.best if self.mode == "min" else value > self.best
        if improved:
            self.best = float(value)
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(state)
            self.stale = 0
        else:
            self.stale += 1
        return self.stale >= self.patience


def median_epoch(values) -> int:
    epochs = list(values)
    if not epochs or any(int(value) < 1 for value in epochs):
        raise ValueError("best epochs must be positive")
    return int(np.ceil(float(np.median(epochs))))


def _counts(frame):
    values = frame.z_star.value_counts().reindex(range(3), fill_value=0).to_numpy()
    return torch.tensor(values, dtype=torch.float32)


def _move(batch):
    return {key: value.to(CFG.device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _teacher_forward(model, batch):
    return model(batch["feat_local"], batch["feat_ctx"], batch["volume"], batch["metadata"])


def _student_forward(model, batch):
    return model(batch["feat_local"], batch["feat_ctx"], batch["metadata"])


def _train_epoch(model, loader, loss_function, class_counts, forward, optimizer, accumulation_steps):
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm3d):
            module.eval()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    if loader.batch_size is None:
        batch_size = loader.batch_sampler.batch_size
        sample_count = loader.batch_sampler.sample_length
    else:
        batch_size = loader.batch_size
        sample_count = len(loader.sampler)
    for batch_index, batch in enumerate(loader):
        batch = _move(batch)
        output = forward(model, batch)
        loss = loss_function(output, batch, class_counts)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        group_start = (batch_index // accumulation_steps) * accumulation_steps
        group_samples = min(
            accumulation_steps * batch_size,
            sample_count - group_start * batch_size,
        )
        (loss * len(batch["z_star"]) / group_samples).backward()
        if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def _validation_metrics(model, loader, loss_function, class_counts, forward, kind):
    model.eval()
    losses = []
    u_losses = []
    scores = []
    targets = []
    for batch in loader:
        batch = _move(batch)
        output = forward(model, batch)
        losses.append(float(loss_function(output, batch, class_counts).cpu()))
        u_losses.append(F.huber_loss(output["score"], batch["U_star"], delta=1.0, reduction="none").cpu())
        scores.append(output["score"].cpu())
        targets.append(batch["U_star"].cpu())
    score_array = torch.cat(scores).numpy()
    target_array = torch.cat(targets).numpy()
    rho = float(spearmanr(score_array, target_array)[0])
    if not np.isfinite(rho):
        raise FloatingPointError("undefined validation Spearman correlation")
    u_loss = float(torch.cat(u_losses).mean())
    selection = u_loss - rho if kind == "teacher" else rho
    return {
        "validation_loss": float(np.mean(losses)),
        "L_U_val": u_loss,
        "rho_sp": rho,
        "selection": float(selection),
    }


def _fit(
    model,
    train_loader,
    valid_loader,
    loss_function,
    class_counts,
    forward,
    learning_rate,
    epochs,
    checkpoint,
    kind,
    patience,
    accumulation_steps=1,
):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=CFG.weight_decay
    )
    selector = CheckpointSelector("min" if kind == "teacher" else "max", patience)
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(
            model, train_loader, loss_function, class_counts, forward, optimizer, accumulation_steps
        )
        metrics = _validation_metrics(
            model, valid_loader, loss_function, class_counts, forward, kind
        )
        row = {"epoch": epoch, "train_loss": train_loss, **metrics}
        history.append(row)
        print(
            f"{checkpoint.parent.name}/{checkpoint.stem} epoch={epoch}/{epochs} "
            f"selection={metrics['selection']:.6f} rho_sp={metrics['rho_sp']:.6f}",
            flush=True,
        )
        if selector.update(metrics["selection"], epoch, model.state_dict()):
            break
    model.load_state_dict(selector.best_state)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(selector.best_state, checkpoint)
    return FitResult(model, history, selector.best_epoch)


def _fit_fixed(
    model,
    train_loader,
    loss_function,
    class_counts,
    forward,
    learning_rate,
    epochs,
    checkpoint,
    accumulation_steps=1,
):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=CFG.weight_decay
    )
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(
            model, train_loader, loss_function, class_counts, forward, optimizer, accumulation_steps
        )
        history.append({"epoch": epoch, "train_loss": train_loss})
        print(f"{checkpoint.parent.name}/{checkpoint.stem} final_epoch={epoch}/{epochs}", flush=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    return FitResult(model, history, epochs)


@torch.no_grad()
def _cache_lessons(model, frame, normalization):
    model.eval()
    lessons = {}
    loader = DataLoader(
        ROIDataset(frame, normalization, load_volume=True), batch_size=CFG.cache_batch_size
    )
    for batch in loader:
        output = _teacher_forward(model, _move(batch))
        for index, roi_id in enumerate(batch["roi_id"]):
            lessons[roi_id] = {
                "teacher_feature": output["feature"][index].cpu(),
                "teacher_logits": output["logits"][index].cpu(),
                "teacher_score": output["score"][index].cpu(),
            }
    return lessons


@torch.no_grad()
def _predict(model, frame, normalization, fold, name, seed):
    model.eval()
    rows = []
    loader = DataLoader(
        ROIDataset(frame, normalization), batch_size=CFG.prediction_batch_size
    )
    for batch in loader:
        output = _student_forward(model, _move(batch))
        probabilities = F.softmax(output["logits"], -1).cpu().numpy()
        scores = output["score"].cpu().numpy()
        for index, roi_id in enumerate(batch["roi_id"]):
            rows.append({
                "roi_id": roi_id,
                "fold": fold,
                "model": name,
                "seed": seed,
                "score": float(scores[index]),
                "prediction": int(probabilities[index].argmax()),
                "p_normal": float(probabilities[index, 0]),
                "p_surface": float(probabilities[index, 1]),
                "p_subsurface": float(probabilities[index, 2]),
            })
    return pd.DataFrame(rows)


def _ensure_mae(frame, path, epochs):
    if not path.exists():
        set_seed(CFG.seed)
        train_mae(frame, path, epochs=epochs, batch_size=CFG.mae_batch_size)
    return path


def _teacher_loader(frame, normalization, training):
    dataset = ROIDataset(frame, normalization, load_volume=True)
    if training:
        return DataLoader(
            dataset,
            batch_sampler=SubsurfaceSampler(
                frame, CFG.teacher_batch_size, CFG.subsurface_fraction
            ),
        )
    return DataLoader(dataset, batch_size=CFG.cache_batch_size)


def _student_loader(frame, normalization, lessons, training):
    dataset = ROIDataset(frame, normalization, lessons=lessons)
    if training:
        return DataLoader(
            dataset,
            batch_sampler=SubsurfaceSampler(
                frame, CFG.batch_size, CFG.subsurface_fraction
            ),
        )
    return DataLoader(dataset, batch_size=CFG.prediction_batch_size)


def _tag_history(history, test_wafer, seed, model_name, phase, validation_wafer=None):
    return [
        {
            "outer_test_wafer": test_wafer,
            "inner_validation_wafer": validation_wafer,
            "seed": seed,
            "model": model_name,
            "phase": phase,
            **row,
        }
        for row in history
    ]


def run_fold(
    manifest,
    test_wafer,
    output_root,
    seeds=(0,),
    mae_epochs=CFG.mae_epochs,
    teacher_epochs=CFG.teacher_epochs,
    student_epochs=CFG.student_epochs,
):
    print(f"Starting fold {test_wafer}; device={CFG.device}", flush=True)
    outer_train, test = make_lowo_split(manifest, test_wafer)
    fold_dir = output_root / f"fold_{test_wafer}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    final_mae_path = _ensure_mae(outer_train, fold_dir / "mae_encoder.pt", mae_epochs)
    set_seed(CFG.seed)
    final_train, final_test, final_normalization = build_final_targets(outer_train, test)
    save_fold_metadata(
        fold_dir / "fold.json", final_train, final_train.iloc[0:0], final_test, final_normalization
    )
    final_train.assign(role="train").to_csv(fold_dir / "targets.csv", index=False)
    test.assign(role="test").to_csv(fold_dir / "test.csv", index=False)
    inner_data = []
    for validation_wafer, raw_train, raw_valid in make_inner_folds(outer_train):
        inner_dir = fold_dir / f"inner_{validation_wafer}"
        mae_path = _ensure_mae(raw_train, inner_dir / "mae_encoder.pt", mae_epochs)
        set_seed(CFG.seed)
        train, valid, _, normalization = build_fold_targets(raw_train, raw_valid, test)
        save_fold_metadata(inner_dir / "fold.json", train, valid, test, normalization)
        pd.concat([train.assign(role="train"), valid.assign(role="validation")]).to_csv(
            inner_dir / "targets.csv", index=False
        )
        inner_data.append({
            "validation_wafer": validation_wafer,
            "directory": inner_dir,
            "mae_path": mae_path,
            "train": train,
            "valid": valid,
            "normalization": normalization,
        })
    variants = {
        "direct": direct_loss,
        "direct_reg": direct_reg_loss,
        "student": student_loss,
    }
    fold_predictions = []
    for seed in seeds:
        best_epochs = {name: [] for name in ("teacher", *variants)}
        inner_history = []
        for data in inner_data:
            validation_wafer = data["validation_wafer"]
            seed_dir = data["directory"] / f"seed_{seed}"
            train = data["train"]
            valid = data["valid"]
            normalization = data["normalization"]
            counts = _counts(train)
            set_seed(seed)
            teacher = Teacher(data["mae_path"]).to(CFG.device)
            teacher_result = _fit(
                teacher,
                _teacher_loader(train, normalization, True),
                _teacher_loader(valid, normalization, False),
                teacher_loss,
                counts,
                _teacher_forward,
                CFG.teacher_learning_rate,
                teacher_epochs,
                seed_dir / "teacher.pt",
                "teacher",
                CFG.teacher_patience,
                CFG.teacher_accumulation_steps,
            )
            best_epochs["teacher"].append(teacher_result.best_epoch)
            inner_history.extend(_tag_history(
                teacher_result.history, test_wafer, seed, "teacher", "inner", validation_wafer
            ))
            lessons = _cache_lessons(
                teacher_result.model, pd.concat([train, valid], ignore_index=True), normalization
            )
            seed_dir.mkdir(parents=True, exist_ok=True)
            torch.save(lessons, seed_dir / "teacher_lessons.pt")
            del teacher, teacher_result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for name, loss_function in variants.items():
                set_seed(seed)
                model = Student().to(CFG.device)
                use_lessons = lessons if name == "student" else None
                result = _fit(
                    model,
                    _student_loader(train, normalization, use_lessons, True),
                    _student_loader(valid, normalization, use_lessons, False),
                    loss_function,
                    counts,
                    _student_forward,
                    CFG.student_learning_rate,
                    student_epochs,
                    seed_dir / f"{name}.pt",
                    "student",
                    CFG.student_patience,
                )
                best_epochs[name].append(result.best_epoch)
                inner_history.extend(_tag_history(
                    result.history, test_wafer, seed, name, "inner", validation_wafer
                ))
        median_epochs = {name: float(np.median(values)) for name, values in best_epochs.items()}
        final_epochs = {name: median_epoch(values) for name, values in best_epochs.items()}
        validation_wafers = [data["validation_wafer"] for data in inner_data]
        selected_inner = [
            row for row in inner_history
            if row["epoch"] == best_epochs[row["model"]][
                validation_wafers.index(row["inner_validation_wafer"])
            ]
        ]
        mean_inner = {}
        for name in best_epochs:
            rows = [row for row in selected_inner if row["model"] == name]
            metrics = ("validation_loss", "L_U_val", "rho_sp", "selection")
            mean_inner[name] = {
                metric: float(np.mean([row[metric] for row in rows]))
                for metric in metrics if metric in rows[0]
            }
        seed_dir = fold_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(inner_history).to_csv(seed_dir / "inner_metrics.csv", index=False)
        epoch_summary = {
            "outer_test_wafer": test_wafer,
            "seed": seed,
            "hyperparameters": CFG.hyperparameters,
            "inner_best_epochs": best_epochs,
            "mean_inner_validation_metrics": mean_inner,
            "median_epochs": median_epochs,
            "final_epochs": final_epochs,
            "final_selection": {"epochs": final_epochs, "rule": "ceil_median_inner_best_epoch"},
            "median_rounding": "ceil",
        }
        (seed_dir / "epochs.json").write_text(json.dumps(epoch_summary, indent=2), encoding="utf-8")
        final_history = []
        counts = _counts(final_train)
        set_seed(seed)
        teacher = Teacher(final_mae_path).to(CFG.device)
        teacher_result = _fit_fixed(
            teacher,
            _teacher_loader(final_train, final_normalization, True),
            teacher_loss,
            counts,
            _teacher_forward,
            CFG.teacher_learning_rate,
            final_epochs["teacher"],
            seed_dir / "teacher.pt",
            CFG.teacher_accumulation_steps,
        )
        final_history.extend(_tag_history(
            teacher_result.history, test_wafer, seed, "teacher", "final"
        ))
        lessons = _cache_lessons(teacher_result.model, final_train, final_normalization)
        torch.save(lessons, seed_dir / "teacher_lessons.pt")
        del teacher, teacher_result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        predictions = []
        for name, loss_function in variants.items():
            set_seed(seed)
            model = Student().to(CFG.device)
            use_lessons = lessons if name == "student" else None
            result = _fit_fixed(
                model,
                _student_loader(final_train, final_normalization, use_lessons, True),
                loss_function,
                counts,
                _student_forward,
                CFG.student_learning_rate,
                final_epochs[name],
                seed_dir / f"{name}.pt",
            )
            final_history.extend(_tag_history(
                result.history, test_wafer, seed, name, "final"
            ))
            predictions.append(_predict(
                result.model, test, final_normalization, test_wafer, name, seed
            ))
        pd.DataFrame(final_history).to_csv(seed_dir / "final_metrics.csv", index=False)
        result = pd.concat(predictions, ignore_index=True).merge(
            test[["roi_id", "wafer", "z_star"]], on="roi_id", validate="many_to_one"
        )
        result.to_csv(seed_dir / "predictions.csv", index=False)
        fold_predictions.append(result)
    return pd.concat(fold_predictions, ignore_index=True)


def _validate_seeds(seeds):
    if not seeds or len(set(seeds)) != len(seeds) or any(seed not in range(5) for seed in seeds):
        raise ValueError("seeds must be unique integers from 0 through 4")
    return tuple(seeds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=CFG.output_root)
    parser.add_argument("--test-wafer", choices=["W1", "W2", "W3", "W4", "W5"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(CFG.seeds))
    parser.add_argument("--mae-epochs", type=int, default=CFG.mae_epochs)
    parser.add_argument("--teacher-epochs", type=int, default=CFG.teacher_epochs)
    parser.add_argument("--student-epochs", type=int, default=CFG.student_epochs)
    args = parser.parse_args()
    try:
        seeds = _validate_seeds(args.seeds)
        if min(args.mae_epochs, args.teacher_epochs, args.student_epochs) < 1:
            raise ValueError("epoch limits must be positive")
    except ValueError as error:
        parser.error(str(error))
    config = replace(
        CFG,
        output_root=args.output,
        seeds=seeds,
        seed=seeds[0],
        mae_epochs=args.mae_epochs,
        teacher_epochs=args.teacher_epochs,
        student_epochs=args.student_epochs,
    )
    save_run_metadata(args.output, config)
    manifest = load_manifest()
    wafers = sorted(map(str, manifest.wafer.unique()))
    selected = [args.test_wafer] if args.test_wafer else wafers
    all_predictions = [
        run_fold(
            manifest,
            test,
            args.output,
            seeds,
            args.mae_epochs,
            args.teacher_epochs,
            args.student_epochs,
        )
        for test in selected
    ]
    pd.concat(all_predictions, ignore_index=True).to_csv(
        args.output / "predictions_all_folds.csv", index=False
    )


if __name__ == "__main__":
    main()
