
from __future__ import annotations

import json
import platform
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler


@dataclass(frozen=True)
class Config:
    data_root: Path = Path(__file__).resolve().parents[1] / "data"
    output_root: Path = Path("outputs/lowo")
    dino_dim: int = 384
    metadata_dim: int = 5
    feature_dim: int = 256
    batch_size: int = 32
    seed: int = 0
    seeds: tuple[int, ...] = (0,)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_threads: int = 4
    mae_epochs: int = 40
    teacher_epochs: int = 150
    student_epochs: int = 100
    teacher_patience: int = 20
    student_patience: int = 15
    mae_learning_rate: float = 3e-5
    teacher_learning_rate: float = 1e-4
    student_learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    mae_batch_size: int = 8
    teacher_batch_size: int = 4
    teacher_accumulation_steps: int = 8
    cache_batch_size: int = 8
    prediction_batch_size: int = 64
    subsurface_fraction: float = 0.30
    hyperparameters: str = "fixed_before_outer_evaluation"


def load_config(path: Path | None = None) -> Config:
    config_path = path or Path(__file__).resolve().with_name("config.json")
    values = json.loads(config_path.read_text(encoding="utf-8"))
    for key in ("data_root", "output_root"):
        if key in values:
            value = Path(values[key])
            values[key] = value if value.is_absolute() else (config_path.parent / value).resolve()
    if "seeds" in values:
        values["seeds"] = tuple(values["seeds"])
    config = Config(**values)
    if config.device == "auto":
        config = Config(**{**asdict(config), "device": "cuda" if torch.cuda.is_available() else "cpu"})
    return config


def config_metadata(config: Config) -> dict:
    values = asdict(config)
    values["data_root"] = str(values["data_root"])
    values["output_root"] = str(values["output_root"])
    values["seeds"] = list(values["seeds"])
    return values


def environment_metadata() -> dict:
    import sklearn
    import torchvision
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpus": gpus,
        "platform": platform.platform(),
    }


def save_run_metadata(output: Path, config: Config) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config_metadata(config), indent=2), encoding="utf-8")
    (output / "environment.json").write_text(json.dumps(environment_metadata(), indent=2), encoding="utf-8")


CFG = load_config()
META_COLUMNS = ("x", "y", "w", "h", "a")
CLASS_NAMES = ("normal", "surface", "subsurface")


def set_seed(seed: int = CFG.seed) -> None:
    torch.set_num_threads(CFG.num_threads)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def candidate_mask(frame: pd.DataFrame) -> np.ndarray:
    if "is_candidate" not in frame.columns:
        raise ValueError("candidate selection requires is_candidate")
    values = frame["is_candidate"]
    missing = values.isna() | values.map(lambda value: isinstance(value, str) and not value.strip())
    if missing.any():
        raise ValueError("is_candidate cannot contain missing values")
    result = np.empty(len(frame), dtype=bool)
    for index, value in enumerate(values):
        if isinstance(value, (bool, np.bool_)):
            result[index] = bool(value)
        elif isinstance(value, (int, np.integer)) and value in (0, 1):
            result[index] = bool(value)
        elif isinstance(value, (float, np.floating)) and np.isfinite(value) and value in (0.0, 1.0):
            result[index] = bool(value)
        elif isinstance(value, str) and value.strip().lower() in {"true", "false", "1", "0"}:
            result[index] = value.strip().lower() in {"true", "1"}
        else:
            raise ValueError("is_candidate must contain only Boolean or 0/1 values")
    return result


def load_manifest(path: Path | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path or CFG.data_root / "manifest.csv")
    required = {"roi_id", "wafer", "z_star", *META_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    if frame.roi_id.duplicated().any():
        raise ValueError("roi_id must be unique")
    return frame


def make_outer_fold(frame: pd.DataFrame, test_wafer: str, validation_wafer: str):
    wafers = set(frame.wafer.astype(str).unique())
    if test_wafer == validation_wafer:
        raise ValueError("test and validation wafers must differ")
    if {test_wafer, validation_wafer} - wafers:
        raise ValueError("unknown test or validation wafer")
    train_wafers = wafers - {test_wafer, validation_wafer}
    train = frame[frame.wafer.astype(str).isin(train_wafers)].copy()
    valid = frame[frame.wafer.astype(str) == validation_wafer].copy()
    test = frame[frame.wafer.astype(str) == test_wafer].copy()
    assert test_wafer not in set(train.wafer.astype(str))
    assert test_wafer not in set(valid.wafer.astype(str))
    return tuple(x.reset_index(drop=True) for x in (train, valid, test))


def make_lowo_split(frame: pd.DataFrame, test_wafer: str):
    wafers = set(frame.wafer.astype(str).unique())
    if test_wafer not in wafers:
        raise ValueError("unknown test wafer")
    train = frame[frame.wafer.astype(str) != test_wafer].copy()
    test = frame[frame.wafer.astype(str) == test_wafer].copy()
    if set(train.wafer.astype(str)) & set(test.wafer.astype(str)):
        raise RuntimeError("outer train and test wafers overlap")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def make_inner_folds(outer_train: pd.DataFrame):
    wafers = sorted(map(str, outer_train.wafer.unique()))
    if len(wafers) != 4:
        raise ValueError("inner validation requires exactly four outer-training wafers")
    folds = []
    for validation_wafer in wafers:
        train = outer_train[outer_train.wafer.astype(str) != validation_wafer].copy()
        valid = outer_train[outer_train.wafer.astype(str) == validation_wafer].copy()
        if set(train.wafer.astype(str)) & {validation_wafer}:
            raise RuntimeError("inner train and validation wafers overlap")
        folds.append((validation_wafer, train.reset_index(drop=True), valid.reset_index(drop=True)))
    return folds


def validation_rotation(wafers: Iterable[str]) -> dict[str, str]:
    ordered = sorted(map(str, wafers))
    if len(ordered) < 3:
        raise ValueError("at least three wafers are required")
    return {w: ordered[(i + 1) % len(ordered)] for i, w in enumerate(ordered)}


def fit_normalization(train: pd.DataFrame) -> dict[str, dict[str, float]]:
    result = {}
    for column in META_COLUMNS:
        std = float(train[column].std(ddof=1))
        result[column] = {
            "mean": float(train[column].mean()),
            "std": std if std > 0 else 1.0,
        }
    return result


def save_fold_metadata(path: Path, train, valid, test, normalization) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_wafers": sorted(map(str, train.wafer.unique())),
        "validation_wafers": sorted(map(str, valid.wafer.unique())),
        "test_wafers": sorted(map(str, test.wafer.unique())),
        "normalization": normalization,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def compute_q_star(frame: pd.DataFrame) -> np.ndarray:
    required = {
        "deff", "dmax", "snr", "snr_ref", "rho_shadow", "rho_sat",
        "c_cont", "has_depth_structure",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"cannot compute q_star; missing {sorted(missing)}")
    present = frame.has_depth_structure.astype(bool).to_numpy()
    depth = np.divide(
        frame.deff, frame.dmax, out=np.zeros(len(frame)), where=frame.dmax != 0
    )
    snr = np.divide(
        frame.snr, frame.snr_ref, out=np.zeros(len(frame)), where=frame.snr_ref != 0
    )
    depth = np.where(present, depth, 0.0)
    snr = np.where(present, np.clip(snr, 0.0, 1.0), 0.0)
    continuity = np.where(present, frame.c_cont.to_numpy(float), 0.0)
    q = (
        depth + snr + (1.0 - frame.rho_shadow.to_numpy(float))
        + (1.0 - frame.rho_sat.to_numpy(float)) + continuity
    ) / 5.0
    return np.clip(q, 0.0, 1.0)


def compute_u_star(q_star, risk, alpha: float = 0.7) -> np.ndarray:
    q_star, risk = np.asarray(q_star, float), np.asarray(risk, float)
    if q_star.shape != risk.shape:
        raise ValueError("q_star and risk must have the same shape")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    return q_star * (alpha + (1.0 - alpha) * risk)


class ROIDataset(Dataset):
    def __init__(self, frame, normalization, load_volume=False, lessons=None):
        self.frame = frame.reset_index(drop=True)
        self.normalization = normalization
        self.load_volume = load_volume
        self.lessons = lessons

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        root = CFG.data_root
        metadata = [
            (float(row[c]) - self.normalization[c]["mean"])
            / self.normalization[c]["std"] for c in META_COLUMNS
        ]
        item = {
            "roi_id": str(row.roi_id),
            "feat_local": torch.as_tensor(
                np.load(root / "patches" / f"{row.roi_id}_local.npy"), dtype=torch.float32
            ),
            "feat_ctx": torch.as_tensor(
                np.load(root / "patches" / f"{row.roi_id}_ctx.npy"), dtype=torch.float32
            ),
            "metadata": torch.tensor(metadata, dtype=torch.float32),
            "z_star": torch.tensor(int(row.z_star), dtype=torch.long),
            "q_star": torch.tensor(float(row.get("q_star", np.nan)), dtype=torch.float32),
            "U_star": torch.tensor(float(row.get("U_star", np.nan)), dtype=torch.float32),
        }
        if self.load_volume:
            item["volume"] = torch.as_tensor(
                np.load(root / "volumes" / f"{row.roi_id}.npy"), dtype=torch.float32
            )
        if self.lessons is not None:
            item.update(self.lessons[str(row.roi_id)])
        return item


class SubsurfaceSampler(Sampler[list[int]]):
    def __init__(self, frame: pd.DataFrame, batch_size: int, fraction: float = CFG.subsurface_fraction):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 0 <= fraction < 1:
            raise ValueError("subsurface fraction must be in [0, 1)")
        self.positive = np.flatnonzero(frame.z_star.to_numpy() == 2)
        self.other = np.flatnonzero(frame.z_star.to_numpy() != 2)
        if len(self.positive) == 0:
            raise ValueError("training fold has no subsurface examples")
        self.fraction = fraction
        self.batch_size = batch_size
        self.sample_length = max(len(frame), int(np.ceil(len(self.other) / (1 - fraction))))
        while True:
            batch_count = int(np.ceil(self.sample_length / batch_size))
            last_size = self.sample_length - (batch_count - 1) * batch_size
            positive_count = (
                (batch_count - 1) * int(np.ceil(fraction * batch_size))
                + int(np.ceil(fraction * last_size))
            )
            missing = len(self.other) - (self.sample_length - positive_count)
            if missing <= 0:
                break
            self.sample_length += missing
        self.length = self.sample_length

    def __iter__(self):
        rng = np.random.default_rng(torch.initial_seed() % (2**32))
        other = rng.permutation(self.other)
        offset = 0
        for start in range(0, self.sample_length, self.batch_size):
            size = min(self.batch_size, self.sample_length - start)
            positive_count = int(np.ceil(self.fraction * size))
            other_count = size - positive_count
            selected_other = other[offset:offset + other_count]
            offset += len(selected_other)
            positive_count += other_count - len(selected_other)
            selected_positive = rng.choice(self.positive, positive_count, replace=True)
            batch = np.concatenate([selected_other, selected_positive])
            rng.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        return int(np.ceil(self.sample_length / self.batch_size))


def weighted_cross_entropy(logits, target, counts):
    weights = 1.0 / counts.float().clamp_min(1.0)
    weights = weights / weights.sum() * len(counts)
    return F.cross_entropy(logits, target, weight=weights.to(logits.device))


def pairwise_rank_loss(scores, targets):
    predicted = scores[:, None] - scores[None, :]
    expected = targets[:, None] - targets[None, :]
    mask = expected > 0
    return -F.logsigmoid(predicted[mask]).mean() if mask.any() else scores.sum() * 0.0


def feature_alignment_loss(projected_student, teacher_feature):
    return 1.0 - F.cosine_similarity(projected_student, teacher_feature, dim=-1).mean()


def logit_distillation_loss(student, teacher, temperature=3.0):
    log_p = F.log_softmax(student / temperature, dim=-1)
    p = F.softmax(teacher / temperature, dim=-1)
    return temperature**2 * F.kl_div(log_p, p, reduction="batchmean")
