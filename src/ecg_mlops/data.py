from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from ecg_mlops.config import resolve_project_path


N_TIMESTEPS = 187
LABEL_COL = 187
CLASS_NAMES = {
    0: "N - Normal",
    1: "S - Supraventricular",
    2: "V - Ventricular",
    3: "F - Fusion",
    4: "Q - No clasificable",
}


@dataclass(frozen=True)
class ECGArrays:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


class ECGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def download_heartbeat_dataset(data_dir: str | Path, dataset_name: str) -> tuple[Path, Path]:
    data_path = resolve_project_path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    train_csv = data_path / "mitbih_train.csv"
    test_csv = data_path / "mitbih_test.csv"
    if train_csv.exists() and test_csv.exists():
        return train_csv, test_csv

    import kagglehub

    dataset_path = Path(kagglehub.dataset_download(dataset_name))
    source_train = next(dataset_path.rglob("mitbih_train.csv"))
    source_test = next(dataset_path.rglob("mitbih_test.csv"))
    shutil.copy2(source_train, train_csv)
    shutil.copy2(source_test, test_csv)
    return train_csv, test_csv


def load_raw_data(
    data_dir: str | Path,
    dataset_name: str = "shayanfazeli/heartbeat",
    download: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_path = resolve_project_path(data_dir)
    train_csv = data_path / "mitbih_train.csv"
    test_csv = data_path / "mitbih_test.csv"

    if download and (not train_csv.exists() or not test_csv.exists()):
        train_csv, test_csv = download_heartbeat_dataset(data_path, dataset_name)

    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(
            "MIT-BIH CSV files not found. Run training with download enabled or place "
            "mitbih_train.csv and mitbih_test.csv in the configured data directory."
        )

    return pd.read_csv(train_csv, header=None), pd.read_csv(test_csv, header=None)


def validate_raw_dataframe(df: pd.DataFrame) -> None:
    expected_columns = N_TIMESTEPS + 1
    if df.shape[1] != expected_columns:
        raise ValueError(f"Expected {expected_columns} columns, found {df.shape[1]}.")

    if df.isna().any().any():
        raise ValueError("Dataset contains missing values. Download may be incomplete.")

    labels = set(df[LABEL_COL].astype(int).unique())
    expected_labels = set(CLASS_NAMES)
    if labels != expected_labels:
        raise ValueError(
            "Dataset labels do not match MIT-BIH classes. "
            f"Missing={sorted(expected_labels - labels)}, unexpected={sorted(labels - expected_labels)}"
        )

    signal = df.iloc[:, :N_TIMESTEPS]
    if signal.min().min() < 0 or signal.max().max() > 1:
        raise ValueError("Signal values must be normalized between 0 and 1.")


def _stratified_limit(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples is None or max_samples >= len(y):
        return X, y

    n_classes = len(np.unique(y))
    train_size = max(max_samples, n_classes)
    selected_idx, _ = train_test_split(
        np.arange(len(y)),
        train_size=train_size,
        random_state=seed,
        stratify=y,
    )
    return X[selected_idx], y[selected_idx]


def prepare_arrays(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    validation_size: float = 0.15,
    seed: int = 42,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
) -> ECGArrays:
    validate_raw_dataframe(df_train)
    validate_raw_dataframe(df_test)

    X_train_full = df_train.iloc[:, :N_TIMESTEPS].values.astype(np.float32)
    y_train_full = df_train[LABEL_COL].values.astype(np.int64)
    X_test = df_test.iloc[:, :N_TIMESTEPS].values.astype(np.float32)
    y_test = df_test[LABEL_COL].values.astype(np.int64)

    X_train_full, y_train_full = _stratified_limit(
        X_train_full, y_train_full, max_train_samples, seed
    )
    X_test, y_test = _stratified_limit(X_test, y_test, max_test_samples, seed)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=validation_size,
        random_state=seed,
        stratify=y_train_full,
    )

    return ECGArrays(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )
