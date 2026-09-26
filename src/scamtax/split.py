"""Train/test splits: naive row-level vs template-grouped."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def random_split(df: pd.DataFrame, test_size: float, seed: int):
    """Row-level stratified split (leaks near-duplicates across the split)."""
    return train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )


def grouped_split(df: pd.DataFrame, groups: np.ndarray, test_size: float, seed: int):
    """Stratified split where every template cluster lands on one side only."""
    n_splits = max(2, round(1 / test_size))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(df, df["label"], groups))
    return df.iloc[train_idx], df.iloc[test_idx]
