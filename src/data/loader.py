"""Data loading and preprocessing for both datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

WINE_PATH = DATA_DIR / "winequality-red.csv"
CSGO_PATH = DATA_DIR / "csgo_task.csv"

WINE_TARGET = "quality"
CSGO_TARGET = "bomb_planted"


# ─── Wine ─────────────────────────────────────────────────────────────────────

def load_wine_raw() -> pd.DataFrame:
    return pd.read_csv(WINE_PATH, sep=";")


def preprocess_wine(df: pd.DataFrame) -> pd.DataFrame:
    """Clean wine dataset.

    Steps:
    - Remove duplicate rows
    - Drop rows where IQR-outliers exist in chlorides / residual sugar
      (extreme physical impossibilities, <1 % of data)
    - No nulls in this dataset
    """
    df = df.drop_duplicates().copy()

    # Cap extreme outliers in two noisiest columns (keep realistic wines)
    for col in ("chlorides", "residual sugar"):
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        df = df[(df[col] >= q1 - 3 * iqr) & (df[col] <= q3 + 3 * iqr)]

    df = df.reset_index(drop=True)
    return df


def get_wine_Xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[WINE_TARGET])
    y = df[WINE_TARGET]
    return X, y


# ─── CS:GO ────────────────────────────────────────────────────────────────────

def load_csgo_raw() -> pd.DataFrame:
    return pd.read_csv(CSGO_PATH)


def preprocess_csgo(df: pd.DataFrame) -> pd.DataFrame:
    """Clean CS:GO dataset.

    Steps:
    - Drop rows with any nulls (4321 / 122410 = 3.5 %, acceptable)
    - Encode 'map' column with LabelEncoder
    - Cast target bool -> int
    - Remove duplicate rows
    """
    df = df.dropna().copy()
    df = df.drop_duplicates()

    le = LabelEncoder()
    df["map"] = le.fit_transform(df["map"])

    df[CSGO_TARGET] = df[CSGO_TARGET].astype(int)
    df = df.reset_index(drop=True)
    return df


def get_csgo_Xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[CSGO_TARGET])
    y = df[CSGO_TARGET]
    return X, y
