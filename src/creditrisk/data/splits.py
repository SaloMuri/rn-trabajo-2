# src/creditrisk/data/splits.py
"""Particiones train/val/test, con partición temporal por fecha de originación."""

from typing import Tuple
import pandas as pd


def temporal_train_val_test_split(
    df: pd.DataFrame,
    date_col: str = "issue_d_dt",
    train_end: str = "2014-06-01",
    val_end: str = "2014-12-01",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Realiza una partición estrictamente temporal (Out-of-Time) según la fecha de originación.
    - Train: issue_d_dt <= train_end
    - Val  : train_end < issue_d_dt <= val_end
    - Test : issue_d_dt > val_end
    """
    df_sorted = df.sort_values(by=date_col).reset_index(drop=True)
    dt_series = pd.to_datetime(df_sorted[date_col])

    t_train = pd.to_datetime(train_end)
    t_val = pd.to_datetime(val_end)

    train_mask = dt_series <= t_train
    val_mask = (dt_series > t_train) & (dt_series <= t_val)
    test_mask = dt_series > t_val

    train_df = df_sorted[train_mask].copy()
    val_df = df_sorted[val_mask].copy()
    test_df = df_sorted[test_mask].copy()

    # Seguro automático: si el dataset tiene otro rango de fechas y algún split queda vacío,
    # particionamos temporalmente por cuantiles (70% train, 15% val, 15% test).
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        n = len(df_sorted)
        i_train = int(n * 0.70)
        i_val = int(n * 0.85)
        train_df = df_sorted.iloc[:i_train].copy()
        val_df = df_sorted.iloc[i_train:i_val].copy()
        test_df = df_sorted.iloc[i_val:].copy()

    return train_df, val_df, test_df