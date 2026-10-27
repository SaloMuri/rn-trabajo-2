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
    Partición estrictamente temporal (Out-of-Time) según la fecha de originación.
    - Train: date_col <= train_end
    - Val  : train_end < date_col <= val_end
    - Test : date_col > val_end
    Si alguna partición queda vacía lanza ValueError (no cambia de método en silencio).
    """
    if date_col not in df.columns:
        raise ValueError(f"No existe la columna de fecha '{date_col}'.")

    df_sorted = df.sort_values(by=date_col).reset_index(drop=True)
    dt_series = pd.to_datetime(df_sorted[date_col])
    t_train, t_val = pd.to_datetime(train_end), pd.to_datetime(val_end)
    if t_train >= t_val:
        raise ValueError("train_end debe ser anterior a val_end.")

    train_df = df_sorted[dt_series <= t_train].copy()
    val_df = df_sorted[(dt_series > t_train) & (dt_series <= t_val)].copy()
    test_df = df_sorted[dt_series > t_val].copy()

    for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
        if len(part) == 0:
            raise ValueError(
                f"La partición '{name}' quedó vacía con train_end={train_end} y val_end={val_end}. "
                "Revisa las fechas en configs/base.yaml."
            )
    return train_df, val_df, test_df