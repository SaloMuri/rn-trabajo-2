# src/creditrisk/data/targets.py
"""Variable objetivo de PD (codificación de loan_status) y de LGD."""

import pandas as pd
import numpy as np

# Mapeo oficial exigido en la guía del Trabajo 2 (Parte 1)
PD_STATUS_MAP = {
    # Buenos pagadores confirmados -> 0
    "Fully Paid": 0,
    "Does not meet the credit policy. Status: Fully Paid": 0,
    "Does not meet the credit policy. Status:Fully Paid": 0,    
    # Malos pagadores confirmados -> 1
    "Charged Off": 1,
    "Default": 1,
    "Late (31-120 days)": 1,
    "Does not meet the credit policy. Status: Charged Off": 1,
    "Does not meet the credit policy. Status:Charged Off": 1,
    # Sin certeza de comportamiento final -> NA (se excluyen del entrenamiento)
    "Current": np.nan,
    "Issued": np.nan,
    "In Grace Period": np.nan,
    "Late (16-30 days)": np.nan,
}


def encode_pd_target(loan_status: pd.Series) -> pd.Series:
    """
    Transforma la columna loan_status en la variable binaria objetivo para PD.
    Retorna una serie con valores {0.0, 1.0, NaN}.
    """
    cleaned = loan_status.astype(str).str.strip()
    return cleaned.map(PD_STATUS_MAP) 


def filter_valid_pd_records(df: pd.DataFrame, status_col: str = "loan_status") -> pd.DataFrame:
    """
    Crea la columna 'target_pd', descarta los casos NA (Current, Issued, etc.)
    y retorna una copia limpia con 'target_pd' como entero (0 o 1).
    """
    df_out = df.copy()
    df_out["target_pd"] = encode_pd_target(df_out[status_col])
    df_valid = df_out.dropna(subset=["target_pd"]).copy()
    df_valid["target_pd"] = df_valid["target_pd"].astype(int)
    return df_valid