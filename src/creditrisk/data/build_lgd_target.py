"""Construye la variable objetivo LGD (Parte 2).

Uso (desde la raíz del repositorio):
    python src/creditrisk/data/build_lgd_target.py
    python src/creditrisk/data/build_lgd_target.py --sample 50000   # demo rápida
    python src/creditrisk/data/build_lgd_target.py --include-default  # análisis de sensibilidad
"""
import argparse
from pathlib import Path

import pandas as pd

CHARGED_OFF = [
    "Charged Off",
    "Does not meet the credit policy. Status:Charged Off",
]
# Incumplimiento sin castigo contable: el cobro aún no empieza (recoveries == 0),
# por lo que su LGD es 1 por construcción (dato censurado). Solo para sensibilidad.
DEFAULT_STATUS = ["Default"]
OUT_MAIN = "data/processed/lgd/lgd_target.parquet"
OUT_WITH_DEFAULT = "data/processed/lgd/lgd_target_with_default.parquet"
COLS = [
    "id",
    "loan_status",
    "issue_d",
    "term",
    "funded_amnt",
    "total_rec_prncp",
    "recoveries",
    "collection_recovery_fee",
]


def build_lgd_target(df: pd.DataFrame, include_default: bool = False) -> pd.DataFrame:
    """Recibe el loan.csv (con COLS) y devuelve un DataFrame con la LGD.

    include_default=False (versión principal): solo créditos castigados (Charged Off).
    include_default=True (sensibilidad): añade también los de estado Default.
    """
    statuses = CHARGED_OFF + (DEFAULT_STATUS if include_default else [])
    d = df[df["loan_status"].isin(statuses)].copy()

    # Parseo robusto de fechas
    d["issue_date"] = pd.to_datetime(d["issue_d"], format="%b-%Y", errors="coerce")
    if d["issue_date"].isna().any():
        d["issue_date"] = d["issue_date"].fillna(
            pd.to_datetime(d["issue_d"], errors="coerce")
        )
    d["issue_year"] = d["issue_date"].dt.year

    # Relleno preventivo de nulos en montos numéricos
    num_cols = ["funded_amnt", "total_rec_prncp", "recoveries", "collection_recovery_fee"]
    d[num_cols] = d[num_cols].fillna(0.0)

    # Cálculo de EAD y filtrado de valores válidos
    d["ead_default"] = d["funded_amnt"] - d["total_rec_prncp"]
    d = d[d["ead_default"] > 0].copy()  # evita división por cero o inconsistencias

    # Cálculo de LGD
    d["net_recovery"] = d["recoveries"] - d["collection_recovery_fee"]
    d["lgd_raw"] = 1.0 - (d["net_recovery"] / d["ead_default"])
    d["lgd"] = d["lgd_raw"].clip(0.0, 1.0)

    keep = [
        "id",
        "loan_status",
        "issue_date",
        "issue_year",
        "term",
        "funded_amnt",
        "ead_default",
        "recoveries",
        "collection_recovery_fee",
        "net_recovery",
        "lgd_raw",
        "lgd",
    ]
    return d.dropna(subset=["lgd"])[keep].reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/raw/loan.csv")
    p.add_argument(
        "--out",
        default=None,
        help=f"por defecto {OUT_MAIN}, o {OUT_WITH_DEFAULT} con --include-default",
    )
    p.add_argument(
        "--include-default",
        action="store_true",
        help="incluye también los créditos en estado Default (análisis de sensibilidad)",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=0,
        help="si >0, lee solo las primeras N filas (modo demo)",
    )
    a = p.parse_args()

    df = pd.read_csv(
        a.raw, usecols=COLS, low_memory=False, nrows=a.sample or None
    )
    out = build_lgd_target(df, include_default=a.include_default)

    out_path = a.out or (OUT_WITH_DEFAULT if a.include_default else OUT_MAIN)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"Filas leídas: {len(df):,} | créditos con LGD: {len(out):,}")
    print(out.groupby("loan_status").lgd.agg(["count", "mean"]).round(3).to_string())
    print(f"LGD media: {out.lgd.mean():.3f} | LGD == 1: {(out.lgd >= 1).mean():.1%}")
    print(f"Guardado en {out_path}")


if __name__ == "__main__":
    main()