"""Lógica compartida entre experimentos (mismos datos y particiones).

Para la Parte 2 (LGD) la partición NO se vuelve a calcular: cada crédito hereda el
split (train/val/test) que tiene en las tablas de la Parte 1, que se construyen con
la partición temporal por fecha de originación. Así ambas partes usan las mismas
particiones y el mismo criterio temporal.
"""
import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from creditrisk.data.features import build_pd_preprocessor

SPLITS = ("train", "val", "test")


def _read(processed_dir: str, name: str, fmt: str) -> pd.DataFrame:
    path = os.path.join(processed_dir, f"{name}.{fmt}")
    return pd.read_parquet(path) if fmt == "parquet" else pd.read_csv(path, low_memory=False)


def load_lgd_splits(
    lgd_path: str, processed_pd_dir: str, fmt: str = "parquet"
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, int]]:
    """Une el objetivo LGD con las variables de originación y el split de la Parte 1.

    Devuelve ({split: DataFrame}, info). Los créditos con LGD que no aparecen en las
    tablas de PD no tienen variables y se descartan; `info["sin_features"]` los cuenta.
    """
    lgd = pd.read_parquet(lgd_path)
    parts = []
    for split in SPLITS:
        pd_df = _read(processed_pd_dir, split, fmt).drop(columns=["issue_d_dt"], errors="ignore")
        pd_df = pd_df.drop(columns=["target_pd"], errors="ignore")
        parts.append(pd_df.assign(split=split))
    feats = pd.concat(parts, ignore_index=True)

    merged = lgd.merge(feats, on="id", how="inner", suffixes=("", "_pd"))
    # `term` existe en ambas tablas: se conserva la numérica limpia de la Parte 1.
    if "term_pd" in merged.columns:
        merged = merged.drop(columns=["term"]).rename(columns={"term_pd": "term"})

    info = {"lgd_total": len(lgd), "con_features": len(merged), "sin_features": len(lgd) - len(merged)}
    return {s: merged[merged["split"] == s].reset_index(drop=True) for s in SPLITS}, info


def subsample_splits(
    splits: Dict[str, pd.DataFrame], fraction: float, seed: int
) -> Dict[str, pd.DataFrame]:
    """Modo demo: muestra una fracción de cada split (misma semilla -> misma muestra)."""
    if fraction >= 1.0:
        return splits
    return {
        s: df.sample(frac=fraction, random_state=seed).reset_index(drop=True)
        for s, df in splits.items()
    }


def fit_lgd_preprocessor(
    train_df: pd.DataFrame, cfg_features: dict, save_path: str | None = None
):
    """Ajusta el preprocesador SOLO con el train de LGD (mismas columnas que la PD)."""
    num = [c for c in cfg_features["numeric"] if c in train_df.columns]
    ordi = [c for c in cfg_features["categorical_ordinal"] if c in train_df.columns]
    nom = [c for c in cfg_features["categorical_nominal"] if c in train_df.columns]
    pre = build_pd_preprocessor(num, ordi, nom)
    pre.fit(train_df)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(pre, save_path)
    return pre


def prepare_lgd_data(cfg: dict, preprocessor_path: str | None = None):
    """Pipeline de datos ÚNICO de la Parte 2 (lo usan el experimento, la búsqueda y las semillas).

    Devuelve (splits, data, info): DataFrames por split, {split: (X, y)} y el resumen de la unión.
    """
    population = cfg["lgd_data"]["population"]
    splits, info = load_lgd_splits(
        cfg["lgd_data"]["paths"][population],
        cfg["paths"]["processed_pd_dir"],
        cfg["paths"].get("output_format", "parquet"),
    )
    splits = subsample_splits(splits, float(cfg["lgd_data"]["sample_fraction"]), int(cfg["seed"]))
    pre = fit_lgd_preprocessor(splits["train"], cfg["features_pd"], preprocessor_path)
    return splits, make_xy(splits, pre), info


def make_xy(
    splits: Dict[str, pd.DataFrame], preprocessor
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Transforma cada split con el MISMO preprocesador y extrae el objetivo `lgd`."""
    return {
        s: (preprocessor.transform(df).astype(np.float32), df["lgd"].to_numpy(dtype=np.float32))
        for s, df in splits.items()
    }
