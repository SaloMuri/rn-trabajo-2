# src/creditrisk/data/features.py
"""Preprocesamiento único compartido por todos los modelos de una misma tarea."""

from typing import List, Tuple
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pd_preprocessor(
    numeric_cols: List[str],
    ordinal_cols: List[str],
    nominal_cols: List[str],
) -> ColumnTransformer:
    """
    Construye el pipeline de preprocesamiento tabular para PD:
    - Numéricas y Ordinales: imputación por mediana + estandarización (media 0, var 1).
    - Categóricas Nominales: imputación por constante 'MISSING' + One-Hot Encoding.
    """
    num_ord_cols = numeric_cols + ordinal_cols

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
            (
                "ohe",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_ord_cols),
            ("cat", cat_pipeline, nominal_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def fit_and_save_preprocessor(
    train_df: pd.DataFrame,
    cfg_features: dict,
    save_path: str = "models/pd/preprocessor.joblib",
) -> Tuple[ColumnTransformer, List[str]]:
    """
    Ajusta el preprocesador SOLO con datos de entrenamiento y lo guarda en disco.
    Retorna el preprocesador ajustado y la lista de nombres de columnas resultantes.
    """
    num_cols = [c for c in cfg_features["numeric"] if c in train_df.columns]
    ord_cols = [c for c in cfg_features["categorical_ordinal"] if c in train_df.columns]
    nom_cols = [c for c in cfg_features["categorical_nominal"] if c in train_df.columns]

    preprocessor = build_pd_preprocessor(num_cols, ord_cols, nom_cols)
    preprocessor.fit(train_df)

    joblib.dump(preprocessor, save_path)
    feature_names = list(preprocessor.get_feature_names_out())
    return preprocessor, feature_names