"""Interpretabilidad de la LGD: importancia por permutación AGRUPADA por variable original.

Importancia por permutación (Breiman, 2001; Fisher, Rudin y Dominici, 2019): se baraja una
variable en el conjunto de evaluación y se mide cuánto EMPEORA el error. Si el RMSE sube
mucho, el modelo dependía de esa variable. Las variables nominales (p. ej. `purpose`) se
convierten en varias columnas one-hot; aquí se barajan JUNTAS, con la misma permutación de
filas, para medir la variable original y no cada categoría por separado.
"""
from typing import Callable, Dict, List

import numpy as np
import pandas as pd


def group_features(feature_names: List[str], nominal_cols: List[str]) -> Dict[str, List[int]]:
    """Agrupa los índices de columnas por variable original.

    `feature_names` viene de ColumnTransformer.get_feature_names_out(): "num__dti",
    "cat__purpose_car", ... Las nominales se reconocen por el prefijo "cat__<variable>_".
    """
    groups: Dict[str, List[int]] = {}
    for i, name in enumerate(feature_names):
        base = name.split("__", 1)[1] if "__" in name else name
        var = next((c for c in nominal_cols if base == c or base.startswith(c + "_")), base)
        groups.setdefault(var, []).append(i)
    return groups


def grouped_permutation_importance(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    groups: Dict[str, List[int]],
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Aumento del RMSE al barajar cada variable (media y desviación en `n_repeats` repeticiones)."""
    rng = np.random.default_rng(random_state)
    rmse = lambda p: float(np.sqrt(np.mean((y - p) ** 2)))
    base = rmse(predict_fn(X))
    rows = []
    for var, idx in groups.items():
        incr = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[:, idx] = X[perm][:, idx]          # misma permutación para todo el grupo
            incr.append(rmse(predict_fn(Xp)) - base)
        rows.append({"variable": var, "aumento_rmse_media": float(np.mean(incr)),
                     "aumento_rmse_std": float(np.std(incr, ddof=1)) if n_repeats > 1 else 0.0})
    out = pd.DataFrame(rows).sort_values("aumento_rmse_media", ascending=False).reset_index(drop=True)
    out.attrs["rmse_base"] = base
    return out
