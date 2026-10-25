# src/creditrisk/interpretability.py
"""Método de interpretabilidad (Permutation Feature Importance) para identificar variables de riesgo."""

from typing import Callable, List
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def compute_permutation_importance(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Calcula la importancia por permutación (caída en AUC-ROC) sobre el conjunto de evaluación.
    Citas de referencia: Breiman (2001); Fisher, Rudin, & Dominici (2019).
    """
    rng = np.random.default_rng(random_state)
    base_probs = predict_fn(X)
    base_auc = float(roc_auc_score(y, base_probs))

    results = []
    X_work = X.copy()

    for col_idx, feat_name in enumerate(feature_names):
        original_col = X_work[:, col_idx].copy()
        drops = []
        for _ in range(n_repeats):
            X_work[:, col_idx] = rng.permutation(original_col)
            perm_probs = predict_fn(X_work)
            perm_auc = float(roc_auc_score(y, perm_probs))
            drops.append(base_auc - perm_auc)
        X_work[:, col_idx] = original_col

        clean_name = feat_name.replace("num__", "").replace("cat__", "")
        results.append(
            {
                "feature": clean_name,
                "importance_auc_drop_mean": float(np.mean(drops)),
                "importance_auc_drop_std": float(np.std(drops)),
            }
        )

    df_imp = pd.DataFrame(results).sort_values(
        by="importance_auc_drop_mean", ascending=False
    )
    return df_imp.reset_index(drop=True)