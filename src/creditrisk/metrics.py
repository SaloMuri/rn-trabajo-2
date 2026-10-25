# src/creditrisk/metrics.py
"""Métricas: AUC, Gini, KS, Brier, curva de calibración (PD); MAE, RMSE (LGD)."""

from typing import Dict, Tuple
import numpy as np
from scipy.stats import ks_2samp
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


def compute_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Calcula el estadístico Kolmogorov-Smirnov (KS) separando las distribuciones
    de probabilidades predichas para buenos (y=0) y malos (y=1).
    """
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    probs_good = y_prob_arr[y_true_arr == 0]
    probs_bad = y_prob_arr[y_true_arr == 1]
    if len(probs_good) == 0 or len(probs_bad) == 0:
        return 0.0
    ks_stat, _ = ks_2samp(probs_good, probs_bad)
    return float(ks_stat)


def evaluate_pd_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Calcula todas las métricas exigidas para la Parte 1 (PD):
    AUC-ROC, Índice de Gini, Estadístico KS y Brier Score.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1.0 - 1e-7)

    auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    gini = 2.0 * auc - 1.0
    ks = compute_ks_statistic(y_true_arr, y_prob_arr)
    brier = float(brier_score_loss(y_true_arr, y_prob_arr))

    return {
        "auc": auc,
        "gini": gini,
        "ks": ks,
        "brier": brier,
    }


def get_calibration_data(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Devuelve (probabilidad_media_predicha, fraccion_real_positivos) por cada bin
    para graficar la curva de calibración.
    """
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    return prob_pred, prob_true


def evaluate_lgd_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Métricas de regresión para la Parte 2 (LGD): MAE y RMSE."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"mae": mae, "rmse": rmse}