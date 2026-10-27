"""Comparación de la distribución de las predicciones de LGD con la observada."""
from typing import Dict

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def distribution_summary(values: np.ndarray, near_one: float = 0.99) -> Dict[str, float]:
    """Media, desviación, cuantiles y proporción de valores cercanos a 1."""
    v = np.asarray(values, dtype=float)
    out = {"media": float(v.mean()), "std": float(v.std())}
    for q in QUANTILES:
        out[f"q{int(q * 100):02d}"] = float(np.quantile(v, q))
    out["frac_ge_0.99"] = float((v >= near_one).mean())
    return out


def compare_distributions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Distancias entre la distribución observada y la predicha.

    - wasserstein: costo medio de "mover" una distribución sobre la otra (unidades de LGD).
    - ks: máxima diferencia entre las dos funciones de distribución acumulada.
    """
    return {
        "wasserstein": float(wasserstein_distance(y_true, y_pred)),
        "ks": float(ks_2samp(y_true, y_pred).statistic),
    }
