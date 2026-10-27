"""Pérdida esperada E[L] = PD * LGD * EAD; definir y justificar la EAD.

Definiciones (unidades coherentes):
- PD  : probabilidad de incumplimiento a partir de las variables de originación (Parte 1).
- LGD : fracción de la deuda AL INCUMPLIR que se pierde (Parte 2).
        LGD = 1 - (recoveries - collection_recovery_fee) / EAD_default,
        con EAD_default = funded_amnt - total_rec_prncp.
- EAD : deuda al incumplir, en dinero. Al originar no se conoce, así que se estima:
    * Opción B (principal): EAD = monto * razón_plazo,
        razón_plazo = media de (EAD_default / funded_amnt) en los créditos castigados de
        TRAIN, por plazo (36 / 60 meses). Es coherente con la LGD: ambas se refieren a la
        deuda en el momento del incumplimiento, que ya incluye lo amortizado hasta entonces.
    * Opción A (sensibilidad, cota superior): EAD = monto, es decir, sin amortización.
"""
from typing import Dict

import numpy as np
import pandas as pd


def fit_ead_ratio(lgd_train: pd.DataFrame) -> Dict[str, float]:
    """Razón media EAD_default / funded_amnt por plazo, calculada SOLO con train."""
    ratio = lgd_train["ead_default"] / lgd_train["funded_amnt"]
    by_term = ratio.groupby(lgd_train["term"]).mean()
    out = {str(int(k)): float(v) for k, v in by_term.items()}
    out["global"] = float(ratio.mean())
    return out


def estimate_ead(amount: np.ndarray, term: np.ndarray, ratios: Dict[str, float], method: str = "B") -> np.ndarray:
    """EAD en dinero. method='A': el monto; method='B': monto * razón del plazo."""
    amount = np.asarray(amount, dtype=float)
    if method == "A":
        return amount.copy()
    if method == "B":
        r = np.array([ratios.get(str(int(t)), ratios["global"]) for t in term], dtype=float)
        return amount * r
    raise ValueError(f"Método de EAD no soportado: {method!r} (use 'A' o 'B')")


def expected_loss(pd_hat: np.ndarray, lgd_hat: np.ndarray, ead: np.ndarray) -> np.ndarray:
    """E[L] = PD * LGD * EAD, por crédito."""
    return np.asarray(pd_hat, dtype=float) * np.asarray(lgd_hat, dtype=float) * np.asarray(ead, dtype=float)


def decile_table(el_hat: np.ndarray, realized: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Ordena por pérdida esperada y compara, por grupo, la pérdida media predicha y la realizada."""
    df = pd.DataFrame({"el_hat": el_hat, "realizada": realized})
    df["decil"] = pd.qcut(df["el_hat"].rank(method="first"), n_bins, labels=False) + 1
    out = df.groupby("decil").agg(n=("el_hat", "size"), el_predicha_media=("el_hat", "mean"),
                                  perdida_realizada_media=("realizada", "mean")).reset_index()
    return out
