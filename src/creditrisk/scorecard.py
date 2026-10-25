# src/creditrisk/scorecard.py
"""Scorecard: puntaje base, odds de referencia, PDO y transformación probabilidad -> puntaje."""

from dataclasses import dataclass
import numpy as np


@dataclass
class CreditScorecard:
    """
    Transforma probabilidades de incumplimiento (PD) en un puntaje crediticio continuo
    donde un mayor puntaje representa menor riesgo (mayores odds de buen pagador).

    Parámetros:
    - base_score (P0): Puntaje de referencia (ej. 600 puntos).
    - base_odds (theta0): Odds (Bueno / Malo) en el puntaje base (ej. 19.0 equivale a PD = 5%).
    - pdo: Puntos para duplicar las odds de ser buen pagador (ej. 20 puntos).
    """

    base_score: float = 600.0
    base_odds: float = 19.0
    pdo: float = 20.0

    @property
    def factor(self) -> float:
        return float(self.pdo / np.log(2.0))

    @property
    def offset(self) -> float:
        return float(self.base_score - self.factor * np.log(self.base_odds))

    def pd_to_score(self, pd_prob: np.ndarray | float) -> np.ndarray | float:
        """Convierte probabilidad de mal pagador P(y=1) en puntaje de la scorecard."""
        p = np.clip(np.asarray(pd_prob, dtype=float), 1e-6, 1.0 - 1e-6)
        good_odds = (1.0 - p) / p
        score = self.offset + self.factor * np.log(good_odds)
        if np.ndim(pd_prob) == 0:
            return float(score)
        return score

    def score_to_pd(self, score: np.ndarray | float) -> np.ndarray | float:
        """Transformación inversa: convierte un puntaje en probabilidad de incumplimiento."""
        s = np.asarray(score, dtype=float)
        good_odds = np.exp((s - self.offset) / self.factor)
        p = 1.0 / (1.0 + good_odds)
        if np.ndim(score) == 0:
            return float(p)
        return p