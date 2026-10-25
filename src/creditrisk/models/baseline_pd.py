# src/creditrisk/models/baseline_pd.py
"""Modelo de referencia para PD: Regresión Logística con interfaz unificada."""

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


class LogisticPDModel:
    """Wrapper de Regresión Logística que expone conteo de parámetros y probabilidades."""

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str | dict | None = None,
        random_state: int = 42,
    ) -> None:
        self.model = LogisticRegression(
            C=C,
            penalty="l2",
            solver="lbfgs",
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LogisticPDModel":
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna únicamente el vector 1D con la probabilidad de incumplimiento P(y=1|x)."""
        return self.model.predict_proba(X)[:, 1]

    def count_parameters(self) -> int:
        """Retorna el número total de parámetros entrenables (pesos w + intercepto b)."""
        if not hasattr(self.model, "coef_"):
            return 0
        return int(self.model.coef_.size + self.model.intercept_.size)

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "LogisticPDModel":
        instance = cls()
        instance.model = joblib.load(path)
        return instance