"""Modelos de referencia para LGD: regresión lineal y predictor constante."""
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge


class LinearLGDModel:
    """Regresión lineal: OLS (alpha=None) o Ridge (alpha>0, penalización L2 sobre los pesos).

    La LGD vive en [0, 1] pero una recta no tiene cota, así que, si `clip=True`, las
    predicciones se recortan a [0, 1] para una comparación justa con la red (cuya
    salida ya está acotada por la sigmoide).
    """

    def __init__(self, clip: bool = True, alpha: float | None = None) -> None:
        self.clip = clip
        self.alpha = alpha
        self.model = LinearRegression() if alpha is None else Ridge(alpha=float(alpha))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearLGDModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred = self.model.predict(X)
        return np.clip(pred, 0.0, 1.0) if self.clip else pred

    def count_parameters(self) -> int:
        """Pesos + intercepto."""
        if not hasattr(self.model, "coef_"):
            return 0
        return int(self.model.coef_.size + np.ravel(self.model.intercept_).size)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "LinearLGDModel":
        return joblib.load(path)


class ConstantLGDModel:
    """Predice siempre la media del train: suelo mínimo que cualquier modelo debe superar."""

    def __init__(self) -> None:
        self.mean_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConstantLGDModel":
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.mean_, dtype=np.float64)

    def count_parameters(self) -> int:
        return 1
