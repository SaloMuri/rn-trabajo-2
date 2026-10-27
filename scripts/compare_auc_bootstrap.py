
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def main():
    ruta = (
        Path(__file__).resolve().parent.parent
        / "reports"
        / "tables"
        / "pd"
        / "test_predictions.csv"
    )
    df = pd.read_csv(ruta)

    columnas = ["y_true", "prob_logreg", "prob_mlp"]
    df = df[columnas].dropna()

    y = df["y_true"].to_numpy()
    prob_lr = df["prob_logreg"].to_numpy()
    prob_mlp = df["prob_mlp"].to_numpy()

    if set(np.unique(y)) != {0, 1}:
        raise ValueError("y_true debe contener las clases 0 y 1.")

    auc_lr = roc_auc_score(y, prob_lr)
    auc_mlp = roc_auc_score(y, prob_mlp)
    diferencia = auc_mlp - auc_lr

    rng = np.random.default_rng(42)
    indices_0 = np.flatnonzero(y == 0)
    indices_1 = np.flatnonzero(y == 1)
    diferencias = []

    for _ in range(5000):
        muestra_0 = rng.choice(
            indices_0, size=len(indices_0), replace=True
        )
        muestra_1 = rng.choice(
            indices_1, size=len(indices_1), replace=True
        )
        indices = np.concatenate([muestra_0, muestra_1])

        auc_lr_boot = roc_auc_score(y[indices], prob_lr[indices])
        auc_mlp_boot = roc_auc_score(y[indices], prob_mlp[indices])
        diferencias.append(auc_mlp_boot - auc_lr_boot)

    inferior, superior = np.percentile(diferencias, [2.5, 97.5])

    print(f"AUC Regresión Logística: {auc_lr:.4f}")
    print(f"AUC MLP:                 {auc_mlp:.4f}")
    print(f"Diferencia MLP - LR:     {diferencia:.4f}")
    print(f"IC 95% de la diferencia: [{inferior:.4f}, {superior:.4f}]")

    if inferior > 0:
        print("Interpretación: el IC respalda una diferencia positiva de AUC.")
    elif superior < 0:
        print("Interpretación: el IC respalda una diferencia negativa de AUC.")
    else:
        print("Interpretación: el IC incluye cero; la diferencia no queda "
              "establecida con este intervalo.")


if __name__ == "__main__":
    main()
