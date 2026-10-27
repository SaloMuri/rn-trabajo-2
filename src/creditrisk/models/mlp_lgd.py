"""Entrenamiento del MLP para LGD (regresión sobre una fracción en [0, 1]).

Reutiliza `TabularMLP` (cuya capa de salida entrega un logit z). La predicción es
LGD_hat = sigmoide(z), así que siempre queda en (0, 1). Dos funciones de pérdida:

- "bce": entropía cruzada binaria con objetivo fraccionario y en [0, 1] (cuasi-máxima
  verosimilitud Bernoulli, estándar para variables fraccionarias).  [opción C]
      L = -[ y*log(p) + (1-y)*log(1-p) ],  p = sigmoide(z)
- "mse": error cuadrático medio sobre p = sigmoide(z).              [opción B, contraste]
      L = (y - p)^2

La parada temprana vigila el RMSE de validación en ambos casos, para que la comparación
entre pérdidas sea justa.
"""
import copy
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from creditrisk.metrics import evaluate_lgd_metrics
from creditrisk.models.mlp import TabularMLP


def _loss_fn(name: str):
    if name == "bce":
        bce = nn.BCEWithLogitsLoss()  # acepta objetivos continuos en [0, 1]
        return lambda logits, y: bce(logits, y)
    if name == "mse":
        return lambda logits, y: torch.mean((torch.sigmoid(logits) - y) ** 2)
    raise ValueError(f"Pérdida no soportada: {name!r} (use 'bce' o 'mse')")


def make_optimizer(name: str, params, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """Optimizadores soportados. `weight_decay` es la penalización L2.

    - adamw: Adam con decaimiento de pesos desacoplado del gradiente (Loshchilov y Hutter, 2019).
    - adam: Adam con la L2 sumada al gradiente.
    - sgd: descenso por gradiente estocástico con momento 0.9.
    - rmsprop: normaliza el gradiente por su media móvil cuadrática.
    """
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Optimizador no soportado: {name!r}")


def predict_lgd_mlp(
    model: TabularMLP, X: np.ndarray, batch_size: int = 4096, device: str = "cpu"
) -> np.ndarray:
    """Devuelve LGD_hat = sigmoide(logit) en NumPy, siempre dentro de (0, 1)."""
    model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=batch_size)
    out = []
    with torch.no_grad():
        for (bx,) in loader:
            out.append(torch.sigmoid(model(bx.to(device))).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float64)


def train_lgd_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    mlp_cfg: dict,
    device: str | None = None,
    verbose: bool = True,
) -> Tuple[TabularMLP, Dict[str, List[float]]]:
    """Entrena el MLP con AdamW y parada temprana sobre el RMSE de validación."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TabularMLP(
        input_dim=X_train.shape[1],
        hidden_dims=list(mlp_cfg.get("hidden_dims", [64, 32])),
        activation=mlp_cfg.get("activation", "relu"),
        dropout=float(mlp_cfg.get("dropout", 0.15)),
        use_batchnorm=bool(mlp_cfg.get("use_batchnorm", True)),
    ).to(device)

    # Sesgo de la capa de salida = logit(media del train): la red arranca prediciendo la media
    # (como el modelo constante) y solo tiene que aprender las desviaciones. Sin esto, la
    # sigmoide parte de 0.5 y, con LGD media ~0.92, gasta muchas épocas en subir.
    if mlp_cfg.get("init_output_bias", True):
        p0 = float(np.clip(np.mean(y_train), 1e-4, 1 - 1e-4))
        with torch.no_grad():
            model.net[-1].bias.fill_(float(np.log(p0 / (1 - p0))))

    loss_name = mlp_cfg.get("loss", "bce")
    criterion = _loss_fn(loss_name)
    optimizer = make_optimizer(
        mlp_cfg.get("optimizer", "adamw"),
        model.parameters(),
        lr=float(mlp_cfg.get("lr", 1e-3)),
        weight_decay=float(mlp_cfg.get("weight_decay", 1e-4)),
    )
    batch_size = int(mlp_cfg.get("batch_size", 512))
    epochs = int(mlp_cfg.get("epochs", 60))
    patience = int(mlp_cfg.get("patience", 8))

    # BatchNorm falla con un lote de 1 muestra: se descarta el último lote en ese caso.
    drop_last = len(X_train) % batch_size == 1
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
    )
    xv = torch.tensor(X_val, dtype=torch.float32).to(device)
    yv = torch.tensor(y_val, dtype=torch.float32).to(device)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_mae": [], "val_rmse": []}
    best_rmse, best_w, bad = float("inf"), copy.deepcopy(model.state_dict()), 0

    for epoch in range(1, epochs + 1):
        model.train()
        running, n_seen = 0.0, 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(by)
            n_seen += len(by)

        model.eval()
        with torch.no_grad():
            v_logits = model(xv)
            v_loss = criterion(v_logits, yv).item()
            v_pred = torch.sigmoid(v_logits).cpu().numpy()
        m = evaluate_lgd_metrics(y_val, v_pred)
        history["train_loss"].append(running / n_seen)
        history["val_loss"].append(v_loss)
        history["val_mae"].append(m["mae"])
        history["val_rmse"].append(m["rmse"])

        if verbose:
            print(f"[{loss_name}] Época {epoch:02d}/{epochs} | train {running / n_seen:.4f} | "
                  f"val {v_loss:.4f} | val MAE {m['mae']:.4f} | val RMSE {m['rmse']:.4f}")

        if m["rmse"] < best_rmse - 1e-6:
            best_rmse, best_w, bad = m["rmse"], copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"[Parada temprana] época {epoch}; mejor val RMSE {best_rmse:.4f}")
                break

    model.load_state_dict(best_w)
    return model, history
