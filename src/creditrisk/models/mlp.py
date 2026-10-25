# src/creditrisk/models/mlp.py
"""Perceptrón multicapa configurable y su entrenamiento (regularización, optimizador, parada temprana)."""

import copy
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from creditrisk.metrics import evaluate_pd_metrics


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "selu":
        return nn.SELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Activación no soportada: {name}")


class TabularMLP(nn.Module):
    """Arquitectura MLP dinámica para datos tabulares."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        activation: str = "relu",
        dropout: float = 0.2,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.config = {
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "activation": activation,
            "dropout": dropout,
            "use_batchnorm": use_batchnorm,
        }

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(get_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # Capa de salida: 1 neurona (entrega logit para PD o valor continuo para LGD)
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def count_parameters(self) -> int:
        """Cuenta los parámetros entrenables de la red."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def predict_proba_mlp(
    model: TabularMLP, X: np.ndarray, batch_size: int = 2048, device: str = "cpu"
) -> np.ndarray:
    """Evalúa el MLP por lotes y retorna las probabilidades P(y=1|x) en NumPy."""
    model.eval()
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probs_list = []
    with torch.no_grad():
        for (bx,) in loader:
            bx = bx.to(device)
            logits = model(bx)
            probs = torch.sigmoid(logits)
            probs_list.append(probs.cpu().numpy())
    return np.concatenate(probs_list, axis=0)


def train_pd_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    mlp_cfg: dict,
    device: str | None = None,
    verbose: bool = True,
) -> Tuple[TabularMLP, Dict[str, List[float]]]:
    """
    Entrena el TabularMLP para clasificación binaria (PD) con AdamW y Early Stopping
    monitoreando el AUC en el conjunto de validación.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TabularMLP(
        input_dim=X_train.shape[1],
        hidden_dims=mlp_cfg.get("hidden_dims", [128, 64, 32]),
        activation=mlp_cfg.get("activation", "relu"),
        dropout=mlp_cfg.get("dropout", 0.25),
        use_batchnorm=mlp_cfg.get("use_batchnorm", True),
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(mlp_cfg.get("lr", 1e-3)),
        weight_decay=float(mlp_cfg.get("weight_decay", 1e-4)),
    )

    batch_size = int(mlp_cfg.get("batch_size", 1024))
    epochs = int(mlp_cfg.get("epochs", 30))
    patience = int(mlp_cfg.get("patience", 5))

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_auc": []}
    best_val_auc = -1.0
    best_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    val_x_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    val_y_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(by)

        train_loss = running_loss / len(X_train)

        # Evaluación en validación
        model.eval()
        with torch.no_grad():
            val_logits = model(val_x_tensor)
            val_loss = criterion(val_logits, val_y_tensor).item()
            val_probs = torch.sigmoid(val_logits).cpu().numpy()

        val_metrics = evaluate_pd_metrics(y_val, val_probs)
        val_auc = val_metrics["auc"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        if verbose:
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}"
            )

        if val_auc > best_val_auc + 1e-5:
            best_val_auc = val_auc
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"[Early Stopping] Detenido en época {epoch}. Mejor Val AUC: {best_val_auc:.4f}")
                break

    model.load_state_dict(best_weights)
    return model, history


def save_mlp_checkpoint(model: TabularMLP, path: str) -> None:
    torch.save({"config": model.config, "state_dict": model.state_dict()}, path)


def load_mlp_checkpoint(path: str, device: str = "cpu") -> TabularMLP:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = TabularMLP(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model