# scripts/run_search.py
"""Búsqueda de hiperparámetros; registrar nº de configuraciones y tiempo de cómputo."""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd
import yaml

from creditrisk.metrics import evaluate_pd_metrics
from creditrisk.models.baseline_pd import LogisticPDModel
from creditrisk.models.mlp import predict_proba_mlp, train_pd_mlp
from creditrisk.seeding import set_all_seeds


def load_split_df(processed_dir: str, split_name: str, fmt: str) -> pd.DataFrame:
    path = os.path.join(processed_dir, f"{split_name}.{fmt}")
    return pd.read_parquet(path) if fmt == "parquet" else pd.read_csv(path, low_memory=False)


def main(base_cfg_path: str = "configs/base.yaml") -> None:
    with open(base_cfg_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    seed = int(base_cfg.get("seed", 42))
    set_all_seeds(seed)

    proc_dir = base_cfg["paths"]["processed_pd_dir"]
    fmt = base_cfg["paths"].get("output_format", "csv")
    os.makedirs("reports/tables/pd", exist_ok=True)

    train_df = load_split_df(proc_dir, "train", fmt)
    val_df = load_split_df(proc_dir, "val", fmt)
    preprocessor = joblib.load("models/pd/preprocessor.joblib")

    X_train = preprocessor.transform(train_df).astype(np.float32)
    y_train = train_df["target_pd"].values.astype(np.int64)
    X_val = preprocessor.transform(val_df).astype(np.float32)
    y_val = val_df["target_pd"].values.astype(np.int64)

    records = []

    # 1. Búsqueda en Regresión Logística (6 configuraciones)
    lr_grid = [
        {"C": 0.001, "class_weight": None},
        {"C": 0.01, "class_weight": None},
        {"C": 0.1, "class_weight": None},
        {"C": 1.0, "class_weight": None},
        {"C": 10.0, "class_weight": None},
        {"C": 1.0, "class_weight": "balanced"},
    ]

    print(f"[Search] Evaluando {len(lr_grid)} configuraciones de Regresión Logística...")
    for idx, cfg in enumerate(lr_grid, 1):
        set_all_seeds(seed)
        t0 = time.time()
        model = LogisticPDModel(C=cfg["C"], class_weight=cfg["class_weight"], random_state=seed)
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        val_probs = model.predict_proba(X_val)
        m = evaluate_pd_metrics(y_val, val_probs)
        records.append(
            {
                "familia": "LogisticRegression",
                "config_id": f"LR_{idx}",
                "descripcion": f"C={cfg['C']}, weight={cfg['class_weight']}",
                "parametros": model.count_parameters(),
                "tiempo_s": round(elapsed, 2),
                "val_auc": round(m["auc"], 4),
                "val_gini": round(m["gini"], 4),
                "val_ks": round(m["ks"], 4),
                "val_brier": round(m["brier"], 4),
            }
        )

    # 2. Búsqueda en Perceptrón Multicapa (6 configuraciones)
    mlp_grid = [
        {"hidden_dims": [64, 32], "activation": "relu", "dropout": 0.15, "lr": 1e-3, "weight_decay": 1e-4},
        {"hidden_dims": [128, 64], "activation": "relu", "dropout": 0.25, "lr": 1e-3, "weight_decay": 1e-4},
        {"hidden_dims": [128, 64, 32], "activation": "relu", "dropout": 0.25, "lr": 1e-3, "weight_decay": 1e-4},
        {"hidden_dims": [128, 64, 32], "activation": "gelu", "dropout": 0.25, "lr": 1e-3, "weight_decay": 1e-4},
        {"hidden_dims": [256, 128, 64], "activation": "relu", "dropout": 0.35, "lr": 5e-4, "weight_decay": 5e-4},
        {"hidden_dims": [128, 64, 32], "activation": "tanh", "dropout": 0.20, "lr": 1e-3, "weight_decay": 1e-4},
    ]

    print(f"[Search] Evaluando {len(mlp_grid)} configuraciones de MLP...")
    for idx, cfg in enumerate(mlp_grid, 1):
        set_all_seeds(seed)
        cfg_run = {**cfg, "use_batchnorm": True, "batch_size": 2048, "epochs": 20, "patience": 4}
        t0 = time.time()
        model, _ = train_pd_mlp(X_train, y_train, X_val, y_val, cfg_run, verbose=False)
        elapsed = time.time() - t0
        val_probs = predict_proba_mlp(model, X_val)
        m = evaluate_pd_metrics(y_val, val_probs)
        print(f"  -> MLP_{idx} ({cfg['hidden_dims']}, {cfg['activation']}) | Val AUC: {m['auc']:.4f} | {elapsed:.1f}s")
        records.append(
            {
                "familia": "TabularMLP",
                "config_id": f"MLP_{idx}",
                "descripcion": f"dims={cfg['hidden_dims']}, act={cfg['activation']}, do={cfg['dropout']}, lr={cfg['lr']}",
                "parametros": model.count_parameters(),
                "tiempo_s": round(elapsed, 2),
                "val_auc": round(m["auc"], 4),
                "val_gini": round(m["gini"], 4),
                "val_ks": round(m["ks"], 4),
                "val_brier": round(m["brier"], 4),
            }
        )

    df_search = pd.DataFrame(records)
    df_search.to_csv("reports/tables/pd/hyperparameter_search.csv", index=False)

    print("\n" + "=" * 95)
    print("RESUMEN DE BÚSQUEDA DE HIPERPARÁMETROS (PRESUPUESTO COMPUTACIONAL Y DESEMPEÑO EN VAL)")
    print("=" * 95)
    print(df_search.to_string(index=False))
    print("=" * 95)


if __name__ == "__main__":
    main()