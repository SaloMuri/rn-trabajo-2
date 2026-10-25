# scripts/run_pd.py
"""Experimento PD: modelo de referencia (Regresión Logística) + MLP. Un solo comando."""

import json
import os
import sys
import time
from pathlib import Path

# Permite encontrar el paquete creditrisk desde cualquier terminal
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd
import yaml

from creditrisk.metrics import evaluate_pd_metrics
from creditrisk.models.baseline_pd import LogisticPDModel
from creditrisk.models.mlp import predict_proba_mlp, save_mlp_checkpoint, train_pd_mlp
from creditrisk.scorecard import CreditScorecard
from creditrisk.seeding import set_all_seeds


def load_split_df(processed_dir: str, split_name: str, fmt: str) -> pd.DataFrame:
    path = os.path.join(processed_dir, f"{split_name}.{fmt}")
    if fmt == "parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def main(
    base_cfg_path: str = "configs/base.yaml",
    pd_cfg_path: str = "configs/pd.yaml",
) -> None:
    with open(base_cfg_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    with open(pd_cfg_path, "r", encoding="utf-8") as f:
        pd_cfg = yaml.safe_load(f)

    seed = int(pd_cfg.get("seed", base_cfg.get("seed", 42)))
    set_all_seeds(seed)

    proc_dir = base_cfg["paths"]["processed_pd_dir"]
    fmt = base_cfg["paths"].get("output_format", "csv")
    os.makedirs("models/pd", exist_ok=True)
    os.makedirs("reports/tables/pd", exist_ok=True)

    is_demo = "demo_sample_frac" in pd_cfg
    sample_frac = float(pd_cfg.get("demo_sample_frac", 1.0))

    print("[1/4] Cargando particiones temporales y preprocesador único...")
    train_df = load_split_df(proc_dir, "train", fmt)
    val_df = load_split_df(proc_dir, "val", fmt)
    test_df = load_split_df(proc_dir, "test", fmt)

    if is_demo and 0.0 < sample_frac < 1.0:
        print(f"[MODO DEMO] Usando muestra rápida del {sample_frac*100:.0f}% de los datos (< 2 min)...")
        train_df = train_df.sample(frac=sample_frac, random_state=seed)
        val_df = val_df.sample(frac=sample_frac, random_state=seed)
        test_df = test_df.sample(frac=sample_frac, random_state=seed)

    preprocessor = joblib.load("models/pd/preprocessor.joblib")

    X_train = preprocessor.transform(train_df).astype(np.float32)
    y_train = train_df["target_pd"].values.astype(np.int64)

    X_val = preprocessor.transform(val_df).astype(np.float32)
    y_val = val_df["target_pd"].values.astype(np.int64)

    X_test = preprocessor.transform(test_df).astype(np.float32)
    y_test = test_df["target_pd"].values.astype(np.int64)

    # 1. ENTRENAMIENTO REGRESIÓN LOGÍSTICA (BASELINE)
    print("\n[2/4] Entrenando Modelo de Referencia (Regresión Logística)...")
    t0 = time.time()
    lr_cfg = pd_cfg["baseline"]
    lr_model = LogisticPDModel(
        C=float(lr_cfg.get("C", 1.0)),
        max_iter=int(lr_cfg.get("max_iter", 1000)),
        class_weight=lr_cfg.get("class_weight", None),
        random_state=seed,
    )
    lr_model.fit(X_train, y_train)
    lr_time = time.time() - t0
    if not is_demo:
        lr_model.save("models/pd/logreg_baseline.joblib")

    lr_val_probs = lr_model.predict_proba(X_val)
    lr_test_probs = lr_model.predict_proba(X_test)
    lr_val_m = evaluate_pd_metrics(y_val, lr_val_probs)
    lr_test_m = evaluate_pd_metrics(y_test, lr_test_probs)

    # 2. ENTRENAMIENTO PERCEPTRÓN MULTICAPA (MLP)
    print("\n[3/4] Entrenando Red Neuronal (TabularMLP)...")
    t0 = time.time()
    mlp_model, history = train_pd_mlp(
        X_train, y_train, X_val, y_val, pd_cfg["mlp"], verbose=True
    )
    mlp_time = time.time() - t0
    if not is_demo:
        save_mlp_checkpoint(mlp_model, "models/pd/mlp_best.pt")
        pd.DataFrame(history).to_csv("reports/tables/pd/mlp_training_history.csv", index=False)

    mlp_val_probs = predict_proba_mlp(mlp_model, X_val)
    mlp_test_probs = predict_proba_mlp(mlp_model, X_test)
    mlp_val_m = evaluate_pd_metrics(y_val, mlp_val_probs)
    mlp_test_m = evaluate_pd_metrics(y_test, mlp_test_probs)

    # 3. CONSTRUCCIÓN DE LA SCORECARD Y DISTRIBUCIÓN DE REFERENCIA
    print("\n[4/4] Configurando Scorecard y evaluando métricas...")
    sc_cfg = pd_cfg["scorecard"]
    scorecard = CreditScorecard(
        base_score=float(sc_cfg["base_score"]),
        base_odds=float(sc_cfg["base_odds"]),
        pdo=float(sc_cfg["pdo"]),
    )
    test_scores = scorecard.pd_to_score(mlp_test_probs)

    if not is_demo:
        scorecard_meta = {
            "base_score": scorecard.base_score,
            "base_odds": scorecard.base_odds,
            "pdo": scorecard.pdo,
            "factor": scorecard.factor,
            "offset": scorecard.offset,
            "test_score_quantiles": {
                str(q): float(np.quantile(test_scores, q))
                for q in np.round(np.linspace(0.01, 0.99, 99), 2)
            },
        }
        with open("models/pd/scorecard_config.json", "w", encoding="utf-8") as f:
            json.dump(scorecard_meta, f, indent=2)

        preds_df = pd.DataFrame(
            {
                "y_true": y_test,
                "prob_logreg": lr_test_probs,
                "prob_mlp": mlp_test_probs,
                "score_mlp": test_scores,
            }
        )
        preds_df.to_csv("reports/tables/pd/test_predictions.csv", index=False)

    comparison_df = pd.DataFrame(
        [
            {
                "Modelo": "Regresión Logística (Baseline)",
                "Parámetros": lr_model.count_parameters(),
                "Tiempo_s": round(lr_time, 2),
                "Val_AUC": round(lr_val_m["auc"], 4),
                "Test_AUC": round(lr_test_m["auc"], 4),
                "Test_Gini": round(lr_test_m["gini"], 4),
                "Test_KS": round(lr_test_m["ks"], 4),
                "Test_Brier": round(lr_test_m["brier"], 4),
            },
            {
                "Modelo": "Perceptrón Multicapa (MLP)",
                "Parámetros": mlp_model.count_parameters(),
                "Tiempo_s": round(mlp_time, 2),
                "Val_AUC": round(mlp_val_m["auc"], 4),
                "Test_AUC": round(mlp_test_m["auc"], 4),
                "Test_Gini": round(mlp_test_m["gini"], 4),
                "Test_KS": round(mlp_test_m["ks"], 4),
                "Test_Brier": round(mlp_test_m["brier"], 4),
            },
        ]
    )
    out_csv = "reports/tables/pd/comparison_demo.csv" if is_demo else "reports/tables/pd/comparison_pd.csv"
    comparison_df.to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print("RESULTADOS DE COMPARACIÓN JUSTA (PARTE 1 - PD)")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ejecuta el experimento PD.")
    parser.add_argument("--config", type=str, default="configs/pd.yaml", help="Ruta al archivo YAML de configuración PD o demo.")
    args = parser.parse_args()
    main(base_cfg_path="configs/base.yaml", pd_cfg_path=args.config)