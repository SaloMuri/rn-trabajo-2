# scripts/run_seeds.py
"""Configuración final con >=30 semillas: media, desviación, mejor y peor de cada métrica; nº de parámetros."""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd
import yaml

from creditrisk.interpretability import compute_permutation_importance
from creditrisk.metrics import evaluate_pd_metrics
from creditrisk.models.baseline_pd import LogisticPDModel
from creditrisk.models.mlp import load_mlp_checkpoint, predict_proba_mlp, train_pd_mlp
from creditrisk.seeding import set_all_seeds


def load_split_df(processed_dir: str, split_name: str, fmt: str) -> pd.DataFrame:
    path = os.path.join(processed_dir, f"{split_name}.{fmt}")
    return pd.read_parquet(path) if fmt == "parquet" else pd.read_csv(path, low_memory=False)


def main(
    base_cfg_path: str = "configs/base.yaml",
    pd_cfg_path: str = "configs/pd.yaml",
    n_seeds: int = 30,
) -> None:
    with open(base_cfg_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    with open(pd_cfg_path, "r", encoding="utf-8") as f:
        pd_cfg = yaml.safe_load(f)

    proc_dir = base_cfg["paths"]["processed_pd_dir"]
    fmt = base_cfg["paths"].get("output_format", "csv")
    os.makedirs("reports/tables/pd", exist_ok=True)
    os.makedirs("models/pd", exist_ok=True)

    train_df = load_split_df(proc_dir, "train", fmt)
    val_df = load_split_df(proc_dir, "val", fmt)
    test_df = load_split_df(proc_dir, "test", fmt)

    preprocessor = joblib.load("models/pd/preprocessor.joblib")
    feature_names = list(preprocessor.get_feature_names_out())

    X_train = preprocessor.transform(train_df).astype(np.float32)
    y_train = train_df["target_pd"].values.astype(np.int64)
    X_val = preprocessor.transform(val_df).astype(np.float32)
    y_val = val_df["target_pd"].values.astype(np.int64)
    X_test = preprocessor.transform(test_df).astype(np.float32)
    y_test = test_df["target_pd"].values.astype(np.int64)

    # Usar la configuración declarada para el MLP final en todas las semillas.
    mlp_cfg = dict(pd_cfg["mlp"])

    seed_records = []
    n_params = 0
    t_start = time.time()

    print(f"[Seeds] Ejecutando evaluación multi-semilla con {n_seeds} semillas independientes...")
    for idx in range(n_seeds):
        current_seed = 100 + idx
        set_all_seeds(current_seed)
        t0 = time.time()
        model, _ = train_pd_mlp(X_train, y_train, X_val, y_val, mlp_cfg, verbose=False)
        elapsed = time.time() - t0
        n_params = model.count_parameters()

        test_probs = predict_proba_mlp(model, X_test)
        m = evaluate_pd_metrics(y_test, test_probs)
        seed_records.append(
            {
                "seed": current_seed,
                "parametros": n_params,
                "tiempo_s": round(elapsed, 2),
                "auc": m["auc"],
                "gini": m["gini"],
                "ks": m["ks"],
                "brier": m["brier"],
            }
        )
        print(
            f"  Semilla {idx+1:02d}/{n_seeds:02d} (seed={current_seed}) | "
            f"Test AUC: {m['auc']:.4f} | Gini: {m['gini']:.4f} | KS: {m['ks']:.4f} | "
            f"Brier: {m['brier']:.4f} ({elapsed:.1f}s)"
        )

    df_seeds = pd.DataFrame(seed_records)
    df_seeds.to_csv("reports/tables/pd/seeds_raw_results.csv", index=False)

    # Resumen estadístico de las >=30 semillas (Media, Desviación, Mejor, Peor)
    summary_rows = []
    for metric in ["auc", "gini", "ks", "brier"]:
        vals = df_seeds[metric].values
        # Para Brier menor es mejor; para AUC, Gini y KS mayor es mejor
        best_val = float(np.min(vals)) if metric == "brier" else float(np.max(vals))
        worst_val = float(np.max(vals)) if metric == "brier" else float(np.min(vals))
        summary_rows.append(
            {
                "metrica": metric.upper(),
                "media": round(float(np.mean(vals)), 4),
                "desviacion_std": round(float(np.std(vals, ddof=1)), 4),
                "mejor": round(best_val, 4),
                "peor": round(worst_val, 4),
                "n_semillas": n_seeds,
                "parametros_mlp": n_params,
            }
        )

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv("reports/tables/pd/seeds_summary_pd.csv", index=False)

    with open("models/pd/seeds_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_seeds": n_seeds,
                "total_time_s": round(time.time() - t_start, 2),
                "n_parameters": n_params,
                "summary": summary_rows,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 80)
    print(f"RESUMEN ESTADÍSTICO EN TEST CON {n_seeds} SEMILLAS (PARÁMETROS MLP: {n_params:,})")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print("=" * 80)

    # -------------------------------------------------------------------------
    # INTERPRETABILIDAD EN TEST (MLP vs Regresión Logística)
    # -------------------------------------------------------------------------
    print("\n[Interpretabilidad] Calculando Permutation Feature Importance sobre Test...")
    best_mlp = load_mlp_checkpoint("models/pd/mlp_best.pt")
    lr_model = LogisticPDModel.load("models/pd/logreg_baseline.joblib")

    imp_mlp = compute_permutation_importance(
        predict_fn=lambda x: predict_proba_mlp(best_mlp, x),
        X=X_test,
        y=y_test,
        feature_names=feature_names,
        n_repeats=5,
        random_state=42,
    )
    imp_lr = compute_permutation_importance(
        predict_fn=lambda x: lr_model.predict_proba(x),
        X=X_test,
        y=y_test,
        feature_names=feature_names,
        n_repeats=5,
        random_state=42,
    )

    imp_merged = pd.merge(
        imp_mlp.rename(
            columns={
                "importance_auc_drop_mean": "mlp_auc_drop_mean",
                "importance_auc_drop_std": "mlp_auc_drop_std",
            }
        ),
        imp_lr[["feature", "importance_auc_drop_mean"]].rename(
            columns={"importance_auc_drop_mean": "lr_auc_drop_mean"}
        ),
        on="feature",
    )
    imp_merged.to_csv("reports/tables/pd/feature_importance_pd.csv", index=False)
    print("\nTOP 10 VARIABLES MÁS IMPORTANTES (CAÍDA EN AUC AL PERMUTAR):")
    print(imp_merged.head(10).to_string(index=False))


if __name__ == "__main__":
    main()