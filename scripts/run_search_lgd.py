"""Búsqueda de hiperparámetros para LGD con presupuesto explícito.

    python scripts/run_search_lgd.py                       # 60 configuraciones MLP + 12 Ridge
    python scripts/run_search_lgd.py --set search.n_mlp_configs=20

- MLP: búsqueda aleatoria (pérdida BCE, opción C) sobre capas, neuronas, activación, dropout,
  L2, optimizador, tasa de aprendizaje y tamaño de lote.
- Ridge: rejilla logarítmica de `alpha`.
- Todas las configuraciones se entrenan con los MISMOS datos y la MISMA semilla; se elige por
  el RMSE de VALIDACIÓN. El test no se evalúa aquí.
- Se guarda el número de configuraciones y el tiempo de cómputo de cada familia, y la mejor
  configuración en `configs/lgd_best.yaml` (usable con `run_lgd.py --best`).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import yaml

from creditrisk.config import load_config
from creditrisk.experiments import prepare_lgd_data
from creditrisk.metrics import evaluate_lgd_metrics
from creditrisk.models.baseline_lgd import LinearLGDModel
from creditrisk.models.mlp_lgd import predict_lgd_mlp, train_lgd_mlp
from creditrisk.search_space import sample_mlp_configs
from creditrisk.seeding import set_all_seeds


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="RUTA=VALOR")
    p.add_argument("--base", default="configs/base.yaml")
    p.add_argument("--config", default="configs/lgd.yaml")
    a = p.parse_args()

    cfg = load_config([a.base, a.config], a.overrides)
    seed = int(cfg["seed"])
    s = cfg["search"]
    population = cfg["lgd_data"]["population"]
    out_tables = os.path.join(cfg["output"]["tables_dir"], population)
    os.makedirs(out_tables, exist_ok=True)
    set_all_seeds(seed)

    print(f"[1/3] Datos LGD (población: {population}; semilla: {seed})")
    _, data, _ = prepare_lgd_data(cfg)
    (Xtr, ytr), (Xva, yva) = data["train"], data["val"]  # el test NO se toca en la búsqueda
    print(f"      train {len(ytr):,} | val {len(yva):,} | variables: {Xtr.shape[1]}")

    # ---- Ridge: rejilla de alpha -------------------------------------------------------
    print(f"[2/3] Ridge: {s['ridge_alphas']} valores de alpha")
    lo, hi = s["ridge_alpha_range"]
    alphas = np.logspace(np.log10(lo), np.log10(hi), int(s["ridge_alphas"]))
    ridge_rows, t_ridge0 = [], time.time()
    for alpha in alphas:
        t0 = time.time()
        m = LinearLGDModel(clip=bool(cfg["baseline"].get("clip", True)), alpha=float(alpha)).fit(Xtr, ytr)
        mv = evaluate_lgd_metrics(yva, m.predict(Xva))
        ridge_rows.append({"alpha": float(f"{alpha:.4g}"), "Val_MAE": mv["mae"], "Val_RMSE": mv["rmse"],
                           "Parámetros": m.count_parameters(), "Tiempo_s": round(time.time() - t0, 3)})
    ridge_total = time.time() - t_ridge0
    ridge_df = pd.DataFrame(ridge_rows)
    ridge_df.round(5).to_csv(os.path.join(out_tables, "search_ridge.csv"), index=False)
    best_ridge = ridge_df.loc[ridge_df["Val_RMSE"].idxmin()]
    print(f"      mejor alpha = {best_ridge['alpha']} (val RMSE {best_ridge['Val_RMSE']:.4f})")

    # ---- MLP: búsqueda aleatoria --------------------------------------------------------
    n = int(s["n_mlp_configs"])
    configs = sample_mlp_configs(s["space"], n, seed)
    print(f"[3/3] MLP (sigmoide+BCE): {n} configuraciones")
    base_mlp = {k: cfg["mlp"][k] for k in ("loss", "epochs", "patience", "init_output_bias")}
    rows, t_mlp0 = [], time.time()
    for i, c in enumerate(configs, start=1):
        mcfg = {**base_mlp, **c}
        set_all_seeds(seed)  # misma semilla para todas: la diferencia viene de la configuración
        t0 = time.time()
        model, hist = train_lgd_mlp(Xtr, ytr, Xva, yva, mcfg, device="cpu", verbose=False)
        fit_s = time.time() - t0
        mv = evaluate_lgd_metrics(yva, predict_lgd_mlp(model, Xva))
        rows.append({"config": i, **{k: (str(v) if k == "hidden_dims" else v) for k, v in c.items()},
                     "Val_MAE": mv["mae"], "Val_RMSE": mv["rmse"],
                     "épocas": len(hist["val_rmse"]), "Parámetros": model.count_parameters(),
                     "Tiempo_s": round(fit_s, 2)})
        print(f"      [{i:02d}/{n}] val RMSE {mv['rmse']:.4f} | {c['optimizer']:<7} {str(c['hidden_dims']):<14} "
              f"{c['activation']:<5} drop {c['dropout']} lr {c['lr']} wd {c['weight_decay']} | {fit_s:.1f}s")
    mlp_total = time.time() - t_mlp0
    mlp_df = pd.DataFrame(rows)
    mlp_df.round(5).to_csv(os.path.join(out_tables, "search_mlp.csv"), index=False)
    best_i = int(mlp_df["Val_RMSE"].idxmin())
    best_cfg = configs[best_i]

    # ---- Presupuesto declarado y mejor configuración --------------------------------------
    budget = {
        "poblacion": population, "semilla": seed,
        "mlp": {"configuraciones": n, "tiempo_total_s": round(mlp_total, 1),
                "tiempo_medio_s": round(mlp_total / n, 2)},
        "ridge": {"configuraciones": int(s["ridge_alphas"]), "tiempo_total_s": round(ridge_total, 2)},
        "pruebas_manuales_previas_declaradas": int(s["declared_prior_trials"]),
        "criterio_de_seleccion": "menor RMSE de validación (el test no se usó)",
        "mejor_mlp": {**best_cfg, "val_rmse": float(mlp_df.loc[best_i, "Val_RMSE"])},
        "mejor_ridge": {"alpha": float(best_ridge["alpha"]), "val_rmse": float(best_ridge["Val_RMSE"])},
    }
    with open(os.path.join(out_tables, "search_budget.json"), "w", encoding="utf-8") as f:
        json.dump(budget, f, indent=2, ensure_ascii=False)

    best_yaml = {"baseline": {"alpha": float(best_ridge["alpha"])},
                 "mlp": {k: best_cfg[k] for k in ("hidden_dims", "activation", "dropout", "use_batchnorm",
                                                  "batch_size", "optimizer", "lr", "weight_decay")}}
    with open(s["best_config_path"], "w", encoding="utf-8") as f:
        f.write("# Mejor configuración de la búsqueda (menor RMSE de validación). Generado por\n"
                "# scripts/run_search_lgd.py; úsese con: python scripts/run_lgd.py --best\n")
        yaml.safe_dump(best_yaml, f, sort_keys=False, allow_unicode=True)

    pd.set_option("display.width", 220)
    print("\n" + "=" * 100)
    print("PRESUPUESTO DE BÚSQUEDA")
    print(f"  MLP  : {n} configuraciones | {mlp_total:.0f} s en total ({mlp_total / n:.1f} s por configuración)")
    print(f"  Ridge: {int(s['ridge_alphas'])} configuraciones | {ridge_total:.2f} s en total")
    print(f"  Pruebas manuales previas declaradas: {int(s['declared_prior_trials'])}")
    print("-" * 100)
    cols = ["config", "optimizer", "hidden_dims", "activation", "dropout", "lr", "weight_decay", "Val_RMSE"]
    print("Top 5 MLP por RMSE de validación:")
    print(mlp_df.sort_values("Val_RMSE").head(5)[cols].round(5).to_string(index=False))
    print(f"\nRidge: mejor alpha = {best_ridge['alpha']} (val RMSE {best_ridge['Val_RMSE']:.4f})")
    print(f"Mejor configuración guardada en {s['best_config_path']}")
    print("=" * 100)


if __name__ == "__main__":
    main()
