"""Experimento LGD: modelo de referencia + MLP. Un solo comando.

    python scripts/run_lgd.py                              # versión principal (Charged Off)
    python scripts/run_lgd.py --population with_default    # sensibilidad (+ Default)
    python scripts/run_lgd.py --demo                       # < 2 min, con muestra de los datos
    python scripts/run_lgd.py --set mlp.lr=0.003 --set mlp.hidden_dims=[128,64]

Entrena, sobre los mismos datos y las mismas particiones temporales de la Parte 1:
constante (suelo) · regresión lineal (referencia) · MLP sigmoide+BCE (opción C, principal)
· MLP sigmoide+MSE (opción B, contraste).
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch

from creditrisk.config import load_config
from creditrisk.experiments import prepare_lgd_data
from creditrisk.lgd_metrics import compare_distributions, distribution_summary
from creditrisk.metrics import evaluate_lgd_metrics
from creditrisk.models.baseline_lgd import ConstantLGDModel, LinearLGDModel
from creditrisk.models.mlp import save_mlp_checkpoint
from creditrisk.models.mlp_lgd import predict_lgd_mlp, train_lgd_mlp
from creditrisk.seeding import set_all_seeds


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--population", choices=["main", "with_default"], default=None)
    p.add_argument("--demo", action="store_true", help="modo demostración (< 2 min)")
    p.add_argument("--best", action="store_true",
                   help="usa la mejor configuración hallada por la búsqueda (configs/lgd_best.yaml)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="RUTA=VALOR",
                   help="cambia un parámetro de la configuración, p. ej. mlp.lr=0.003")
    p.add_argument("--base", default="configs/base.yaml")
    p.add_argument("--config", default="configs/lgd.yaml")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    paths = [a.base, a.config]
    if a.best:
        paths.append("configs/lgd_best.yaml")
    if a.demo:
        paths.append("configs/lgd_demo.yaml")
    overrides = list(a.overrides)
    if a.population:
        overrides.append(f"lgd_data.population={a.population}")
    cfg = load_config(paths, overrides)

    seed = int(cfg["seed"])
    population = cfg["lgd_data"]["population"]
    out_models = os.path.join(cfg["output"]["models_dir"], population)
    out_tables = os.path.join(cfg["output"]["tables_dir"], population)
    os.makedirs(out_models, exist_ok=True)
    os.makedirs(out_tables, exist_ok=True)
    set_all_seeds(seed)

    # ---- 1. Datos: mismas particiones temporales que la Parte 1 -------------------------
    print(f"[1/4] Datos LGD (población: {population}; semilla: {seed})")
    splits, data, info = prepare_lgd_data(cfg, os.path.join(out_models, "preprocessor.joblib"))
    if info["sin_features"]:
        print(f"      AVISO: {info['sin_features']:,} créditos con LGD no están en las tablas de PD "
              f"(sin variables) y se descartan.")
    # Para comparar poblaciones en igualdad de condiciones: métricas también sobre el test
    # SIN créditos Default (el test de la versión principal).
    mask_te = (splits["test"]["loan_status"] != "Default").to_numpy()
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data["train"], data["val"], data["test"]
    print("      " + " | ".join(f"{s}: {len(data[s][1]):,}" for s in ("train", "val", "test"))
          + f" | variables tras preprocesar: {Xtr.shape[1]}")

    results, preds, histories = [], {}, {}

    def register(name, model_params, fit_s, pva, pte):
        mv, mt = evaluate_lgd_metrics(yva, pva), evaluate_lgd_metrics(yte, pte)
        mt0 = evaluate_lgd_metrics(yte[mask_te], pte[mask_te])
        results.append({"Modelo": name, "Parámetros": model_params, "Tiempo_s": round(fit_s, 2),
                        "Val_MAE": mv["mae"], "Val_RMSE": mv["rmse"],
                        "Test_MAE": mt["mae"], "Test_RMSE": mt["rmse"],
                        "Test_MAE_sinDefault": mt0["mae"], "Test_RMSE_sinDefault": mt0["rmse"]})
        preds[name] = pte

    # ---- 2. Referencias -----------------------------------------------------------------
    print("[2/4] Referencias: constante y regresión lineal")
    t0 = time.time()
    const = ConstantLGDModel().fit(Xtr, ytr)
    register("Constante (media train)", const.count_parameters(), time.time() - t0,
             const.predict(Xva), const.predict(Xte))

    t0 = time.time()
    alpha = cfg["baseline"].get("alpha")
    lin = LinearLGDModel(clip=bool(cfg["baseline"].get("clip", True)), alpha=alpha).fit(Xtr, ytr)
    lin_s = time.time() - t0
    lin.save(os.path.join(out_models, "linear_baseline.joblib"))
    register("Regresión lineal (referencia)", lin.count_parameters(), lin_s, lin.predict(Xva), lin.predict(Xte))
    print(f"      referencia lineal: {'OLS' if alpha is None else f'Ridge alpha={alpha}'}")

    # ---- 3. MLP: opción C (principal) y opción B (contraste) ----------------------------
    runs = [("MLP sigmoide+BCE (C)", dict(cfg["mlp"]), "mlp_bce"),
            ("MLP sigmoide+MSE (B, contraste)", {**cfg["mlp"], **cfg["contrast_mlp"]}, "mlp_mse")]
    for i, (name, mcfg, tag) in enumerate(runs, start=1):
        print(f"[3/4] Entrenando {name}")
        set_all_seeds(seed)  # misma inicialización y mismo orden de lotes en ambas pérdidas
        t0 = time.time()
        model, hist = train_lgd_mlp(Xtr, ytr, Xva, yva, mcfg, device="cpu", verbose=True)
        fit_s = time.time() - t0
        save_mlp_checkpoint(model, os.path.join(out_models, f"{tag}.pt"))
        histories[tag] = pd.DataFrame(hist)
        histories[tag].to_csv(os.path.join(out_tables, f"{tag}_history.csv"), index=False)
        register(name, model.count_parameters(), fit_s, predict_lgd_mlp(model, Xva), predict_lgd_mlp(model, Xte))

    # ---- 4. Resultados ------------------------------------------------------------------
    print("[4/4] Guardando tablas")
    comp = pd.DataFrame(results)
    comp.round(4).to_csv(os.path.join(out_tables, "comparison_lgd.csv"), index=False)

    pd.DataFrame({"y_true": yte, **{k: v for k, v in preds.items()}}).to_csv(
        os.path.join(out_tables, "test_predictions.csv"), index=False)

    dist_rows = [{"Serie": "Observada (test)", **distribution_summary(yte), "wasserstein": 0.0, "ks": 0.0}]
    for name, p in preds.items():
        dist_rows.append({"Serie": name, **distribution_summary(p), **compare_distributions(yte, p)})
    pd.DataFrame(dist_rows).round(4).to_csv(os.path.join(out_tables, "distribution_test.csv"), index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 100)
    print(f"RESULTADOS LGD ({population}) — test = créditos originados después de {cfg['temporal_split']['val_end']}")
    print("=" * 100)
    print(comp.round(4).to_string(index=False))
    print("-" * 100)
    print(pd.DataFrame(dist_rows).round(3).to_string(index=False))
    print("=" * 100)


if __name__ == "__main__":
    main()
