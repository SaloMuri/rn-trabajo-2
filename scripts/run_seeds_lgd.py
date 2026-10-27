"""30 semillas para la configuración final de las redes de LGD.

    python scripts/run_seeds_lgd.py                 # 30 semillas, configuración de lgd_best.yaml
    python scripts/run_seeds_lgd.py --demo          # 3 semillas, muestra de los datos
    python scripts/run_seeds_lgd.py --set seeds.n_seeds=10

Por cada semilla se entrenan el MLP sigmoide+BCE (C) y el de contraste sigmoide+MSE (B) con
la MISMA configuración, los mismos datos y las mismas particiones. La semilla cambia la
inicialización de pesos, el orden de los lotes y el dropout. Las referencias (constante y
regresión lineal) son deterministas: se calculan una sola vez (desviación = 0).
Reporta media, desviación estándar, mejor y peor valor de cada métrica.
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

from creditrisk.config import load_config
from creditrisk.experiments import prepare_lgd_data
from creditrisk.metrics import evaluate_lgd_metrics
from creditrisk.models.baseline_lgd import ConstantLGDModel, LinearLGDModel
from creditrisk.models.mlp_lgd import predict_lgd_mlp, train_lgd_mlp
from creditrisk.seeding import set_all_seeds

METRICS = ["Val_MAE", "Val_RMSE", "Test_MAE", "Test_RMSE"]


def summarize(df: pd.DataFrame, model: str) -> list:
    rows = []
    for m in METRICS:
        v = df[m].to_numpy()
        rows.append({"Modelo": model, "Métrica": m, "Media": v.mean(),
                     "Desv_est": v.std(ddof=1) if len(v) > 1 else 0.0,
                     "Mejor": v.min(), "Peor": v.max(), "n": len(v)})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo", action="store_true", help="3 semillas y muestra de los datos")
    p.add_argument("--best", action=argparse.BooleanOptionalAction, default=True,
                   help="usa configs/lgd_best.yaml (por defecto sí)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="RUTA=VALOR")
    p.add_argument("--base", default="configs/base.yaml")
    p.add_argument("--config", default="configs/lgd.yaml")
    a = p.parse_args()

    paths = [a.base, a.config]
    if a.best:
        paths.append("configs/lgd_best.yaml")
    overrides = list(a.overrides)
    if a.demo:
        paths.append("configs/lgd_demo.yaml")
        overrides.append("seeds.n_seeds=3")
    cfg = load_config(paths, overrides)

    population = cfg["lgd_data"]["population"]
    n_seeds, first = int(cfg["seeds"]["n_seeds"]), int(cfg["seeds"]["first_seed"])
    out_tables = os.path.join(cfg["output"]["tables_dir"], population)
    os.makedirs(out_tables, exist_ok=True)

    print(f"[1/3] Datos LGD (población: {population})")
    _, data, _ = prepare_lgd_data(cfg)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data["train"], data["val"], data["test"]
    print(f"      train {len(ytr):,} | val {len(yva):,} | test {len(yte):,}")
    print(f"      configuración MLP: { {k: cfg['mlp'][k] for k in ('hidden_dims','activation','dropout','use_batchnorm','batch_size','optimizer','lr','weight_decay')} }")

    def metrics_of(pva, pte):
        mv, mt = evaluate_lgd_metrics(yva, pva), evaluate_lgd_metrics(yte, pte)
        return {"Val_MAE": mv["mae"], "Val_RMSE": mv["rmse"], "Test_MAE": mt["mae"], "Test_RMSE": mt["rmse"]}

    print("[2/3] Referencias deterministas")
    const = ConstantLGDModel().fit(Xtr, ytr)
    lin = LinearLGDModel(clip=bool(cfg["baseline"].get("clip", True)), alpha=cfg["baseline"].get("alpha")).fit(Xtr, ytr)
    ref_rows = {"Constante (media train)": metrics_of(const.predict(Xva), const.predict(Xte)),
                "Regresión lineal (referencia)": metrics_of(lin.predict(Xva), lin.predict(Xte))}

    runs = [("MLP sigmoide+BCE (C)", dict(cfg["mlp"])),
            ("MLP sigmoide+MSE (B, contraste)", {**cfg["mlp"], **cfg["contrast_mlp"]})]
    raw, params, t_start = [], {}, time.time()
    print(f"[3/3] {n_seeds} semillas x {len(runs)} redes = {n_seeds * len(runs)} entrenamientos")
    for i in range(n_seeds):
        seed = first + i
        for name, mcfg in runs:
            set_all_seeds(seed)
            t0 = time.time()
            model, hist = train_lgd_mlp(Xtr, ytr, Xva, yva, mcfg, device="cpu", verbose=False)
            fit_s = time.time() - t0
            params[name] = model.count_parameters()
            raw.append({"Modelo": name, "semilla": seed, **metrics_of(predict_lgd_mlp(model, Xva), predict_lgd_mlp(model, Xte)),
                        "épocas": len(hist["val_rmse"]), "Tiempo_s": round(fit_s, 2)})
        last = {r["Modelo"]: r for r in raw[-len(runs):]}
        print(f"      semilla {seed:>3} | " + " | ".join(f"{k[:14]}: test RMSE {v['Test_RMSE']:.4f}" for k, v in last.items()))
    total_s = time.time() - t_start

    raw_df = pd.DataFrame(raw)
    raw_df.round(5).to_csv(os.path.join(out_tables, "seeds_raw_results.csv"), index=False)

    rows = []
    for name, r in ref_rows.items():
        rows += [{"Modelo": name, "Métrica": m, "Media": r[m], "Desv_est": 0.0, "Mejor": r[m], "Peor": r[m], "n": 1}
                 for m in METRICS]
    for name, _ in runs:
        rows += summarize(raw_df[raw_df["Modelo"] == name], name)
    summary = pd.DataFrame(rows)
    summary["Parámetros"] = summary["Modelo"].map({**params, "Constante (media train)": 1,
                                                  "Regresión lineal (referencia)": lin.count_parameters()})
    summary.round(5).to_csv(os.path.join(out_tables, "seeds_summary.csv"), index=False)

    # Comparaciones pareadas (misma semilla): ¿cuántas veces la red supera a la referencia?
    c, b = (raw_df[raw_df["Modelo"] == n].set_index("semilla") for n, _ in runs)
    lin_t = ref_rows["Regresión lineal (referencia)"]
    extra = {
        "n_semillas": n_seeds, "tiempo_total_s": round(total_s, 1),
        "tiempo_medio_por_entrenamiento_s": round(float(raw_df["Tiempo_s"].mean()), 2),
        "C_gana_a_lineal_test_rmse": f"{int((c['Test_RMSE'] < lin_t['Test_RMSE']).sum())}/{n_seeds}",
        "C_gana_a_lineal_test_mae": f"{int((c['Test_MAE'] < lin_t['Test_MAE']).sum())}/{n_seeds}",
        "C_gana_a_B_test_rmse": f"{int((c['Test_RMSE'] < b['Test_RMSE']).sum())}/{n_seeds}",
        "diferencia_media_C_menos_B_test_rmse": float((c["Test_RMSE"] - b["Test_RMSE"]).mean()),
        "desv_est_de_la_diferencia_C_menos_B": float((c["Test_RMSE"] - b["Test_RMSE"]).std(ddof=1)) if n_seeds > 1 else 0.0,
    }
    with open(os.path.join(out_tables, "seeds_extra.json"), "w", encoding="utf-8") as f:
        json.dump(extra, f, indent=2, ensure_ascii=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 100)
    print(f"RESULTADOS CON {n_seeds} SEMILLAS ({population}) — media ± desviación estándar [mejor, peor]")
    print("=" * 100)
    for model in summary["Modelo"].unique():
        s = summary[summary["Modelo"] == model].set_index("Métrica")
        print(f"{model:34s} (parámetros: {int(s['Parámetros'].iloc[0]):>6,})")
        for m in METRICS:
            r = s.loc[m]
            print(f"    {m:10s} {r['Media']:.4f} ± {r['Desv_est']:.4f}   [{r['Mejor']:.4f}, {r['Peor']:.4f}]")
    print("-" * 100)
    for k, v in extra.items():
        print(f"  {k}: {v}")
    print("=" * 100)


if __name__ == "__main__":
    main()
