"""Regenera las figuras y tablas de interpretabilidad de la Parte 2 (LGD) desde el código.

    python scripts/make_figures_lgd.py

Requiere haber ejecutado antes (en este orden): run_lgd.py --best, run_search_lgd.py,
run_seeds_lgd.py y run_expected_loss.py. Las figuras se guardan en reports/figures/lgd/.
Color por entidad (igual en todas las figuras): observada = gris, lineal = verde azulado,
MLP sigmoide+BCE (C) = azul, MLP sigmoide+MSE (B) = naranja.
"""
import os
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401
import numpy as np
import pandas as pd

from creditrisk.config import load_config
from creditrisk.experiments import load_lgd_splits
from creditrisk.lgd_interpretability import group_features, grouped_permutation_importance
from creditrisk.models.baseline_lgd import LinearLGDModel
from creditrisk.models.mlp import load_mlp_checkpoint
from creditrisk.models.mlp_lgd import predict_lgd_mlp

# --- Paleta de referencia (validada: 3 primeras ranuras, todos los pares, modo claro) ---
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
C_OBS, C_LIN, C_BCE, C_MSE = INK2, "#1baf7a", "#2a78d6", "#eb6834"
SOURCE = "Fuente: elaboración propia con datos de Lending Club (Kaggle, ranadeep/credit-risk-dataset)."

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9.5, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "lines.linewidth": 1.8, "legend.frameon": False,
})


def finish(fig, title: str, path: str, note: str = "") -> None:
    w = fig.get_figwidth()
    t = textwrap.fill(title, int(w * 8.3)); n_t = t.count("\n") + 1
    foot = [textwrap.fill(note, int(w * 15))] if note else []
    foot.append(textwrap.fill(SOURCE, int(w * 15)))
    foot_txt = "\n".join(foot); n_f = foot_txt.count("\n") + 1
    fig.suptitle(t, x=0.01, ha="left", fontsize=11.5, fontweight="bold", color=INK)
    fig.text(0.01, 0.006, foot_txt, fontsize=7.5, color=MUTED, ha="left", va="bottom", linespacing=1.3)
    fig.tight_layout(rect=(0, 0.018 + 0.028 * n_f, 1, 0.97 - 0.032 * n_t))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("  ", path)


def label_end(ax, x, y, text, color, dx=0.0, dy=0.0, ha="left"):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", color=color,
                fontsize=8.5, fontweight="bold", ha=ha, va="center")


def main() -> None:
    cfg = load_config(["configs/base.yaml", "configs/lgd.yaml", "configs/lgd_best.yaml"])
    T = os.path.join(cfg["output"]["tables_dir"], "main")
    models_dir = os.path.join(cfg["output"]["models_dir"], "main")
    out = "reports/figures/lgd"
    os.makedirs(out, exist_ok=True)
    comp = pd.read_csv(os.path.join(T, "comparison_lgd.csv")).set_index("Modelo")
    lin_val = comp.loc["Regresión lineal (referencia)", "Val_RMSE"]
    const_val = comp.loc["Constante (media train)", "Val_RMSE"]

    # ---------------- L1: distribución del objetivo y censura por año ----------------
    d = pd.read_parquet(cfg["lgd_data"]["paths"]["main"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4))
    edges = np.linspace(0, 1.0, 51)
    cnt, _ = np.histogram(d.loc[d["lgd"] < 1, "lgd"], bins=edges)
    a.bar(edges[:-1], cnt / len(d) * 100, width=0.02, align="edge", color=C_BCE, edgecolor=SURFACE, linewidth=0.8)
    share1 = (d["lgd"] >= 1).mean() * 100
    a.bar([1.06], [share1], width=0.05, color=C_BCE, edgecolor=SURFACE, linewidth=0.8)   # LGD = 1 exacto, aparte
    a.annotate(f"{share1:.1f} % tiene\nLGD = 1 exacto\n(no se recuperó nada)", (1.06, share1), xytext=(-22, -6),
               textcoords="offset points", ha="right", va="top", fontsize=8.5, color=INK)
    a.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.06]); a.set_xticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0", "= 1"])
    a.set_xlabel("LGD observada"); a.set_ylabel("% de créditos (barras de 0.02)")
    a.set_title("(a) Distribución de la LGD", loc="left", fontsize=10, color=INK)
    by = d.groupby("issue_year")["lgd"].agg(["mean", "size"])
    b.plot(by.index, by["mean"], color=C_BCE, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
    b.set_xticks(by.index); b.set_xticklabels([f"{y}\nn={n:,}" for y, n in zip(by.index, by["size"])], fontsize=7.5)
    b.set_ylabel("LGD media"); b.set_ylim(0.9, 1.0)
    label_end(b, by.index[-1], by["mean"].iloc[-1], f"{by['mean'].iloc[-1]:.3f}", C_BCE, dx=-6, dy=9, ha="right")
    b.set_title("(b) LGD media por año de originación", loc="left", fontsize=10, color=INK)
    finish(fig, "Figura L1. La LGD se concentra en 1 y sube en los créditos más recientes", f"{out}/figL1_distribucion_lgd.png",
           "Créditos castigados (Charged Off), n = 46,009. El aumento hacia 2015 refleja censura: el cobro de esos créditos aún no termina.")

    # ---------------- L2: curvas de entrenamiento ----------------
    hb = pd.read_csv(os.path.join(T, "mlp_bce_history.csv")); hm = pd.read_csv(os.path.join(T, "mlp_mse_history.csv"))
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for h, col, nm in ((hb, C_BCE, "MLP sigmoide+BCE (C)"), (hm, C_MSE, "MLP sigmoide+MSE (B)")):
        ep = np.arange(1, len(h) + 1)
        ax.plot(ep, h["val_rmse"], color=col, label=nm)
        i = int(h["val_rmse"].idxmin())
        ax.plot(ep[i], h["val_rmse"][i], "o", color=col, markersize=11 if col == C_BCE else 6, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5)
    ax.axhline(lin_val, color=C_LIN, linestyle="--", linewidth=1.4); ax.axhline(const_val, color=MUTED, linestyle=":", linewidth=1.4)
    xmax = len(hb); ax.set_xlim(0.5, xmax + 6.5)
    label_end(ax, xmax + 0.5, lin_val, f"Regresión lineal\n({lin_val:.4f})", C_LIN)
    label_end(ax, xmax + 0.5, const_val, f"Constante\n({const_val:.4f})", INK2)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True)); ax.set_xlabel("Época"); ax.set_ylabel("RMSE de validación"); ax.legend(loc="lower right", fontsize=8.5)
    finish(fig, "Figura L2. El RMSE de validación oscila entre épocas: elegir la mejor época es en parte suerte",
           f"{out}/figL2_curvas_entrenamiento.png", "Configuración final, semilla 42. El círculo marca la mejor época (mínimo RMSE de validación); la parada temprana espera 8 épocas más.")

    # ---------------- L3: distribución predicha vs observada (test) ----------------
    pr = pd.read_csv(os.path.join(T, "test_predictions.csv"))
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for col, c, nm in (("y_true", C_OBS, "Observada"), ("Regresión lineal (referencia)", C_LIN, "Regresión lineal"),
                       ("MLP sigmoide+BCE (C)", C_BCE, "MLP sigmoide+BCE (C)")):
        v = np.sort(pr[col].to_numpy()); ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=c, label=nm)
    ax.set_xlim(0.78, 1.005); ax.set_xlabel("LGD"); ax.set_ylabel("Fracción acumulada de créditos")
    obs1 = (pr["y_true"] >= 0.99).mean() * 100
    ax.annotate(f"{obs1:.0f} % de los créditos observados\ntiene LGD ≥ 0.99; la red no predice\nnada por encima de {pr['MLP sigmoide+BCE (C)'].max():.2f}", (0.995, 0.35),
                ha="right", fontsize=8.5, color=INK)
    ax.legend(loc="upper left", fontsize=8.5)
    finish(fig, "Figura L3. Las predicciones son estrechas: no reproducen la masa en LGD = 1",
           f"{out}/figL3_distribucion_pred_vs_obs.png", "Función de distribución acumulada en el test (n = 2,773, originados en 2015); eje horizontal recortado a [0.78, 1].")

    # ---------------- L4: 30 semillas ----------------
    sd = pd.read_csv(os.path.join(T, "seeds_raw_results.csv")); ss = pd.read_csv(os.path.join(T, "seeds_summary.csv"))
    ref = lambda m, met: float(ss[(ss["Modelo"] == m) & (ss["Métrica"] == met)]["Media"].iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    rng = np.random.default_rng(0)
    for ax, met, ttl in zip(axes, ("Test_RMSE", "Test_MAE"), ("(a) RMSE en test", "(b) MAE en test")):
        for k, (nm, col) in enumerate((("MLP sigmoide+BCE (C)", C_BCE), ("MLP sigmoide+MSE (B, contraste)", C_MSE))):
            v = sd[sd["Modelo"] == nm][met].to_numpy()
            ax.scatter(k + rng.uniform(-0.12, 0.12, len(v)), v, s=40, color=col, edgecolor=SURFACE, linewidth=0.8, alpha=0.9, zorder=3)
            ax.hlines(v.mean(), k - 0.25, k + 0.25, color=INK, linewidth=2, zorder=4)
        lin, con = ref("Regresión lineal (referencia)", met), ref("Constante (media train)", met)
        ax.axhline(lin, color=C_LIN, linestyle="--", linewidth=1.4); ax.axhline(con, color=MUTED, linestyle=":", linewidth=1.4)
        label_end(ax, 1.45, lin, f"Lineal {lin:.4f}", C_LIN, dx=0, dy=7 if met == "Test_RMSE" else 7, ha="right")
        label_end(ax, 1.45, con, f"Constante {con:.4f}", INK2, dx=0, dy=-7 if met == "Test_RMSE" else 7, ha="right")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["MLP sigmoide+BCE (C)", "MLP sigmoide+MSE (B)"]); ax.set_xlim(-0.5, 1.5)
        ax.set_ylabel(met.replace("Test_", "")); ax.set_title(ttl, loc="left", fontsize=10, color=INK)
    finish(fig, "Figura L4. Con 30 semillas ambas redes superan en promedio a la lineal; entre C y B la ventaja es pequeña",
           f"{out}/figL4_semillas.png", "Cada punto es una semilla (n = 30); la raya negra es la media. Las referencias son deterministas.")

    # ---------------- L5: búsqueda de hiperparámetros ----------------
    s = pd.read_csv(os.path.join(T, "search_mlp.csv"))
    order = s.groupby("optimizer")["Val_RMSE"].median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for k, o in enumerate(order):
        v = s[s["optimizer"] == o]["Val_RMSE"].to_numpy()
        ax.scatter(v, k + rng.uniform(-0.15, 0.15, len(v)), s=36, color=C_BCE, edgecolor=SURFACE, linewidth=0.8, alpha=0.9, zorder=3)
    bi = s["Val_RMSE"].idxmin(); ko = order.index(s.loc[bi, "optimizer"])
    ax.scatter([s.loc[bi, "Val_RMSE"]], [ko], s=120, facecolors="none", edgecolors=INK, linewidths=1.6, zorder=4)
    ax.annotate("mejor configuración", (s.loc[bi, "Val_RMSE"], ko), xytext=(8, 12), textcoords="offset points", fontsize=8.5, color=INK)
    ax.axvline(lin_val, color=C_LIN, linestyle="--", linewidth=1.4); ax.axvline(const_val, color=MUTED, linestyle=":", linewidth=1.4)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([f"{o} (n={int((s['optimizer'] == o).sum())})" for o in order])
    ax.set_xlabel("RMSE de validación"); ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(order) - 0.1)
    label_end(ax, lin_val, len(order) - 0.3, "Lineal", C_LIN, dx=-4, ha="right"); label_end(ax, const_val, len(order) - 0.3, "Constante", INK2, dx=4)
    finish(fig, "Figura L5. SGD con momento rinde peor; los demás optimizadores quedan muy cerca",
           f"{out}/figL5_busqueda_hiperparametros.png", "60 configuraciones aleatorias del MLP (sigmoide+BCE), semilla 42, RMSE de validación.")

    # ---------------- L6: pérdida esperada por deciles ----------------
    dec = pd.read_csv(os.path.join(T, "expected_loss_deciles_B.csv"))
    fig, ax = plt.subplots(figsize=(8.5, 4.4)); w = 0.4; x = dec["decil"].to_numpy()
    ax.bar(x - w / 2, dec["el_predicha_media"], w, color=C_BCE, edgecolor=SURFACE, linewidth=1.2, label="Pérdida esperada (PD·LGD·EAD)")
    ax.bar(x + w / 2, dec["perdida_realizada_media"], w, color=C_OBS, edgecolor=SURFACE, linewidth=1.2, label="Pérdida realizada")
    ax.set_xticks(x); ax.set_xlabel("Decil de pérdida esperada (1 = menor riesgo)"); ax.set_ylabel("Dólares por crédito")
    ax.legend(loc="upper left", fontsize=8.5)
    finish(fig, "Figura L6. La pérdida esperada ordena bien el riesgo, pero sobreestima su nivel",
           f"{out}/figL6_perdida_esperada_deciles.png", "Test (n = 25,757 créditos con pérdida conocida). EAD opción B. La PD de la Parte 1 incluye Late y Default como malos.")

    # ---------------- L7: interpretabilidad ----------------
    splits, _ = load_lgd_splits(cfg["lgd_data"]["paths"]["main"], cfg["paths"]["processed_pd_dir"], "parquet")
    pre = joblib.load(os.path.join(models_dir, "preprocessor.joblib"))
    names = list(pre.get_feature_names_out())
    Xva, yva = pre.transform(splits["val"]).astype(np.float32), splits["val"]["lgd"].to_numpy(dtype=np.float32)
    groups = group_features(names, cfg["features_pd"]["categorical_nominal"])
    mlp = load_mlp_checkpoint(os.path.join(models_dir, "mlp_bce.pt")); lin = LinearLGDModel.load(os.path.join(models_dir, "linear_baseline.joblib"))
    imp_m = grouped_permutation_importance(lambda M: predict_lgd_mlp(mlp, M), Xva, yva, groups, n_repeats=10)
    imp_l = grouped_permutation_importance(lambda M: lin.predict(M), Xva, yva, groups, n_repeats=10)
    imp = imp_m.merge(imp_l, on="variable", suffixes=("_mlp", "_lineal"))
    imp.round(5).to_csv(os.path.join(T, "feature_importance_lgd.csv"), index=False)
    coef = pd.DataFrame({"variable": [n.split("__", 1)[1] for n in names], "coef_estandarizado": lin.model.coef_})
    coef = coef.reindex(coef["coef_estandarizado"].abs().sort_values(ascending=False).index)
    coef.round(5).to_csv(os.path.join(T, "ridge_coefficients_lgd.csv"), index=False)

    top = imp.head(10).iloc[::-1].reset_index(drop=True)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    y = np.arange(len(top)); h = 0.38
    a.barh(y + h / 2, top["aumento_rmse_media_mlp"], h, xerr=top["aumento_rmse_std_mlp"], color=C_BCE, edgecolor=SURFACE, linewidth=1.2,
           error_kw={"elinewidth": 1, "ecolor": INK2, "capsize": 2}, label="MLP sigmoide+BCE (C)")
    a.barh(y - h / 2, top["aumento_rmse_media_lineal"], h, xerr=top["aumento_rmse_std_lineal"], color=C_LIN, edgecolor=SURFACE, linewidth=1.2,
           error_kw={"elinewidth": 1, "ecolor": INK2, "capsize": 2}, label="Regresión lineal")
    a.set_yticks(y); a.set_yticklabels(top["variable"]); a.set_xlabel("Aumento del RMSE de validación al barajar la variable")
    a.legend(loc="lower right", fontsize=8.5); a.grid(axis="y", visible=False)
    a.set_title("(a) Importancia por permutación (10 repeticiones)", loc="left", fontsize=10, color=INK)
    num = coef[coef["variable"].isin([n.split("__", 1)[1] for n in names if n.startswith("num__")])].head(10).iloc[::-1]
    b.barh(np.arange(len(num)), num["coef_estandarizado"], 0.6, color=C_LIN, edgecolor=SURFACE, linewidth=1.2)
    b.axvline(0, color=INK2, linewidth=1); b.set_yticks(np.arange(len(num))); b.set_yticklabels(num["variable"]); b.grid(axis="y", visible=False)
    b.set_xlabel("Coeficiente sobre variables estandarizadas (+ = mayor LGD)")
    b.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5)); a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    b.set_title("(b) Regresión lineal: dirección del efecto", loc="left", fontsize=10, color=INK)
    finish(fig, "Figura L7. Qué variables usa cada modelo para predecir la LGD", f"{out}/figL7_interpretabilidad.png",
           "Validación. Con variables correlacionadas (p. ej. grado y tasa de interés) los coeficientes individuales son inestables.")
    print("Tablas: feature_importance_lgd.csv y ridge_coefficients_lgd.csv en", T)


if __name__ == "__main__":
    main()
