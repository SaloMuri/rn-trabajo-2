# scripts/make_figures.py
"""Regenera todas las figuras del reporte desde el código."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_curve

from creditrisk.metrics import get_calibration_data

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


def load_split_df(processed_dir: str, split_name: str, fmt: str) -> pd.DataFrame:
    path = os.path.join(processed_dir, f"{split_name}.{fmt}")
    return pd.read_parquet(path) if fmt == "parquet" else pd.read_csv(path, low_memory=False)


def plot_eda_hypotheses(train_df: pd.DataFrame, out_dir: str) -> None:
    """Figura 1: Panel de validación visual de las 3 hipótesis en el análisis descriptivo."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # Panel A: NUEVA Hipótesis 1 - Tasa de interés asignada según variable nominal (purpose)
    top_purposes_rate = train_df["purpose"].value_counts().head(7).index
    df_rate = train_df[train_df["purpose"].isin(top_purposes_rate)]
    purp_rate = df_rate.groupby("purpose")["int_rate"].mean().sort_values(ascending=True)
    
    axes[0].barh(purp_rate.index, purp_rate.values, color="#1f77b4", edgecolor="black", alpha=0.85)
    axes[0].set_title("H1: Tasa de Interés Promedio por Propósito")
    axes[0].set_xlabel("Tasa de Interés Promedio (%)")
    axes[0].set_ylabel("Propósito del Crédito")

    # Panel B: Hipótesis 2 - Interacción Plazo (36 vs 60 meses) y Cuartiles de DTI
    df_dti = train_df.dropna(subset=["dti", "term"]).copy()
    df_dti["dti_bin"] = pd.qcut(df_dti["dti"], q=4, labels=["Q1 (Bajo)", "Q2", "Q3", "Q4 (Alto)"])
    piv = df_dti.pivot_table(index="dti_bin", columns="term", values="target_pd", aggfunc="mean", observed=False) * 100
    x = np.arange(len(piv.index))
    width = 0.35
    if 36.0 in piv.columns and 60.0 in piv.columns:
        axes[1].bar(x - width / 2, piv[36.0], width, label="36 meses", color="#2ca02c", alpha=0.85)
        axes[1].bar(x + width / 2, piv[60.0], width, label="60 meses", color="#d62728", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(piv.index)
    axes[1].legend(title="Plazo (term)")
    axes[1].set_title("H2: Impago según Cuartil de DTI y Plazo")
    axes[1].set_xlabel("Cuartiles de Relación Deuda-Ingreso (DTI)")
    axes[1].set_ylabel("Tasa de Incumplimiento (%)")

    # Panel C: Hipótesis 3 - Tasa de incumplimiento por Propósito (Top 7 más frecuentes)
    top_purposes = train_df["purpose"].value_counts().head(7).index
    df_p = train_df[train_df["purpose"].isin(top_purposes)]
    purp_pd = (df_p.groupby("purpose")["target_pd"].mean() * 100).sort_values(ascending=True)
    axes[2].barh(purp_pd.index, purp_pd.values, color="#ff7f0e", edgecolor="black", alpha=0.85)
    axes[2].set_title("H3: Tasa de Impago por Propósito")
    axes[2].set_xlabel("Tasa de Incumplimiento (%)")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig1_eda_hipotesis.png"), bbox_inches="tight")
    plt.close()


def plot_training_history(out_dir: str) -> None:
    """Figura 2: Curvas de pérdida y evolución del AUC en validación del MLP."""
    hist_path = "reports/tables/pd/mlp_training_history.csv"
    if not os.path.exists(hist_path):
        return
    df_h = pd.read_csv(hist_path)
    epochs = np.arange(1, len(df_h) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, df_h["train_loss"], marker="o", label="Train Loss (BCE)", color="#1f77b4")
    axes[0].plot(epochs, df_h["val_loss"], marker="s", label="Val Loss (BCE)", color="#d62728")
    axes[0].set_title("Evolución de la Pérdida (Entropía Cruzada Binaria)")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("BCE Loss")
    axes[0].legend()

    axes[1].plot(epochs, df_h["val_auc"], marker="^", color="#2ca02c", linewidth=2, label="Val AUC-ROC")
    best_ep = int(np.argmax(df_h["val_auc"])) + 1
    best_auc = df_h["val_auc"].max()
    axes[1].axvline(best_ep, linestyle="--", color="gray", label=f"Mejor época ({best_ep}: AUC={best_auc:.4f})")
    axes[1].set_title("Evolución del AUC-ROC en Validación (Early Stopping)")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("AUC-ROC")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_curvas_entrenamiento_mlp.png"), bbox_inches="tight")
    plt.close()


def plot_roc_and_ks(preds_df: pd.DataFrame, out_dir: str) -> None:
    """Figura 3: Comparación de Curvas ROC y Separación KS en Test Out-of-Time."""
    y_true = preds_df["y_true"].values
    p_lr = preds_df["prob_logreg"].values
    p_mlp = preds_df["prob_mlp"].values

    fpr_lr, tpr_lr, _ = roc_curve(y_true, p_lr)
    fpr_mlp, tpr_mlp, _ = roc_curve(y_true, p_mlp)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ROC Curve
    axes[0].plot(fpr_lr, tpr_lr, label="Regresión Logística (AUC = 0.6899)", color="#1f77b4", lw=2)
    axes[0].plot(fpr_mlp, tpr_mlp, label="Perceptrón Multicapa (AUC = 0.6983)", color="#d62728", lw=2)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Azar (AUC = 0.50)")
    axes[0].set_title("Curva ROC en Conjunto de Prueba (Out-of-Time)")
    axes[0].set_xlabel("Tasa de Falsos Positivos (FPR)")
    axes[0].set_ylabel("Tasa de Verdaderos Positivos (TPR)")
    axes[0].legend(loc="lower right")

    # KS Curve para el MLP
    thresholds = np.linspace(0, 1, 200)
    probs_good = p_mlp[y_true == 0]
    probs_bad = p_mlp[y_true == 1]
    cdf_good = [np.mean(probs_good <= t) for t in thresholds]
    cdf_bad = [np.mean(probs_bad <= t) for t in thresholds]
    diffs = np.abs(np.array(cdf_good) - np.array(cdf_bad))
    max_idx = int(np.argmax(diffs))
    ks_val = diffs[max_idx]
    best_t = thresholds[max_idx]

    axes[1].plot(thresholds, cdf_good, label="CDF Buenos Pagadores (y=0)", color="#2ca02c", lw=2)
    axes[1].plot(thresholds, cdf_bad, label="CDF Malos Pagadores (y=1)", color="#d62728", lw=2)
    axes[1].vlines(
        best_t,
        cdf_bad[max_idx],
        cdf_good[max_idx],
        colors="black",
        linestyles="--",
        lw=2,
        label=f"KS = {ks_val:.4f} (umbral p={best_t:.2f})",
    )
    axes[1].set_title("Curva Kolmogorov-Smirnov (KS) del MLP en Test")
    axes[1].set_xlabel("Probabilidad de Incumplimiento Predicha (PD)")
    axes[1].set_ylabel("Probabilidad Acumulada (CDF)")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3_roc_y_ks.png"), bbox_inches="tight")
    plt.close()


def plot_calibration(preds_df: pd.DataFrame, out_dir: str) -> None:
    """Figura 4: Curva de Calibración (Reliability Diagram) en Test Out-of-Time."""
    y_true = preds_df["y_true"].values
    p_lr = preds_df["prob_logreg"].values
    p_mlp = preds_df["prob_mlp"].values

    pred_lr, true_lr = get_calibration_data(y_true, p_lr, n_bins=10)
    pred_mlp, true_mlp = get_calibration_data(y_true, p_mlp, n_bins=10)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", label="Calibración Perfecta")
    ax.plot(pred_lr, true_lr, marker="o", lw=2, color="#1f77b4", label="Regresión Logística (Brier = 0.1756)")
    ax.plot(pred_mlp, true_mlp, marker="s", lw=2, color="#d62728", label="Perceptrón Multicapa (Brier = 0.1731)")
    ax.set_title("Curva de Calibración en Test Out-of-Time")
    ax.set_xlabel("Probabilidad Media Predicha")
    ax.set_ylabel("Fracción Empírica de Incumplimiento (Observada)")
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig4_curva_calibracion.png"), bbox_inches="tight")
    plt.close()


def plot_scorecard_distribution(preds_df: pd.DataFrame, out_dir: str) -> None:
    """Figura 5: Distribución de Puntajes de la Scorecard para Buenos vs Malos."""
    y_true = preds_df["y_true"].values
    scores = preds_df["score_mlp"].values

    scores_good = scores[y_true == 0]
    scores_bad = scores[y_true == 1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    axes[0].hist(scores_good, bins=45, density=True, alpha=0.6, color="#2ca02c", label="Buenos (y=0)")
    axes[0].hist(scores_bad, bins=45, density=True, alpha=0.6, color="#d62728", label="Malos (y=1)")
    axes[0].axvline(np.median(scores_good), color="#2ca02c", linestyle="--", lw=1.8, label=f"Mediana Buenos ({np.median(scores_good):.0f})")
    axes[0].axvline(np.median(scores_bad), color="#d62728", linestyle="--", lw=1.8, label=f"Mediana Malos ({np.median(scores_bad):.0f})")
    axes[0].set_title("Distribución del Puntaje Crediticio (Scorecard MLP)")
    axes[0].set_xlabel("Puntaje de Crédito (Mayor = Menor Riesgo)")
    axes[0].set_ylabel("Densidad")
    axes[0].legend()

    # Tasa de incumplimiento por deciles de puntaje
    df_sc = pd.DataFrame({"score": scores, "y": y_true})
    df_sc["decile"] = pd.qcut(df_sc["score"], q=10)
    dec_summary = df_sc.groupby("decile", observed=False).agg(mean_score=("score", "mean"), bad_rate=("y", "mean"))
    axes[1].plot(dec_summary["mean_score"], dec_summary["bad_rate"] * 100, marker="o", color="#9467bd", lw=2.2)
    axes[1].set_title("Tasa de Incumplimiento Real según Decil de Puntaje")
    axes[1].set_xlabel("Puntaje Promedio del Decil")
    axes[1].set_ylabel("Tasa de Incumplimiento Observada (%)")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig5_scorecard_distribucion.png"), bbox_inches="tight")
    plt.close()


def plot_feature_importance(out_dir: str) -> None:
    """Figura 6: Top 12 Variables más importantes por Permutación (MLP vs Regresión Logística)."""
    imp_path = "reports/tables/pd/feature_importance_pd.csv"
    if not os.path.exists(imp_path):
        return
    df_imp = pd.read_csv(imp_path).head(12).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y_pos = np.arange(len(df_imp))
    width = 0.38

    ax.barh(y_pos + width / 2, df_imp["mlp_auc_drop_mean"], width, label="MLP (Red Neuronal)", color="#d62728", alpha=0.85)
    ax.barh(y_pos - width / 2, df_imp["lr_auc_drop_mean"], width, label="Regresión Logística", color="#1f77b4", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_imp["feature"])
    ax.set_xlabel("Caída Media en AUC-ROC al Permutar la Variable (Mayor = Más Importante)")
    ax.set_title("Interpretabilidad: Top 12 Variables de Riesgo en Test (Permutation Importance)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig6_interpretabilidad_pfi.png"), bbox_inches="tight")
    plt.close()


def main(base_cfg_path: str = "configs/base.yaml") -> None:
    with open(base_cfg_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    proc_dir = base_cfg["paths"]["processed_pd_dir"]
    fmt = base_cfg["paths"].get("output_format", "csv")
    out_dir = "reports/figures/pd"
    os.makedirs(out_dir, exist_ok=True)

    print("[Figuras] Generando gráficas del reporte en reports/figures/pd/ ...")
    train_df = load_split_df(proc_dir, "train", fmt)
    plot_eda_hypotheses(train_df, out_dir)
    plot_training_history(out_dir)

    preds_path = "reports/tables/pd/test_predictions.csv"
    if os.path.exists(preds_path):
        preds_df = pd.read_csv(preds_path)
        plot_roc_and_ks(preds_df, out_dir)
        plot_calibration(preds_df, out_dir)
        plot_scorecard_distribution(preds_df, out_dir)

    plot_feature_importance(out_dir)
    print(f"[OK] Figuras generadas exitosamente en: {out_dir}")


if __name__ == "__main__":
    main()