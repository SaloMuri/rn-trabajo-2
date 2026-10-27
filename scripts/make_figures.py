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
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss

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
    """Figura 1: exploración descriptiva de tres hipótesis de riesgo crediticio.

    Los paneles muestran asociaciones en el conjunto de entrenamiento; no prueban
    causalidad ni, por sí solos, mecanismos de mediación.
    """
    df = train_df.copy()
    target_col = next(
        (col for col in ("target_pd", "target", "default") if col in df.columns),
        None,
    )
    if target_col is None:
        raise ValueError(
            "No se encontró la variable objetivo: target_pd, target o default."
        )

    required = ["int_rate", "dti", "revol_util", "annual_inc"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "Faltan columnas requeridas para la Figura 1: " + ", ".join(missing)
        )

    # Reconstruir home_ownership si el preprocesamiento la expandió en dummies.
    if "home_ownership" in df.columns:
        df["home_ownership_cat"] = df["home_ownership"].astype(str)
    else:
        home_categories = ["MORTGAGE", "OWN", "RENT", "OTHER", "ANY"]
        conditions = []
        for category in home_categories:
            col = f"home_ownership_{category}"
            if col in df.columns:
                conditions.append(pd.to_numeric(df[col], errors="coerce").fillna(0).eq(1))
            else:
                conditions.append(pd.Series(False, index=df.index))
        df["home_ownership_cat"] = np.select(
            conditions, home_categories, default="OTRA/NO IDENTIFICADA"
        )

    for col in [target_col, "int_rate", "dti", "revol_util", "annual_inc"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    def p_label(p_value: float) -> str:
        if not np.isfinite(p_value):
            return "p no disponible"
        if p_value < 0.001:
            return "p < 0.001"
        return f"p = {p_value:.3f}"

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
    })
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150)
    navy, crimson, amber, emerald, slate = (
        "#0A192F", "#DC2626", "#D97706", "#059669", "#475569"
    )
    width = 0.36

    # PANEL A — H1: tenencia de vivienda, tasa de interés y default.
    ax1 = axes[0]
    grp_ho = (
        df.dropna(subset=[target_col, "int_rate"])
        .groupby("home_ownership_cat", observed=False)
        .agg(
            pd_rate=(target_col, lambda x: x.mean() * 100),
            int_mean=("int_rate", "mean"),
            n=(target_col, "count"),
        )
    )
    preferred_order = ["MORTGAGE", "OWN", "RENT", "OTHER", "ANY", "OTRA/NO IDENTIFICADA"]
    order_ho = [cat for cat in preferred_order if cat in grp_ho.index]
    order_ho += [cat for cat in grp_ho.index if cat not in order_ho]
    grp_ho = grp_ho.reindex(order_ho)
    samples_int = [
        df.loc[df["home_ownership_cat"] == cat, "int_rate"].dropna().to_numpy()
        for cat in order_ho
    ]
    samples_int = [sample for sample in samples_int if len(sample) > 0]
    if len(samples_int) >= 2:
        f_stat_ho, p_val_ho = stats.f_oneway(*samples_int)
    else:
        f_stat_ho, p_val_ho = np.nan, np.nan

    x_pos = np.arange(len(order_ho))
    bars_pd = ax1.bar(
        x_pos - width / 2, grp_ho["pd_rate"], width,
        color=navy, alpha=0.9, label="Incumplimiento observado (%)",
    )
    ax1_twin = ax1.twinx()
    bars_ir = ax1_twin.bar(
        x_pos + width / 2, grp_ho["int_mean"], width,
        color=amber, alpha=0.85, label="Tasa de interés media (%)",
    )
    f_text = f"F = {f_stat_ho:.1f}" if np.isfinite(f_stat_ho) else "F no disponible"
    ax1.set_title(f"A. Vivienda, interés y default\n(ANOVA de int_rate: {f_text}; {p_label(p_val_ho)})")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"{cat}\n(n={int(grp_ho.loc[cat, 'n']):,})" for cat in order_ho], rotation=10)
    ax1.set_ylabel("Tasa de incumplimiento (%)", color=navy)
    ax1_twin.set_ylabel("Tasa de interés media (%)", color=amber)
    if len(grp_ho):
        ax1.set_ylim(0, max(float(grp_ho["pd_rate"].max()) * 1.3, 1))
        ax1_twin.set_ylim(0, max(float(grp_ho["int_mean"].max()) * 1.3, 1))
    for bar in bars_pd:
        ax1.annotate(
            f"{bar.get_height():.1f}%",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=7, xytext=(0, 2), textcoords="offset points",
        )
    for bar in bars_ir:
        ax1_twin.annotate(
            f"{bar.get_height():.1f}%",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=7, xytext=(0, 2), textcoords="offset points",
        )
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7, frameon=True)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    # PANEL B — H2: DTI y utilización del crédito rotativo.
    ax2 = axes[1]
    df_b = df.dropna(subset=["dti", "revol_util", target_col]).copy()
    df_b["dti_clipped"] = df_b["dti"].clip(lower=0, upper=45)
    df_b["dti_q_idx"] = pd.qcut(
        df_b["dti_clipped"], q=4, labels=False, duplicates="drop"
    )
    n_dti_bins = int(df_b["dti_q_idx"].nunique())
    dti_labels = [f"Q{i + 1}" for i in range(n_dti_bins)]
    df_b["dti_q"] = df_b["dti_q_idx"].map(
        {i: label for i, label in enumerate(dti_labels)}
    )
    df_b["dti_q"] = pd.Categorical(df_b["dti_q"], categories=dti_labels, ordered=True)
    df_b["revol_group"] = np.where(
        df_b["revol_util"] >= 75.0, "Utilización ≥ 75%", "Utilización < 75%"
    )
    piv_h2 = df_b.pivot_table(
        index="dti_q", columns="revol_group", values=target_col,
        aggfunc="mean", observed=False,
    ) * 100
    rho_dti, p_dti = stats.spearmanr(df_b["dti"], df_b[target_col], nan_policy="omit")
    x_q = np.arange(len(piv_h2.index))
    cols_h2 = [
        col for col in ["Utilización < 75%", "Utilización ≥ 75%"]
        if col in piv_h2.columns
    ]
    colors_h2 = {"Utilización < 75%": slate, "Utilización ≥ 75%": crimson}
    n_cols = max(len(cols_h2), 1)
    for idx, col_name in enumerate(cols_h2):
        offset = (idx - (n_cols - 1) / 2) * width
        vals = piv_h2[col_name].to_numpy(dtype=float)
        rects = ax2.bar(
            x_q + offset, vals, width, color=colors_h2[col_name],
            alpha=0.9, label=col_name,
        )
        for rect in rects:
            if np.isfinite(rect.get_height()):
                ax2.annotate(
                    f"{rect.get_height():.1f}%",
                    (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    ha="center", va="bottom", fontsize=7, xytext=(0, 2), textcoords="offset points",
                )
    rho_text = f"ρ = {rho_dti:.3f}" if np.isfinite(rho_dti) else "ρ no disponible"
    ax2.set_title(f"B. DTI y utilización rotativa (n={len(df_b):,})\n(Spearman DTI: {rho_text}; {p_label(p_dti)})")
    ax2.set_xticks(x_q)
    ax2.set_xticklabels([str(label) for label in piv_h2.index])
    ax2.set_xlabel("Cuartiles de DTI (recortado a [0, 45])")
    ax2.set_ylabel("Tasa de incumplimiento (%)")
    if len(piv_h2):
        max_rate = np.nanmax(piv_h2.to_numpy())
        ax2.set_ylim(0, max(max_rate * 1.25, 1))
    ax2.legend(loc="upper left", fontsize=7, frameon=True)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    # PANEL C — H3: ingreso anual e incumplimiento.
    ax3 = axes[2]
    df_c = df.dropna(subset=["annual_inc", target_col]).copy()
    df_c = df_c[df_c["annual_inc"] > 0]
    df_c["inc_q_idx"] = pd.qcut(
        df_c["annual_inc"], q=4, labels=False, duplicates="drop"
    )
    n_inc_bins = int(df_c["inc_q_idx"].nunique())
    inc_labels = [f"Q{i + 1}" for i in range(n_inc_bins)]
    df_c["inc_q"] = df_c["inc_q_idx"].map(
        {i: label for i, label in enumerate(inc_labels)}
    )
    df_c["inc_q"] = pd.Categorical(df_c["inc_q"], categories=inc_labels, ordered=True)
    grp_inc = df_c.groupby("inc_q", observed=False).agg(
        pd_rate=(target_col, lambda x: x.mean() * 100),
        n=(target_col, "count"),
    )
    rho_inc, p_inc = stats.spearmanr(
        df_c["annual_inc"], df_c[target_col], nan_policy="omit"
    )
    x_inc = np.arange(len(grp_inc))
    bars_inc = ax3.bar(
        x_inc, grp_inc["pd_rate"], width=0.52, color=emerald, alpha=0.88,
        label="Incumplimiento observado (%)",
    )
    ax3.plot(x_inc, grp_inc["pd_rate"], color=navy, marker="o", linewidth=2)
    for bar in bars_inc:
        ax3.annotate(
            f"{bar.get_height():.1f}%",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=8, xytext=(0, 3), textcoords="offset points",
        )
    rho_text = f"ρ = {rho_inc:.3f}" if np.isfinite(rho_inc) else "ρ no disponible"
    ax3.set_title(f"C. Ingreso anual y default (n={len(df_c):,})\n(Spearman: {rho_text}; {p_label(p_inc)})")
    ax3.set_xticks(x_inc)
    ax3.set_xticklabels(inc_labels)
    ax3.set_xlabel("Cuartiles de ingreso anual")
    ax3.set_ylabel("Tasa de incumplimiento (%)")
    if len(grp_inc):
        ax3.set_ylim(0, max(float(grp_inc["pd_rate"].max()) * 1.25, 1))
    ax3.legend(loc="upper right", fontsize=7, frameon=True)
    ax3.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(
        f"Figura 1. Exploración descriptiva de hipótesis de riesgo crediticio (N = {len(df):,})",
        fontsize=12, fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fig1_eda_hipotesis.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Figura 1 (hipótesis de riesgo) guardada en: {out_path}")

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

    auc_lr = roc_auc_score(y_true, p_lr)
    auc_mlp = roc_auc_score(y_true, p_mlp)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ROC Curve
    axes[0].plot(fpr_lr, tpr_lr,label=f"Regresión Logística (AUC = {auc_lr:.4f})",color="#1f77b4", lw=2)
    axes[0].plot(fpr_mlp, tpr_mlp, label=f"Perceptrón Multicapa (AUC = {auc_mlp:.4f})",color="#d62728", lw=2)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Azar (AUC = 0.50)")
    axes[0].set_title("Curva ROC en Conjunto de Prueba (Out-of-Time)")
    axes[0].set_xlabel("Tasa de Falsos Positivos (FPR)")
    axes[0].set_ylabel("Tasa de Verdaderos Positivos (TPR)")
    axes[0].legend(loc="lower right")



    # KS Curve para el MLP: evaluar los umbrales observados, no una cuadrícula aproximada.
    thresholds = np.sort(np.unique(p_mlp))
    probs_good = p_mlp[y_true == 0]
    probs_bad = p_mlp[y_true == 1]
    cdf_good = np.array([np.mean(probs_good <= t) for t in thresholds])
    cdf_bad = np.array([np.mean(probs_bad <= t) for t in thresholds])
    diffs = np.abs(cdf_good - cdf_bad)
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

    brier_lr = brier_score_loss(y_true, p_lr)
    brier_mlp = brier_score_loss(y_true, p_mlp)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", label="Calibración Perfecta")
    ax.plot(pred_lr, true_lr, marker="o", lw=2, color="#1f77b4",label=f"Regresión Logística (Brier = {brier_lr:.4f})")
    ax.plot(pred_mlp, true_mlp, marker="s", lw=2,color="#d62728",label=f"Perceptrón Multicapa (Brier = {brier_mlp:.4f})")
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