# app/app.py
"""Aplicación web: puntaje, probabilidad de incumplimiento, percentil y escenarios 'qué pasaría si'.
Enlaces al reporte y al video.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

from creditrisk.models.mlp import load_mlp_checkpoint
from creditrisk.scorecard import CreditScorecard

st.set_page_config(
    page_title="Scorecard de Riesgo Crediticio — UNAL",
    page_icon="🏦",
    layout="wide",
)

@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("models/pd/preprocessor.joblib")
    mlp_model = load_mlp_checkpoint("models/pd/mlp_best.pt", device="cpu")
    with open("models/pd/scorecard_config.json", "r", encoding="utf-8") as f:
        sc_meta = json.load(f)
    scorecard = CreditScorecard(
        base_score=sc_meta["base_score"],
        base_odds=sc_meta["base_odds"],
        pdo=sc_meta["pdo"],
    )
    return preprocessor, mlp_model, scorecard, sc_meta


def predict_single_profile(profile_dict: dict, preprocessor, mlp_model, scorecard: CreditScorecard):
    df_in = pd.DataFrame([profile_dict])
    X_in = preprocessor.transform(df_in).astype(np.float32)
    with torch.no_grad():
        logit = mlp_model(torch.tensor(X_in, dtype=torch.float32)).item()
        pd_prob = float(1.0 / (1.0 + np.exp(-logit)))
    score = float(scorecard.pd_to_score(pd_prob))
    return pd_prob, score


def compute_percentile(score: float, quantiles_dict: dict) -> int:
    q_keys = sorted([float(k) for k in quantiles_dict.keys()])
    q_vals = [quantiles_dict[f"{k:.2f}" if f"{k:.2f}" in quantiles_dict else str(k)] for k in q_keys]
    idx = np.searchsorted(q_vals, score)
    return int(np.clip(idx + 1, 1, 99))


def get_risk_tier(score: float):
    if score >= 565:
        return "🟢 Riesgo Bajo — Aprobación Estándar", "#2ca02c"
    if score >= 535:
        return "🟡 Riesgo Moderado — Revisión Analista", "#ff7f0e"
    return "🔴 Riesgo Alto — Rechazo / Garantía Adicional", "#d62728"

def estimate_grade_and_rate(dti: float, revol_util: float, delinq_2yrs: int, inq_6m: int, pub_rec: int, term: int):
    """
    Heurística de pre-cotización (Pricing Engine) calibrada con las medias históricas
    de Lending Club para usuarios que aún no conocen su Grado ni Tasa asignada.
    """
    risk_points = 0.0
    risk_points += max(0.0, (dti - 10.0) * 0.08)
    risk_points += max(0.0, (revol_util - 30.0) * 0.03)
    risk_points += delinq_2yrs * 1.2
    risk_points += inq_6m * 0.45
    risk_points += pub_rec * 1.5
    if term == 60:
        risk_points += 1.1

    grade_est = int(np.clip(np.round(1.0 + risk_points), 1, 7))
    # Tasas promedio históricas por grado en Lending Club (A -> ~7.2%, G -> ~25.8%)
    typical_rates = {1: 7.2, 2: 10.5, 3: 13.6, 4: 16.8, 5: 19.9, 6: 23.4, 7: 25.8}
    return grade_est, typical_rates[grade_est]

def main():
    st.title("🏦 Simulador de Riesgo Crediticio y Scorecard (Parte 1 — PD)")
    st.markdown(
        "**Trabajo 2 — Redes Neuronales y Algoritmos Bioinspirados (UNAL Facultad de Minas)**  \n"
        "🔗 [Reporte Técnico (Blog)](https://salomuri.github.io/rn-trabajo-2/blog/) · 🎥 [Video Promocional](#) · 💻 [Repositorio GitHub](https://github.com/SaloMuri/rn-trabajo-2.git)"
    )
    st.divider()

    try:
        preprocessor, mlp_model, scorecard, sc_meta = load_artifacts()
    except Exception as e:
        st.error(
            f"No se encontraron los modelos entrenados (`{e}`). "
            "Ejecuta primero `python scripts/run_pd.py`."
        )
        return

# -------------------------------------------------------------------------
    # BARRA LATERAL: PERFIL DEL SOLICITANTE EN ORIGINACIÓN
    # -------------------------------------------------------------------------
    st.sidebar.header("📋 Datos del Solicitante (Originación)")
    st.sidebar.caption(
        "Ingresa las características conocidas al momento de pedir el crédito:"
    )

    loan_amnt = st.sidebar.number_input(
        "Monto solicitado (USD)", min_value=1000, max_value=40000, value=12000, step=500, key="sidebar_loan_amnt"
    )
    term = st.sidebar.selectbox("Plazo del préstamo (meses)", options=[36, 60], index=0, key="sidebar_term")

    annual_inc = st.sidebar.number_input(
        "Ingreso anual reportado (USD)",
        min_value=10000,
        max_value=500000,
        value=65000,
        step=2500,
        key="sidebar_annual_inc"
    )
    dti = st.sidebar.slider(
        "Relación Deuda-Ingreso mensual (DTI %)",
        min_value=0.0,
        max_value=60.0,
        value=18.0,
        step=0.5,
        key="sidebar_dti"
    )
    revol_util = st.sidebar.slider(
        "Uso de cupo en tarjetas rotativas (%)",
        min_value=0.0,
        max_value=120.0,
        value=45.0,
        step=1.0,
        key="sidebar_revol_util"
    )
    revol_bal = st.sidebar.number_input(
        "Saldo rotativo actual (USD)",
        min_value=0,
        max_value=200000,
        value=11000,
        step=500,
        key="sidebar_revol_bal"
    )

    home_ownership = st.sidebar.selectbox(
        "Tipo de vivienda", options=["MORTGAGE", "RENT", "OWN", "OTHER"], index=1, key="sidebar_home_ownership"
    )
    purpose = st.sidebar.selectbox(
        "Propósito del préstamo",
        options=[
            "debt_consolidation",
            "credit_card",
            "home_improvement",
            "other",
            "major_purchase",
            "small_business",
            "car",
            "medical",
        ],
        index=0,
        key="sidebar_purpose"
    )
    verification_status = st.sidebar.selectbox(
        "Estado de verificación de ingresos",
        options=["Verified", "Source Verified", "Not Verified"],
        index=1,
        key="sidebar_verification_status"
    )

    with st.sidebar.expander("⚙️ Historial crediticio detallado"):
        emp_length = st.slider("Antigüedad laboral (años, 10 = 10+)", 0, 10, 5, key="sidebar_emp_length")
        credit_history_years = st.slider(
            "Antigüedad del historial crediticio (años)", 1, 40, 12, key="sidebar_credit_history_years"
        )
        open_acc = st.number_input("Líneas de crédito abiertas actualmente", 1, 50, 10, key="sidebar_open_acc")
        total_acc = st.number_input("Total histórico de líneas de crédito", 1, 100, 22, key="sidebar_total_acc")
        inq_last_6mths = st.slider(
            "Consultas a central de riesgo (últimos 6 meses)", 0, 10, 1, key="sidebar_inq_last_6mths"
        )
        delinq_2yrs = st.slider("Moras >30 días en los últimos 2 años", 0, 10, 0, key="sidebar_delinq_2yrs")
        pub_rec = st.slider(
            "Registros públicos adversos (bancarrota/embargos)", 0, 5, 0, key="sidebar_pub_rec"
        )
        has_prev_delinq = st.selectbox(
            "¿Ha tenido alguna mora histórica?",
            options=[0, 1],
            format_func=lambda x: "Sí" if x == 1 else "No",
            key="sidebar_has_prev_delinq"
        )

    # -------------------------------------------------------------------------
    # GESTIÓN INTELIGENTE DE GRADE E INT_RATE (PRE-COTIZACIÓN AUTOMÁTICA)
    # -------------------------------------------------------------------------
    auto_pricing = st.sidebar.checkbox(
        "✨ Estimar Grado y Tasa automáticamente",
        value=True,
        help="Calcula tu grado y tasa según tu nivel de endeudamiento e historial si aún no tienes oferta del banco.",
    )

    if auto_pricing:
        grade, int_rate = estimate_grade_and_rate(
            dti=dti, revol_util=revol_util, delinq_2yrs=delinq_2yrs, 
            inq_6m=inq_last_6mths, pub_rec=pub_rec, term=term
        )
        grade_letters = {1: "A (Excelente)", 2: "B (Bueno)", 3: "C (Medio)", 4: "D (Moderado)", 5: "E (Riesgoso)", 6: "F (Alto Riesgo)", 7: "G (Crítico)"}
        st.sidebar.success(f"📌 **Grado estimado:** {grade_letters[grade]}  \n📈 **Tasa estimada:** {int_rate:.2f}%")
    else:
        grade_map = {"A (Excelente)": 1, "B (Bueno)": 2, "C (Medio)": 3, "D (Moderado)": 4, "E (Riesgoso)": 5, "F (Alto Riesgo)": 6, "G (Crítico)": 7}
        grade_label = st.sidebar.selectbox("Calificación crediticia (Grado)", options=list(grade_map.keys()), index=2)
        grade = grade_map[grade_label]
        int_rate = st.sidebar.slider("Tasa de interés efectiva anual (%)", min_value=5.0, max_value=30.0, value=13.5, step=0.25)

    # Cálculo financiero automático de la cuota mensual (installment)
    r_monthly = (int_rate / 100.0) / 12.0
    installment = float(loan_amnt * (r_monthly * (1 + r_monthly) ** term) / ((1 + r_monthly) ** term - 1))
    st.sidebar.info(f"💵 **Cuota mensual estimada:** ${installment:,.2f} USD")
    
    # Cálculo financiero automático de la cuota mensual (installment)
    r_monthly = (int_rate / 100.0) / 12.0
    installment = float(loan_amnt * (r_monthly * (1 + r_monthly) ** term) / ((1 + r_monthly) ** term - 1))
    st.sidebar.info(f"💵 **Cuota mensual estimada:** ${installment:,.2f} USD")

    annual_inc = st.sidebar.number_input("Ingreso anual reportado (USD)", min_value=10000, max_value=500000, value=65000, step=2500)
    dti = st.sidebar.slider("Relación Deuda-Ingreso mensual (DTI %)", min_value=0.0, max_value=60.0, value=18.0, step=0.5)
    revol_util = st.sidebar.slider("Uso de cupo en tarjetas rotativas (%)", min_value=0.0, max_value=120.0, value=45.0, step=1.0)
    revol_bal = st.sidebar.number_input("Saldo rotativo actual (USD)", min_value=0, max_value=200000, value=11000, step=500)

    home_ownership = st.sidebar.selectbox("Tipo de vivienda", options=["MORTGAGE", "RENT", "OWN", "OTHER"], index=1)
    purpose = st.sidebar.selectbox(
        "Propósito del préstamo",
        options=[
            "debt_consolidation",
            "credit_card",
            "home_improvement",
            "other",
            "major_purchase",
            "small_business",
            "car",
            "medical",
        ],
        index=0,
    )
    verification_status = st.sidebar.selectbox(
        "Estado de verificación de ingresos",
        options=["Verified", "Source Verified", "Not Verified"],
        index=1,
    )

    with st.sidebar.expander("⚙️ Historial crediticio detallado"):
        emp_length = st.slider("Antigüedad laboral (años, 10 = 10+)", 0, 10, 5)
        credit_history_years = st.slider("Antigüedad del historial crediticio (años)", 1, 40, 12)
        open_acc = st.number_input("Líneas de crédito abiertas actualmente", 1, 50, 10)
        total_acc = st.number_input("Total histórico de líneas de crédito", 1, 100, 22)
        inq_last_6mths = st.slider("Consultas a central de riesgo (últimos 6 meses)", 0, 10, 1)
        delinq_2yrs = st.slider("Moras >30 días en los últimos 2 años", 0, 10, 0)
        pub_rec = st.slider("Registros públicos adversos (bancarrota/embargos)", 0, 5, 0)
        has_prev_delinq = st.selectbox("¿Ha tenido alguna mora histórica?", options=[0, 1], format_func=lambda x: "Sí" if x == 1 else "No")

    base_profile = {
        "loan_amnt": float(loan_amnt),
        "int_rate": float(int_rate),
        "installment": float(installment),
        "annual_inc": float(annual_inc),
        "dti": float(dti),
        "delinq_2yrs": float(delinq_2yrs),
        "inq_last_6mths": float(inq_last_6mths),
        "open_acc": float(open_acc),
        "pub_rec": float(pub_rec),
        "revol_bal": float(revol_bal),
        "revol_util": float(revol_util),
        "total_acc": float(total_acc),
        "credit_history_months": float(credit_history_years * 12.0),
        "has_prev_delinq": int(has_prev_delinq),
        "term": float(term),
        "emp_length": float(emp_length),
        "grade": float(grade),
        "home_ownership": home_ownership,
        "verification_status": verification_status,
        "purpose": purpose,
    }

    pd_prob, score = predict_single_profile(base_profile, preprocessor, mlp_model, scorecard)
    pct = compute_percentile(score, sc_meta["test_score_quantiles"])
    tier_text, tier_color = get_risk_tier(score)

    # -------------------------------------------------------------------------
    # PANEL PRINCIPAL: RESULTADOS DEL PERFIL ACTUAL
    # -------------------------------------------------------------------------
    st.subheader("1. Evaluación Crediticia del Perfil Actual (Red Neuronal MLP)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Puntaje Scorecard", f"{score:.0f} pts", help="Escala continua: Base 600 pts (PD=5%), PDO = 20 pts.")
    col2.metric("Probabilidad de Impago (PD)", f"{pd_prob * 100:.2f}%", help="Probabilidad estimada por el MLP de incumplir el crédito.")
    col3.metric("Percentil Poblacional", f"Percentil {pct}°", help=f"Tu puntaje supera al {pct}% de los solicitantes evaluados.")

    st.markdown(
        f"<div style='padding:14px;border-radius:8px;background-color:{tier_color}22;border-left:6px solid {tier_color};font-size:18px;'>"
        f"<b>Dictamen sugerido:</b> {tier_text}</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # -------------------------------------------------------------------------
    # ESCENARIOS "¿QUÉ PASARÍA SI...?" (WHAT-IF INTERACTIVO)
    # -------------------------------------------------------------------------
    st.subheader("2. Laboratorio de Escenarios: ¿Qué pasaría si modificas tu solicitud?")
    st.caption(
        "Mueve los controles de abajo para simular cómo cambiarían tu Puntaje y tu Probabilidad de Incumplimiento "
        "si ajustaras las condiciones del préstamo o tu nivel de endeudamiento antes de aplicar:"
    )

    w_col1, w_col2, w_col3, w_col4 = st.columns(4)
    with w_col1:
        sim_term = st.radio("Cambiar plazo a (meses):", options=[36, 60], index=0 if term == 36 else 1)
    with w_col2:
        sim_loan_amnt = st.slider("Simular monto solicitado (USD):", 1000, 40000, int(loan_amnt), step=500)
    with w_col3:
        sim_dti = st.slider("Simular relación Deuda/Ingreso (DTI %):", 0.0, 60.0, float(dti), step=0.5)
    with w_col4:
        sim_revol_util = st.slider("Simular uso de cupo en tarjetas (%):", 0.0, 120.0, float(revol_util), step=1.0)

    sim_profile = dict(base_profile)
    sim_profile["term"] = float(sim_term)
    sim_profile["loan_amnt"] = float(sim_loan_amnt)
    sim_profile["dti"] = float(sim_dti)
    sim_profile["revol_util"] = float(sim_revol_util)
    sim_profile["installment"] = float(
        sim_loan_amnt * (r_monthly * (1 + r_monthly) ** sim_term) / ((1 + r_monthly) ** sim_term - 1)
    )

    sim_pd, sim_score = predict_single_profile(sim_profile, preprocessor, mlp_model, scorecard)
    sim_pct = compute_percentile(sim_score, sc_meta["test_score_quantiles"])

    delta_score = sim_score - score
    delta_pd = (sim_pd - pd_prob) * 100.0
    delta_pct = sim_pct - pct

    s_col1, s_col2, s_col3 = st.columns(3)
    s_col1.metric(
        "Nuevo Puntaje Simulado",
        f"{sim_score:.0f} pts",
        delta=f"{delta_score:+.1f} pts vs. actual",
    )
    s_col2.metric(
        "Nueva Probabilidad de Impago (PD)",
        f"{sim_pd * 100:.2f}%",
        delta=f"{delta_pd:+.2f}% vs. actual",
        delta_color="inverse",
    )
    s_col3.metric(
        "Nuevo Percentil Poblacional",
        f"Percentil {sim_pct}°",
        delta=f"{delta_pct:+d} posiciones",
    )


if __name__ == "__main__":
    main()