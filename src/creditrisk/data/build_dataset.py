# src/creditrisk/data/build_dataset.py
"""Archivo original -> datos procesados que usan los modelos.
Imprime registros iniciales, finales, variables y periodo temporal.
"""

import os
import re
import sys
from pathlib import Path

# Esto le enseña a Python dónde está la carpeta 'src' automáticamente en Windows/Mac/Linux
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml
import numpy as np
import pandas as pd
from creditrisk.data.targets import filter_valid_pd_records
def clean_term(series: pd.Series) -> pd.Series:
    """Convierte ' 36 months' o '60 months' en entero (36 o 60)."""
    return series.astype(str).str.extract(r"(\d+)")[0].astype(float)


def clean_percentage(series: pd.Series) -> pd.Series:
    """Limpia columnas que puedan venir con '%' (ej. int_rate, revol_util) a float."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    cleaned = series.astype(str).str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def clean_emp_length(series: pd.Series) -> pd.Series:
    """
    Convierte emp_length a escala numérica de 0 a 10:
    '< 1 year' -> 0, '1 year' -> 1, ..., '10+ years' -> 10, NaN/n/a -> NaN
    """
    s = series.astype(str).str.lower().str.strip()
    s = s.replace({"< 1 year": "0", "10+ years": "10", "n/a": np.nan, "nan": np.nan})
    extracted = s.str.extract(r"(\d+)")[0]
    return pd.to_numeric(extracted, errors="coerce")


def clean_grade(series: pd.Series) -> pd.Series:
    """Mapea el grado de riesgo de Lending Club de forma ordinal: A=1, B=2, ..., G=7."""
    grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
    return series.astype(str).str.strip().str.upper().map(grade_map)


def clean_home_ownership(series: pd.Series) -> pd.Series:
    """Agrupa categorías residuales (ANY, NONE) dentro de OTHER para evitar ruido."""
    s = series.astype(str).str.strip().str.upper()
    return s.replace({"ANY": "OTHER", "NONE": "OTHER"})


def build_pd_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Ejecuta la limpieza integral de la tabla para el modelo de Probabilidad de Incumplimiento (PD).
    Garantiza que solo se utilicen variables de originación.
    """
    # 1. Filtrar solo préstamos con desenlace conocido (Buenos=0, Malos=1)
    df = filter_valid_pd_records(df_raw, status_col="loan_status")

    # 2. Parsear fechas de originación (issue_d) y primera línea de crédito (earliest_cr_line)
    # Formato típico en Lending Club: 'Dec-2011' (%b-%Y)
    df["issue_d_dt"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df["earliest_cr_line_dt"] = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")

    # Descartar filas corruptas sin fecha de originación
    df = df.dropna(subset=["issue_d_dt"]).copy()

    # 3. Calcular antigüedad crediticia en meses al momento de la solicitud
    diff_days = (df["issue_d_dt"] - df["earliest_cr_line_dt"]).dt.days
    df["credit_history_months"] = np.maximum(0.0, diff_days / 30.4375)

    # 4. Crear bandera binaria de morosidad histórica previa
    if "mths_since_last_delinq" in df.columns:
        df["has_prev_delinq"] = df["mths_since_last_delinq"].notna().astype(int)
    else:
        df["has_prev_delinq"] = (df["delinq_2yrs"].fillna(0) > 0).astype(int)

    # 5. Limpiar formatos de texto en variables ordinales y porcentajes
    df["term"] = clean_term(df["term"])
    df["int_rate"] = clean_percentage(df["int_rate"])
    df["revol_util"] = clean_percentage(df["revol_util"])
    df["emp_length"] = clean_emp_length(df["emp_length"])
    df["grade"] = clean_grade(df["grade"])
    df["home_ownership"] = clean_home_ownership(df["home_ownership"])

    # Unificar nombre de columna de verificación de ingresos (en algunas versiones se llama is_inc_v)
    if "verification_status" not in df.columns and "is_inc_v" in df.columns:
        df["verification_status"] = df["is_inc_v"]
    df["verification_status"] = df["verification_status"].astype(str).str.strip()
    df["purpose"] = df["purpose"].astype(str).str.strip().str.lower()

    # 6. Corrección de errores de captura evidentes (sin filtrar por el futuro)
    # DTI negativos o mayores a 100% en préstamos individuales suelen ser errores de registro
    df["dti"] = df["dti"].clip(lower=0.0, upper=100.0)
    # Utilización rotativa acotada a un tope físico razonable (0% - 150%)
    df["revol_util"] = df["revol_util"].clip(lower=0.0, upper=150.0)

    # 7. Seleccionar únicamente las columnas de originación + fecha + target
    keep_cols = [
        "id",
        "issue_d_dt",
        "target_pd",
        # Numéricas
        "loan_amnt",
        "int_rate",
        "installment",
        "annual_inc",
        "dti",
        "delinq_2yrs",
        "inq_last_6mths",
        "open_acc",
        "pub_rec",
        "revol_bal",
        "revol_util",
        "total_acc",
        "credit_history_months",
        "has_prev_delinq",
        # Ordinales
        "term",
        "emp_length",
        "grade",
        # Nominales
        "home_ownership",
        "verification_status",
        "purpose",
    ]
    # Conservar solo las que existan efectivamente en el dataframe
    existing_cols = [c for c in keep_cols if c in df.columns]
    df_clean = df[existing_cols].copy()

    return df_clean

def save_df(df: pd.DataFrame, path_no_ext: str, fmt: str) -> str:
    """Guarda el DataFrame en formato csv o parquet según la configuración."""
    full_path = f"{path_no_ext}.{fmt}"
    if fmt == "parquet":
        df.to_parquet(full_path, index=False)
    else:
        df.to_csv(full_path, index=False)
    return full_path


def main(config_path: str = "configs/base.yaml") -> None:
    from creditrisk.data.splits import temporal_train_val_test_split
    from creditrisk.data.features import fit_and_save_preprocessor

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_path = cfg["paths"]["raw_csv"]
    out_dir = cfg["paths"]["processed_pd_dir"]
    out_fmt = cfg["paths"].get("output_format", "csv")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("models/pd", exist_ok=True)

    print(f"[INFO] Cargando dataset original desde: {raw_path} ...")
    df_raw = pd.read_csv(raw_path, low_memory=False)
    n_raw_rows, n_raw_cols = df_raw.shape

    df_clean = build_pd_table(df_raw)
    n_clean_rows, n_clean_cols = df_clean.shape

    # Partición temporal Out-of-Time
    split_cfg = cfg["temporal_split"]
    train_df, val_df, test_df = temporal_train_val_test_split(
        df_clean,
        date_col="issue_d_dt",
        train_end=split_cfg["train_end"],
        val_end=split_cfg["val_end"],
    )

    # Ajustar y guardar el preprocesador único (fit SOLO en train)
    _, feat_names = fit_and_save_preprocessor(
        train_df, cfg["features_pd"], save_path="models/pd/preprocessor.joblib"
    )

    # Guardar particiones y muestra de referencia para la App Web
    save_df(train_df, os.path.join(out_dir, "train"), out_fmt)
    save_df(val_df, os.path.join(out_dir, "val"), out_fmt)
    save_df(test_df, os.path.join(out_dir, "test"), out_fmt)

    ref_sample = train_df.sample(n=min(10000, len(train_df)), random_state=cfg["seed"])
    save_df(ref_sample, os.path.join(out_dir, "reference_dist"), out_fmt)

    min_date = df_clean["issue_d_dt"].min().strftime("%Y-%m")
    max_date = df_clean["issue_d_dt"].max().strftime("%Y-%m")
    bad_rate = df_clean["target_pd"].mean() * 100

    print("=" * 65)
    print("RESUMEN DE CONSTRUCCIÓN Y PARTICIÓN TEMPORAL (PARTE 1 - PD)")
    print("=" * 65)
    print(f"Registros originales (raw)    : {n_raw_rows:,} filas x {n_raw_cols} columnas")
    print(f"Registros válidos (PD 0/1)    : {n_clean_rows:,} filas x {n_clean_cols} columnas")
    print(f"Periodo de originación        : {min_date} a {max_date}")
    print(f"Tasa global de incumplimiento : {bad_rate:.2f}%")
    print("-" * 65)
    print(f"Train Set (<= {split_cfg['train_end']}) : {len(train_df):,} filas (PD: {train_df['target_pd'].mean()*100:.2f}%)")
    print(f"Val Set   (<= {split_cfg['val_end']}) : {len(val_df):,} filas (PD: {val_df['target_pd'].mean()*100:.2f}%)")
    print(f"Test Set  (>  {split_cfg['val_end']}) : {len(test_df):,} filas (PD: {test_df['target_pd'].mean()*100:.2f}%)")
    print(f"Features tras One-Hot Encoding: {len(feat_names)} columnas")
    print("=" * 65)
if __name__ == "__main__":
    main()