"""Tabla descriptiva de censura: proporción de créditos con etiqueta PD clasificable por año de originación."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import yaml

from creditrisk.data.targets import encode_pd_target

with open("configs/base.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

df = pd.read_csv(cfg["paths"]["raw_csv"], usecols=["loan_status", "issue_d"], low_memory=False)
df["anio"] = pd.to_datetime(df["issue_d"], format="%b-%Y").dt.year
df["y"] = encode_pd_target(df["loan_status"])      # NaN = sin desenlace conocido

t = df.groupby("anio").agg(originados=("y", "size"), con_etiqueta_pd=("y", "count"), malos_clasificados=("y", "sum"))
t["pct_con_etiqueta_pd"] = (100 * t["con_etiqueta_pd"] / t["originados"]).round(1)
t["tasa_malos_entre_etiquetados"] = (100 * t["malos_clasificados"] / t["con_etiqueta_pd"]).round(1)
t["malos_clasificados"] = t["malos_clasificados"].astype(int)

t.to_csv("reports/tables/pd/censura_por_anio.csv")
print(t.to_string())