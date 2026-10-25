# Datos Originales (No Versionados en Git)

El archivo original excede el límite de 100 MB por archivo de GitHub (~450 MB descomprimido), por lo que se excluye mediante `.gitignore` y se versionan las particiones procesadas y comprimidas en formato Apache Parquet dentro de `data/processed/pd/`.

- **Fuente oficial:** Kaggle — *Lending Club Loan Data* (`https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset`)
- **Fecha de descarga:** Octubre de 2026
- **Nombre del archivo esperado:** `data/raw/loan.csv` (887,379 registros y 74 columnas, periodo `2007-06` a `2015-12`).
- **Diccionario de variables:** `LCDataDictionary.xlsx` (hoja `LoanStats`), incluido en el conjunto de datos de Kaggle.
- **Cómo regenerar los datos procesados desde el archivo original:**
  Colocar `loan.csv` en `data/raw/loan.csv` y ejecutar desde la raíz del proyecto:
  ```bash
  python src/creditrisk/data/build_dataset.py