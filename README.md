# Credit Risk NN: ¿me prestas?

Trabajo 2 de *Redes neuronales y algoritmos bioinspirados* (Universidad Nacional de Colombia, Sede Medellín, Facultad de Minas, 2026-02). Redes neuronales para datos tabulares aplicadas al riesgo de crédito con el conjunto *Lending Club*.

| Recurso | Enlace |
|---|---|
| Repositorio | https://github.com/SaloMuri/rn-trabajo-2 |
| Reporte técnico | [`blog/index.html`](https://salomuri.github.io/rn-trabajo-2/blog/) (versión web, se abre en el navegador) y [`blog/reporte.md`](blog/reporte.md) (fuente) |
| Aplicación web | https://rn-trabajo-2-app.streamlit.app |
| Video promocional | (pendiente) |

**Integrantes:** Salome Murillo Gaviria, Nicolas David Zambrano Murcia, Oky Ruiz De La Rosa y Andrés Pérez Cano.

## Videos de contribución individual

En cumplimiento de los lineamientos de evaluación del curso, se incluyen los enlaces a los videos de contribución individual:

| # | Integrante | Rol / módulos principales desarrollados | Video |
| :-: | :--- | :--- | :---: |
| 1 | Salomé Murillo Gaviria | Scorecard, interpretabilidad por importancia por permutación, evaluación con búsqueda de hiperparámetros y 30 semillas, y reporte web. | [Video de contribución 1](https://youtu.be/iFEk_I-LNzQ) |
| 2 | Nicolás David Zambrano Murcia | Arquitectura neuronal MLP y estabilidad en 30 semillas, derivación de *scorecard*, modelamiento LGD y aplicación. | [Video de contribución 2](https://youtu.be/e_y7hSoHltw) |
| 3 | Andrés Pérez Cano | Construcción del objetivo LGD (con variante `--include-default`), carga de datos, configuración y métricas de distribución de la Parte 2, y cálculo de la pérdida esperada PD × LGD × EAD. | [Video de contribución 3](https://www.youtube.com/watch?v=WwKmzS14pFE) |
| 4 | Oky Ruiz de la Rosa | Pérdida esperada, análisis de sensibilidad LGD, pruebas unitarias y cacería de alucinaciones. | [Video de contribución 4](https://youtube.com/REEMPLAZAR_LINK_4) |

## Qué hace el proyecto

Se estima la **pérdida esperada** de un crédito, `EL = PD × LGD × EAD`, con redes neuronales y modelos de referencia lineales, bajo las mismas particiones temporales (entrenamiento hasta 2014-06, validación hasta 2014-12 y prueba después).

| Parte | Tarea | Modelos | Resultado |
|---|---|---|---|
| 1 | Probabilidad de incumplimiento (PD) y *scorecard* | Regresión logística frente a MLP | Ver sección 4 del reporte |
| 2 | Pérdida dado el incumplimiento (LGD, regresión) y pérdida esperada | Constante, regresión lineal (Ridge), MLP sigmoide + BCE y MLP sigmoide + MSE | Ver sección 5 del reporte |

### **Resultados de la Parte 1 (PD y Scorecard — Conjunto de prueba Out-of-Time, media ± desviación estándar con 30 semillas)**

| Modelo | Parámetros Entrenables | ROC-AUC | Índice de Gini | Estadístico KS | Brier Score (Calibrado) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Regresión Logística (Baseline)** | 39 | 0.6899 | 0.3799 | 0.2829 | 0.1756 |
| **Perceptrón Multicapa (MLPGanador)** | 4,801 | 0.6983 | 0.3966 | 0.2923 | 0.1731 |
| **MLP (Estabilidad 30 Semillas)** | 4,801 | **0.6970 ± 0.0014** | **0.3940 ± 0.0029** | **0.2918 ± 0.0035** | **0.1732 ± 0.0003** |

**Resultados de la Parte 2** (conjunto de prueba, media ± desviación estándar con 30 semillas; menor es mejor):

| Modelo | Parámetros | MAE | RMSE |
|---|---:|---:|---:|
| Constante (media del entrenamiento) | 1 | 0.0807 | 0.0922 |
| Regresión lineal (Ridge) | 39 | 0.0780 | 0.0924 |
| MLP sigmoide + BCE | 4,609 | 0.0729 ± 0.0030 | 0.0885 ± 0.0020 |
| MLP sigmoide + MSE | 4,609 | 0.0729 ± 0.0057 | 0.0876 ± 0.0039 |

La pérdida esperada total del conjunto de prueba (EAD B) es 1.65 veces la pérdida realizada, por una diferencia de definiciones entre la PD y la LGD. El reporte explica esta y las demás limitaciones (censura de la LGD en los créditos recientes, entre otras) en la sección 5.

## Instalación

Requiere **Python 3.10 o superior**. Las versiones de las librerías están fijadas en [`requirements.txt`](requirements.txt). Los modelos guardados con `joblib` se generaron con `scikit-learn==1.7.2`, así que conviene no cambiar esa versión.

**Linux y macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
source setup.sh
```

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
setup
```

Para comprobar la instalación:

```bash
pytest -q
```

Deben pasar las 20 pruebas.

## Datos

- **Fuente:** Kaggle, *Lending Club Loan Data* (`ranadeep/credit-risk-dataset`). Detalles en [`data/raw/README.md`](data/raw/README.md).
- **Archivo original** (`data/raw/loan.csv`, unos 450 MB): **no se versiona** en git. Solo hace falta para regenerar los datos procesados.
- **Datos procesados:** sí están versionados, en formato Parquet: las particiones de PD en `data/processed/pd/` y el objetivo de LGD en `data/processed/lgd/`. Con ellos se pueden reproducir todos los experimentos sin descargar el original.

Para regenerarlos desde el original (poner `loan.csv` en `data/raw/`):

```bash
python src/creditrisk/data/build_dataset.py        # particiones de PD
python src/creditrisk/data/build_lgd_target.py      # objetivo de LGD (población principal)
python src/creditrisk/data/build_lgd_target.py --include-default   # variante con `Default` (sensibilidad)
```

## Ejecución: un comando por experimento

Cada experimento se ejecuta con un comando desde la raíz del proyecto. Con `make` (Linux y macOS) o con el comando de Python (cualquier sistema, incluido Windows sin `make`):

| Experimento | `make` | Comando de Python |
|---|---|---|
| Parte 1: PD (referencia y MLP) | `make pd` | `python scripts/run_pd.py` |
| Parte 1: búsqueda de hiperparámetros | `make search` | `python scripts/run_search.py` |
| Parte 1: 30 semillas | `make seeds` | `python scripts/run_seeds.py` |
| Parte 1: figuras | `make figures` | `python scripts/make_figures.py` |
| Parte 2: LGD (referencias y MLP) | `make lgd` | `python scripts/run_lgd.py` |
| Parte 2: LGD con `Default` (sensibilidad) | `make lgd-sens` | `python scripts/run_lgd.py --population with_default` |
| Parte 2: búsqueda de hiperparámetros | `make search-lgd` | `python scripts/run_search_lgd.py` |
| Parte 2: 30 semillas | `make seeds-lgd` | `python scripts/run_seeds_lgd.py` |
| Parte 2: pérdida esperada | `make el` | `python scripts/run_expected_loss.py` |
| Parte 2: figuras e interpretabilidad | `make figures-lgd` | `python scripts/make_figures_lgd.py` |
| Parte 1: demo (muestra pequeña) | `make demo` | `python scripts/run_pd.py --config configs/demo.yaml` |
| Parte 2: demo completa (modelos, pérdida esperada y figuras) | `make demo-lgd` | `python scripts/run_lgd.py --demo` y luego `python scripts/run_expected_loss.py` y `python scripts/make_figures_lgd.py` |
| Aplicación web | `make app` | `streamlit run app/app.py` |
| Pruebas | `make test` | `pytest -q` |

Orden para reproducir la Parte 2 completa:

```bash
python scripts/run_lgd.py            # 1. referencias y MLP con la configuración por defecto
python scripts/run_search_lgd.py     # 2. búsqueda (escribe configs/lgd_best.yaml)
python scripts/run_lgd.py --best     # 3. modelos finales con la mejor configuración
python scripts/run_seeds_lgd.py      # 4. 30 semillas
python scripts/run_expected_loss.py  # 5. pérdida esperada (necesita los modelos de PD de la Parte 1)
python scripts/make_figures_lgd.py   # 6. figuras e interpretabilidad
```

Los parámetros se cambian sin editar código, con los archivos de `configs/` o con `--set ruta=valor`; por ejemplo, `python scripts/run_lgd.py --set mlp.epochs=30`.

### Modo demostración (menos de 2 minutos)

```bash
make demo        # Parte 1: PD con una muestra pequeña (unos 24 s)
make demo-lgd    # Parte 2: modelos de LGD, pérdida esperada y figuras (unos 30 s)
```

Sin `make`, los comandos equivalentes son:

```bash
python scripts/run_pd.py --config configs/demo.yaml    # Parte 1: PD con una muestra pequeña
python scripts/run_lgd.py --demo                       # Parte 2: modelos de LGD con una muestra pequeña
python scripts/run_expected_loss.py                    # Parte 2: pérdida esperada con los modelos finales
python scripts/make_figures_lgd.py                     # Parte 2: figuras e interpretabilidad
```

Los tiempos son de una prueba en un entorno Linux en la nube (PD unos 24 segundos; LGD, pérdida esperada y figuras unos 30 segundos en total). Las demos de los modelos se guardan en carpetas aparte (`models/lgd_demo/`, `reports/tables/lgd_demo/`, `reports/tables/pd/comparison_demo.csv`) y no sobrescriben los resultados completos. Con la muestra reducida, las cifras de `run_lgd.py --demo` son distintas de las del reporte; la pérdida esperada y las figuras usan los modelos finales, por lo que sus cifras coinciden con las del reporte.

La demo de la Parte 2 no incluye la construcción del objetivo de LGD (necesita `data/raw/loan.csv`, que no se versiona), la búsqueda de hiperparámetros (unos 8 minutos) ni las 30 semillas completas (unos 8 minutos). Para ver el análisis de la búsqueda y de las semillas, abrir `reports/tables/lgd/main/search_budget.json` y `seeds_summary.csv`.

## Semillas y reproducibilidad

- **Semilla global:** 42 (en `configs/base.yaml`, `configs/pd.yaml` y `configs/lgd.yaml`). Fija la inicialización, el orden de los lotes y el *dropout*.
- **Treinta semillas:** la configuración final de cada red se entrena con las semillas 0 a 29 y se reporta media, desviación estándar, mejor y peor valor. En el modelo de LGD la semilla cambia la inicialización, el orden de los lotes y el *dropout*; los datos y las particiones son fijos.
- **Versiones fijas:** `requirements.txt` fija las versiones exactas de todas las librerías.
- **Datos y modelos guardados:** los datos procesados, los modelos (`models/`) y las tablas (`reports/tables/`) están versionados.
- **Diferencias entre equipos:** dos equipos con las mismas versiones dan resultados iguales salvo un ruido numérico pequeño (cuarta cifra decimal), por el tipo de procesador y los hilos. Los rankings de configuraciones casi empatadas pueden cambiar, pero las conclusiones no.

## Hardware y tiempos

Parte 2 medida en un equipo con CPU AMD Ryzen 5 150 con gráficos Radeon (3.30 GHz), 16 GB de RAM y Windows 11:

| Experimento | Tiempo aproximado |
|---|---|
| Búsqueda de hiperparámetros del MLP de LGD (60 configuraciones) | 8 minutos |
| 30 semillas × 2 redes | 8 minutos |
| Un entrenamiento del MLP de LGD | 8 a 15 segundos |

Con otro hardware los tiempos cambian.

## Estructura

| Ruta | Para qué existe |
|---|---|
| `configs/` | Parámetros de los experimentos; permiten cambiarlos sin editar código. `lgd_best.yaml` lo genera la búsqueda. |
| `data/raw/` | Documenta el origen de los datos originales (no se versionan). |
| `data/processed/` | Datos procesados que usan los modelos (versionados). |
| `src/creditrisk/` | Código de la librería: datos, modelos, métricas, *scorecard*, interpretabilidad, pérdida esperada. |
| `scripts/` | Puntos de entrada de los experimentos, de las figuras y del reporte web (`build_blog.py`). |
| `models/` | Modelos entrenados (versionados). |
| `reports/` | Figuras y tablas generadas por el código. |
| `app/` | Aplicación web. |
| `blog/` | Fuente del reporte técnico (`reporte.md`) y su versión web (`index.html`, se genera con `python scripts/build_blog.py`). |
| `docs/` | Bitácoras de uso de IA y cacería de la alucinación. |
| `tests/` | Pruebas. |

## Uso de IA y cacería de la alucinación

- [`docs/uso_ia.md`](docs/uso_ia.md): registro de las interacciones con asistentes de IA (Gemini en la Parte 1 y Claude en la Parte 2).
- [`docs/caceria_alucinacion.md`](docs/caceria_alucinacion.md): errores de la IA encontrados, con su evidencia, y verificaciones sin hallazgo.
