# 🏦 Redes Neuronales para Datos Tabulares: ¿Me Prestas?
## Estimación de Probabilidad de Incumplimiento (PD), Scorecard y Pérdida Dado el Incumplimiento (LGD)

**Universidad Nacional de Colombia — Sede Medellín (Facultad de Minas)**  
**Departamento de Ciencias de la Computación y de la Decisión**  
**Curso:** Redes Neuronales y Algoritmos Bioinspirados (Semestre 2026-02)

---

## 🔗 Enlaces Principales del Proyecto

* 📊 **[Reporte Técnico Interactivo (Blog Web en 5 Pestañas)](https://salomuri.github.io/rn-trabajo-2/blog/)** *(Versión Markdown en [`blog/reporte.md`](blog/reporte.md))*
* 🚀 **[Aplicación Web Interactiva (Scorecard & Simulador Contrafactual)](https://rn-trabajo-2-app.streamlit.app/)** *(Código en [`app/app.py`](app/app.py))*
* 🎬 **[Video Promocional del Proyecto (YouTube)](https://youtube.com/)** *(Reemplazar con enlace definitivo)*

---

## 🎥 Videos de Contribución Individual e Integrantes

En cumplimiento de los lineamientos de evaluación del curso, a continuación se presentan los enlaces a los **4 videos de contribución individual** donde cada integrante expone y sustenta sus aportes técnicos al repositorio:

| # | Integrante | Rol / Módulos Principales Desarrollados | Enlace al Video de Contribución Individual |
| :-: | :--- | :--- | :---: |
| **1** | **Salomé Murillo Gaviria** | Pipeline de Datos PD (`build_dataset.py`), Evaluación *Out-of-Time*, Hipótesis de Riesgo (`make_figures.py`), Interpretabilidad PFI y Reporte Web (`build_blog.py`). | [▶️ Ver Video Contribución 1](https://youtube.com/REEMPLAZAR_LINK_1) |
| **2** | **Nicolás David Zambrano Murcia** | Arquitectura Neuronal MLP y Estabilidad en 30 Semillas (`run_pd.py`), Derivación de Scorecard Bancaria (`scorecard.py`), Modelamiento LGD (`run_lgd.py`) y App (`app.py`). | [▶️ Ver Video Contribución 2]https://youtu.be/edYUph7pUao?si=n_rByvtvuN6mSLIf|
| **3** | **Andres Perez Cano/ Contribución Técnica 3**  Experimentación de Hiperparámetros (`run_search.py`), Calibración Isotónica y Auditoría Matemática. | [▶️ Ver Video Contribución 3](https://youtube.com/REEMPLAZAR_LINK_3) |
| **4** | **Oky Ruiz de la Rosa / Contribución Técnica 4** | Pérdida Esperada ($\mathbb{E}[L]$), Análisis de Sensibilidad LGD (`--include-default`), Pruebas Unitarias (`tests/`) y Cacería de Alucinaciones. | [▶️ Ver Video Contribución 4](https://youtube.com/REEMPLAZAR_LINK_4) |

---

## 📌 Resumen Ejecutivo y Resultados Clave

Este proyecto implementa un sistema reproducible de extremo a extremo (*end-to-end*) para la cuantificación del riesgo de crédito de consumo sobre el conjunto histórico de **Lending Club (`2007-06` a `2015-12`, 887,379 registros brutos $\times$ 74 columnas)**:

1. **Parte 1 — Probabilidad de Incumplimiento (PD) y Scorecard (`268,530` créditos cerrados válidos):**
   * **Prevención de *Data Leakage*:** Selección estricta de variables disponibles exclusivamente en el instante de originación ($t_0$), expandidas a **38 características** tras *One-Hot Encoding*.
   * **Partición Temporal Estricta (*Out-of-Time*):**
     * `Train` ($\le \text{Jun } 2014$): **202,707** registros (75.49% | Tasa de incumplimiento: **20.46%**).
     * `Val` ($\text{Jul } 2014 - \text{Dic } 2014$): **34,988** registros (13.03% | Tasa de incumplimiento: **27.12%**).
     * `Test OOT` ($\text{Ene } 2015 - \text{Dic } 2015$): **30,835** registros (11.48% | Tasa de incumplimiento: **25.46%**).
   * **Desempeño en Prueba (*Out-of-Time*):** El modelo **MLP `[64, 32]`** (`4,801` parámetros entrenables, `BatchNorm`, `ReLU`, `Dropout=0.2`, `AdamW`) alcanza **ROC-AUC = `0.6983`**, **Gini = `0.3966`**, **KS = `0.2923`** y **Brier Score calibrado = `0.1731`**, superando de forma consistente a la Regresión Logística (`39` parámetros, AUC = `0.6939`) a lo largo de **30 semillas independientes** ($\mu_{\text{AUC}} = 0.6970 \pm 0.0014$).
   * **Scorecard Bancaria (`300–850` puntos):** Escalamiento analítico con $P_0 = 600$, $\text{Odds}_0 = 19:1$ ($\text{PD}_0 = 5\%$) y $\text{PDO} = 20$ ($\text{Factor} \approx 28.8539$, $\text{Offset} \approx 515.04$).

2. **Parte 2 — Pérdida Dado el Incumplimiento (LGD) y Pérdida Esperada ($\mathbb{E}[L]$):**
   * Estimación de la severidad de pérdida $\text{LGD} \in [0, 1]$ sobre los **46,009 préstamos castigados (`Charged Off`)** (y análisis de sensibilidad con **47,227 registros** incluyendo `Default`), comparando regresores lineales (Ridge), modelos de dos etapas y redes neuronales MLP con salida sigmoide acotada para el cálculo de la provisión financiera $\mathbb{E}[L]_i = \text{PD}_i \cdot \text{LGD}_i \cdot \text{EAD}_i$.

---

## 📂 Estructura del Repositorio

```text
rn-trabajo-2/
├── app/
│   └── app.py                         # Aplicación web interactiva en Streamlit (Scorecard, Percentil y Contrafactual)
├── blog/
│   ├── reporte.md                     # Fuente Markdown del reporte técnico completo
│   └── index.html                     # Reporte web autocontenido (5 pestañas, MathJax e imágenes Base64)
├── configs/
│   ├── base.yaml                      # Semilla global (seed=42), rutas y parámetros de Scorecard
│   ├── pd.yaml                        # Configuración completa del pipeline de Parte 1 (PD y MLP)
│   ├── demo.yaml                      # Configuración rápida para pruebas de humo (demo)
│   ├── lgd.yaml                       # Configuración de búsqueda de hiperparámetros para Parte 2 (LGD)
│   ├── lgd_best.yaml                  # Configuración del modelo ganador de LGD para 30 semillas
│   └── lgd_demo.yaml                  # Configuración rápida de demostración para LGD
├── data/
│   ├── raw/                           # Datos crudos de Kaggle (loan.csv y LCDataDictionary.xlsx)
│   └── processed/
│       ├── pd/                        # Particiones procesadas (train.parquet, test.parquet, reference_dist.parquet)
│       └── lgd/                       # Población objetivo LGD (lgd_target.parquet y lgd_target_with_default.parquet)
├── docs/
│   ├── uso_ia.md                      # Bitácora transparente de prompts e impacto de la IA
│   └── caceria_alucinacion.md         # 5 hallazgos reales documentados en la Cacería de la Alucinación
├── models/
│   ├── pd/                            # Pesos entrenados (mlp_best.pt, lr_baseline.joblib, preprocessor.joblib)
│   └── lgd/                           # Artefactos y pesos entrenados del modelo LGD
├── outputs/
│   ├── figures/                       # Gráficas de alta resolución (300 DPI) para PD y LGD
│   └── tables/                        # Resultados de búsqueda, estabilidad en 30 semillas, PFI y métricas JSON
├── scripts/
│   ├── build_dataset.py               # Construcción y auditoría del dataset a partir de data/raw/loan.csv
│   ├── run_pd.py                      # Pipeline completo de entrenamiento, evaluación, 30 semillas y PFI (Parte 1)
│   ├── run_search.py                  # Búsqueda estructurada de arquitecturas e hiperparámetros PD
│   ├── run_seeds.py                   # Evaluación de estabilidad estocástica en 30 semillas para PD
│   ├── make_figures.py                # Generación de las 6 figuras de la Parte 1 con pruebas estadísticas
│   ├── build_lgd_target.py            # Construcción de la variable objetivo LGD (Charged Off / Default)
│   ├── eda_lgd.py                     # Análisis exploratorio y gráficas de distribución bimodal de LGD
│   ├── run_lgd.py                     # Entrenamiento de modelos de referencia y MLP para LGD (Parte 2)
│   ├── run_lgd_seeds.py               # Evaluación de estabilidad en 30 semillas para el mejor MLP de LGD
│   ├── eval_expected_loss.py          # Integración de PD x LGD x EAD y evaluación de Pérdida Esperada
│   └── build_blog.py                  # Compilador del blog HTML en 5 pestañas con imágenes Base64
├── src/creditrisk/                    # Paquete modular Python (data, features, models, evaluation, scorecard)
├── tests/                             # Suite de pruebas unitarias automatizadas (pytest)
├── Makefile                           # Automatización de comandos de ejecución
└── requirements.txt                   # Dependencias del proyecto con versiones fijadas, es importante correrlo antes de realizar cualquier experimento o sección del trabajo.