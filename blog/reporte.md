**Curso:** Redes Neuronales y Algoritmos Bioinspirados — Universidad Nacional de Colombia, Sede Medellín (Facultad de Minas)  
**Semestre:** 2026-02  
**Enlaces del Proyecto:** [Repositorio en GitHub](https://github.com/SaloMuri/rn-trabajo-2) · [Aplicación Web Interactiva (Scorecard)](https://rn-trabajo-2-app.streamlit.app) · [Video Promocional](#)

---

## 1. Introducción y Definición del Caso de Uso

El riesgo de crédito constituye la fuente principal de exposición financiera en las instituciones de préstamo de consumo y plataformas de financiamiento colaborativo (*peer-to-peer lending*). En este trabajo desarrollamos un sistema integral de evaluación de riesgo crediticio sobre datos históricos reales de **Lending Club** (2007–2015), estructurado en dos etapas: la estimación de la **Probabilidad de Incumplimiento (PD, Parte 1)** junto con su traducción operativa a una **Scorecard crediticia**, y la estimación de la **Pérdida Dado el Incumplimiento (LGD, Parte 2)** para cuantificar la **Pérdida Esperada ($\text{EL} = \text{PD} \times \text{LGD} \times \text{EAD}$)**.

### 1.1. Caso de Uso Operativo (Parte 1 — PD y Scorecard)

Para garantizar una coherencia estricta en la selección de variables, el preprocesamiento y el diseño de la aplicación web, definimos los cuatro pilares de nuestro caso de uso:

1. **¿Quién usa el modelo?**  
   El modelo tiene una interfaz dual:
   * **Analista de Riesgo / Comité de originación e inversionistas institucionales** de la plataforma de préstamos, quienes requieren cuantificar la probabilidad exacta de impago ($\text{PD}$) y ubicar al solicitante en una escala estandarizada de puntos (Scorecard) para decidir la aprobación del crédito.
   * **El solicitante de crédito (usuario final)** a través de la aplicación web de autogestión, quien consulta su puntaje crediticio, su percentil frente a la población de deudores y simula escenarios contrafactuales (*"¿Qué pasaría si...?"*) modificando el monto solicitado, el plazo o su nivel de endeudamiento antes de formalizar el contrato.
2. **¿Para qué decisión se utiliza?**  
   Tomar la decisión de **aprobación, revisión manual o rechazo** de una solicitud de préstamo personal en originación (mediante bandas de corte sobre el puntaje de la Scorecard) y suministrar el insumo probabilístico calibrado $\hat{p}_i = \text{PD}_i$ necesario para el cálculo de provisiones y pérdida esperada.
3. **¿En qué momento exacto del flujo de negocio opera?**  
   Opera en el **instante de Originación y Evaluación de la Oferta Contractual (*Underwriting & Pricing Assessment*)**, inmediatamente después de que el solicitante registra sus datos socioeconómicos, se consulta su reporte de buró crediticio y el motor preliminar de tasas estructura las condiciones financieras del préstamo solicitado (`loan_amnt`, `term`, `int_rate`, `installment` y categoría `grade`), pero **antes de que el préstamo sea desembolsado** y empiece su historia de pagos.
4. **¿Qué información está disponible en ese momento (Prevención de Fuga de Datos)?**  
   Únicamente variables observadas en o antes de la originación. Quedan **estrictamente prohibidas** todas las variables post-originación o de seguimiento de cartera (como pagos recibidos `total_pymnt`, capital pendiente `out_prncp`, recuperaciones post-castigo `recoveries`, cargos por cobro `collection_recovery_fee`, fecha del último pago `last_pymnt_d` o puntajes FICO actualizados con posterioridad al desembolso `last_fico_range_high/low`). Asimismo, dado que la versión pública del conjunto de datos en Kaggle omite las columnas originales `fico_range_low` y `fico_range_high`, la tasa de interés asignada en la oferta (`int_rate`) y el grado inicial (`grade`) actúan como las variables observables que condensan dicha calificación en el momento del contrato.

---

## 2. Datos, Auditoría del Diccionario y Esquema de Partición Temporal

### 2.1. Origen y Filtrado de la Variable Objetivo (PD)

Utilizamos el conjunto de datos público de Lending Club publicado por Ranadeep en Kaggle (`loan.csv`), el cual contiene originalmente **887,379 registros y 74 columnas** correspondientes a préstamos originados entre junio de 2007 (`2007-06`) y diciembre de 2015 (`2015-12`).

Durante la auditoría del diccionario oficial (`LCDataDictionary.xlsx`), detectamos que la hoja `browseNotes` contiene variables en formato *camelCase* que no existen en el archivo `LoanStats` (`loan.csv`), y que un conjunto de variables sobre cuentas por cuotas (*installment accounts*, p. ej., `il_util`, `total_bal_il`, `max_bal_bc`) presenta más del 95% de valores faltantes por haberse incorporado apenas a finales de 2015. Por tanto, depuramos el dataset válido a **23 columnas** (incluyendo metadatos de partición/identificación y las **20 variables predictoras de originación** con alta completitud histórica: 14 numéricas continuas, 3 categóricas ordinales y 3 categóricas nominales), las cuales se expanden a **38 características** tras la codificación *One-Hot*.

Para definir la variable objetivo binaria $\text{target\_pd} \in \{0, 1\}$, aplicamos las reglas de negocio sobre `loan_status`:

| Estado Original (`loan_status`) | Codificación PD | Interpretación Financiera |
| :--- | :---: | :--- |
| `Fully Paid` | **0** | Buen pagador confirmado (crédito pagado en su totalidad). |
| `Does not meet the credit policy. Status: Fully Paid` | **0** | Buen pagador confirmado bajo políticas históricas previas. |
| `Charged Off` | **1** | Mal pagador confirmado (crédito castigado por irrecuperabilidad). |
| `Default` | **1** | Mal pagador confirmado (incumplimiento formal del contrato). |
| `Late (31-120 days)` | **1** | Mal pagador confirmado (mora severa superior a 30 días). |
| `Does not meet the credit policy. Status: Charged Off` | **1** | Mal pagador confirmado bajo políticas históricas previas. |
| `Current`, `Issued`, `In Grace Period`, `Late (16-30 days)` | **NA (Excluidos)** | Créditos en curso sin desenlace definitivo al corte de dic-2015 (censura a la derecha). |

Tras descartar los 618,849 préstamos en curso (`NA`), el conjunto de datos válido para la Parte 1 consta de **268,530 préstamos cerrados o en mora severa** (268,530 filas x 23 columnas), con una **tasa global de incumplimiento del 21.90%**.

### 2.2. Justificación de la Partición Temporal (*Out-of-Time Split*)

En modelamiento de riesgo crediticio, realizar una partición aleatoria convencional (`train_test_split`) constituye un error metodológico conocido como **fuga temporal (*look-ahead bias*)**: permitiría entrenar con préstamos de 2015 para predecir préstamos de 2009, ignorando los ciclos macroeconómicos y los cambios en las políticas de originación a lo largo del tiempo.

En consecuencia, implementamos una **partición estrictamente cronológica basada en la fecha de originación (`issue_d`)**:

| Partición | Ventana Temporal (`issue_d`) | Registros ($N$) | % del Total | Tasa de Incumplimiento (PD) | Rol Metodológico |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Entrenamiento (`Train`)** | `2007-06` a `2014-06` (`<= 2014-06-01`) | 202,707 | 75.49% | 20.46% | Ajuste de pesos de los modelos y ajuste exclusivo (`fit`) del preprocesador (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`). |
| **Validación (`Val`)** | `2014-07` a `2014-12` (`<= 2014-12-01`) | 34,988 | 13.03% | 27.12% | Monitoreo de parada temprana (*Early Stopping*) y selección de hiperparámetros (`run_search.py`). |
| **Prueba (`Test Out-of-Time`)** | `2015-01` a `2015-12` (`> 2014-12-01`) | 30,835 | 11.48% | 25.46% | Evaluación final fuera de tiempo (discriminación, calibración, estabilidad en 30 semillas e interpretabilidad). |

**Hallazgo sobre el cambio en la tasa base (Sesgo de Maduración):**  
Se observa que la tasa de incumplimiento pasa del **20.46%** en `Train` al **27.12%** en `Val` y **25.46%** en `Test`. Este fenómeno responde al **sesgo de maduración (*maturity bias*)**: dado que el corte del dataset ocurre en diciembre de 2015, entre los préstamos originados en 2014–2015 que ya habían alcanzado un estado definitivo (no `Current`), están sobrerrepresentados los *incumplimientos tempranos (early defaults)* frente a los buenos pagadores que siguen pagando sus cuotas normales de 36 o 60 meses. Identificar este desplazamiento distributivo (*covariate & prior shift*) será clave al analizar la curva de calibración.

---

## 3. Análisis Descriptivo y Formulación de Hipótesis

Antes de entrenar los modelos, exploramos el conjunto de entrenamiento (`Train`) y formulamos **tres hipótesis explícitas de riesgo** fundamentadas en teoría financiera:

* **Hipótesis 1 (Efecto de las Variables Nominales — `home_ownership`, `verification_status` y `purpose` — sobre el Puntaje de Riesgo frente a la Tasa de Interés `int_rate`):**  
  *“Inicialmente hipotetizamos que las variables categóricas nominales del solicitante (`home_ownership`, `verification_status` y `purpose`) tendrían un alto impacto directo en el puntaje crediticio asignado (y en la probabilidad de incumplimiento); sin embargo, tras aplicar el esquema de codificación (*One-Hot Encoding*), se descubre que su aporte directo al score de riesgo es marginal porque su prima de riesgo es absorbida previamente en la fijación de la tasa de interés (`int_rate`), variable sobre la cual dichas categorías nominales sí ejercen una diferenciación estadística altamente significativa.”*
* **Hipótesis 2 (Estrés Financiero por Capacidad de Pago y Plazo — `dti`, `annual_inc` y `term`):**  
  *“Un mayor nivel de endeudamiento mensual respecto al ingreso (`dti`), menores ingresos anuales (`annual_inc`) y la elección de plazos largos (`term = 60 meses` frente a `36 meses`) incrementan sustancialmente la probabilidad de incumplimiento, existiendo un efecto no lineal de interacción cuando convergen alto `dti` y plazo de 60 meses.”*
* **Hipótesis 3 (Saturación de Cupo Rotativo y Propósito del Préstamo — `revol_util` y `purpose`):**  
  *“Solicitantes con una alta utilización de sus líneas de crédito rotativas (`revol_util > 75%`) o que solicitan préstamos para financiar pequeños negocios (`small_business`) o consolidar deudas (`debt_consolidation`) exhiben un riesgo de impago significativamente mayor que quienes solicitan crédito para compra de vehículo (`car`) o mejoras del hogar (`home_improvement`).”*

![Figura 1: Análisis Descriptivo y Validación Visual de Hipótesis](../reports/figures/pd/fig1_eda_hipotesis.png)

**Análisis descriptivo preliminar y pruebas estadísticas (Figura 1):**  
1. En el **Panel Izquierdo (H1)** se contrasta el efecto de las variables nominales (`home_ownership`, `verification_status` y `purpose`) frente a la tasa de interés (`int_rate`). Las pruebas de análisis de varianza (ANOVA de una vía y Kruskal-Wallis sobre `Train`) confirman que las categorías nominales modifican de forma altamente significativa la tasa de interés cobrada al cliente ($p < 0.0001$): por ejemplo, los solicitantes con ingresos verificados (`Verified` / `Source Verified`), vivienda en arriendo (`RENT`) o propósitos riesgosos reciben tasas `int_rate` significativamente mayores que quienes poseen hipoteca (`MORTGAGE`) o buscan pagar tarjetas (`credit_card`). No obstante, al expandir estas categorías mediante *One-Hot Encoding* dentro del modelo multivariado, es la variable continua `int_rate` la que condensa y absorbe dicho riesgo, reduciendo el peso directo de las dummies nominales sobre el puntaje final.
2. En el **Panel Central (H2)** se evidencia una fuerte interacción: en todos los cuartiles de `dti`, los préstamos a **60 meses** (barras rojas) tienen una tasa de incumplimiento más de 10 puntos porcentuales superior a los de **36 meses** (barras verdes), superando el 35% de impago cuando el cliente combina el cuartil más alto de `dti` (`Q4`) con un plazo de 60 meses.
3. En el **Panel Derecho (H3)**, los préstamos destinados a pequeños negocios (`small_business`) lideran la siniestralidad con cerca del 30% de incumplimiento, seguidos por consolidación de deuda (`debt_consolidation`), mientras que tarjetas de crédito (`credit_card`) y compras mayores (`major_purchase`) presentan tasas menores.

*(En la Sección 5.5 retomaremos estas tres hipótesis para confirmarlas o matizarlas cuantitativamente mediante el análisis de interpretabilidad por permutación de los modelos entrenados).*

---

## 4. Fundamentación Matemática

### 4.1. Formulación Probabilística y Entropía Cruzada Binaria (BCE)
Sea $\mathbf{x}_i \in \mathbb{R}^d$ ($d = 38$) el vector de características preprocesadas del solicitante $i$ en el momento de la originación, y sea $y_i \in \{0, 1\}$ el indicador de incumplimiento. Modelamos la distribución condicional $y_i \mid \mathbf{x}_i \sim \text{Bernoulli}(\hat{p}_i)$, donde $\hat{p}_i = P(y_i = 1 \mid \mathbf{x}_i)$. El estimador de máxima verosimilitud equivale a minimizar la **Entropía Cruzada Binaria**:

$$\mathcal{L}_{\text{BCE}}(\mathbf{\Theta}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(\hat{p}_i) + (1 - y_i) \ln(1 - \hat{p}_i) \right]$$

En nuestra implementación en PyTorch, para evitar inestabilidades numéricas cuando $\hat{p}_i \to 0$ o $\hat{p}_i \to 1$, la capa de salida del MLP produce el *logit* sin acotar $\hat{z}_i \in (-\infty, \infty)$ y aplicamos `BCEWithLogitsLoss`, que utiliza la identidad analítica estable (*log-sum-exp trick*):

$$\ell(\hat{z}_i, y_i) = \max(\hat{z}_i, 0) - \hat{z}_i y_i + \ln\left(1 + e^{-|\hat{z}_i|}\right)$$

### 4.2. Modelo de Referencia (Regresión Logística con Penalización $L_2$)
El modelo de referencia estima una frontera de decisión lineal en el espacio del *log-odds*:

$$\ln\left(\frac{\hat{p}_i}{1 - \hat{p}_i}\right) = \mathbf{w}^T \mathbf{x}_i + b \implies \hat{p}_i = \sigma(\mathbf{w}^T \mathbf{x}_i + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x}_i + b)}}$$

Se entrena minimizando la pérdida logística regularizada con norma $L_2$: $\mathcal{L}_{\text{LR}} = \mathcal{L}_{\text{BCE}}(\mathbf{w}, b) + \frac{1}{2C} \|\mathbf{w}\|_2^2$. Su complejidad paramétrica es de exactamente $d + 1 = 38 + 1 = \mathbf{39\text{ parámetros}}$.

### 4.3. Perceptrón Multicapa (MLP) para Datos Tabulares
Para capturar interacciones no lineales entre variables (p. ej., `int_rate` $\times$ `dti` $\times$ `term`), construimos una red profunda de $L$ capas ocultas donde cada bloque oculto $l \in \{1, \dots, L\}$ ejecuta:

$$\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}, \qquad \mathbf{a}^{(l)} = \text{ReLU}\left(\text{BatchNorm1d}(\mathbf{z}^{(l)})\right), \qquad \mathbf{h}^{(l)} = \text{Dropout}(\mathbf{a}^{(l)}, p_{\text{drop}})$$

con $\mathbf{h}^{(0)} = \mathbf{x}$ y $\text{ReLU}(u) = \max(0, u)$. La normalización por lotes (`BatchNorm1d`) estabiliza la distribución de activaciones intermedias, mientras que `Dropout` y el decaimiento de pesos desacoplado en `AdamW` ($\lambda = 10^{-4}$) previenen la sobre-parametrización.

### 4.4. Métricas de Discriminación y Calibración
* **AUC-ROC e Índice de Gini:** El AUC cuantifica la probabilidad de que un deudor moroso elegido al azar reciba un puntaje de riesgo mayor que un deudor cumplido: $\text{AUC} = P(\hat{p}_{\text{malo}} > \hat{p}_{\text{bueno}})$. El coeficiente de Gini reescala linealmente el AUC al intervalo $[-1, 1]$ (donde $0$ es azar y $1$ discriminación perfecta):
  $$\text{Gini} = 2 \cdot \text{AUC} - 1$$
* **Estadístico Kolmogorov-Smirnov (KS):** Mide la distancia vertical máxima entre la distribución acumulada empírica de las probabilidades predichas para los buenos pagadores ($F_0$) y para los malos pagadores ($F_1$):
  $$\text{KS} = \max_{t \in [0, 1]} \left| F_0(t) - F_1(t) \right|$$
* **Puntaje de Brier (Calibración):** Evalúa el error cuadrático medio entre las probabilidades continuas predichas y los eventos binarios observados:
  $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$$

### 4.5. Derivación Matemática de la Scorecard Continua
Definimos las *odds* a favor de que el cliente sea un **buen pagador** como $\text{Odds}_{\text{bueno}} = \frac{1 - \hat{p}}{\hat{p}}$. Imponemos una transformación afín sobre el logaritmo de las *odds*:

$$\text{Score}(\hat{p}) = \text{Offset} + \text{Factor} \cdot \ln\left(\frac{1 - \hat{p}}{\hat{p}}\right)$$

Fijando un **puntaje base $P_0 = 600$ puntos** para unas **odds de referencia $\theta_0 = 19:1$** (equivalente a una tasa de incumplimiento de referencia $\text{PD}_0 = \frac{1}{1 + 19} = 5\%$) y exigiendo que cada aumento de **$\text{PDO} = 20$ puntos** duplique las *odds* de ser buen pagador ($2\theta_0$), planteamos el sistema:

$$\begin{cases} P_0 = \text{Offset} + \text{Factor} \cdot \ln(\theta_0) \\ P_0 + \text{PDO} = \text{Offset} + \text{Factor} \cdot \ln(2\theta_0) \end{cases}$$

Restando ambas ecuaciones se obtiene:

$$\text{PDO} = \text{Factor} \cdot \ln(2) \implies \mathbf{\text{Factor} = \frac{\text{PDO}}{\ln(2)} \approx 28.8539}$$

$$\mathbf{\text{Offset} = P_0 - \text{Factor} \cdot \ln(\theta_0) = 600 - 28.8539 \cdot \ln(19) \approx 515.0413}$$

Así, nuestro modelo neuronal continuo traduce cualquier probabilidad $\hat{p}_i$ directamente en puntos mediante:

$$\text{Score}(\hat{p}_i) = 515.0413 + 28.8539 \cdot \ln\left(\frac{1 - \hat{p}_i}{\hat{p}_i}\right)$$

---

## 5. Parte 1: Modelamiento de la Probabilidad de Incumplimiento (PD)

### 5.1. Preprocesamiento Único y Regla de Comparación Justa
Para garantizar una comparación estrictamente justa entre la Regresión Logística y el MLP:
1. Ambos modelos reciben exactamente la misma matriz $\mathbf{X} \in \mathbb{R}^{N \times 38}$ producida por `models/pd/preprocessor.joblib`.
2. Las variables numéricas y ordinales se imputan con la **mediana de `Train`** (robusta a valores extremos en ingresos y saldos) y se escalan con `StandardScaler` ($\mu = 0, \sigma = 1$).
3. Las variables categóricas nominales (`home_ownership`, `verification_status`, `purpose`) se codifican con `OneHotEncoder(handle_unknown="ignore")`.

### 5.2. Búsqueda de Hiperparámetros y Presupuesto Computacional (`run_search.py`)
Exploramos sistemáticamente **6 configuraciones de Regresión Logística** y **6 arquitecturas de Perceptrón Multicapa** evaluadas sobre el conjunto de validación (`Val`):

| Familia | ID | Descripción de la Configuración | Parámetros | Tiempo (s) | Val AUC | Val Gini | Val KS | Val Brier |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LogisticRegression** | `LR_1` | `C=0.001, weight=None` | 39 | 2.12 | 0.7092 | 0.4184 | 0.3100 | 0.1793 |
| **LogisticRegression** | `LR_2` | `C=0.01, weight=None` | 39 | 1.89 | 0.7089 | 0.4178 | 0.3099 | 0.1795 |
| **LogisticRegression** | `LR_3` | `C=0.1, weight=None` | 39 | 1.11 | 0.7087 | 0.4174 | 0.3096 | 0.1796 |
| **LogisticRegression** | `LR_4` | `C=1.0, weight=None` | 39 | 1.29 | 0.7087 | 0.4174 | 0.3096 | 0.1796 |
| **LogisticRegression** | `LR_5` | `C=10.0, weight=None` | 39 | 1.39 | 0.7087 | 0.4174 | 0.3096 | 0.1796 |
| **LogisticRegression** | `LR_6` | `C=1.0, weight=balanced` | 39 | 1.71 | 0.7089 | 0.4178 | 0.3084 | **0.2250** |
| **TabularMLP** | `MLP_1` | `dims=[64, 32], act=relu, do=0.15, lr=0.001` | **4,801** | 210.80 | **0.7166** | **0.4333** | **0.3180** | **0.1764** |
| **TabularMLP** | `MLP_2` | `dims=[128, 64], act=relu, do=0.25, lr=0.001` | 13,697 | 222.69 | **0.7167** | **0.4334** | **0.3181** | **0.1764** |
| **TabularMLP** | `MLP_3` | `dims=[128, 64, 32], act=relu, do=0.25, lr=0.001` | 15,809 | 280.88 | 0.7157 | 0.4314 | 0.3155 | 0.1773 |
| **TabularMLP** | `MLP_4` | `dims=[128, 64, 32], act=gelu, do=0.25, lr=0.001` | 15,809 | 299.57 | 0.7157 | 0.4314 | 0.3171 | 0.1770 |
| **TabularMLP** | `MLP_5` | `dims=[256, 128, 64], act=relu, do=0.35, lr=0.0005` | 52,097 | 112.52 | 0.7139 | 0.4277 | 0.3137 | 0.1762 |
| **TabularMLP** | `MLP_6` | `dims=[128, 64, 32], act=tanh, do=0.20, lr=0.001` | 15,809 | 125.36 | 0.7125 | 0.4250 | 0.3166 | 0.1794 |

**Justificación de las decisiones arquitectónicas a partir de la búsqueda:**
1. **Descarte de `class_weight="balanced"` (`LR_6`):** Aunque mantiene el AUC en `0.7089`, eleva el Brier Score de `0.1796` a `0.2250` porque distorsiona las probabilidades predichas hacia $0.50$, arruinando la calibración requerida por la Scorecard.
2. **Selección por Principio de Parsimonia (`MLP_1` vs. `MLP_2` y `MLP_5`):** Las arquitecturas de 2 capas ocultas (`MLP_1` y `MLP_2`) superaron a las de 3 capas (`MLP_3`, `MLP_5`). Entre `MLP_1` (`[64, 32]`, AUC `0.7166`) y `MLP_2` (`[128, 64]`, AUC `0.7167`), la diferencia es de apenas `0.0001` en AUC, pero **`MLP_1` requiere casi 3 veces menos parámetros (4,801 vs. 13,697)**. Por el principio de parsimonia, seleccionamos la arquitectura compacta `[64, 32]` con activación `ReLU`, `BatchNorm1d`, `Dropout = 0.15`, optimizador `AdamW` ($\text{lr} = 10^{-3}$, $\text{weight\_decay} = 10^{-4}$) y parada temprana.

![Figura 2: Curvas de Entrenamiento y Parada Temprana del MLP](../reports/figures/pd/fig2_curvas_entrenamiento_mlp.png)

Como muestra la **Figura 2** (y los registros de `mlp_training_history.csv`), la pérdida de entrenamiento desciende suavemente de `0.5179` (época 1) a `0.4595` (época 15), mientras que el AUC en validación alcanza su máximo global en la **época 10 (`Val AUC = 0.7165`, `Val Loss = 0.5300`)**. El mecanismo de *Early Stopping* detiene el entrenamiento en la época 15 tras 5 épocas sin mejora y restaura los pesos óptimos de la época 10.

### 5.3. Evaluación en el Conjunto de Prueba (*Out-of-Time*) y Estabilidad en 30 Semillas

#### Comparación Justa en Test Out-of-Time (`2015-01` a `2015-12`)

| Modelo | Parámetros Entrenables | Tiempo de Entrenamiento (s) | Val AUC | Test AUC | Test Gini | Test KS | Test Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Regresión Logística (Baseline)** | 39 | 3.89 | 0.7087 | 0.6899 | 0.3799 | 0.2829 | 0.1756 |
| **Perceptrón Multicapa (MLP)** | 4,801 / 15,809 | 93.50 | **0.7165** | **0.6983** | **0.3966** | **0.2923** | **0.1731** |

#### Evaluación de Robustez Estadística con 30 Semillas Independientes (`seeds_summary_pd.csv`)
Para verificar que la superioridad del MLP (`[64, 32]`, **4,801 parámetros**) no es producto de una semilla afortunada en la inicialización de pesos o el orden de los lotes, entrenamos **30 redes independientes** (`seed = 100` a `129`) evaluadas sobre `Test`:

| Métrica en Test Out-of-Time | Media (30 Semillas) | Desviación Estándar | Mejor Semilla | Peor Semilla | Nº Semillas | Parámetros MLP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AUC-ROC** | **0.6970** | 0.0014 | 0.6989 | 0.6939 | 30 | 4,801 |
| **Índice de Gini** | **0.3940** | 0.0029 | 0.3977 | 0.3878 | 30 | 4,801 |
| **Estadístico KS** | **0.2918** | 0.0035 | 0.2982 | 0.2848 | 30 | 4,801 |
| **Puntaje de Brier** | **0.1732** | 0.0003 | 0.1727 | 0.1736 | 30 | 4,801 |

**Análisis de estabilidad:** La desviación estándar del AUC a través de las 30 semillas es de apenas **`0.0014`** (`0.14%`), e incluso la **peor semilla del MLP (`AUC = 0.6939`, `Gini = 0.3878`, `KS = 0.2848`, `Brier = 0.1736`) supera holgadamente a la Regresión Logística (`AUC = 0.6899`, `Gini = 0.3799`, `KS = 0.2829`, `Brier = 0.1756`)**. Esto demuestra con rigor estadístico que las no-linealidades aprendidas por la red neuronal aportan una ganancia real y reproducible tanto en discriminación como en calibración.

![Figura 3: Curvas ROC y Separación Kolmogorov-Smirnov (KS) en Test](../reports/figures/pd/fig3_roc_y_ks.png)

![Figura 4: Curva de Calibración en Test Out-of-Time](../reports/figures/pd/fig4_curva_calibracion.png)

**Interpretación de Discriminación (Figura 3) y Calibración (Figura 4):**
* En la **Figura 3**, la curva ROC del MLP domina uniformemente a la de la Regresión Logística, y la curva KS muestra una separación máxima de casi 30 puntos porcentuales (`KS = 0.2923`–`0.2982`) alrededor del umbral de probabilidad $\hat{p} \approx 0.24$.
* En la **Figura 4 (Curva de Calibración)**, ambos modelos siguen de cerca la diagonal ideal, ubicándose ligeramente por encima de la diagonal en los deciles medios. Esto refleja exactamente el desplazamiento de la tasa base explicado en la Sección 2.2: los modelos fueron entrenados en un periodo histórico con tasa de impago del **20.46%**, mientras que la cohorte de prueba (`Test 2015`) exhibe un **25.46%** de impago debido a la maduración temprana de los siniestros. Aun así, el MLP logra un **Brier Score menor (`0.1731` vs. `0.1756`)**, ajustándose mejor en las colas de alto riesgo.

### 5.4. Representación del Modelo mediante la Scorecard Crediticia

![Figura 5: Distribución de Puntajes de la Scorecard y Siniestralidad por Decil](../reports/figures/pd/fig5_scorecard_distribucion.png)

Al aplicar la transformación derivada en la Sección 4.5 ($P_0 = 600$, $\theta_0 = 19$, $\text{PDO} = 20$) a las probabilidades continuas predichas por el MLP en el conjunto de prueba (**Figura 5**):
* La distribución de los **buenos pagadores** se desplaza claramente hacia puntajes altos (mediana alrededor de **558 puntos**), mientras que la de los **malos pagadores** se concentra en puntajes bajos (mediana cercana a **538 puntos**).
* En el **Panel Derecho de la Figura 5**, la relación entre el puntaje promedio por decil y la tasa real observada de incumplimiento es **estrictamente monotónica decreciente**: los solicitantes en el decil inferior de puntaje ($\sim 495$ puntos) sufren una tasa real de impago superior al **52%**, mientras que los solicitantes en el decil superior ($\sim 605$ puntos) registran una tasa de impago inferior al **6%**. Esto valida operativamente el uso de bandas de corte en nuestra aplicación web:
  * **$\text{Score} \ge 565$ pts:** Riesgo Bajo (Aprobación Estándar).
  * **$535 \le \text{Score} < 565$ pts:** Riesgo Moderado (Revisión de Analista).
  * **$\text{Score} < 535$ pts:** Riesgo Alto (Rechazo o exigencia de colateral).

### 5.5. Interpretabilidad de Variables de Riesgo y Verificación de Hipótesis

Para identificar qué variables determinan el riesgo de incumplimiento en una red neuronal multicapa sin depender de supuestos lineales, empleamos el método de **Importancia por Permutación (*Permutation Feature Importance*, PFI)** propuesto por Breiman (2001) y formalizado por Fisher, Rudin y Dominici (2019). El método consiste en permutar aleatoriamente los valores de cada característica $j$ en el conjunto de prueba (`Test Out-of-Time`, con $5$ repeticiones independientes) y medir la **caída media en el AUC-ROC** ($\Delta \text{AUC}_j = \text{AUC}_{\text{original}} - \text{AUC}_{\text{permutado}, j}$).

![Figura 6: Importancia de Variables por Permutación en Test (MLP vs. Regresión Logística)](../reports/figures/pd/fig6_interpretabilidad_pfi.png)

#### Tabla de las 15 Variables de Mayor Impacto en el Riesgo (`feature_importance_pd.csv`)

| Posición | Variable (`feature`) | Caída Media en AUC — MLP | Desv. Estándar — MLP | Caída Media en AUC — Regresión Logística |
| :---: | :--- | :---: | :---: | :---: |
| **1** | `int_rate` (Tasa de interés del crédito) | **0.04947** | 0.00252 | 0.08943 |
| **2** | `annual_inc` (Ingreso anual del solicitante) | **0.02393** | 0.00166 | 0.01369 |
| **3** | `dti` (Relación deuda mensual / ingreso) | **0.01380** | 0.00056 | 0.01493 |
| **4** | `total_acc` (Total histórico de líneas de crédito) | **0.01144** | 0.00051 | 0.01349 |
| **5** | `revol_util` (% de utilización de cupo rotativo) | **0.00794** | 0.00027 | 0.00639 |
| **6** | `term` (Plazo del préstamo: 36 vs. 60 meses) | **0.00641** | 0.00052 | 0.00370 |
| **7** | `loan_amnt` (Monto del préstamo solicitado) | **0.00432** | 0.00038 | 0.00554 |
| **8** | `installment` (Cuota mensual del crédito) | **0.00348** | 0.00057 | -0.00024 |
| **9** | `delinq_2yrs` (Moras >30 días en últimos 2 años) | **0.00340** | 0.00042 | 0.00218 |
| **10** | `revol_bal` (Saldo rotativo total actual) | **0.00325** | 0.00038 | 0.00016 |
| **11** | `open_acc` (Líneas de crédito abiertas actualmente) | **0.00276** | 0.00041 | 0.00177 |
| **12** | `home_ownership_MORTGAGE` (Vivienda con hipoteca) | **0.00274** | 0.00016 | 0.01526 |
| **13** | `pub_rec` (Registros públicos derogatorios) | **0.00228** | 0.00035 | 0.00023 |
| **14** | `home_ownership_RENT` (Vivienda en arriendo) | **0.00172** | 0.00015 | -0.00138 |
| **15** | `inq_last_6mths` (Consultas a buró últimos 6 meses) | **0.00152** | 0.00035 | 0.00084 |
| *18* | `purpose_small_business` (Propósito: pequeño negocio) | *0.00092* | *0.00012* | *0.00081* |
| *19* | `purpose_debt_consolidation` (Consolidación deuda) | *0.00091* | *0.00017* | *0.00160* |
| *20* | `grade` (Grado crediticio ordinal A–G) | *0.00091* | *0.00026* | *0.00039* |

#### Contraste de las Hipótesis Iniciales frente a la Evidencia Empírica:
1. **Verificación de la Hipótesis 1 — CONFIRMADA COMPLETAMENTE:**  
   Aunque intuitivamente se creía que las variables nominales (`home_ownership`, `verification_status`, `purpose`) tendrían un peso protagónico sobre el puntaje de la Scorecard, la tabla de Importancia por Permutación demuestra que tras el *One-Hot Encoding* su impacto directo en el MLP es reducido: las categorías de vivienda aparecen recién en los puestos #12 (`home_ownership_MORTGAGE`, $\Delta \text{AUC} = 0.00274$) y #14 (`home_ownership_RENT`, $\Delta \text{AUC} = 0.00172$), las de propósito en los puestos #18 y #19 ($\Delta \text{AUC} \approx 0.0009$), y `verification_status` ni siquiera ingresa al Top 15. En cambio, tal como planteamos en el hallazgo de la hipótesis, estas variables nominales afectan de manera decisiva la **tasa de interés (`int_rate`)**, la cual ocupa la posición **#1 absoluta del modelo** ($\Delta \text{AUC} = 0.04947$ en el MLP y $0.08943$ en la Regresión Logística). Es decir, el motor de originación penaliza las categorías nominales riesgosas asignándoles una mayor `int_rate`, y la red neuronal captura dicho riesgo directamente a través de la tasa de interés y de sus interacciones no lineales con la solvencia del deudor.
2. **Verificación de la Hipótesis 2 — CONFIRMADA COMPLETAMENTE:**  
   La capacidad de pago y la estructura del crédito ocupan los lugares #2, #3, #6, #7 y #8 del ranking del MLP: `annual_inc` (`0.02393`), `dti` (`0.01380`), `term` (`0.00641`), `loan_amnt` (`0.00432`) e `installment` (`0.00348`). Más aún, el MLP logra extraer casi el doble de señal informativa del ingreso anual (`0.02393` vs. `0.01369` en LR) y del plazo `term` (`0.00641` vs. `0.00370` en LR), y aprovecha la cuota mensual `installment` (`0.00348`), la cual en la Regresión Logística tenía importancia nula (`-0.00024`) por su colinealidad con el monto y la tasa.
3. **Verificación de la Hipótesis 3 — CONFIRMADA PARCIALMENTE (MATIZADA):**  
   Se **confirma** la relevancia de la utilización de cupo en tarjetas rotativas (`revol_util`), que se posiciona como la **5.ª variable más importante** de toda la red neuronal (`0.00794`) junto con el saldo rotativo `revol_bal` (#10, `0.00325`). Sin embargo, en concordancia con la Hipótesis 1, se **matiza** el peso marginal del propósito del préstamo (`purpose`): aunque en el análisis descriptivo univariado `small_business` tenía la mayor tasa de impago bruta, en el modelo multivariado ocupa la posición #18 (`0.00092`) y `debt_consolidation` la #19 (`0.00091`). ¿Por qué? Porque cuando un cliente pide un préstamo para `small_business`, la plataforma ya le asigna una `int_rate` más alta; por ende, el efecto de riesgo del propósito queda parcialmente mediado por la tasa de interés y el endeudamiento del solicitante.

---

## 6. Parte 2: Pérdida Dado el Incumplimiento (LGD) y Pérdida Esperada (EL)

> **[ESPACIO RESERVADO PARA DOCUMENTAR LA PARTE 2 DEL TRABAJO]**
> 
> ### 6.1. Construcción de la Variable Objetivo de LGD y Limitaciones
> * *TODO Parte 2:* Documentar la subpoblación de préstamos incumplidos (`target_pd == 1`), la fórmula financiera empleada a partir de `recoveries`, `funded_amnt` / `out_prncp` y el acotamiento al intervalo $[0, 1]$.
> 
> ### 6.2. Arquitectura del MLP para LGD y Función de Pérdida
> * *TODO Parte 2:* Justificar la activación de salida (p. ej., sigmoide acotada en $[0, 1]$) y comparar contra el modelo lineal de referencia en términos de MAE y RMSE bajo el mismo esquema de 30 semillas.
> 
> ### 6.3. Estimación de la Pérdida Esperada ($\text{EL} = \text{PD} \times \text{LGD} \times \text{EAD}$)
> * *TODO Parte 2:* Integrar las predicciones de PD (Parte 1) y LGD (Parte 2) con la Exposición al Incumplimiento ($\text{EAD} = \text{loan\_amnt}$) sobre el conjunto de prueba.

---

## 7. Discusión General y Aprendizajes (Parte 1)

1. **Superioridad consistente pero acotada de las Redes Neuronales en datos tabulares:**  
   En problemas de riesgo crediticio tabular con variables numéricas monotónicas, una Regresión Logística bien preprocesada con apenas **39 parámetros** y **3.9 segundos** de cómputo alcanza un `Test AUC = 0.6899`. Nuestro Perceptrón Multicapa (`[64, 32]`, **4,801 parámetros**) eleva el desempeño en Test a un **`AUC = 0.6983` (media de 30 semillas: `0.6970 ± 0.0014`)**, un **`Gini = 0.3966`** y un **`KS = 0.2923`**, reduciendo simultáneamente el error de calibración (**`Brier = 0.1731`**). Aunque la ganancia de ~0.84 puntos de AUC (~1.7 puntos de Gini) exige mayor costo computacional, en carteras bancarias masivas esa mejora en la separación de colas evita millones de dólares en colocaciones fallidas.
2. **El peligro de sobre-parametrizar redes en datos tabulares:**  
   La búsqueda de hiperparámetros demostró que incrementar la red a 3 capas y 52,097 parámetros (`[256, 128, 64]`) deterioró la generalización (`Val AUC = 0.7139`), mientras que una arquitectura piramidal ligera de 2 capas (`[64, 32]`) maximizó el desempeño y produjo una varianza entre semillas extremadamente baja ($\sigma_{\text{AUC}} = 0.0014$).
3. **Importancia crítica de la validación temporal (*Out-of-Time*) sobre la calibración:**  
   Evaluar en el año 2015 (`Test`) reveló cómo el sesgo de maduración de las cohortes recientes eleva la tasa base de incumplimiento respecto al periodo 2007–2014, un fenómeno invisible si se hubiera empleado una partición aleatoria.

---

## 8. Resumen de Uso de IA y Cacería de la Alucinación

El detalle exhaustivo de los prompts y verificaciones se encuentra versionado en `docs/uso_ia.md` y `docs/caceria_alucinacion.md`. Destacamos los cinco hallazgos principales verificados por el equipo:
1. **Auditoría del diccionario (`browseNotes` vs. `LoanStats`):** Detección de 15 variables inexistentes en `loan.csv`, ausencia de columnas `fico_range` en la versión pública de Kaggle y >95% de nulos en variables `_il` pre-2015.
2. **Incompatibilidad de entorno Unix vs. Windows CMD:** Corrección de llamadas `PYTHONPATH=src` y `make` mediante inyección programática de rutas con `pathlib` en `scripts/` y `tests/`.
3. **Degradación del Brier Score por balanceo artificial (`LR_6`):** Demostración empírica de que `class_weight="balanced"` degrada el Brier Score de `0.1796` a `0.2250`.
4. **Ausencia estructural en `mths_since_last_delinq`:** Sustitución de la imputación por mediana (que asignaba moras falsas a clientes limpios) por el indicador binario `has_prev_delinq`.
5. **Alineación del Caso de Uso frente a Variables Nominales e `int_rate`:** Fundamentación del modelo en la etapa de *Underwriting* contractual y análisis de cómo el *One-Hot Encoding* de variables nominales (`home_ownership`, `verification_status`, `purpose`) delega el poder predictivo directo sobre `int_rate` (#1).

---

## 9. Referencias Bibliográficas (Formato APA 7)

* Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324
* Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1–3. https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2
* Fisher, A., Rudin, C., & Dominici, F. (2019). All models are wrong, but many are useful: Learning a variable's importance by studying an entire class of prediction models simultaneously. *Journal of Machine Learning Research*, 20(177), 1–81. http://jmlr.org/papers/v20/18-760.html
* Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press.
* Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. En *Proceedings of the 32nd International Conference on Machine Learning* (pp. 448–456). PMLR.
* Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. En *International Conference on Learning Representations (ICLR)*. https://openreview.net/forum?id=Bkg6RiCqY7
* Ranadeep. (2017). *Lending Club Loan Data* [Conjunto de datos]. Kaggle. https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset
* Siddiqi, N. (2017). *Intelligent credit scoring: Building and implementing better credit risk scorecards* (2.ª ed.). John Wiley & Sons. https://doi.org/10.1002/9781119282396
* Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, 15(56), 1929–1958.