# Estimación de Probabilidad de Incumplimiento (PD) Y Pérdida Dado el Incumplimiento (LGD) Mediante Redes Neuronales 

**Curso:** Redes Neuronales y Algoritmos Bioinspirados — Semestre 2026-02

**Institución:** Universidad Nacional de Colombia, Sede Medellín — Facultad de Minas

**Autores:** Salome Murillo Gaviria, Nicolas David Zambrano Murcia, Oky Ruiz De La Rosa, Andrés Pérez Cano

**Enlaces del Proyecto:** [Repositorio en GitHub](https://github.com/SaloMuri/rn-trabajo-2) · [Aplicación Web Interactiva (Scorecard)](https://rn-trabajo-2-app.streamlit.app) · [Video Promocional](#)


---

## 1. Introducción y Motivación

La gestión cuantitativa del riesgo de crédito constituye la piedra angular de la estabilidad del sistema financiero moderno y de la asignación eficiente de capital en entidades bancarias y plataformas *fintech*. Bajo los marcos regulatorios internacionales de **Basilea III** (Basel Committee on Banking Supervision [BCBS], 2017) y el estándar contable **IFRS 9** (International Accounting Standards Board [IASB], 2014), las instituciones crediticias están obligadas a provisionar capital en función de la **Pérdida Esperada (*Expected Loss*, EL)** de sus portafolios de préstamos. Formalmente, para un crédito individual $i$, la pérdida esperada se descompone como el producto de tres parámetros fundamentales:

$$\text{EL}_i = \text{PD}_i \cdot \text{LGD}_i \cdot \text{EAD}_i$$

donde $\text{PD}_i \in (0,1)$ representa la **Probabilidad de Incumplimiento (*Probability of Default*)**, $\text{LGD}_i \in [0,1]$ denota la **Pérdida Dado el Incumplimiento (*Loss Given Default*)** y $\text{EAD}_i > 0$ corresponde a la **Exposición al Momento del Incumplimiento (*Exposure at Default*)** (Siddiqi, 2017; Thomas, Crook, & Edelman, 2017).

Históricamente, la industria bancaria ha estimado la $\text{PD}$ mediante modelos lineales generalizados, particularmente la **Regresión Logística**, debido a su interpretabilidad directa y facilidad para traducir coeficientes en tarjetas de puntuación (*scorecards*) aditivas (Hand & Henley, 1997). Sin embargo, la relación entre las variables financieras de un solicitante —tales como su razón deuda-ingreso (*Debt-to-Income*, DTI), utilización de cupo rotativo e historial de morosidad— y el evento de incumplimiento exhibe fuertes no-linealidades e interacciones de orden superior que un hiperplano lineal no puede capturar sin ingeniería manual exhaustiva de variables (Lessmann, Baesens, Seow, & Thomas, 2015).

El presente reporte técnico aborda rigurosamente la **Parte 1 (Estimación de PD y Construcción de Scorecard)** y la **Parte 2 (Estimación de LGD y Pérdida Esperada)**, desarrollada en la sección 5. Evaluamos comparativamente un modelo de referencia lineal (**Regresión Logística regularizada**) frente a una arquitectura de **Perceptrón Multicapa (*Multilayer Perceptron*, MLP)** entrenada con PyTorch sobre el conjunto de datos real de préstamos personales de *LendingClub* (2007–2015).

---

## 2. Caso de Uso y Formulación del Problema de Negocio

### 2.1. Contexto Operativo
Consideremos el comité de riesgo de crédito de una entidad originadora de préstamos de consumo. Cuando un solicitante demanda un crédito por un monto $M$ a un plazo de 36 o 60 meses, el sistema de originación debe responder dos preguntas críticas en tiempo real:
1. **Decisión de Originación y *Pricing*:** ¿Cuál es la probabilidad $\hat{p}_i = \mathbb{P}(Y_i = 1 \mid \mathbf{x}_i)$ de que el deudor incumpla sus obligaciones contractuales antes del vencimiento?
2. **Clasificación en Escala Maestra de Riesgo (*Scorecard*):** ¿Cómo transformar dicha probabilidad en un puntaje crediticio continuo \(S_i\), comprensible para analistas de crédito y útil para apoyar la evaluación del riesgo de cada solicitud?

### 2.2. Costos de Error Asimétricos
En el otorgamiento de crédito, los errores de clasificación tienen impactos financieros fuertemente asimétricos:
* **Falso Negativo (Aprobar a un deudor que entra en *default*):** Implica la pérdida directa del capital no recuperado ($\text{LGD} \cdot \text{EAD}$), la cual suele oscilar entre el $50\%$ y el $85\%$ del principal desembolsado.
* **Falso Positivo (Rechazar a un buen pagador):** Representa un costo de oportunidad equivalente al margen financiero neto (intereses menos costo de fondeo), típicamente entre el $8\%$ y el $18\%$ anual.

Por consiguiente, un modelo de riesgo crediticio no puede evaluarse únicamente por su exactitud global (*accuracy*), sino por su **capacidad de ordenamiento y discriminación** (**ROC-AUC**, **Índice de Gini**, **Estadístico KS**) y por la **calibración probabilística** de sus predicciones (**Brier Score**), requisito indispensable para que el cálculo de provisiones $\text{EL}_i$ no subestime ni sobreestime el capital regulatorio requerido.

---

## 3. Datos, Prevención de Fuga de Información (*Data Leakage*) e Hipótesis

### 3.1. Construcción de la Variable Objetivo y Filtrado del Universo
El conjunto de datos original de *LendingClub* (`accepted_2007_to_2018Q4.csv`) contiene **887,379 registros** y **74 variables**. Para construir un problema de clasificación binaria libre de sesgo de supervivencia, se filtraron exclusivamente los préstamos con desenlace crediticio definitivo:
* **Buenos Pagadores ($Y = 0$):** Estado `Fully Paid`.
* **Incumplimiento / Default ($Y = 1$):** Estados `Charged Off`, `Default`, `Does not meet the credit policy. Status:Charged Off` y mora severa `Late (31-120 days)`.

Los préstamos activos (`Current`, `In Grace Period`, `Issued`) equivalentes a **618,849 registros** fueron excluidos del modelado de PD debido a que su estado final aún no es observable (censura por la derecha). El universo limpio resultante consta exactamente de **268,530 préstamos cerrados** originados entre junio de 2007 y diciembre de 2015, con una **tasa global de incumplimiento del 21.90%**.

### 3.2. Auditoría Anti-*Leakage* y Partición Temporal (*Out-of-Time*)
Uno de los errores metodológicos más graves en la literatura aplicada sobre *LendingClub* es incluir variables registradas *después* de la originación o después del evento de mora (por ejemplo: `recoveries`, `total_rec_prncp`, `last_pymnt_amnt`, `out_prncp`). Nuestro pipeline (`src/creditrisk/data/features.py`) restringe estrictamente el espacio de entrada a **19 variables ex-ante** conocidas al instante de la solicitud (ver detalle completo en la **Tabla 1**), las cuales tras el preprocesamiento y codificación *One-Hot* de variables categóricas se expanden a una matriz numérica de **$D = 38$ características** (conformando un dataset limpio de 23 columnas al incluir metadatos de fecha y estado).

**Tabla 1.** *Resumen de la construcción del universo de datos y partición temporal Out-of-Time para el modelo PD.*

| Partición / Subconjunto | Ventana Temporal (`issue_d`) | Número de Registros ($N$) | Proporción del Total (%) | Tasa de Incumplimiento ($\bar{Y}$) | Propósito en el Pipeline |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Entrenamiento (*Train Set*)** | $\le \text{2014-06-01}$ | $202,707$ | $75.49\%$ | $20.46\%$ | Ajuste de pesos ($\mathbf{W}$), escaladores y cuantiles de referencia PSI |
| **Validación (*Val Set*)** | $(\text{2014-06-01}, \text{2014-12-01}]$ | $34,988$ | $13.03\%$ | $27.12\%$ | *Early stopping* y selección de hiperparámetros |
| **Prueba (*Test Set Out-of-Time*)** | $> \text{2014-12-01}$ | $30,835$ | $11.48\%$ | $25.46\%$ | Evaluación final ciega de generalización temporal |
| **Total Universo Válido (PD 0/1)** | **2007-06 a 2015-12** | **$268,530$** | **$100.00\%$** | **$21.90\%$** | **Portafolio cerrado auditado sin fuga de información** |

*Fuente:* Elaboración propia a partir de la ejecución del pipeline de datos (`src/creditrisk/data/build_dataset.py`) sobre el dataset de *LendingClub* (2007–2015).

Asimismo, en lugar de una partición aleatoria que mezclaría el futuro con el pasado, implementamos una **partición estrictamente temporal (*Out-of-Time Split*)** basada en la fecha de emisión (`issue_d`), cuyos volúmenes y tasas de incumplimiento se detallan en la **Tabla 1**. Se observa un desplazamiento temporal real (*covariate & prior shift*): las cohortes más recientes (2014–2015) presentan tasas de incumplimiento superiores ($25.46\%–27.12\%$) frente al histórico previo ($20.46\%$), lo cual somete a los modelos a una prueba realista de estrés operacional.

### 3.3. Hipótesis del análisis exploratorio

El análisis exploratorio busca identificar relaciones entre las características de los solicitantes y el incumplimiento crediticio (*default*). A partir de la teoría financiera y la exploración estadística del conjunto de entrenamiento ($N = 202,707$ préstamos cerrados en `train.parquet`), se plantean tres hipótesis explícitas y falsables que orientan el análisis, pero no implican que las relaciones observadas sean causales. 

Estas hipótesis se contrastarán mediante estadísticas descriptivas, visualizaciones y pruebas de asociación, y serán reevaluadas al final del modelamiento mediante análisis estadístico e Importancia por Permutación (PFI).

### Hipótesis de Investigación

**H1. Tipo de vivienda, tasa de interés e incumplimiento (Panel A de la Figura 1).** Planteamos como hipótesis inicial ingenua que las variables categóricas nominales del solicitante (`home_ownership`, `verification_status` y `purpose`) tendrían un peso dominante directo en la predicción del puntaje de incumplimiento (PD). Sin embargo, postulamos que, tras controlar por la codificación One-Hot y la estructura financiera del crédito, esta premisa podría refutarse parcialmente en su efecto directo (su aporte marginal individual en PFI sería menor frente a las variables cuantitativas y ordinales), debido a que parte del riesgo asociado al perfil nominal podría estar reflejado en la tasa de interés (`int_rate`) fijada por la entidad al originar el crédito. Para explorar esta hipótesis, se comparan las tasas de incumplimiento y las tasas de interés medias entre categorías de tenencia de vivienda, por ejemplo, arrendatarios (`RENT`) frente a deudores hipotecarios (`MORTGAGE`). Las diferencias observadas permiten identificar asociaciones, pero no demostrar que la tasa de interés medie causalmente la relación entre la vivienda y el incumplimiento.

**H2. Endeudamiento, utilización del crédito e incumplimiento (Panel B de la Figura 1).** Se plantea que los solicitantes con una mayor relación deuda-ingreso (`dti`), combinada con una utilización elevada de su cupo de crédito rotativo (`revol_util` $\geq 75\%$), presentan una mayor fragilidad financiera ante choques de liquidez, lo que se reflejaría en un incremento de la tasa observada de incumplimiento a través de los cuartiles de endeudamiento. Para explorar esta hipótesis, se compara la tasa de incumplimiento entre grupos definidos por intervalos de `dti` y por el umbral de utilización del crédito renovable del 75 %. Este umbral se utiliza como criterio descriptivo de segmentación y no como un límite universal de riesgo.

**H3. Ingresos e incumplimiento (Panel C de la Figura 1).** Se plantea que el ingreso anual percibido (`annual_inc`) opera como un amortiguador de solvencia frente al servicio de la deuda: a medida que el solicitante se ubica en cuartiles superiores de ingreso anual, se espera que la frecuencia empírica de incumplimiento disminuya de forma monótona y sostenida. Para evaluarlo, se comparan las tasas observadas de incumplimiento entre cuartiles de ingresos y se examina la asociación monotónica entre los ingresos y la variable objetivo mediante la correlación de Spearman. Los resultados permitirán determinar si los datos respaldan esta hipótesis.



![Figura 1: Validación exploratoria de las tres hipótesis de negocio en el conjunto de entrenamiento.](../reports/figures/pd/fig1_eda_hipotesis.png)
**Figura 1.** *Análisis exploratorio de las tres hipótesis de riesgo en el conjunto de entrenamiento ($N = 202,707$): (A) tenencia de vivienda frente a tasa de interés e incumplimiento; (B) tasa de incumplimiento por cuartiles de DTI y utilización rotativa; (C) tasa de incumplimiento por cuartiles de ingreso anual.*  
*Fuente:* Elaboración propia mediante el módulo de visualización `scripts/make_figures.py`.

---

## 4F. Fundamentación Matemática

### 4F.1. Función de Pérdida: Entropía Cruzada Binaria Ponderada y Estabilidad Numérica
Sea $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ el conjunto de entrenamiento, donde $\mathbf{x}_i \in \mathbb{R}^D$ ($D=38$) y $y_i \in \{0, 1\}$. El modelo produce un *logit* real $z_i = f_{\boldsymbol{\theta}}(\mathbf{x}_i) \in \mathbb{R}$, el cual se proyecta al espacio de probabilidades mediante la función logística sigmoide:

$$\hat{p}_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}}$$

Bajo el principio de Máxima Verosimilitud sobre una distribución de Bernoulli, minimizar el logaritmo negativo de la verosimilitud equivale a minimizar la **Entropía Cruzada Binaria (*Binary Cross-Entropy*, BCE)**. Para mitigar eventuales desbalances y controlar el costo relativo de la clase minoritaria ($y_i=1$) mediante un peso positivo $w_+ > 0$, la pérdida por observación se define como:

$$\mathcal{L}_i(z_i, y_i) = -\left[ w_+ y_i \ln \sigma(z_i) + (1 - y_i) \ln (1 - \sigma(z_i)) \right]$$

Evaluar $\ln \sigma(z_i)$ de forma ingenua provoca desbordamiento numérico (*underflow/overflow*) cuando $|z_i| \gg 0$. En `PyTorch` (`BCEWithLogitsLoss`), sustituimos analíticamente $\sigma(z_i)$ aplicando la identidad **Log-Sum-Exp**:

$$\mathcal{L}_i(z_i, y_i) = (1 - y_i) z_i + \left(1 + (w_+ - 1)y_i\right) \left[ \max(0, -z_i) + \ln\left(1 + e^{-|z_i|}\right) \right]$$

Esta formulación es numéricamente estable para cualquier $z_i \in (-\infty, +\infty)$. Su gradiente respecto al *logit* $z_i$ adopta la forma limpia:

$$\frac{\partial \mathcal{L}_i}{\partial z_i} = \sigma(z_i)\left(1 + (w_+ - 1)y_i\right) - w_+ y_i$$

Cuando $w_+ = 1$, se recupera la expresión clásica $\frac{\partial \mathcal{L}_i}{\partial z_i} = \hat{p}_i - y_i$, mostrando que el impulso de retropropagación es directamente proporcional al error de calibración residual.

### 4F.2. Modelo de Referencia: Regresión Logística con Regularización $L_2$
El modelo base (*baseline*) parametriza el *logit* como una combinación afín de las características estandarizadas:

$$z_i = \mathbf{w}^\top \mathbf{x}_i + b \implies \ln\left(\frac{\hat{p}_i}{1 - \hat{p}_i}\right) = \sum_{j=1}^{D} w_j x_{ij} + b$$

Esto implica que la frontera de decisión $\{\mathbf{x} \in \mathbb{R}^D : \hat{p}(\mathbf{x}) = \tau\}$ es estrictamente un hiperplano lineal y que el cambio en el *log-odds* ante una variación unitaria en $x_j$ es constante ($w_j$), independientemente de los valores que tomen las demás variables. El problema de optimización incorpora penalización de Tikhonov ($L_2$):

$$\min_{\mathbf{w}, b} \frac{1}{N} \sum_{i=1}^N \mathcal{L}_i(\mathbf{w}^\top \mathbf{x}_i + b, y_i) + \frac{\lambda}{2} \|\mathbf{w}\|_2^2$$

### 4F.3. Arquitectura de Red Neuronal Multicapa (MLP)
Para superar la rigidez aditiva de la Regresión Logística, implementamos en `src/creditrisk/models/mlp.py` un Perceptrón Multicapa de $L$ capas ocultas. Para una entrada $\mathbf{h}^{(0)} = \mathbf{x}$, cada bloque oculto $l \in \{1, \dots, L\}$ ejecuta la composición de cuatro operadores:

$$\begin{aligned}
\mathbf{u}^{(l)} &= \mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}, \quad &\mathbf{W}^{(l)} \in \mathbb{R}^{d_l \times d_{l-1}} \\
\tilde{\mathbf{u}}^{(l)} &= \text{BatchNorm}\left(\mathbf{u}^{(l)}\right) = \boldsymbol{\gamma}^{(l)} \odot \frac{\mathbf{u}^{(l)} - \boldsymbol{\mu}_{\mathcal{B}}}{\sqrt{\boldsymbol{\sigma}_{\mathcal{B}}^2 + \epsilon}} + \boldsymbol{\beta}^{(l)} \\
\mathbf{a}^{(l)} &= \phi\left(\tilde{\mathbf{u}}^{(l)}\right) \\
\mathbf{h}^{(l)} &= \text{Dropout}_p\left(\mathbf{a}^{(l)}\right) = \frac{1}{1 - p} (\mathbf{m}^{(l)} \odot \mathbf{a}^{(l)}), \quad &m_k^{(l)} \sim \text{Bernoulli}(1-p)
\end{aligned}$$

Y la capa de salida proyecta la última representación latente al escalar *logit*:

$$z = \mathbf{w}_{\text{out}}^\top \mathbf{h}^{(L)} + b_{\text{out}}$$

**Justificación de los componentes arquitectónicos:**
1. **Batch Normalization (Ioffe & Szegedy, 2015):** Estabiliza la distribución de las activaciones intermedias frente a variables financieras con asimetría residual, permitiendo tasas de aprendizaje mayores y actuando como regularizador implícito.
2. **Activación No Lineal $\phi(\cdot)$ (ReLU / GELU):** Introduce particiones poliedrales (con $\text{ReLU}(u) = \max(0, u)$) o curvatura suave ($\text{GELU}(u) = u \Phi(u)$), habilitando al modelo para aprender interacciones multiplicativas implícitas como $\text{DTI} \times \text{Tasa de Interés}$ sin especificarlas manualmente.
3. **Dropout (Srivastava et al., 2014) y Decaimiento de Pesos ($L_2$ en AdamW):** Previenen la co-adaptación de neuronas ocultas sobre ruido idiosincrático del conjunto de entrenamiento.

### 4F.4. Métricas de Evaluación en Riesgo Crediticio
1. **Área Bajo la Curva ROC (ROC-AUC) e Índice de Gini:**  
   El $\text{AUC}$ equivale a la probabilidad de que el modelo asigne una $\text{PD}$ estrictamente mayor a un cliente incumplido elegido al azar que a un cliente cumplido elegido al azar:
   $$\text{AUC} = \mathbb{P}\left(\hat{p}(\mathbf{x}_+) > \hat{p}(\mathbf{x}_-) \mid Y_+ = 1, Y_- = 0\right)$$
   En la banca comercial, se reporta estándarmente el **Coeficiente de Gini**, relacionado linealmente con el AUC mediante:
   $$\text{Gini} = 2 \cdot \text{AUC} - 1 \in [-1, 1]$$
2. **Estadístico de Kolmogorov-Smirnov (KS):**  
   Cuantifica la distancia vertical máxima entre las funciones de distribución acumulada empírica de las probabilidades predichas para los clientes en *default* ($F_1(t)$) y los pagadores ($F_0(t)$):
   $$\text{KS} = \sup_{t \in [0, 1]} \left| F_1(t) - F_0(t) \right|$$
   El umbral $t^*$ donde se alcanza el supremo define el punto de corte estadísticamente óptimo para separar ambas poblaciones.
3. **Brier Score (Métrica de Calibración):**  
   Mide el error cuadrático medio entre las probabilidades pronosticadas y los eventos binarios realizados:
   $$\text{BS} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$$

### 4F.5. Derivación Matemática del *Scorecard* Bancario (*PDO Scaling*)
En la industria financiera, las probabilidades $\hat{p}_i$ se transforman en un puntaje aditivo mediante una función afín del logaritmo de las probabilidades a favor de pago ($\text{Odds}_{\text{bueno}} = \frac{1 - \hat{p}_i}{\hat{p}_i}$) (Siddiqi, 2017):

$$\text{Score}(\hat{p}_i) = \text{Offset} + \text{Factor} \cdot \ln\left(\frac{1 - \hat{p}_i}{\hat{p}_i}\right)$$

Para determinar las constantes $\text{Offset}$ y $\text{Factor}$, se imponen dos condiciones de diseño institucional:
1. Un puntaje base $S_0$ debe corresponder a unos *odds* de referencia $\theta_0$:
   $$S_0 = \text{Offset} + \text{Factor} \cdot \ln(\theta_0)$$
2. Un incremento de **$\text{PDO}$ (*Points to Double the Odds*)** puntos debe duplicar exactamente los *odds* de buen pagador ($2\theta_0$):
   $$S_0 + \text{PDO} = \text{Offset} + \text{Factor} \cdot \ln(2\theta_0)$$

Restando la primera ecuación de la segunda, se despejan analíticamente los parámetros implementados en `src/creditrisk/scorecard.py`:

$$\text{Factor} = \frac{\text{PDO}}{\ln(2)}, \qquad \text{Offset} = S_0 - \text{Factor} \cdot \ln(\theta_0)$$

Con nuestra configuración estándar ($S_0 = 600$, $\theta_0 = 19$ equivalente a $\text{PD}_0 = 5\%$, y $\text{PDO} = 20$), se obtiene $\text{Factor} \approx 28.8539$ y $\text{Offset} \approx 515.04$. Notemos que, dado que $\ln\left(\frac{1-\hat{p}_i}{\hat{p}_i}\right) = -z_i$, el puntaje bancario es simplemente una reescalación lineal inversa del *logit* de salida de la red neuronal: $\text{Score}_i = \text{Offset} - \text{Factor} \cdot z_i$, expresado en una escala de puntuación con un puntaje base de 600 puntos, odds de referencia de 19:1 y un PDO de 20 puntos.

### 4F.6. Regresión Fraccionaria para la LGD: Salida Sigmoide, BCE como Cuasi-Verosimilitud y Contraste con MSE

**Problema de regresión fraccionaria.** El objetivo $y_i = \text{LGD}_i \in [0,1]$ tiene masa en $y=1$ (46.4 %) y valores continuos en $(0,1)$. Una distribución Beta no sirve, porque su densidad no admite $y=0$ ni $y=1$ exactos. Se modela la media condicional $\mathbb{E}[y \mid \mathbf{x}]$ con una red de salida sigmoide:

$$\hat{y}_i = \sigma(z_i), \quad z_i = f_{\boldsymbol{\theta}}(\mathbf{x}_i), \quad \sigma(z)=\frac{1}{1+e^{-z}} \in (0,1)$$

La media condicional es lo que necesita la pérdida esperada, porque $\mathbb{E}[\text{EL}] = \text{PD}\cdot\mathbb{E}[\text{LGD}]\cdot\text{EAD}$ trabaja con valores esperados.

**Pérdida C: entropía cruzada binaria (BCE) con objetivo fraccionario.** Si $y \in \{0,1\}$, la verosimilitud de Bernoulli es $p^{y}(1-p)^{1-y}$ y su logaritmo negativo da la BCE. La misma expresión se puede evaluar cuando $y\in[0,1]$:

$$\mathcal{L}_{\text{BCE}}(y,p) = -\left[\,y\ln p + (1-y)\ln(1-p)\,\right]$$

Ya no es una verosimilitud exacta, sino una **cuasi-verosimilitud** (Gourieroux, Monfort, & Trognon, 1984; Papke & Wooldridge, 1996). Tiene tres propiedades:

1. *Descomposición.* $\mathcal{L}_{\text{BCE}}(y,p) = H(y) + D_{\text{KL}}\!\left(\text{Bern}(y)\,\|\,\text{Bern}(p)\right)$, con $H(y) = -y\ln y - (1-y)\ln(1-y)$. El término $D_{\text{KL}} \ge 0$ vale cero solo si $p=y$. Por tanto, **el mínimo de la BCE no es cero**, sino la entropía del objetivo $H(y)$. Por eso las pérdidas de entrenamiento de nuestras redes están alrededor de 0.27 y eso no indica un mal ajuste.
2. *Consistencia de la media.* Para $\mathbb{E}[y\mid\mathbf{x}]=m$, $\;\frac{d}{dp}\mathbb{E}[\mathcal{L}]= -\frac{m}{p} + \frac{1-m}{1-p} = 0 \Rightarrow p = m$, y la segunda derivada es positiva. El mínimo se alcanza en la media condicional **aunque $y$ no sea Bernoulli**. Esto exige acertar solo el primer momento, no la distribución completa.
3. *Gradiente limpio.* Con $p=\sigma(z)$, $\;\partial\mathcal{L}_{\text{BCE}}/\partial z = p - y$.

**Pérdida B (contraste): error cuadrático con salida sigmoide.** Si se supone $y\mid\mathbf{x}\sim\mathcal{N}(p,\sigma^2)$, entonces $-\ln f(y) = \frac{(y-p)^2}{2\sigma^2} + \text{cte}$, y maximizar la verosimilitud equivale a minimizar $\mathcal{L}_{\text{MSE}}=(y-p)^2$. Pero un objetivo acotado y con masa en 1 no es gaussiano, y el gradiente respecto al logit es:

$$\frac{\partial\mathcal{L}_{\text{MSE}}}{\partial z} = 2\,(p-y)\,p\,(1-p)$$

Este gradiente **se anula cuando $p\to 0$ o $p\to 1$**: justo en la región donde está casi la mitad de los datos, la red recibe señal débil aunque se equivoque. Ese es el argumento teórico para preferir BCE. La Sección 5.4 (Resultados y evaluación) verifica si se sostiene en la práctica.

### 4F.7. Métricas de Evaluación de la LGD: MAE y RMSE

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|, \qquad \text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}$$

El MAE equivale al estimador de máxima verosimilitud bajo ruido de Laplace y su minimizador es la **mediana** condicional. El RMSE equivale al de ruido gaussiano y su minimizador es la **media** condicional. Con una masa tan grande en 1, la mediana y la media pueden diferir bastante. Los modelos estiman la media, así que el RMSE es la métrica coherente con su pérdida, y el MAE se reporta como complemento. El RMSE penaliza más los errores grandes.

### 4F.8. Inicialización de la Salida del MLP de LGD

El sesgo de la capa de salida se inicializa en $\text{logit}(\bar{y}_{\text{train}})$, de modo que la red parta prediciendo la media del entrenamiento. Sin esta inicialización, en la prueba en modo demo la red predecía en promedio 0.70 (frente a 0.92 observado) y subajustaba; se reproduce con `python scripts/run_lgd.py --demo --set mlp.init_output_bias=false`.

### 4F.9. Pérdida Esperada como Valor Esperado

La pérdida esperada de un crédito se obtiene combinando los tres componentes. Se supone que, dado $\mathbf{x}$, la severidad (LGD, EAD) es independiente de que el incumplimiento ocurra, de modo que $\mathbb{E}[\text{pérdida}\mid\mathbf{x}] \approx \text{PD}(\mathbf{x})\cdot\mathbb{E}[\text{LGD}\mid\mathbf{x}]\cdot\mathbb{E}[\text{EAD}\mid\mathbf{x}]$.

---

## 4. Parte 1: Modelamiento de Probabilidad de Incumplimiento (PD)

### 4.1. Diseño Experimental y Búsqueda de Hiperparámetros
Para comparar ambas familias de modelos, se realizó una búsqueda de hiperparámetros mediante `scripts/run_search.py`. Se evaluaron seis configuraciones de Regresión Logística y seis de Perceptrón Multicapa (MLP), utilizando el ROC-AUC del conjunto de validación (\(N = 34{,}988\)) como principal criterio de comparación. La Tabla 2 presenta los resultados de las 12 configuraciones. Aunque `MLP_4` obtuvo el mayor ROC-AUC de validación, la evaluación final se realizó con la configuración `MLP_1`, que tiene una arquitectura más pequeña y menor tiempo de entrenamiento registrado. Por tanto, `MLP_1` se presenta como la configuración utilizada en la evaluación final, no como la ganadora absoluta de la búsqueda por ROC-AUC.

**Tabla 2.** *Resultados de la búsqueda de hiperparámetros para Regresión Logística y Perceptrón Multicapa (MLP).*

| ID | Familia | Configuración | Parámetros | Tiempo (s) | Val AUC | Val Gini | Val KS | Val Brier |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|
| LR_1 | Regresión Logística | C=0.001; sin ponderación | 39 | 0.42 | 0.7089 | 0.4178 | 0.3081 | 0.1793 |
| LR_2 | Regresión Logística | C=0.01; sin ponderación | 39 | 0.68 | 0.7086 | 0.4172 | 0.3101 | 0.1794 |
| LR_3 | Regresión Logística | C=0.1; sin ponderación | 39 | 0.43 | 0.7084 | 0.4168 | 0.3094 | 0.1795 |
| LR_4 | Regresión Logística | C=1.0; sin ponderación | 39 | 0.41 | 0.7084 | 0.4168 | 0.3093 | 0.1795 |
| LR_5 | Regresión Logística | C=10.0; sin ponderación | 39 | 0.42 | 0.7084 | 0.4168 | 0.3093 | 0.1794 |
| LR_6 | Regresión Logística | C=1.0; ponderación balanceada | 39 | 0.54 | 0.7085 | 0.4171 | 0.3083 | 0.2257 |
| MLP_1 | MLP | Capas [64, 32]; ReLU; dropout 0.15; LR=0.001 | 4.801 | 20.99 | 0.7146 | 0.4292 | 0.3139 | 0.1769 |
| MLP_2 | MLP | Capas [128, 64]; ReLU; dropout 0.25; LR=0.001 | 13.697 | 24.31 | 0.7143 | 0.4286 | 0.3149 | 0.1768 |
| MLP_3 | MLP | Capas [128, 64, 32]; ReLU; dropout 0.25; LR=0.001 | 15.809 | 29.96 | 0.7158 | 0.4316 | 0.3153 | 0.1788 |
| MLP_4 | MLP | Capas [128, 64, 32]; GELU; dropout 0.25; LR=0.001 | 15.809 | 51.76 | 0.7160 | 0.4321 | 0.3146 | 0.1774 |
| MLP_5 | MLP | Capas [256, 128, 64]; ReLU; dropout 0.35; LR=0.0005 | 52.097 | 70.30 | 0.7154 | 0.4308 | 0.3151 | 0.1763 |
| MLP_6 | MLP | Capas [128, 64, 32]; tanh; dropout 0.20; LR=0.001 | 15.809 | 52.05 | 0.7122 | 0.4244 | 0.3143 | 0.1796 |

*Fuente:* Elaboración propia a partir de `reports/tables/pd/hyperparameter_search.csv`, generado por `scripts/run_search.py`. Las métricas de esta tabla corresponden al conjunto de validación; no deben interpretarse como métricas de prueba.

**Hallazgos de la búsqueda de hiperparámetros (Tabla 2):**

1. **Mayor AUC de validación:** la configuración `MLP_4`, con capas `[128, 64, 32]` y activación GELU, obtiene el AUC de validación más alto de las configuraciones evaluadas (0.7160). Sin embargo, requiere un tiempo registrado de 51.76 segundos, frente a 20.99 segundos para `MLP_1`, cuya arquitectura `[64, 32]` tiene 4,801 parámetros entrenables.

2. **Efecto de la ponderación de clases en la calibración:** la configuración logística `LR_6`, con `class_weight=balanced`, obtiene un Brier Score de 0.2257, superior al de las configuraciones logísticas sin ponderación (entre 0.1793 y 0.1795). En este experimento, la ponderación balanceada empeora el Brier Score; este resultado no demuestra por sí solo que la causa sea un desplazamiento específico de los logits ni que el mismo efecto ocurra en todas las configuraciones.

3. **Configuración usada en la evaluación final:** las métricas posteriores de comparación, scorecard e interpretación corresponden al MLP configurado en `configs/pd.yaml` (`[64, 32]`, ReLU, dropout 0.15), no a `MLP_4`. Por ello, los resultados finales deben presentarse como la evaluación de esa configuración concreta, sin afirmar que es la configuración con el mayor AUC de validación de toda la búsqueda.

### 4.2. Dinámica de Convergencia y Regularización del MLP
La **Figura 2** muestra la evolución de la pérdida *Binary Cross-Entropy* (BCE) y del ROC-AUC en validación durante el entrenamiento del MLP con arquitectura `[64, 32]`. El entrenamiento utiliza el optimizador AdamW, una tasa de aprendizaje de 0.001, *dropout* de 0.15, normalización por lotes (*Batch Normalization*) y un criterio de parada temprana (*Early Stopping*) con paciencia de 4 épocas, según `configs/pd.yaml` y `src/creditrisk/models/mlp.py`. No se utiliza un planificador de tasa de aprendizaje `ReduceLROnPlateau` en esta implementación.

En el historial registrado (`reports/tables/pd/mlp_training_history.csv`) aparecen 11 épocas. El mayor ROC-AUC de validación registrado es **0.7146 en la época 7**. La pérdida de entrenamiento disminuye a lo largo del proceso, mientras que las métricas de validación fluctúan ligeramente; por ello, las curvas permiten observar la evolución del ajuste, pero no bastan por sí solas para afirmar que no existe sobreajuste. El procedimiento conserva los pesos de la época con mejor ROC-AUC de validación.

![Figura 2: Curvas de convergencia de pérdida BCE y evolución de ROC-AUC durante el entrenamiento del MLP.](../reports/figures/pd/fig2_curvas_entrenamiento_mlp.png)
**Figura 2.** *Dinámica de convergencia de la pérdida Binary Cross-Entropy (izquierda) y evolución del ROC-AUC (derecha) por época para el modelo MLP [64, 32].*  
*Fuente:* Elaboración propia a partir del historial `reports/tables/pd/mlp_training_history.csv`.

### 4.3. Evaluación Comparativa en Conjunto de Prueba (*Out-of-Time*) y Poder Discriminante
Al evaluar ambos modelos en el conjunto de prueba temporal (*Test Set Out-of-Time*), el Perceptrón Multicapa obtiene mejores resultados que la Regresión Logística en las cuatro métricas registradas en `reports/tables/pd/comparison_pd.csv`. El ROC-AUC aumenta de 0.6902 a 0.6960, el índice de Gini de 0.3804 a 0.3919 y el estadístico KS de 0.2839 a 0.2941. El Brier Score disminuye de 0.1754 a 0.1735, lo que indica una mejora en esta métrica de error probabilístico. Estas diferencias describen los resultados de la ejecución registrada; por sí solas, no demuestran que la mejora sea estadísticamente significativa ni que se mantenga en todas las semillas.

**Tabla 3.** *Comparación de desempeño de la Regresión Logística y el MLP en la evaluación final.*

| Modelo | Parámetros entrenables | Tiempo registrado (s) | AUC validación | AUC prueba | Gini prueba | KS prueba | Brier prueba |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Regresión Logística (baseline) | 39 | 0.39 | 0.7084 | 0.6902 | 0.3804 | 0.2839 | 0.1754 |
| MLP `[64, 32]` | 4.801 | 23.58 | 0.7146 | 0.6960 | 0.3919 | 0.2941 | 0.1735 |

*Fuente:* Elaboración propia a partir de `reports/tables/pd/comparison_pd.csv`. Los valores corresponden a la ejecución registrada en ese archivo; los tiempos dependen del entorno de ejecución.

![Figura 3: Comparación de Curvas ROC y separación de distribuciones acumuladas KS en el conjunto de prueba.](../reports/figures/pd/fig3_roc_y_ks.png)
**Figura 3.** *Comparación de curvas ROC (izquierda) y separación de distribuciones acumuladas KS (derecha) en el conjunto de prueba Out-of-Time ($N = 30,835$).*  
*Fuente:* Elaboración propia mediante `scripts/make_figures.py` a partir de las predicciones de prueba (`reports/tables/pd/test_predictions.csv`).

### 4.4. Calibración Probabilística, Estabilidad Estocástica (30 Semillas) y *Scorecard*
La **Figura 4** presenta las curvas de calibración (*Reliability Diagram*) de ambos modelos, construidas con 10 intervalos de igual amplitud en el conjunto de prueba. La tasa global observada de incumplimiento es del 25.46 %, mientras que las probabilidades medias predichas son del 22.76 % para la Regresión Logística y del 23.89 % para el MLP. En promedio, ambos modelos subestiman la frecuencia observada de incumplimiento en aproximadamente 2.70 y 1.57 puntos porcentuales, respectivamente. Sin embargo, las diferencias entre las probabilidades predichas y las frecuencias observadas varían entre intervalos, por lo que no puede afirmarse que exista una subestimación uniforme del 3–4 %. El cambio temporal entre los datos históricos y la cohorte de prueba constituye un posible factor contextual, pero estos resultados descriptivos no bastan para establecer que sea la causa de las diferencias de calibración.

![Figura 4: Curva de calibración probabilística (Reliability Diagram) en el conjunto de prueba.](../reports/figures/pd/fig4_curva_calibracion.png)
**Figura 4.** *Curvas de calibración probabilística (Reliability Diagram) de la Regresión Logística y el MLP en el conjunto de prueba ($N = 30,835$).* 
*Fuente:* Elaboración propia mediante `scripts/make_figures.py`.

Para evaluar la estabilidad del Perceptrón Multicapa, se realizaron 30 ejecuciones con distintas semillas aleatorias, identificadas en el script `scripts/run_seeds.py`. Se resumen la media, la desviación estándar, el mejor resultado y el peor resultado de cada métrica de evaluación. Esta evaluación permite observar la variabilidad entre ejecuciones del modelo; por sí sola, no demuestra significancia estadística frente a la Regresión Logística ni garantiza que el rendimiento se mantenga en otros conjuntos de datos.

**Tabla 4.** *Variabilidad de las métricas del MLP en 30 ejecuciones con distintas semillas.*

| Métrica | Media | Desviación estándar | Mejor resultado | Peor resultado | Número de semillas |
|:---|---:|---:|---:|---:|---:|
| ROC-AUC | 0.6968 | 0.0014 | 0.7002 | 0.6943 | 30 |
| Gini | 0.3937 | 0.0027 | 0.4003 | 0.3887 | 30 |
| KS | 0.2916 | 0.0025 | 0.2957 | 0.2862 | 30 |
| Brier Score | 0.1732 | 0.0003 | 0.1726 | 0.1737 | 30 |

*Fuente:* Elaboración propia a partir de `reports/tables/pd/seeds_summary_pd.csv`. Los resultados corresponden a las 30 ejecuciones registradas. En ROC-AUC, Gini y KS, un valor mayor representa un mejor resultado; en Brier Score, un valor menor representa un menor error probabilístico.

Como análisis complementario, se comparó el ROC-AUC de ambos modelos sobre las mismas observaciones del conjunto de prueba mediante un bootstrap pareado y estratificado por clase, con 5.000 remuestreos y semilla aleatoria fija. La Regresión Logística obtuvo un ROC-AUC de 0.6902, mientras que el MLP alcanzó 0.6960, con una diferencia de 0.0058 a favor de la red neuronal. El intervalo de confianza bootstrap del 95% para esta diferencia fue [0.0036, 0.0080], sin incluir cero. Estos resultados respaldan una ventaja de discriminación del MLP en el conjunto de prueba analizado. Sin embargo, no se reporta un valor p porque no se realizó una prueba de hipótesis específica, y la diferencia observada no demuestra por sí sola que la mejora se deba a la captura de relaciones no lineales.

Las probabilidades estimadas por el MLP se transforman en puntajes mediante la escala de *scorecard* definida en `src/creditrisk/scorecard.py`. En el conjunto de prueba, los puntajes observados se encuentran aproximadamente entre 500.50 y 744.71 puntos. La Figura 5 presenta la distribución de los puntajes para buenos pagadores e incumplidos y la tasa de incumplimiento observada por decil de puntaje. Esta tasa disminuye del 48.70 % en el decil de menor puntaje al 5.80 % en el decil de mayor puntaje. El resultado muestra una asociación descriptiva entre mayores puntajes y menor riesgo observado en este conjunto de prueba; no obstante, estos deciles no constituyen por sí solos umbrales operativos validados para aprobar o rechazar solicitudes.

![Figura 5: Distribución del Credit Score y tasa de incumplimiento real por banda de riesgo del Scorecard.](../reports/figures/pd/fig5_scorecard_distribucion.png)
**Figura 5.** *Distribución de puntajes del Scorecard por estado crediticio (izquierda) y tasa de incumplimiento observada por decil de puntaje (derecha).* 
*Fuente:* Elaboración propia mediante `scripts/make_figures.py` y `src/creditrisk/scorecard.py`.

### 4.5. Interpretabilidad Post-Hoc (PFI) y Verificación de Hipótesis
Para interpretar el modelo MLP se empleó la importancia por permutación (*Permutation Feature Importance*, PFI), utilizando la disminución del AUC en el conjunto de evaluación al alterar los valores de cada variable. Una mayor disminución indica que el modelo depende más de esa característica para discriminar entre solicitantes que incumplen y aquellos que no lo hacen. Esta medida refleja la importancia de una variable para el modelo, pero no implica causalidad.

Los resultados registrados en `reports/tables/pd/feature_importance_pd.csv` muestran que las variables con mayor importancia para el MLP son:


**Tabla 5.** *Top 10 de variables según la caída media del ROC-AUC del MLP en el conjunto de prueba.*

| Ranking | Variable | Definición | Caída media en AUC del MLP | Desviación estándar |
|:---:|:---|:---|---:|---:|
| 1 | `int_rate` | Tasa de interés del préstamo | 0.03906 | 0.00237 |
| 2 | `annual_inc` | Ingreso anual declarado por el prestatario durante el registro | 0.02128 | 0.00158 |
| 3 | `dti` | Relación entre los pagos mensuales de deuda del prestatario —excluyendo la hipoteca y el préstamo solicitado— y sus ingresos mensuales declarados | 0.01355 | 0.00045 |
| 4 | `revol_util` | Tasa de utilización del crédito rotativo: proporción del crédito disponible que está utilizando el prestatario | 0.01095 | 0.00056 |
| 5 | `total_acc` | Número total de líneas de crédito registradas actualmente en el historial crediticio del prestatario | 0.00980 | 0.00024 |
| 6 | `loan_amnt` | Importe del préstamo solicitado por el prestatario; refleja el importe registrado por el departamento de crédito si este lo reduce | 0.00740 | 0.00033 |
| 7 | `term` | Número de pagos del préstamo, expresado en meses; puede ser de 36 o 60 meses | 0.00650 | 0.00055 |
| 8 | `grade` | Calificación crediticia asignada al préstamo por LendingClub | 0.00409 | 0.00037 |
| 9 | `revol_bal` | Saldo total del crédito rotativo | 0.00351 | 0.00037 |
| 10 | `delinq_2yrs` | Número de episodios de morosidad de 30 días o más registrados en el historial crediticio del prestatario durante los dos años anteriores | 0.00262 | 0.00030 |

*Fuente:* Elaboración propia a partir de `reports/tables/pd/feature_importance_pd.csv`, generado por `src/creditrisk/interpretability.py`. Las definiciones se basan en `LCDataDictionary.xlsx`, hoja `LoanStats`.

La tasa de interés (`int_rate`) presenta la mayor importancia por permutación, seguida de los ingresos anuales (`annual_inc`), la relación deuda-ingreso (`dti`) y la utilización del crédito renovable (`revol_util`). Estos resultados indican que las condiciones del crédito, los ingresos y el nivel de endeudamiento aportan información para la predicción del incumplimiento.

También aparecen variables relacionadas con el historial y las condiciones del préstamo, como `total_acc`, `loan_amnt`, `term` y `grade`. En cambio, las variables categóricas individuales asociadas a la tenencia de vivienda presentan una importancia menor que las primeras variables de la clasificación. Por ejemplo, `home_ownership_MORTGAGE` ocupa la posición 11 y `home_ownership_RENT`, la posición 15 en los resultados registrados.

Estos resultados deben interpretarse con cautela. La importancia por permutación puede verse afectada por la correlación entre variables y por la representación de las categorías mediante variables indicadoras (*one-hot encoding*). Además, la importancia predictiva no permite concluir que modificar una variable produzca un cambio causal en el incumplimiento. En consecuencia, la PFI se utiliza como complemento de las métricas de desempeño y del análisis exploratorio, no como prueba causal de las hipótesis planteadas.

![Figura 6: Importancia de variables mediante Permutación (PFI) sobre el modelo MLP en el conjunto de prueba.](../reports/figures/pd/fig6_interpretabilidad_pfi.png)
**Figura 6.** *Importancia de variables por permutación (PFI), medida como la caída media en ROC-AUC ($\Delta \text{AUC}$), para las 12 variables principales del MLP en el conjunto de prueba.*
*Fuente:* Elaboración propia mediante `scripts/make_figures.py` a partir de `reports/tables/pd/feature_importance_pd.csv`.

**Contraste de los resultados con las hipótesis de la Sección 3.3 (a partir de la Figura 1, la Tabla 5 y la Figura 6):**


**Contrastación y Veredicto Final de las Hipótesis de Riesgo:**

*🔄 **Hipótesis 1 — Premisa de Dominancia Directa Refutada Parcialmente / Asociación con la Tasa de Interés (Panel A de la Figura 1 y Tabla 5):** La creencia inicial de que las variables nominales (`home_ownership`, `verification_status` y `purpose`), expandidas mediante One-Hot Encoding en columnas dummy, dominarían directamente el puntaje de la red neuronal no encuentra respaldo en los resultados actuales de importancia por permutación (PFI). En la tabla PFI, `int_rate` ocupa el **primer puesto**, con una caída media de AUC de \(\Delta \text{AUC} = 0.03906 \pm 0.00237\). En contraste, `home_ownership_MORTGAGE` ocupa el puesto 11, con \(\Delta \text{AUC} = 0.00250 \pm 0.00007\), mientras que `home_ownership_RENT` ocupa el puesto 15, con \(\Delta \text{AUC} = 0.00174 \pm 0.00008\). Esto respalda la mayor importancia predictiva individual de la tasa de interés frente a esas dos categorías de vivienda, aunque no permite concluir que todas las variables nominales tengan poca importancia conjunta. En el **Panel A de la Figura 1**, la tasa de interés media es del 13,5 % para `MORTGAGE`, 13,9 % para `OWN`, 14,1 % para `RENT` y 13,5 % para `OTHER`; el ANOVA de una vía detecta diferencias estadísticamente significativas entre las medias de la tasa de interés según la tenencia de vivienda (\(F = 317.3\), \(p < 0.001\)). Además, las tasas de incumplimiento observadas son del 18,7 % para `MORTGAGE`, 21,5 % para `OWN`, 22,3 % para `RENT` y 20,3 % para `OTHER`. Por tanto, los resultados descriptivos respaldan una asociación entre la tenencia de vivienda, la tasa de interés y el incumplimiento. No obstante, estos resultados no demuestran que la tasa de interés sea un mecanismo causal de canalización del riesgo, ni que las diferencias de incumplimiento entre categorías de vivienda sean estadísticamente significativas.*

*✅ **Hipótesis 2 — Respaldada por los Patrones Descriptivos y la Asociación Estadística (Panel B de la Figura 1 y Tabla 5):** Los resultados respaldan la relación positiva entre endeudamiento e incumplimiento. En el **Panel B de la Figura 1**, la tasa de incumplimiento aumenta monótonamente a través de los cuartiles de `dti`: para los solicitantes con utilización rotativa inferior al 75 %, pasa del 14,2 % en Q1 al 25,5 % en Q4; para quienes presentan una utilización igual o superior al 75 %, pasa del 20,3 % al 29,6 %. Además, la correlación de Spearman entre `dti` y el incumplimiento es positiva y estadísticamente significativa (\(\rho = 0.109\), \(p < 0.001\)), aunque de magnitud pequeña. En el modelo neuronal (**Tabla 5**), `dti` ocupa el **tercer puesto** de importancia por permutación, con \(\Delta \text{AUC} = 0.01355 \pm 0.00045\), mientras que `revol_util` ocupa el **cuarto puesto**, con \(\Delta \text{AUC} = 0.01095 \pm 0.00056\). En conjunto, la tendencia observada, la asociación estadística y la importancia predictiva de ambas variables respaldan la hipótesis. Sin embargo, las diferencias entre los grupos de utilización rotativa se presentan como evidencia descriptiva, pues la figura no incluye una prueba estadística específica para compararlos dentro de cada cuartil de `dti`.*

*✅ **Hipótesis 3 — Respaldada por la Tendencia Descriptiva y la Asociación Inversa (Panel C de la Figura 1 y Tabla 5):** Los resultados respaldan la hipótesis de que un mayor ingreso anual está asociado con una menor frecuencia de incumplimiento. Como muestra el **Panel C de la Figura 1**, la tasa observada desciende de forma monótona desde el 24,8 % en Q1 hasta el 15,7 % en Q4; los cuartiles intermedios registran tasas del 22,1 % y 18,9 %, respectivamente. La correlación de Spearman entre `annual_inc` y el incumplimiento es negativa y estadísticamente significativa (\(\rho = -0.088\), \(p < 0.001\)), aunque de magnitud pequeña. En concordancia, el análisis PFI (**Tabla 5** y **Figura 6**) posiciona a `annual_inc` como la **segunda variable con mayor importancia predictiva individual** en la red neuronal, con \(\Delta \text{AUC} = 0.02128 \pm 0.00158\). Por tanto, la evidencia descriptiva, la asociación inversa y la importancia predictiva respaldan la hipótesis. La conclusión se refiere a una asociación observada y no implica que aumentar el ingreso, por sí solo, cause una reducción del incumplimiento.*

En conjunto, el análisis PFI del MLP identifica `int_rate`, `annual_inc`, `dti` y `revol_util` como las cuatro variables con mayor importancia predictiva individual, con caídas medias de AUC de \(0.03906\), \(0.02128\), \(0.01355\) y \(0.01095\), respectivamente. Estos resultados respaldan la relevancia de las condiciones del préstamo, los ingresos y el endeudamiento para la clasificación del riesgo crediticio. No obstante, la importancia predictiva de una variable no demuestra causalidad. Por ello, los veredictos anteriores integran la PFI con las tasas observadas y las asociaciones estadísticas calculadas sobre el conjunto de entrenamiento.



---

## 5. Parte 2: Estimación de la Pérdida Dado el Incumplimiento (LGD) y de la Pérdida Esperada

### 5.1. Construcción de la variable objetivo LGD

**Definición.** La LGD es la fracción de la exposición que no se recupera. Para un crédito castigado $i$:

$$\text{EAD}^{\text{def}}_i = \text{funded_amnt}_i - \text{total_rec_prncp}_i, \qquad \text{LGD}_i = \text{clip}\!\left(1 - \frac{\text{recoveries}_i - \text{collection_recovery_fee}_i}{\text{EAD}^{\text{def}}_i},\; 0,\; 1\right)$$

- `funded_amnt`: monto desembolsado.
- `total_rec_prncp`: capital ya pagado antes del castigo. Por eso la exposición al incumplimiento es el capital pendiente, y no el monto original.
- `recoveries`: dinero recuperado después del castigo (cobranza, venta de cartera).
- `collection_recovery_fee`: costo de esa cobranza, que se resta porque reduce la recuperación neta.
- El recorte a $[0,1]$ evita valores fuera de rango por cargos y comisiones. Se descartan los créditos con $\text{EAD}^{\text{def}} \le 0$.

Estas variables son posteriores a la originación. Son legítimas como **objetivo**, pero no entran como variables de entrada: el modelo usa las mismas 19 variables ex-ante de la Parte 1 (38 columnas tras la codificación).

**Población.** La población principal son los créditos `Charged Off` y `Does not meet the credit policy. Status:Charged Off`: **46,009 créditos**. Los demás estados se tratan así:

- `Default` (1,218 créditos con EAD válida): sus `recoveries` son cero por construcción, porque la cobranza aún no ha empezado. Si se incluyeran, tendrían LGD = 1 artificialmente. Se excluyen de la población principal y se usan solo en un análisis de sensibilidad (47,227 créditos en total).
- `Late (31-120 days)` (11,591 créditos): su desenlace aún no se conoce, así que no tienen LGD observable.
- Créditos pagados o vigentes: no tienen pérdida.

**Tabla 6.** *Distribución de la LGD observada por partición temporal (población principal).*

| Partición | $N$ | LGD media | Desv. est. | % con LGD = 1 | Error de predecir siempre 1 (MAE / RMSE) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Entrenamiento (≤ 2014-06) | 36,974 | 0.9215 | 0.1013 | 39.2 % | 0.0785 / 0.1282 |
| Validación (2014-06 a 2014-12) | 6,262 | 0.9547 | 0.0772 | 69.7 % | 0.0453 / 0.0895 |
| Prueba (> 2014-12) | 2,773 | 0.9807 | 0.0707 | 89.0 % | 0.0193 / 0.0732 |
| **Total** | **46,009** | **0.9296** | — | **46.4 %** | — |

*Fuente:* Elaboración propia a partir de `data/processed/lgd/lgd_target.parquet` y las particiones de `data/processed/pd/`.

![Figura 7: Distribución de la LGD y LGD media por año de originación.](../reports/figures/lgd/figL1_distribucion_lgd.png)
**Figura 7.** *La LGD se concentra en 1 y sube en los créditos más recientes: (a) distribución de la LGD observada; (b) LGD media por año de originación.*  
*Fuente:* Elaboración propia mediante `scripts/make_figures_lgd.py`.

**Limitaciones del objetivo.**

1. **Censura por la derecha.** La LGD media sube de 0.92 (créditos antiguos) a 0.98 (créditos de 2015, Figura 7b). Los créditos recientes se castigaron hace poco y su cobranza aún no termina, así que se observa menos recuperación de la que habrá. Esto no es riesgo real más alto: es información incompleta, y afecta justo al conjunto de prueba (Tabla 6).
2. **Sin valor del dinero en el tiempo.** Las recuperaciones no se descuentan, porque el conjunto de datos no trae sus fechas.
3. **La población excluye `Default` y `Late`.** Eso es coherente con la LGD observable, pero crea un desajuste con la PD de la Parte 1 (ver 5.5).
4. **Intereses ignorados** en la exposición: la EAD usa solo capital.

### 5.2. Fundamentos matemáticos

La formulación de la LGD se desarrolla en la sección de Fundamentación Matemática (4F.6 a 4F.9): regresión fraccionaria con salida sigmoide, la BCE como cuasi-verosimilitud frente al contraste con MSE, las métricas MAE y RMSE, la inicialización de la salida y la pérdida esperada como valor esperado. Aquí se usan sus resultados: el MLP-C estima la media condicional de la LGD $\mathbb{E}[y\mid\mathbf{x}]$, que es lo que necesita la pérdida esperada, y el RMSE es la métrica coherente con esa pérdida, mientras que el MAE se reporta como complemento.

### 5.3. Modelos y comparación justa

Todos los modelos usan el mismo preprocesador (ajustado solo con entrenamiento) y las mismas particiones (Tabla 6).

- **Constante:** predice la media del entrenamiento. 1 parámetro.
- **Referencia lineal:** regresión lineal (Ridge, $\alpha$ elegido por validación), con la predicción recortada a $[0,1]$. 39 parámetros.
- **MLP-C:** red con salida sigmoide y pérdida BCE (la propuesta principal).
- **MLP-B:** la misma arquitectura con pérdida MSE (contraste).

**Presupuesto de búsqueda.** Las configuraciones se muestrean al azar, porque la búsqueda aleatoria suele ser más eficiente que una malla cuando pocos hiperparámetros importan (Bergstra & Bengio, 2012). La selección usa solo el RMSE de validación; el conjunto de prueba no interviene.

**Tabla 7.** *Presupuesto de búsqueda y configuración elegida.*

| Familia | Configuraciones | Tiempo total | Selección | Resultado |
| :--- | :---: | :---: | :--- | :--- |
| MLP (BCE) | 60 | 479.8 s (8.0 s por configuración) | menor RMSE de validación | `[64, 32]`, selu, dropout 0.1, rmsprop, lr 0.002, weight decay 0.0025, lote 1024 |
| Ridge | 12 valores de $\alpha$ (0.01 a $10^4$) | 0.14 s | menor RMSE de validación | $\alpha = 0.01$ (val RMSE 0.0831) |
| Constante | 0 | — | — | — |

Además hubo **1 prueba manual previa** declarada. Se buscó solo para el MLP con BCE y la configuración elegida se reutilizó para el MLP-B; esto es una ventaja de diseño para C. El presupuesto es desigual por naturaleza (el MLP tiene más hiperparámetros que Ridge). Los tiempos se midieron en un equipo con CPU AMD Ryzen 5 150 con gráficos Radeon (3.30 GHz), 16 GB de RAM y Windows 11; con otro hardware cambian, pero no los rankings. Los espacios de búsqueda y las semillas están en `configs/lgd.yaml`.

**Treinta semillas.** La configuración final de cada red se entrena con 30 semillas (0 a 29). La semilla cambia la inicialización, el orden de los lotes y el dropout; los datos son fijos. Por eso la variabilidad reportada es la del **entrenamiento**, no la del muestreo de datos.

**Tabla 8.** *Resultados en el conjunto de prueba con 30 semillas: media ± desviación estándar [mejor, peor].*

| Modelo | Parámetros | MAE | RMSE |
| :--- | :---: | :---: | :---: |
| Constante (media del entrenamiento) | 1 | 0.0807 | 0.0922 |
| Regresión lineal (Ridge) | 39 | 0.0780 | 0.0924 |
| **MLP sigmoide + BCE (C)** | **4,609** | **0.0729 ± 0.0030** [0.0658, 0.0781] | **0.0885 ± 0.0020** [0.0842, 0.0928] |
| MLP sigmoide + MSE (B) | 4,609 | 0.0729 ± 0.0057 [0.0614, 0.0828] | 0.0876 ± 0.0039 [0.0808, 0.0952] |

*Nota:* La constante y la lineal son deterministas (desviación 0). *Fuente:* Elaboración propia a partir de `reports/tables/lgd/main/seeds_summary.csv` (`scripts/run_seeds_lgd.py`; 30 semillas × 2 redes en 480 s).

![Figura 8: Resultados por semilla.](../reports/figures/lgd/figL4_semillas.png)
**Figura 8.** *RMSE y MAE en prueba de cada una de las 30 semillas para las dos redes, con las referencias deterministas.*  
*Fuente:* Elaboración propia mediante `scripts/make_figures_lgd.py`.

### 5.4. Resultados y evaluación

**Aporte de la red sobre la línea base.** El MLP-C reduce el RMSE de prueba de 0.0924 a 0.0885 (**−4.2 %**) y el MAE de 0.0780 a 0.0729 (**−6.5 %**). Supera a la lineal en RMSE en 29 de 30 semillas y en MAE en 29 de 30. A cambio usa unas 118 veces más parámetros (4,609 frente a 39). Este resultado es coherente con la literatura sobre LGD: Loterman et al. (2012) encuentran que los métodos no lineales, en particular las máquinas de vectores de soporte y las redes neuronales, superan a la regresión lineal, pero que gran parte de la varianza de la LGD queda sin explicar (R² promedio entre 4 % y 43 %). La mejora existe, pero es modesta, y depende de la semilla: el RMSE del MLP-C va de 0.0842 a 0.0928, y su peor caso (0.0928) es peor que el de la lineal (0.0924).

**BCE frente a MSE: la ventaja teórica no aparece en el error medio.**

- C gana a B en RMSE en solo 13 de 30 semillas. La diferencia media (C − B) es +0.00086, con desviación 0.0034: no es significativa (prueba $t$ pareada $p=0.17$; Wilcoxon $p=0.28$). En MAE las medias son iguales (0.0729).
- Lo que sí se observa es que **C es más estable**: su desviación en RMSE (0.0020) es la mitad de la de B (0.0039), y el peor caso de B (0.0952) es peor que el de C (0.0928). Es consistente con el argumento del gradiente de 5.2, pero no lo prueba.
- Por tanto, BCE se justifica por la teoría (salida acotada, cuasi-verosimilitud, gradiente que no se anula) y por su estabilidad, **no** por un error medio menor.

**Entrenamiento inestable.** Con la configuración elegida, el RMSE de validación oscila entre épocas (por ejemplo, de 0.0798 a 0.0939 en la ejecución final), y la parada temprana actúa alrededor de la época 18 en promedio. Elegir la mejor época y la mejor configuración es en parte suerte (*winner's curse*): la configuración ganadora tuvo val RMSE 0.0798 en la búsqueda, pero su media en 30 semillas es 0.0809. Por eso se reportan medias, y no el mejor valor.

![Figura 9: Curvas de entrenamiento.](../reports/figures/lgd/figL2_curvas_entrenamiento.png)
**Figura 9.** *RMSE de validación por época de las dos redes con la configuración final.*  
*Fuente:* Elaboración propia mediante `scripts/make_figures_lgd.py`.

![Figura 10: Búsqueda de hiperparámetros.](../reports/figures/lgd/figL5_busqueda_hiperparametros.png)
**Figura 10.** *RMSE de validación de las 60 configuraciones del MLP, por optimizador.*  
*Fuente:* Elaboración propia mediante `scripts/make_figures_lgd.py`.

**Distribución predicha frente a observada.**

**Tabla 9.** *Distribución de la LGD en el conjunto de prueba (modelo guardado, semilla 42).*

| Serie | Media | Desv. est. | Q05 | Q95 | % ≥ 0.99 | Wasserstein | KS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Observada | 0.981 | 0.071 | 0.850 | 1.000 | 89.1 % | — | — |
| Constante | 0.922 | 0.000 | 0.922 | 0.922 | 0.0 % | 0.081 | 0.893 |
| Regresión lineal | 0.924 | 0.021 | 0.905 | 0.942 | 0.0 % | 0.074 | 0.891 |
| MLP-C | 0.935 | 0.020 | 0.917 | 0.948 | 0.0 % | 0.066 | 0.891 |
| MLP-B | 0.932 | 0.013 | 0.922 | 0.941 | 0.0 % | 0.069 | 0.891 |

*Fuente:* Elaboración propia a partir de `reports/tables/lgd/main/distribution_test.csv`.

Ningún modelo reproduce la masa en LGD = 1: el 89 % de los créditos observados tiene LGD ≥ 0.99, y los modelos no predicen nada por encima de 0.97 (Figura 11). Las predicciones son estrechas y centradas cerca de 0.93, con distancia de Wasserstein baja (0.066 a 0.081) frente a una distribución observada casi degenerada en 1. Esto es esperable para un estimador de la media cuando los datos de prueba son más extremos que los de entrenamiento.

![Figura 11: Distribución acumulada predicha y observada.](../reports/figures/lgd/figL3_distribucion_pred_vs_obs.png)
**Figura 11.** *Función de distribución acumulada de la LGD en el conjunto de prueba (n = 2,773), observada frente a predicha.*  
*Fuente:* Elaboración propia mediante `scripts/make_figures_lgd.py`.

**Advertencia crítica: el error absoluto del conjunto de prueba depende del desplazamiento temporal.** Como la LGD observada sube con el tiempo (Tabla 6; censura), una predicción constante de 1 tiene en prueba un error **menor** que todos los modelos: MAE 0.0193 y RMSE 0.0732, frente a 0.0729 y 0.0885 del MLP-C. En validación ocurre lo contrario (RMSE de predecir 1: 0.0895, frente a 0.0809 del MLP). Esta no es una referencia que se haya entrenado: es un chequeo de cordura calculado a posteriori. Muestra que, en la cohorte de 2015, los modelos entrenados con el pasado subestiman la LGD, y que las diferencias entre modelos (centésimas) son menores que el efecto del desplazamiento. Las conclusiones sobre "mejor modelo" deben leerse como comparaciones **entre modelos bajo el mismo desplazamiento**, no como evidencia de una LGD bien estimada en 2015.

**Sensibilidad: incluir `Default`.** Con los 1,218 créditos `Default` añadidos (LGD = 1 por construcción), el MLP-C obtiene en prueba un RMSE de 0.0871 frente a 0.0898 de la lineal, y el MLP-B 0.0870 (una ejecución, semilla 42; `reports/tables/lgd/with_default/`). La conclusión relativa no cambia. Para comparar poblaciones distintas sin sesgo, el archivo de resultados incluye las columnas `*_sinDefault`, que evalúan sobre el mismo subconjunto de prueba.

### 5.5. Pérdida esperada ($\text{EL}=\text{PD}\cdot\text{LGD}\cdot\text{EAD}$)

**Componentes.** PD: MLP de la Parte 1. LGD: MLP-C de la Parte 2. EAD: lo que se debe cuando el crédito incumple. Para un crédito nuevo no se conoce, así que se estima con dos definiciones:

- **EAD A (sin amortización):** $\widehat{\text{EAD}} = \text{monto}$. Es simple y conservadora.
- **EAD B (principal):** $\widehat{\text{EAD}} = \text{monto}\times r_{\text{plazo}}$, con $r_{\text{plazo}}$ la razón media $\text{EAD}^{\text{def}}/\text{monto}$ de los créditos castigados del entrenamiento: 0.658 (36 meses), 0.818 (60 meses), 0.713 (global). Refleja que el cliente ya pagó parte del capital antes de caer, y que a 60 meses se amortiza más despacio. A queda como análisis de sensibilidad.

Los supuestos y la formulación de la pérdida esperada como valor esperado están en 4F.9.

**Evaluación.** Se calcula sobre los créditos del conjunto de prueba con pérdida conocida (25,757 de 30,835). La pérdida realizada es $\text{LGD}_i\cdot\text{EAD}^{\text{def}}_i$ para los castigados y cero para los pagados; se excluyen los `Default` y `Late` (5,078), cuya pérdida final se desconoce.

**Tabla 10.** *Pérdida esperada frente a pérdida realizada en prueba (25,757 créditos).*

| EAD | EL total (USD) | EL / realizada | Spearman por crédito | MAE de EAD en castigados (USD) | Sesgo de EAD (USD) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| A | 84,935,697 | 2.20 | 0.176 | 901 | +901 |
| **B (principal)** | **63,927,574** | **1.65** | **0.174** | 3,049 | −2,974 |
| Realizada | 38,666,227 | 1.00 | — | — | — |

*Fuente:* Elaboración propia a partir de `reports/tables/lgd/main/expected_loss_summary.json` (`scripts/run_expected_loss.py`).

**Tabla 11.** *Pérdida esperada y pérdida realizada media por decil de pérdida esperada (EAD B), USD por crédito.*

| Decil | $N$ | EL media predicha | Pérdida realizada media | Razón |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 2,576 | 197.0 | 91.4 | 2.16 |
| 2 | 2,576 | 401.3 | 250.0 | 1.61 |
| 3 | 2,575 | 607.3 | 454.0 | 1.34 |
| 4 | 2,576 | 859.9 | 596.7 | 1.44 |
| 5 | 2,576 | 1,190.8 | 784.1 | 1.52 |
| 6 | 2,575 | 1,679.9 | 1,196.5 | 1.40 |
| 7 | 2,576 | 2,421.7 | 1,710.2 | 1.42 |
| 8 | 2,575 | 3,533.9 | 2,282.0 | 1.55 |
| 9 | 2,576 | 5,162.4 | 3,080.5 | 1.68 |
| 10 | 2,576 | 8,764.5 | 4,566.3 | 1.92 |

*Fuente:* Elaboración propia a partir de `reports/tables/lgd/main/expected_loss_deciles_B.csv`.

![Figura 12: Pérdida esperada frente a pérdida realizada por decil.](../reports/figures/lgd/figL6_perdida_esperada_deciles.png)
**Figura 12.** *La pérdida esperada ordena bien el riesgo, pero sobreestima su nivel: media por decil de pérdida esperada, esperada frente a realizada (EAD B).*  
*Fuente:* Elaboración propia mediante `scripts/make_figures_lgd.py`.

**Lectura.**

1. **Ordena el riesgo por grupos, no por crédito.** La pérdida realizada crece con el decil (de 91 a 4,566 USD), pero la correlación de Spearman por crédito es solo 0.17: el modelo discrimina poco entre créditos individuales, y la utilidad está en el agregado.
2. **Sobreestima el nivel** (1.65 veces la pérdida realizada con B). La causa principal es un **desajuste de definiciones**: la PD de la Parte 1 cuenta como "malos" los `Charged Off`, `Default` y `Late (31-120)`, mientras que la pérdida realizada solo observa castigados. La PD media predicha en el subconjunto evaluado es 0.225, frente a una tasa de castigados observada de 0.108 (2.1 veces), antes de tocar la LGD o la EAD.
3. **La LGD subestima** (media predicha 0.935 frente a 0.981 observada en los castigados), por la censura y el desplazamiento temporal (5.1). Este sesgo actúa en sentido contrario y atenúa la sobreestimación total.
4. **A frente a B: no basta con el total.** En el total, B se acerca más a la pérdida realizada (1.65 frente a 2.20). Pero a nivel de crédito, A estima mejor la EAD en prueba (MAE 901 frente a 3,049 USD): en la cohorte de 2015 los créditos castigados lo hicieron pronto, con razón EAD/monto cercana a 0.94 y no 0.71 (censura otra vez). B subestima la EAD, y su total queda más cerca solo porque compensa la sobreestimación de la PD. Por tanto, el resultado de B no es evidencia de que sea mejor estimador de la EAD.

### 5.6. Interpretabilidad

Se usa la importancia por permutación (Breiman, 2001; Fisher, Rudin, & Dominici, 2019): se baraja una variable en el conjunto de prueba y se mide cuánto **aumenta el RMSE**. Las variables nominales (`purpose`, `home_ownership`, etc.) se expanden en varias columnas, que aquí se barajan **juntas** con la misma permutación para medir la variable original.

![Figura 13: Interpretabilidad de la LGD.](../reports/figures/lgd/figL7_interpretabilidad.png)
**Figura 13.** *Aumento del RMSE al barajar cada variable original, para el MLP-C y para la regresión lineal.*  
*Fuente:* Elaboración propia mediante `scripts/make_figures_lgd.py` y `src/creditrisk/lgd_interpretability.py`.

- **`int_rate` domina en ambos modelos** (aumento de RMSE: 0.00038 en el MLP y 0.00184 en la lineal), igual que en la PD (Tabla 5).
- **El MLP reparte algo más el peso:** en el MLP, `annual_inc` (0.00017) equivale al 45 % de `int_rate`; en la lineal, `annual_inc` (0.00016) es solo el 9 % de `int_rate`. `term` aporta 0.00005 en ambos. Ninguna otra variable pasa de 0.00005 en el MLP.
- **Las magnitudes son pequeñas** (milésimas de punto de RMSE, sobre un error de 0.0885), lo que concuerda con que la mejora sobre la constante es modesta: casi toda la señal está en la tasa de interés.
- **Los coeficientes de la regresión lineal son inestables:** `int_rate` sale con signo negativo y `grade` con positivo, porque ambas variables son casi colineales (la tasa se asigna según el grado). Por eso los coeficientes individuales no deben leerse como efectos aislados; la importancia por permutación es más fiable.

### 5.7. Reproducibilidad

```
python scripts/run_lgd.py                  # referencias y MLP con la configuración por defecto
python scripts/run_search_lgd.py           # búsqueda (escribe configs/lgd_best.yaml)
python scripts/run_lgd.py --best           # modelos finales
python scripts/run_seeds_lgd.py            # 30 semillas
python scripts/run_expected_loss.py        # pérdida esperada
python scripts/make_figures_lgd.py         # figuras e interpretabilidad
```

Equivalentes en el `Makefile`: `lgd`, `search-lgd`, `seeds-lgd`, `el`, `figures-lgd`. El modo `--demo` ejecuta una versión reducida. Con las mismas versiones de las librerías (`requirements.txt`; los modelos de PD se guardaron con scikit-learn 1.7.2), dos máquinas dan resultados iguales salvo ruido numérico del orden de la cuarta cifra decimal; los rankings de configuraciones casi empatadas pueden cambiar.

### 5.8. Discusión, limitaciones y aprendizajes de la Parte 2

1. **Qué aportó la red.** Mejora la línea base lineal en 4.2 % (RMSE) y 6.5 % (MAE), de forma consistente (29 de 30 semillas), con 118 veces más parámetros. Es una mejora real pero pequeña, y menor que el efecto del desplazamiento temporal sobre el error (5.4). Para un banco, el MLP solo se justifica si la mejora vale el costo de mantener un modelo menos transparente.
2. **BCE no supera a MSE en error medio.** Se justifica teóricamente y por su menor variabilidad entre semillas, pero la diferencia de error no es significativa.
3. **La censura es el principal límite.** La LGD de créditos recientes está subestimada en recuperación (sobreestimada como pérdida) porque la cobranza no termina. Una extensión natural es modelar la recuperación como función del tiempo transcurrido desde el castigo, o excluir cohortes sin madurez suficiente.
4. **La pérdida esperada necesita definiciones coherentes entre partes.** El mayor error del total (1.65 veces) viene de que la PD y la pérdida realizada no miden el mismo evento. Alinear la definición de "incumplimiento" entre PD y LGD es más importante que mejorar la red.
5. **La mejor corrida no es el resultado.** Con curvas inestables y selección sobre una sola semilla, el valor reportable es la media de 30 semillas.
6. **Qué se aprendió del proceso.** Verificar el conteo de estados antes de modelar (el estado `Late` no tiene LGD observable), comprobar con un baseline trivial que la métrica tiene sentido, y no tomar cifras ni citas de un asistente de IA sin revisarlas contra los datos o la fuente.

---

## 6. Discusión Crítica, Limitaciones y Recomendaciones

1. **Magnitud de la Mejora del MLP sobre Datos Tabulares:** En el conjunto de prueba, la Regresión Logística obtuvo un ROC-AUC de 0.6902, mientras que el MLP alcanzó 0.6960, lo que representa una diferencia de 0.0058 a favor de la red neuronal. El intervalo de confianza bootstrap del 95 % para esta diferencia fue [0.0036, 0.0080], sin incluir cero. En cuanto al estadístico KS, la Regresión Logística obtuvo 0.2839 y el MLP 0.2941, una diferencia de 0.0102. Aunque los resultados favorecen al MLP en ambas métricas, la mejora en ROC-AUC es moderada y debe interpretarse en el contexto de la partición temporal utilizada. Estos resultados no demuestran por sí solos que la ventaja provenga de la capacidad del MLP para capturar relaciones no lineales.
2. **Diferencias Temporales en la Tasa de Incumplimiento:** La tasa de incumplimiento observada aumenta del 20.46 % en el conjunto de entrenamiento al 27.12 % en validación y al 25.46 % en prueba. Estas diferencias muestran que la frecuencia del evento objetivo no es constante entre las particiones temporales y justifican evaluar la calibración del modelo al aplicarlo a nuevas cohortes. Sin embargo, este resultado no permite concluir por sí solo que exista deriva de covariables (*covariate drift*) o deriva del concepto (*concept drift*), ni establecer sus causas. Como trabajo futuro, se recomienda implementar un cálculo del Índice de Estabilidad Poblacional (PSI) para monitorizar los cambios en la distribución de las variables de entrada, junto con evaluaciones periódicas de calibración y discriminación.
3. **Umbral de Decisión y Consideraciones de Negocio:** El umbral que maximiza la separación entre las distribuciones de probabilidades de buenos y malos pagadores puede utilizarse como referencia para analizar la capacidad discriminativa del modelo. En la curva KS generada por `scripts/make_figures.py`, dicho umbral es aproximadamente $\tau^* = 0.216$, equivalente a un puntaje de alrededor de 552 puntos en la escala de la *scorecard*. Sin embargo, este valor no debe interpretarse automáticamente como el punto de corte óptimo para aprobar o rechazar solicitudes: el estadístico KS mide separación entre distribuciones, pero no incorpora por sí solo los ingresos por intereses, las pérdidas por incumplimiento ni los demás costes de la decisión. Para definir una política operativa, sería necesario especificar y justificar los costes y beneficios relevantes, incluida la pérdida dado el incumplimiento (LGD), y evaluar la decisión resultante en datos apropiados.
4. **Censura y evaluación temporal en la LGD (Parte 2):** La LGD observada crece con el año de originación (0.92 en entrenamiento, 0.98 en prueba) porque la cobranza de los créditos recientes aún no termina. Por eso una predicción constante de 1 tiene en prueba un error menor (MAE 0.0193, RMSE 0.0732) que los modelos entrenados con el pasado (RMSE 0.0885 del MLP-C), y lo contrario ocurre en validación. Las comparaciones entre modelos de la Sección 1P son válidas bajo el mismo desplazamiento, pero no demuestran una LGD bien estimada en la cohorte de 2015.
5. **Coherencia entre PD y LGD en la pérdida esperada:** La pérdida esperada total supera 1.65 veces la realizada (EAD B). La causa principal es que la PD de la Parte 1 trata como "malos" a `Charged Off`, `Default` y `Late (31-120)`, mientras que la pérdida realizada solo observa castigados (PD media predicha 0.225 frente a 0.108 observada en el subconjunto evaluado). Alinear la definición de incumplimiento entre PD y LGD es más importante que mejorar la red.

---

## 1A. Declaración de Uso de Inteligencia Artificial y Cacería de Alucinaciones

En cumplimiento de los lineamientos de transparencia académica del curso, el detalle de las interacciones con modelos de lenguaje de gran escala (LLMs) se encuentra documentado en dos archivos auditables del repositorio. La **Parte 1** (PD y *Scorecard*) se desarrolló con **Gemini / Argon** y la **Parte 2** (LGD y Pérdida Esperada) con **Claude (Claude Code)**.

* **Registro de Uso de IA (`docs/uso_ia.md`):** Contiene ocho entradas, cinco de la Parte 1 y tres de la Parte 2.
  * *Parte 1:* estructura de carpetas y auditoría de variables del diccionario de datos, scripts de limpieza, codificación del objetivo y partición temporal, resolución de errores de consola en Windows, diseño del preprocesador, de la regresión logística y del MLP, y fundamentación del caso de uso.
  * *Parte 2:* revisión de lo que ya existía en el repositorio y de lo que pedía el enunciado, definición de la población y del objetivo LGD (incluida la sensibilidad con `Default`) e implementación del MLP con salida sigmoide y pérdida BCE, con su contraste con MSE.
  * En ambas partes, las decisiones de diseño (prevención de *data leakage*, partición temporal, población de la LGD, definición de la EAD y validación estadística) fueron tomadas y verificadas por los autores, y las cifras del reporte provienen de los resultados ejecutados por el equipo en el repositorio.
* **Cacería de Alucinaciones (`docs/caceria_alucinacion.md`):** Registra cuatro hallazgos con evidencia y una verificación sin hallazgo:
  1. *Versiones inexistentes en `requirements.txt` (Parte 1, Gemini):* la IA afirmó que versiones como `torch==2.14.1` no existían en PyPI y que `pip install` fallaría; el equipo comprobó en el registro de paquetes que sí existían y descartó la advertencia.
  2. *Script inexistente `build_blog.py` (Parte 1, Gemini):* la IA indicó ejecutar un script que no estaba en el repositorio; el equipo lo detectó al revisar la carpeta `scripts/` y lo creó.
  3. *Pérdida de 761 créditos por un espacio en una cadena (Parte 1):* el mapa de estados del *pipeline* de PD buscaba `Status: Charged Off` con un espacio que el CSV no tiene, por lo que esos créditos se descartaban; el equipo lo detectó al contrastar los conteos y lo corrigió cubriendo las variantes con y sin espacio.
  4. *Porcentaje mal calculado (Parte 2, Claude):* la IA reportó una mejora de "3.5 % en RMSE" que correspondía al MAE (la del RMSE era 3.7 %); se recalculó desde las tablas y el reporte usa las cifras finales (−4.2 % en RMSE y −6.5 % en MAE).
  5. *Verificación sin hallazgo (Parte 2):* se comprobó la reproducibilidad de los resultados entre equipos, los conteos del objetivo (46,009 y 47,227 créditos) y las pruebas automáticas, sin encontrar discrepancias.

  En conjunto, los errores detectados (de concepto, código, datos del mundo real y cálculo) llevaron a verificar las afirmaciones de la IA contra el código, los datos y las fuentes antes de incorporarlas al reporte.

---

## Referencias Bibliográficas (Formato APA 7.ª Edición)

* Basel Committee on Banking Supervision [BCBS]. (2017). *Basel III: Finalising post-crisis reforms*. Bank for International Settlements. https://www.bis.org/bcbs/publ/d424.htm
* Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, *13*, 281–305. https://jmlr.org/papers/v13/bergstra12a.html
* Breiman, L. (2001). Random forests. *Machine Learning*, *45*(1), 5–32. https://doi.org/10.1023/A:1010933404324
* Fisher, A., Rudin, C., & Dominici, F. (2019). All models are wrong, but many are useful: Learning a variable's importance by studying an entire class of prediction models simultaneously. *Journal of Machine Learning Research*, *20*(177), 1–81. https://www.jmlr.org/papers/v20/18-760.html
* Gourieroux, C., Monfort, A., & Trognon, A. (1984). Pseudo maximum likelihood methods: Theory. *Econometrica*, *52*(3), 681–700. https://econometricsociety.org/publications/econometrica/1984/05/01/pseudo-maximum-likelihood-methods-theory
* Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *Advances in Neural Information Processing Systems (NeurIPS)*, *35*, 507–520.
* Hand, D. J., & Henley, W. E. (1997). Statistical classification methods in consumer credit scoring: A review. *Journal of the Royal Statistical Society: Series A (Statistics in Society)*, *160*(3), 523–541. https://doi.org/10.1111/j.1467-985X.1997.00078.x
* International Accounting Standards Board [IASB]. (2014). *IFRS 9 Financial Instruments*. IFRS Foundation.
* Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. En *Proceedings of the 32nd International Conference on Machine Learning (ICML)* (pp. 448–456). PMLR.
* Lessmann, S., Baesens, B., Seow, H.-V., & Thomas, L. C. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research. *European Journal of Operational Research*, *247*(1), 124–136. https://doi.org/10.1016/j.ejor.2015.05.030
* Loterman, G., Brown, I., Martens, D., Mues, C., & Baesens, B. (2012). Benchmarking regression algorithms for loss given default modeling. *International Journal of Forecasting*, *28*(1), 161–170. https://ideas.repec.org/a/eee/intfor/v28y2012i1p161-170.html
* Papke, L. E., & Wooldridge, J. M. (1996). Econometric methods for fractional response variables with an application to 401(k) plan participation rates. *Journal of Applied Econometrics*, *11*(6), 619–632.
* Siddiqi, N. (2017). *Intelligent credit scoring: Building and implementing better credit risk scorecards* (2nd ed.). John Wiley & Sons. https://doi.org/10.1002/9781119282396
* Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, *15*(56), 1929–1958.
* Thomas, L., Crook, J., & Edelman, D. (2017). *Credit scoring and its applications* (2nd ed.). Society for Industrial and Applied Mathematics (SIAM). https://doi.org/10.1137/1.9781611974560