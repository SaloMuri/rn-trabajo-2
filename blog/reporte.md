# ¿Me prestas? Modelamiento de Riesgo Crediticio con Redes Neuronales en Datos Tabulares

**Curso:** Redes Neuronales y Algoritmos Bioinspirados — Universidad Nacional de Colombia, Sede Medellín (Facultad de Minas)  
**Semestre:** 2026-02  
**Enlaces del Proyecto:** [Repositorio en GitHub](#) · [Aplicación Web Interactiva (Scorecard)](#) · [Video Promocional](#)

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

Durante la auditoría del diccionario oficial (`LCDataDictionary.xlsx`), detectamos que la hoja `browseNotes` contiene variables en formato *camelCase* que no existen en el archivo `LoanStats` (`loan.csv`), y que un conjunto de variables sobre cuentas por cuotas (*installment accounts*, p. ej., `il_util`, `total_bal_il`, `max_bal_bc`) presenta más del 95% de valores faltantes por haberse incorporado apenas a finales de 2015. Por tanto, depuramos el espacio de características a **20 variables predictoras de originación** con alta completitud histórica (14 numéricas continuas, 3 categóricas ordinales y 3 categóricas nominales), las cuales se expanden a **38 características** tras la codificación *One-Hot*.

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

Tras descartar los 621,598 préstamos en curso (`NA`), el conjunto de datos válido para la Parte 1 consta de **265,781 préstamos cerrados o en mora severa**, con una **tasa global de incumplimiento del 21.84%**.

### 2.2. Justificación de la Partición Temporal (*Out-of-Time Split*)

En modelamiento de riesgo crediticio, realizar una partición aleatoria convencional (`train_test_split`) constituye un error metodológico conocido como **fuga temporal (*look-ahead bias*)**: permitiría entrenar con préstamos de 2015 para predecir préstamos de 2009, ignorando los ciclos macroeconómicos y los cambios en las políticas de originación a lo largo del tiempo.

En consecuencia, implementamos una **partición estrictamente cronológica basada en la fecha de originación (`issue_d`)**:

| Partición | Ventana Temporal (`issue_d`) | Registros ($N$) | % del Total | Tasa de Incumplimiento (PD) | Rol Metodológico |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Entrenamiento (`Train`)** | `2007-06` a `2014-06` | 199,958 | 75.23% | 20.36% | Ajuste de pesos de los modelos y ajuste exclusivo (`fit`) del preprocesador (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`). |
| **Validación (`Val`)** | `2014-07` a `2014-12` | 34,988 | 13.16% | 27.12% | Monitoreo de parada temprana (*Early Stopping*) y selección de hiperparámetros (`run_search.py`). |
| **Prueba (`Test Out-of-Time`)** | `2015-01` a `2015-12` | 30,835 | 11.60% | 25.46% | Evaluación final fuera de tiempo (discriminación, calibración, estabilidad en 30 semillas e interpretabilidad). |

**Hallazgo sobre el cambio en la tasa base (Sesgo de Maduración):**  
Se observa que la tasa de incumplimiento pasa del **20.36%** en `Train` al **27.12%** en `Val` y **25.46%** en `Test`. Este fenómeno responde al **sesgo de maduración (*maturity bias*)**: dado que el corte del dataset ocurre en diciembre de 2015, entre los préstamos originados en 2014–2015 que ya habían alcanzado un estado definitivo (no `Current`), están sobrerrepresentados los *incumplimientos tempranos (early defaults)* frente a los buenos pagadores que siguen pagando sus cuotas normales de 36 o 60 meses. Identificar este desplazamiento distributivo (*covariate & prior shift*) será clave al analizar la curva de calibración.

---

## 3. Análisis Descriptivo y Formulación de Hipótesis

Antes de entrenar los modelos, exploramos el conjunto de entrenamiento (`Train`) y formulamos **tres hipótesis explícitas de riesgo** fundamentadas en teoría financiera:

* **Hipótesis 1 (Prima de Riesgo en Tasa y Calificación Inicial):**  
  *“La tasa de interés asignada (`int_rate`) y el grado crediticio (`grade`) presentan una relación monotónica creciente con la probabilidad de incumplimiento; sin embargo, al ser `int_rate` una medida continua y granular del riesgo percibido en originación, absorberá la mayor parte del poder predictivo frente a la variable ordinal `grade`.”*
* **Hipótesis 2 (Estrés Financiero por Capacidad de Pago y Plazo — `dti`, `annual_inc` y `term`):**  
  *“Un mayor nivel de endeudamiento mensual respecto al ingreso (`dti`), menores ingresos anuales (`annual_inc`) y la elección de plazos largos (`term = 60 meses` frente a `36 meses`) incrementan sustancialmente la probabilidad de incumplimiento, existiendo un efecto no lineal de interacción cuando convergen alto `dti` y plazo de 60 meses.”*
* **Hipótesis 3 (Saturación de Cupo Rotativo y Propósito del Préstamo — `revol_util` y `purpose`):**  
  *“Solicitantes con una alta utilización de sus líneas de crédito rotativas (`revol_util > 75%`) o que solicitan préstamos para financiar pequeños negocios (`small_business`) o consolidar deudas (`debt_consolidation`) exhiben un riesgo de impago significativamente mayor que quienes solicitan crédito para compra de vehículo (`car`) o mejoras del hogar (`home_improvement`).”*

![Figura 1: Análisis Descriptivo y Validación Visual de Hipótesis](../reports/figures/pd/fig1_eda_hipotesis.png)

**Análisis descriptivo preliminar (Figura 1):**  
1. En el **Panel Izquierdo (H1)** se observa una progresión monotónica perfecta en los datos de entrenamiento: mientras el grado `A` registra una tasa de impago inferior al 7%, los grados `F` y `G` superan el 40%–45% de incumplimiento.
2. En el **Panel Central (H2)** se evidencia una fuerte interacción: en todos los cuartiles de `dti`, los préstamos a **60 meses** (barras rojas) tienen una tasa de incumplimiento más de 10 puntos porcentuales superior a los de **36 meses** (barras verdes), superando el 35% de impago cuando el cliente combina el cuartil más alto de `dti` (`Q4`) con un plazo de 60 meses.
3. En el **Panel Derecho (H3)**, los préstamos destinados a pequeños negocios (`small_business`) lideran la siniestralidad con cerca del 30% de incumplimiento, seguidos por consolidación de deuda (`debt_consolidation`), mientras que tarjetas de crédito (`credit_card`) y compras mayores (`major_purchase`) presentan tasas menores.

*(En la Sección 6.5 retomaremos estas tres hipótesis para confirmarlas o matizarlas cuantitativamente mediante el análisis de interpretabilidad de los modelos entrenados).*

---