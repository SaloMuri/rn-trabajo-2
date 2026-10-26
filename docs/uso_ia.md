# Bitácora de Uso de Inteligencia Artificial

Registro transparente de las interacciones con el asistente de IA (Gemini / Argon) durante el desarrollo de la Parte 1 (Probabilidad de Incumplimiento y Scorecard).

| # | Fecha | Herramienta | Prompt / Consulta Realizada | Qué hicimos con la respuesta | Impacto en el Proyecto |
|---|---|---|---|---|---|
| 1 | 2026-10 | Gemini | Solicitud de estructura de carpetas sobre el repositorio base y auditoría de las 43 variables resaltadas en `Copia de LCDataDictionary.xlsx`. | Descubrimos que habíamos resaltado variables en la pestaña `browseNotes` (nombres en camelCase como `annualInc`) en lugar de `LoanStats` (snake_case `annual_inc`), y que 15 variables marcadas no existían en `loan.csv` o tenían >95% de nulos. Ajustamos la selección a 20 variables estrictas de originación. | Crítico: evitó errores de columnas inexistentes y previno *data leakage* de variables post-originación. |
| 2 | 2026-10 | Gemini | Solicitud del script de limpieza (`build_dataset.py`), codificación de `loan_status` (`targets.py`) y partición temporal (`splits.py`). | Implementamos el filtro de estados definitivos (descartando `Current`, `Issued`, `In Grace Period` y `Late (16-30 days)`), limpiamos strings como `term` y `emp_length`, y creamos la antigüedad crediticia dividiendo por `30.4375` días/mes. | Alto: redujo los 887,379 registros originales a 265,781 préstamos con desenlace conocido entre 2007-06 y 2015-12. |
| 3 | 2026-10 | Gemini | Resolución de errores de consola en Windows (`"PYTHONPATH" no se reconoce como un comando...`, `make` no reconocido y `ModuleNotFoundError: No module named 'creditrisk'`). | Comprendimos la diferencia entre variables de entorno en Bash (`PYTHONPATH=src`) y CMD de Windows (`set PYTHONPATH=src`). Incorporamos `sys.path.insert` dinámico en los scripts y `conftest.py` para `pytest`. | Medio: garantizó portabilidad de ejecución en Windows sin depender de GNU Make. |
| 4 | 2026-10 | Gemini | Diseño del preprocesador compartido (`features.py`), Regresión Logística (`baseline_pd.py`), arquitectura `TabularMLP` (`mlp.py`) y búsqueda de hiperparámetros (`run_search.py`). | Ejecutamos 6 configuraciones de LR y 6 de MLP. Verificamos empíricamente que `class_weight='balanced'` degradaba el Brier Score de 0.1795 a 0.2250 y que una red compacta `[64, 32]` (4,801 parámetros) superaba a redes grandes de 52,097 parámetros. | Alto: fundamentó la elección de la arquitectura final bajo el principio de parsimonia. |
| 5 | 2026-10 | Gemini | Cuestionamiento propio del equipo sobre la validez del Caso de Uso al incluir `grade` e `int_rate` en la aplicación web si un usuario externo aún no conoce la calificación del banco, y solicitud de fundamentación matemática. | Formalizamos el caso de uso en la etapa de *Underwriting / Originación contractual* sobre el marketplace P2P de Lending Club (donde `grade` e `int_rate` actúan como condiciones de oferta inicial y *proxy* del FICO ausente en Kaggle) y documentamos las ecuaciones de BCE, Scorecard y KS. | Crítico: alineó la coherencia entre el negocio, el modelo matemático y el simulador web. |

---

# 7. Cacería de la alucinación 🎯

Las herramientas de IA producen errores que suenan convincentes. Su misión es cazarlos. A continuación se documentan los hallazgos en los que el asistente de IA generó información incorrecta, imprecisa, desactualizada o subóptima durante el desarrollo del trabajo.

### Hallazgo 1: Gemini afirma que en `requirements.txt` se usan versiones de librerías inexistentes
1. **Prompt utilizado:**
   > Argon, este es el repo final, todo corre bien; antes de finalizar, verifica los requerimientos 3 y 4.
2. **Parte relevante de la respuesta de la IA:**
   > 🔴 **ALERTA CRÍTICA 1:** Versiones inexistentes en `requirements.txt` (¡Romperá `pip install` en otro PC!)  
   > En tu `requirements.txt` actual pusieron números de versión que no existen en PyPI (por ejemplo, `torch==2.14.1`, `pyarrow==25.0.1`, `streamlit==1.65.0`, `pytest==9.1.1`). Si el profesor o ustedes hacen un clon limpio en otra máquina y corren `pip install -r requirements.txt`, pip va a arrojar `ERROR: No matching distribution found for torch==2.14.1` inmediatamente.
3. **Cómo sospechamos que había un error:**
   Todo funcionaba correctamente dentro del entorno de desarrollo e incluso en ejecuciones dentro de un clon limpio del repositorio.
4. **Evidencia que lo demuestra:**
   Buscamos directamente en el registro público de paquetes en PyPI a través del navegador y verificamos que las versiones especificadas existían y estaban disponibles.
5. **Categoría:** Concepto / Código
6. **La corrección:**
   Verificamos manualmente en el navegador que las versiones sí existían en PyPI, por lo que ignoramos la advertencia generada por la IA.

---

### Hallazgo 2: Gemini reporta y sugiere ejecutar un script inexistente (`build_blog.py`)
1. **Prompt utilizado:**
   > Listo, aquí tenés los resultados de las tablas, para construir el reporte y las bitácoras; ojo, en el uso de IA y alucinación solo pon lo que notamos, no inventes nada. Configuremos el demo, y finalmente, creemos el reporte en un html, para poder cumplir con la entrada de blog.
2. **Parte relevante de la respuesta de la IA:**
   > Solo te queda pegar estos textos en `configs/demo.yaml` y `blog/reporte.md`, y correr: `python scripts/build_blog.py`
3. **Cómo sospechamos que había un error:**
   Al verificar la estructura interna del repositorio y los scripts existentes, no contábamos con ninguna carpeta o archivo llamado `build_blog.py`.
4. **Evidencia que lo demuestra:**
   Inspección del directorio de `scripts/` en el árbol del proyecto, constatando la ausencia de dicho archivo.
5. **Categoría:** Código
6. **La corrección:**
   Creamos la carpeta necesaria y ajustamos el código correspondiente para asegurar un flujo de trabajo correcto.

---

### Hallazgo 3: Pérdida de datos por un espacio en blanco en la cadena de texto
1. **Prompt utilizado:**
   > Notamos este error: los `Does not meet the credit policy. Status:Charged Off`, porque la PD los busca con un espacio después de `Status:` y el CSV no lo tiene. El PD está perdiendo esos 761 malos pagadores. Probablemente pierde también los de `Status:Fully Paid`.
2. **Parte relevante de la respuesta de la IA:**
   > ¡Excelente ojo analítico! Por qué se produce el error de los 761 créditos fantasma y cómo solucionarlo de raíz...
3. **Cómo sospechamos que había un error:**
   Para el punto 2, no veíamos la misma cantidad de datos reflejada en los resultados, existiendo una pérdida que afectaba la comparación justa (*fair comparison*).
4. **Evidencia que lo demuestra:**
   Conteo directo sobre las etiquetas del dataset que reveló la discrepancia de 761 registros clasificados como `Charged Off` al realizar la búsqueda de la cadena exacta.
5. **Categoría:** Concepto / Código / Datos del mundo real
6. **La corrección:**
   Tras la consulta con la IA, descubrimos que al crear nuestro mapa de targets habíamos incluido un espacio de más. Ajustamos el mapeo agregando tanto `Fully Paid` como `Charged Off` controlando las variantes con y sin espacio para evitar inconsistencias.

---

### Hallazgo 4: [Pendiente de completar]
1. **Prompt utilizado:**
   > [Escribe aquí el prompt literal]
2. **Parte relevante de la respuesta de la IA:**
   > [Copia aquí la respuesta relevante de la IA]
3. **Cómo sospechamos que había un error:**
   [Explica la sospecha]
4. **Evidencia que lo demuestra:**
   [Detalla la derivación, experimento o prueba]
5. **Categoría:** [Matemáticas / Código / Datos del mundo real / Bibliografía / Conceptos]
6. **La corrección:**
   [Describe la solución]

---

### Hallazgo 5: [Pendiente de completar]
1. **Prompt utilizado:**
   > [Escribe aquí el prompt literal]
2. **Parte relevante de la respuesta de la IA:**
   > [Copia aquí la respuesta relevante de la IA]
3. **Cómo sospechamos que había un error:**
   [Explica la sospecha]
4. **Evidencia que lo demuestra:**
   [Detalla la derivación, experimento o prueba]
5. **Categoría:** [Matemáticas / Código / Datos del mundo real / Bibliografía / Conceptos]
6. **La corrección:**
   [Describe la solución]