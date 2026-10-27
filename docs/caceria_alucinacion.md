# Cacería de la alucinación 🎯

Las herramientas de IA producen errores que suenan convincentes. Su misión es cazarlos. A continuación se documentan los hallazgos en los que el asistente de IA generó información incorrecta, imprecisa, desactualizada o subóptima durante el desarrollo del trabajo.

Este documento reúne cuatro hallazgos con evidencia (los tres de la Parte 1, desarrollada con Gemini / Argon, y uno de la Parte 2, desarrollada con Claude) y una verificación sin hallazgo (Verificación 5, sobre la reproducibilidad de la Parte 2). Cada hallazgo incluye el prompt, la parte relevante de la respuesta de la IA, cómo se sospechó el error, la evidencia, la categoría, la corrección y la lección aprendida. En la verificación se describe qué se comprobó, cómo y con qué resultado.

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
7. **Lección Aprendida:**
   Revisar y contrastar los datos entregados y confiar en la misma creación del proceso.
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
7. **Lección Aprendida:**
   No hacer literal lo que dice la IA y verificar que la linea de trabajo que se esta haciendo si este: en primer lugar bien establecida y en segundo lugar bien realizada.

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
7. **Lección Aprendida:**
   Ser minuiciosos al momento de escribir variables asociadas al conjunto de datos, verificando que sean identicas o en defecto, cubrir las posibles variantes.
---

### Hallazgo 4: La IA reporta una mejora de "3.5 % en RMSE" que era la del MAE
1. **Prompt utilizado:**
   > Listo, ya corrí `run_lgd.py --best`, `run_seeds_lgd.py` y `run_expected_loss.py`. Aquí está la salida. Sigamos con las figuras y la interpretabilidad, y dime qué muestra cada figura y qué resultados importantes tenemos para el reporte.
2. **Parte relevante de la respuesta de la IA:**
   > L4: el MLP gana en promedio sobre las 30 semillas. La ganancia es de 3.5 % en RMSE frente a la regresión lineal, con unas 1,000 veces más parámetros (51,201 frente a 39).
3. **Cómo sospechamos que había un error:**
   Al revisar los resultados de nuestra ejecución (30 semillas), el número no coincidía con la tabla de RMSE.
4. **Evidencia que lo demuestra:**
   Con los resultados de esa ejecución (previa a la corrección de los datos, con 51,201 parámetros): RMSE lineal 0.0924 frente a MLP-C 0.0890, es decir, −3.7 %. El 3.5 % correspondía al MAE (0.0779 frente a 0.0752). La IA corrigió la cifra en un mensaje posterior. Es un error pequeño, pero real y comprobable con la fórmula $(0.0924-0.0890)/0.0924$.
5. **Categoría:** Matemáticas (cálculo de un porcentaje)
6. **La corrección:**
   Se recalculó con los CSV (`seeds_summary.csv`). Con los datos corregidos (761 créditos recuperados) el reporte usa las cifras finales: −4.2 % en RMSE y −6.5 % en MAE.
7. **Lección Aprendida:**
   Todo porcentaje que dé la IA hay que recalcularlo a mano desde las tablas antes de ponerlo en el reporte.

---

### Verificación 5 (sin hallazgo): Reproducibilidad y consistencia de los resultados de la Parte 2
1. **Qué se verificó:**
   Que los resultados de la Parte 2 fueran reproducibles entre equipos y coherentes con los datos, para descartar que la IA hubiera reportado cifras que no salen del código.
2. **Cómo se verificó:**
   - *Reproducibilidad entre equipos:* se ejecutaron `run_lgd.py --best`, `run_seeds_lgd.py` y `run_expected_loss.py` en el entorno de la IA y en el equipo de la autora de la Parte 2, y se compararon las salidas. La comparación se hizo antes de corregir el bug de PD (761 créditos); tras la corrección solo se re-ejecutó en el equipo de la autor de la Parte 2.
   - *Conteos del objetivo:* se comparó la salida de `build_lgd_target.py` con el `value_counts` de `loan_status` del dataset.
   - *Pruebas automáticas:* se ejecutó `pytest -q tests`.
3. **Resultado:**
   - Las cifras coincidieron salvo ruido numérico de la cuarta cifra decimal (por ejemplo, MLP-C: RMSE de prueba 0.0887 y MLP-B 0.0881 en ambos; pérdida esperada con EAD B / realizada 1.6174 en ambos). Solo cambiaron los tiempos de ejecución, por el hardware.
   - Los conteos cuadran: 45,248 `Charged Off` + 761 de política = 46,009 filas; 1,219 `Default` menos 1 con EAD ≤ 0 (se filtra) = 1,218, que sumados dan 47,227. La diferencia de 1 frente a lo esperado inicialmente se explica por ese `Default`.
   - `pytest -q tests`: pasan las 20 pruebas.
4. **Evidencia:**
   Salidas de terminal de `run_lgd.py`, `run_seeds_lgd.py` y `run_expected_loss.py` en ambos equipos; `reports/tables/lgd/main/`; `tests/`.
5. **Categoría:** Verificación (sin error detectado)
6. **Lección Aprendida:**
   Fijar semillas, versiones de las librerías y datos procesados versionados permite reproducir resultados en otra máquina; las pequeñas diferencias numéricas pueden cambiar el orden de configuraciones casi empatadas, pero no las conclusiones.