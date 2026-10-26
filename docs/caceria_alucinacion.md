# Bitácora de Uso de Inteligencia Artificial
# Cacería de la alucinación 🎯

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