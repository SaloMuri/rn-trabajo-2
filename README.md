# Credit Risk NN
Trabajo 2 — Redes neuronales y algoritmos bioinspirados (UNAL Facultad de Minas, 2026-02).
Enlaces: reporte (TODO) · aplicación web (TODO) · video promocional (TODO)

# Configuración

# Configuración en MacOS y Linux

Ejecute los siguientes comandos en el terminal:

bash
python3 -m venv .venv
source .venv/bin/activate
source setup.sh


# Configuración en Windows

Ejecute los siguientes comandos en el terminal:

bash
python3 -m venv .venv
.venv\Scripts\activate
setup


# Ejecución de pruebas

Ejecute el siguiente comando en el terminal:

bash
pytest

## Instalación
TODO

## Ejecución
TODO: un comando por experimento (ver `Makefile`), modo demo, app web.

## Semillas
TODO

## Estructura
| Ruta | Para qué existe |
|---|---|
| `configs/` | Parámetros de los experimentos; permiten cambiarlos sin editar código. |
| `data/raw/` | Documenta el origen de los datos originales (no se versionan). |
| `data/processed/` | Datos procesados que usan los modelos (versionados). |
| `src/creditrisk/` | Código de la librería: datos, modelos, métricas, scorecard, interpretabilidad, pérdida esperada. |
| `scripts/` | Puntos de entrada de los experimentos y de las figuras. |
| `models/` | Modelos entrenados (versionados). |
| `reports/` | Figuras y tablas generadas por el código. |
| `app/` | Aplicación web. |
| `blog/` | Fuente del reporte técnico. |
| `docs/` | Bitácoras de uso de IA y cacería de la alucinación. |
| `tests/` | Pruebas. |
