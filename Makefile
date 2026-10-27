export PYTHONPATH := src

.PHONY: setup data lgd-data pd lgd lgd-sens lgd-demo demo-lgd search-lgd search seeds seeds-lgd censura el figures figures-lgd demo app test

# Instalación y datos
setup:     ; pip install -r requirements.txt
data:      ; python -m creditrisk.data.build_dataset
lgd-data:  ; python src/creditrisk/data/build_lgd_target.py

# Parte 1 (PD)
pd:        ; python scripts/run_pd.py
search:    ; python scripts/run_search.py
seeds:     ; python scripts/run_seeds.py
censura:   ; python scripts/censura_etiquetas.py
figures:   ; python scripts/make_figures.py

# Parte 2 (LGD y pérdida esperada)
lgd:       ; python scripts/run_lgd.py
lgd-sens:  ; python scripts/run_lgd.py --population with_default
lgd-demo:  ; python scripts/run_lgd.py --demo
search-lgd: ; python scripts/run_search_lgd.py
seeds-lgd: ; python scripts/run_seeds_lgd.py
el:        ; python scripts/run_expected_loss.py
figures-lgd: ; python scripts/make_figures_lgd.py

# Demostración (menos de 2 minutos)
demo-lgd:  ; python scripts/run_lgd.py --demo && python scripts/run_expected_loss.py && python scripts/make_figures_lgd.py
demo:      ; python scripts/run_pd.py --config configs/demo.yaml

# Aplicación y pruebas
app:       ; streamlit run app/app.py
test:      ; pytest -q