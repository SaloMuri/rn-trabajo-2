export PYTHONPATH := src

# TODO: completar cada objetivo cuando exista el script correspondiente.
setup:   ; pip install -r requirements.txt
data:    ; python -m creditrisk.data.build_dataset
pd:      ; python scripts/run_pd.py
lgd:     ; python scripts/run_lgd.py
lgd-sens: ; python scripts/run_lgd.py --population with_default
lgd-demo: ; python scripts/run_lgd.py --demo
search-lgd: ; python scripts/run_search_lgd.py
search:  ; python scripts/run_search.py
seeds:   ; python scripts/run_seeds.py
seeds-lgd: ; python scripts/run_seeds_lgd.py
censura: ; python scripts/censura_etiquetas.py
el: ; python scripts/run_expected_loss.py
figures: ; python scripts/make_figures.py
figures-lgd: ; python scripts/make_figures_lgd.py
demo:    ; python scripts/run_pd.py --config configs/demo.yaml
app:     ; streamlit run app/app.py
test:    ; pytest -q
