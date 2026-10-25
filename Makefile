export PYTHONPATH := src

# TODO: completar cada objetivo cuando exista el script correspondiente.
setup:   ; pip install -r requirements.txt
data:    ; python -m creditrisk.data.build_dataset
pd:      ; python scripts/run_pd.py
lgd:     ; python scripts/run_lgd.py
search:  ; python scripts/run_search.py
seeds:   ; python scripts/run_seeds.py
figures: ; python scripts/make_figures.py
demo: 	 ; python scripts/run_pd.py --config configs/demo.yaml
app:     ; streamlit run app/app.py
test:    ; pytest -q
