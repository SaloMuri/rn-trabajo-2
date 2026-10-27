"""Pérdida esperada de cada crédito del test: E[L] = PD * LGD * EAD.

    python scripts/run_expected_loss.py
    python scripts/run_expected_loss.py --set expected_loss.ead_primary=A

Requiere: la PD de la Parte 1 (models/pd/) y la LGD (python scripts/run_lgd.py --best).
EAD: B (principal) = monto x razón media EAD/monto por plazo (train); A (sensibilidad) = monto.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from creditrisk.config import load_config
from creditrisk.expected_loss import decile_table, estimate_ead, expected_loss, fit_ead_ratio
from creditrisk.experiments import load_lgd_splits
from creditrisk.models.mlp import load_mlp_checkpoint, predict_proba_mlp
from creditrisk.models.mlp_lgd import predict_lgd_mlp


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="RUTA=VALOR")
    p.add_argument("--base", default="configs/base.yaml")
    p.add_argument("--config", default="configs/lgd.yaml")
    a = p.parse_args()
    cfg = load_config([a.base, a.config], a.overrides)
    el = cfg["expected_loss"]
    fmt = cfg["paths"].get("output_format", "parquet")
    out_tables = os.path.join(cfg["output"]["tables_dir"], "main")
    os.makedirs(out_tables, exist_ok=True)

    print("[1/4] Razón EAD/monto por plazo (solo créditos castigados de TRAIN)")
    lgd_splits, _ = load_lgd_splits(cfg["lgd_data"]["paths"]["main"], cfg["paths"]["processed_pd_dir"], fmt)
    ratios = fit_ead_ratio(lgd_splits["train"])
    print("      " + " | ".join(f"{k}: {v:.3f}" for k, v in ratios.items()))
    with open(os.path.join(cfg["output"]["models_dir"], "main", "ead_ratio.json"), "w", encoding="utf-8") as f:
        json.dump(ratios, f, indent=2)

    print("[2/4] PD y LGD de cada crédito del test (todos los créditos de la Parte 1)")
    test = pd.read_parquet(os.path.join(cfg["paths"]["processed_pd_dir"], f"test.{fmt}")) if fmt == "parquet" \
        else pd.read_csv(os.path.join(cfg["paths"]["processed_pd_dir"], f"test.{fmt}"), low_memory=False)
    X_pd = joblib.load(el["pd_preprocessor"]).transform(test).astype(np.float32)
    pd_hat = predict_proba_mlp(load_mlp_checkpoint(el["pd_model"]), X_pd)
    X_lgd = joblib.load(el["lgd_preprocessor"]).transform(test).astype(np.float32)
    lgd_hat = predict_lgd_mlp(load_mlp_checkpoint(el["lgd_model"]), X_lgd)

    print("[3/4] EAD y pérdida esperada")
    amount, term = test["loan_amnt"].to_numpy(), test["term"].to_numpy()
    res = pd.DataFrame({"id": test["id"].to_numpy(), "target_pd": test["target_pd"].to_numpy(),
                        "pd": pd_hat, "lgd": lgd_hat})
    for m in ("A", "B"):
        res[f"ead_{m}"] = estimate_ead(amount, term, ratios, m)
        res[f"el_{m}"] = expected_loss(pd_hat, lgd_hat, res[f"ead_{m}"])

    # Pérdida realizada, solo donde se conoce: buenos (0) y castigados (LGD observada x EAD_default).
    obs = lgd_splits["test"][["id", "lgd", "ead_default"]].rename(columns={"lgd": "lgd_obs", "ead_default": "ead_obs"})
    res = res.merge(obs, on="id", how="left")
    res["conocida"] = (res["target_pd"] == 0) | res["lgd_obs"].notna()
    res["perdida_realizada"] = np.where(res["target_pd"] == 0, 0.0, res["lgd_obs"] * res["ead_obs"])
    res.loc[~res["conocida"], "perdida_realizada"] = np.nan
    res.to_csv(os.path.join(out_tables, "expected_loss_test.csv"), index=False)

    print("[4/4] Evaluación")
    k = res[res["conocida"]]
    castig = res[res["lgd_obs"].notna()]
    summary = {
        "n_test": int(len(res)), "n_con_perdida_conocida": int(len(k)),
        "n_excluidos_default_o_late": int((~res["conocida"]).sum()),
        "pd_media_predicha_test": float(res["pd"].mean()),
        "tasa_incumplimiento_observada_test_(target_pd)": float(res["target_pd"].mean()),
        "pd_media_predicha_subconjunto": float(k["pd"].mean()),
        "tasa_castigados_subconjunto": float(k["lgd_obs"].notna().mean()),
        "lgd_media_predicha_en_castigados": float(castig["lgd"].mean()),
        "lgd_media_observada_en_castigados": float(castig["lgd_obs"].mean()),
        "perdida_realizada_total": float(k["perdida_realizada"].sum()),
        "ead_mae_en_castigados": {m: float((castig[f"ead_{m}"] - castig["ead_obs"]).abs().mean()) for m in ("A", "B")},
        "ead_sesgo_medio_en_castigados_(pred-obs)": {m: float((castig[f"ead_{m}"] - castig["ead_obs"]).mean()) for m in ("A", "B")},
    }
    for m in ("A", "B"):
        tot = float(k[f"el_{m}"].sum())
        summary[f"el_total_{m}"] = tot
        summary[f"el_total_{m}_sobre_realizada"] = tot / summary["perdida_realizada_total"]
        summary[f"spearman_el_{m}_vs_realizada"] = float(spearmanr(k[f"el_{m}"], k["perdida_realizada"]).statistic)
    prim = el["ead_primary"]
    decs = decile_table(k[f"el_{prim}"].to_numpy(), k["perdida_realizada"].to_numpy())
    decs.round(2).to_csv(os.path.join(out_tables, f"expected_loss_deciles_{prim}.csv"), index=False)
    summary["ead_principal"] = prim
    with open(os.path.join(out_tables, "expected_loss_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 100)
    print(f"PÉRDIDA ESPERADA EN TEST (EAD principal: {prim})")
    print("=" * 100)
    for key, v in summary.items():
        print(f"  {key}: {v if not isinstance(v, float) else round(v, 4)}")
    print("-" * 100)
    print(f"Por deciles de pérdida esperada (EAD {prim}), dólares por crédito:")
    print(decs.round(1).to_string(index=False))
    print("=" * 100)


if __name__ == "__main__":
    main()
