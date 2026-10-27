import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from creditrisk.lgd_interpretability import group_features, grouped_permutation_importance


def test_group_features_joins_one_hot_columns():
    names = ["num__dti", "num__term", "cat__purpose_car", "cat__purpose_credit_card", "cat__home_ownership_RENT"]
    g = group_features(names, ["purpose", "home_ownership"])
    assert g == {"dti": [0], "term": [1], "purpose": [2, 3], "home_ownership": [4]}


def test_permutation_importance_finds_the_informative_variable():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 3))
    y = 0.5 + 0.2 * X[:, 0]                       # solo la variable 0 importa
    predict = lambda M: 0.5 + 0.2 * M[:, 0]
    imp = grouped_permutation_importance(predict, X, y, {"a": [0], "b": [1], "c": [2]}, n_repeats=5)
    assert imp.iloc[0]["variable"] == "a" and imp.iloc[0]["aumento_rmse_media"] > 0.1
    assert imp[imp["variable"] != "a"]["aumento_rmse_media"].abs().max() < 1e-12   # barajar lo irrelevante no cambia nada
