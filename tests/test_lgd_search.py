import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
import torch
import yaml

from creditrisk.models.baseline_lgd import LinearLGDModel
from creditrisk.models.mlp_lgd import make_optimizer
from creditrisk.search_space import sample_mlp_configs

ROOT = Path(__file__).resolve().parents[1]


def _space():
    return yaml.safe_load((ROOT / "configs" / "lgd.yaml").read_text(encoding="utf-8"))["search"]["space"]


def test_sampling_is_reproducible_distinct_and_in_range():
    space = _space()
    a = sample_mlp_configs(space, 60, seed=42)
    b = sample_mlp_configs(space, 60, seed=42)
    assert a == b                                              # misma semilla -> mismas configuraciones
    assert len({repr(sorted(c.items())) for c in a}) == 60     # todas distintas
    for c in a:
        lo, hi = space["lr_range"][c["optimizer"]]
        assert lo * 0.9 <= c["lr"] <= hi * 1.1                 # (el redondeo a 2 cifras puede desplazar un poco)
        assert 1e-6 * 0.9 <= c["weight_decay"] <= 1e-2 * 1.1
        assert c["activation"] in space["activation"] and c["batch_size"] in space["batch_size"]
    assert sample_mlp_configs(space, 60, seed=7) != a          # otra semilla -> otras configuraciones


def test_optimizers_and_error_on_unknown():
    p = [torch.nn.Parameter(torch.zeros(2))]
    for name in ("adamw", "adam", "sgd", "rmsprop"):
        assert isinstance(make_optimizer(name, p, 1e-3, 1e-4), torch.optim.Optimizer)
    with pytest.raises(ValueError):
        make_optimizer("lbfgs", p, 1e-3, 0.0)


def test_ridge_shrinks_coefficients_and_counts_params():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4)).astype(np.float32)
    y = np.clip(0.8 + 0.1 * X[:, 0] + 0.05 * rng.normal(size=300), 0, 1).astype(np.float32)
    ols = LinearLGDModel(alpha=None).fit(X, y)
    ridge = LinearLGDModel(alpha=1000.0).fit(X, y)
    assert np.linalg.norm(ridge.model.coef_) < np.linalg.norm(ols.model.coef_)
    assert ridge.count_parameters() == ols.count_parameters() == 5
