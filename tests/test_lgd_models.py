import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from creditrisk.config import apply_override, deep_update
from creditrisk.lgd_metrics import compare_distributions
from creditrisk.metrics import evaluate_lgd_metrics
from creditrisk.models.baseline_lgd import ConstantLGDModel, LinearLGDModel
from creditrisk.models.mlp_lgd import _loss_fn, predict_lgd_mlp, train_lgd_mlp


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5)).astype(np.float32)
    y = np.clip(0.9 - 0.05 * X[:, 0] + 0.02 * rng.normal(size=n), 0, 1).astype(np.float32)
    return X, y


def test_config_override_and_merge():
    base = {"mlp": {"lr": 0.001, "dropout": 0.1}, "seed": 1}
    merged = deep_update(base, {"mlp": {"lr": 0.01}})
    assert merged["mlp"] == {"lr": 0.01, "dropout": 0.1} and base["mlp"]["lr"] == 0.001
    apply_override(merged, "mlp.hidden_dims=[128,64]")
    apply_override(merged, "mlp.use_batchnorm=false")
    assert merged["mlp"]["hidden_dims"] == [128, 64] and merged["mlp"]["use_batchnorm"] is False


def test_lgd_metrics_known_values():
    m = evaluate_lgd_metrics(np.array([0.0, 1.0]), np.array([0.5, 0.5]))
    assert np.isclose(m["mae"], 0.5) and np.isclose(m["rmse"], 0.5)


def test_bce_with_fractional_target_matches_formula():
    z = torch.tensor([0.3, -1.2])
    y = torch.tensor([0.8, 1.0])
    p = torch.sigmoid(z)
    manual = -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()
    assert torch.isclose(_loss_fn("bce")(z, y), manual, atol=1e-6)
    assert torch.isclose(_loss_fn("mse")(z, y), ((p - y) ** 2).mean(), atol=1e-6)


def test_linear_baseline_is_clipped_and_constant_is_mean():
    X, y = _toy()
    lin = LinearLGDModel(clip=True).fit(X, y)
    assert lin.predict(X * 100).min() >= 0.0 and lin.predict(X * 100).max() <= 1.0
    assert lin.count_parameters() == X.shape[1] + 1
    assert np.isclose(ConstantLGDModel().fit(X, y).predict(X)[0], y.mean())


def test_mlp_predictions_in_unit_interval_and_bias_init():
    X, y = _toy()
    cfg = {"hidden_dims": [8], "epochs": 3, "patience": 3, "batch_size": 64, "loss": "bce"}
    model, hist = train_lgd_mlp(X[:300], y[:300], X[300:], y[300:], cfg, device="cpu", verbose=False)
    pred = predict_lgd_mlp(model, X)
    assert pred.min() > 0.0 and pred.max() < 1.0
    assert len(hist["val_rmse"]) >= 1
    # arranca en la media: tras 0 épocas el sesgo es logit(media del train)
    cfg0 = dict(cfg, epochs=1, lr=0.0)
    m0, _ = train_lgd_mlp(X[:300], y[:300], X[300:], y[300:], cfg0, device="cpu", verbose=False)
    p0 = y[:300].mean()
    assert np.isclose(m0.net[-1].bias.item(), np.log(p0 / (1 - p0)), atol=1e-4)


def test_identical_distributions_have_zero_distance():
    y = np.random.default_rng(1).uniform(size=500)
    d = compare_distributions(y, y)
    assert d["wasserstein"] == 0.0 and d["ks"] == 0.0
