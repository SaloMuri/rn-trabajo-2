import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from creditrisk.expected_loss import decile_table, estimate_ead, expected_loss, fit_ead_ratio


def test_expected_loss_is_product():
    el = expected_loss(np.array([0.2, 0.5]), np.array([0.9, 1.0]), np.array([1000.0, 200.0]))
    assert np.allclose(el, [180.0, 100.0])


def test_ead_ratio_by_term_and_estimates():
    train = pd.DataFrame({"term": [36, 36, 60], "funded_amnt": [1000.0, 1000.0, 1000.0],
                          "ead_default": [600.0, 700.0, 800.0]})
    r = fit_ead_ratio(train)
    assert np.isclose(r["36"], 0.65) and np.isclose(r["60"], 0.8)
    amount, term = np.array([2000.0, 1000.0]), np.array([36, 60])
    assert np.allclose(estimate_ead(amount, term, r, "A"), amount)             # A: sin amortización
    assert np.allclose(estimate_ead(amount, term, r, "B"), [1300.0, 800.0])    # B: monto x razón del plazo
    assert np.isclose(estimate_ead(np.array([1000.0]), np.array([48]), r, "B")[0], 1000.0 * r["global"])


def test_decile_table_orders_groups_and_keeps_all_rows():
    rng = np.random.default_rng(0)
    el_hat = rng.uniform(0, 100, 1000)
    realized = el_hat + rng.normal(0, 5, 1000)
    t = decile_table(el_hat, realized)
    assert t["n"].sum() == 1000 and len(t) == 10
    assert t["el_predicha_media"].is_monotonic_increasing
