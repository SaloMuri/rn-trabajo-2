import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from creditrisk.metrics import evaluate_pd_metrics

def test_pd_metrics_bounds():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    m = evaluate_pd_metrics(y_true, y_prob)
    assert m["auc"] == 1.0
    assert m["gini"] == 1.0
    assert m["ks"] == 1.0
    assert 0.0 <= m["brier"] <= 1.0