import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from creditrisk.scorecard import CreditScorecard

def test_pdo_doubles_good_odds():
    sc = CreditScorecard(base_score=600.0, base_odds=19.0, pdo=20.0)
    assert np.isclose(sc.pd_to_score(0.05), 600.0, atol=1e-4)
    assert np.isclose(sc.pd_to_score(1.0 / 39.0), 620.0, atol=1e-4)