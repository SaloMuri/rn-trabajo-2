import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from creditrisk.data.targets import encode_pd_target

def test_pd_target_encoding():
    s = pd.Series(["Fully Paid", "Charged Off", "Current", "Default"])
    res = encode_pd_target(s)
    assert res.iloc[0] == 0.0
    assert res.iloc[1] == 1.0
    assert pd.isna(res.iloc[2])
    assert res.iloc[3] == 1.0