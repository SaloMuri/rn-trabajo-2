import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest
from creditrisk.data.splits import temporal_train_val_test_split


def _df():
    f = pd.to_datetime(["2013-01-01", "2013-06-01", "2014-06-01", "2014-09-01", "2014-12-01", "2015-03-01"])
    return pd.DataFrame({"issue_d_dt": f, "x": range(6)})


def test_limites_temporales():
    tr, va, te = temporal_train_val_test_split(_df(), train_end="2014-06-01", val_end="2014-12-01")
    assert (len(tr), len(va), len(te)) == (3, 2, 1)
    assert tr["issue_d_dt"].max() <= pd.Timestamp("2014-06-01") < va["issue_d_dt"].min()
    assert va["issue_d_dt"].max() <= pd.Timestamp("2014-12-01") < te["issue_d_dt"].min()


def test_particion_vacia_falla():
    with pytest.raises(ValueError):
        temporal_train_val_test_split(_df(), train_end="2014-06-01", val_end="2030-01-01")