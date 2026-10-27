import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from creditrisk.data.build_lgd_target import build_lgd_target


def _loans() -> pd.DataFrame:
    rows = [
        # id, loan_status, issue_d, term, funded, rec_prncp, recoveries, fee
        (1, "Charged Off", "Jan-2013", " 36 months", 10000, 4000, 600, 100),
        (2, "Does not meet the credit policy. Status:Charged Off", "Feb-2012", " 36 months", 5000, 0, 0, 0),
        (3, "Fully Paid", "Jan-2013", " 36 months", 10000, 10000, 0, 0),
        (4, "Default", "Jan-2013", " 36 months", 10000, 2000, 0, 0),
        (5, "Charged Off", "Jan-2013", " 36 months", 1000, 1000, 0, 0),    # EAD = 0: se descarta
        (6, "Charged Off", "Jan-2013", " 36 months", 1000, 0, 2000, 0),    # recupera más que EAD
    ]
    cols = ["id", "loan_status", "issue_d", "term", "funded_amnt",
            "total_rec_prncp", "recoveries", "collection_recovery_fee"]
    return pd.DataFrame(rows, columns=cols)


def test_lgd_formula_and_clipping():
    out = build_lgd_target(_loans()).set_index("id")
    assert np.isclose(out.loc[1, "lgd"], 1 - (600 - 100) / 6000)  # EAD = 10000 - 4000
    assert out.loc[2, "lgd"] == 1.0
    assert out.loc[6, "lgd"] == 0.0                                # recorte inferior
    assert out["lgd"].between(0, 1).all()


def test_main_version_excludes_default_and_non_defaulted():
    ids = set(build_lgd_target(_loans())["id"])
    assert ids == {1, 2, 6}  # sin Fully Paid, sin Default, sin EAD <= 0


def test_include_default_adds_only_default_loans():
    out = build_lgd_target(_loans(), include_default=True).set_index("id")
    assert set(out.index) == {1, 2, 4, 6}
    assert out.loc[4, "lgd"] == 1.0  # sin recuperaciones registradas -> LGD = 1
