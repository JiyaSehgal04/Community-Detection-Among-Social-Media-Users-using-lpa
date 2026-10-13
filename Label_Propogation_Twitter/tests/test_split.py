import pandas as pd

from pipeline import split as split_mod


def _egos() -> pd.DataFrame:
    n = 43  # 40 with circles, 3 without; n_nodes has many ties
    return pd.DataFrame({
        "ego_id": range(1000, 1000 + n),
        "n_nodes": [11 + (i % 7) for i in range(n)],
        "n_edges": [20] * n,
        "n_circles": [1] * n,
        "has_circles": [i < 40 for i in range(n)],
    })


def test_split_is_deterministic_disjoint_and_covers_egos_with_circles():
    egos = _egos()
    a = split_mod.make_split(egos, seed=42)
    b = split_mod.make_split(egos.sample(frac=1, random_state=3), seed=42)  # row order must not matter
    assert a == b

    train, val, test = set(a["train"]), set(a["val"]), set(a["test"])
    assert not (train & val) and not (train & test) and not (val & test)
    assert train | val | test == set(egos.ego_id[egos.has_circles])
    assert len(a["train"]) + len(a["val"]) + len(a["test"]) == 40
