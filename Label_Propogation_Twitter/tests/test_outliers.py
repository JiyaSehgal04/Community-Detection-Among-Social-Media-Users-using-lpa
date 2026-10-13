import pandas as pd
import pytest

from backend import graphs
from backend.outliers import OutlierConfig, filter_tokens


def _interests() -> pd.DataFrame:
    rows = [(1, "#common"), (1, "#a"), (1, "#b"), (2, "#common"), (2, "#a"),
            (3, "#common"), (3, "#b"), (4, "#common"), (5, "#common")]
    return pd.DataFrame({"user_id": [u for u, _ in rows], "token": [t for _, t in rows],
                         "type": "hashtag"})


def _dataset(interests: pd.DataFrame, n_users: int = 6) -> graphs.Dataset:
    users = pd.DataFrame({"user_id": list(range(1, n_users + 1)), "ego_ids": [[100]] * n_users,
                          "n_hashtags": [0] * n_users, "n_mentions": [0] * n_users})
    # Star 1-{2,3,4,5} plus edge 2-3 in ego 100.
    follows = pd.DataFrame({"src": [1, 1, 1, 1, 2], "dst": [2, 3, 4, 5, 3], "ego_id": [100] * 5})
    egos = pd.DataFrame({"ego_id": [100], "n_nodes": [n_users], "n_edges": [5],
                         "n_circles": [0], "has_circles": [False]})
    return graphs.Dataset(users, interests, follows, egos, circles={})


def test_token_rule_drops_common_tokens_and_off_is_noop():
    interests = _interests()
    kept = filter_tokens(interests, 6, OutlierConfig(token_max_share=0.5))  # drops tokens used by > 3 users
    assert set(kept.token) == {"#a", "#b"}

    pd.testing.assert_frame_equal(filter_tokens(interests, 6, OutlierConfig()), interests)

    ds = _dataset(interests)
    index, matrix = ds.tfidf("both", OutlierConfig(token_max_share=0.0001))  # every token filtered out
    assert index == {} and matrix.shape[0] == 0
    empty = _dataset(interests.iloc[0:0])  # no interests at all
    assert empty.tfidf("both")[0] == {}
    assert graphs.interest_graph(ds, 100, outliers=OutlierConfig(token_max_share=0.0001)).number_of_edges() == 0


def test_hub_rule_rescales_only_hub_edges_and_off_is_noop():
    interests = pd.DataFrame({"user_id": [2, 3], "token": ["#a", "#a"], "type": "hashtag"})
    ds = _dataset(interests)  # degrees: node 1 -> 4, nodes 2 and 3 -> 2, nodes 4 and 5 -> 1

    cap2 = graphs.follow_graph(ds, 100, outliers=OutlierConfig(hub_cap=2))
    assert cap2[1][2]["weight"] == pytest.approx(0.5)  # hub 1: 2/4
    assert cap2[2][3]["weight"] == pytest.approx(1.0)  # no hub end

    cap1 = graphs.follow_graph(ds, 100, outliers=OutlierConfig(hub_cap=1))
    assert cap1[2][3]["weight"] == pytest.approx(0.25)   # both ends hubs: 0.5 * 0.5
    assert cap1[1][2]["weight"] == pytest.approx(0.125)  # 0.25 * 0.5

    for off in (None, OutlierConfig(), OutlierConfig(hub_cap=10)):  # 10 >= max degree
        g = graphs.follow_graph(ds, 100, outliers=off)
        assert all(d["weight"] == 1.0 for *_, d in g.edges(data=True))

    hybrid = graphs.hybrid_graph(ds, 100, k=2, min_sim=0.0, outliers=OutlierConfig(hub_cap=2))
    assert hybrid[1][2]["weight"] == pytest.approx(0.3 * 0.5)  # follow part scaled
    assert hybrid[2][3]["weight"] == pytest.approx(1.0 + 0.3)  # interest edge untouched


def test_heavy_user_rule_keeps_lowest_document_frequency_tokens():
    interests = _interests()
    cfg = OutlierConfig(max_tokens=2)
    kept = filter_tokens(interests, 6, cfg)
    per_user = kept.groupby("user_id").size()
    original = interests.groupby("user_id").size()
    assert (per_user == original.clip(upper=2)).all()
    assert set(kept[kept.user_id == 1].token) == {"#a", "#b"}  # #common (5 users) loses to #a, #b (2 each)
    pd.testing.assert_frame_equal(kept, filter_tokens(interests, 6, cfg))  # deterministic
