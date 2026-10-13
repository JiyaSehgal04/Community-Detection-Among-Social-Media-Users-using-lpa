import math

import pandas as pd
import pytest

from backend import graphs
from pipeline import experiments


@pytest.fixture
def dataset():
    # Ego 100: two follow triangles {1,2,3} and {4,5,6} sharing interests; ego 200 has no circles.
    nodes100 = [1, 2, 3, 4, 5, 6, 100]
    users = pd.DataFrame({
        "user_id": nodes100 + [7, 200],
        "ego_ids": [[100]] * 7 + [[200], [200]],
        "n_hashtags": [1] * 9, "n_mentions": [0] * 9,
    })
    interests = pd.DataFrame({
        "user_id": [1, 2, 3, 4, 5, 6, 7],
        "token": ["#a", "#a", "#a", "#b", "#b", "#b", "#a"],
        "type": ["hashtag"] * 7,
    })
    tri = [(1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4), (3, 4)]
    edges = tri + [(b, a) for a, b in tri] + [(100, n) for n in range(1, 7)]
    follows_by_ego = pd.DataFrame(
        [(a, b, 100) for a, b in edges] + [(200, 7, 200)], columns=["src", "dst", "ego_id"])
    egos = pd.DataFrame({"ego_id": [100, 200], "n_nodes": [7, 2], "n_edges": [len(edges), 1],
                         "n_circles": [2, 0], "has_circles": [True, False]})
    circles = {"100": {"0": [1, 2, 3], "1": [4, 5, 6]}, "200": {}}
    return graphs.Dataset(users, interests, follows_by_ego, egos, circles)


def test_run_produces_rows_for_every_mode_and_algorithm(dataset):
    args = experiments.parse_args(["--egos", "100,200", "--runs", "3", "--k", "3", "--min-sim", "0.1"])
    metrics, runs = experiments.run(dataset, args)

    assert len(metrics) == 2 * 3 * 3
    assert set(metrics["mode"]) == {"follow", "interest", "hybrid"}
    assert set(metrics["algorithm"]) == {"lpa", "nx_lpa", "louvain"}
    assert metrics.loc[metrics.ego_id == 200, "circle_f1"].isna().all()
    assert metrics.loc[metrics.ego_id == 200, "circle_f1_one_community"].isna().all()
    # one community {1..6, 100} vs circles {1,2,3}, {4,5,6}: F1 = 2*(3/6)/(1+3/6) = 2/3
    assert (metrics.loc[metrics.ego_id == 100, "circle_f1_one_community"] == 2 / 3).all()
    # egos x modes x runs, LPA only; ego 200's interest graph has no edges, so it has no runs
    assert len(runs) == 2 * 3 * 3 - 3
    assert not ((runs.ego_id == 200) & (runs["mode"] == "interest")).any()


def test_lpa_recovers_planted_communities(dataset):
    args = experiments.parse_args(["--egos", "100", "--mode", "hybrid", "--runs", "5",
                                   "--k", "3", "--min-sim", "0.1"])
    metrics, _ = experiments.run(dataset, args)
    row = metrics[(metrics.algorithm == "lpa")].iloc[0]
    assert row.circle_f1 == pytest.approx(1.0)
    # ego node 100 has no tokens but joins a community: 6 of 7 grouped users share the top token
    assert row.interest_purity == pytest.approx(6 / 7)


def test_summary_table_averages_per_mode_and_algorithm(dataset):
    args = experiments.parse_args(["--egos", "100,200", "--runs", "2", "--k", "3", "--min-sim", "0.1"])
    metrics, _ = experiments.run(dataset, args)
    summary = experiments.summarize(metrics)
    assert len(summary) == 9
    assert (summary.n_egos == 2).all()
    assert (summary.n_egos_with_circles == 1).all()
    assert not math.isnan(summary.circle_f1.iloc[0])
