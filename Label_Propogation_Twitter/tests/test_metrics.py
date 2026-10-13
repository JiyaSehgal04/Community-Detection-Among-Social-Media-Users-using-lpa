import math

import networkx as nx
import pytest

from backend import metrics


def test_circle_f1_is_one_when_communities_equal_circles():
    circles = {"0": [1, 2, 3], "1": [4, 5]}
    communities = [{1, 2, 3}, {4, 5}]
    assert metrics.circle_f1(communities, circles) == pytest.approx(1.0)


def test_circle_f1_is_nan_without_circles():
    assert math.isnan(metrics.circle_f1([{1, 2}], {}))


def test_circle_f1_ignores_users_outside_circles():
    circles = {"0": [1, 2], "1": [3, 4]}
    # 10, 11, 12 are in no circle; they must not count as errors
    communities = [{1, 2, 10, 11}, {3, 4, 12}]
    assert metrics.circle_f1(communities, circles) == pytest.approx(1.0)


def test_circle_f1_partial_match():
    circles = {"0": [1, 2, 3, 4]}
    communities = [{1, 2}, {3, 4}]
    # circle->best: F1({1,2,3,4},{1,2}) = 2*1*0.5/1.5 = 2/3
    # community->best: each community's best is the circle, F1 = 2/3
    assert metrics.circle_f1(communities, circles) == pytest.approx(2 / 3)


def test_interest_purity():
    tokens = {1: {"#a"}, 2: {"#a"}, 3: {"#b"}, 4: {"#b"}, 5: {"#b"}, 6: set()}
    # {1,2,3}: best token #a covers 2/3; {4,5,6}: #b covers 2/3; singleton {7} ignored
    communities = [{1, 2, 3}, {4, 5, 6}, {7}]
    assert metrics.interest_purity(communities, tokens) == pytest.approx(2 / 3)


def test_interest_purity_nan_when_no_groups():
    assert math.isnan(metrics.interest_purity([{1}, {2}], {1: {"#a"}}))


def test_coverage_and_community_count():
    stats = metrics.community_stats([{1, 2, 3}, {4, 5}, {6}, {7}])
    assert stats == {"n_communities": 2, "n_singletons": 2, "coverage": pytest.approx(5 / 7),
                     "largest_share": pytest.approx(3 / 7)}


def test_modularity_nan_for_graph_without_edges():
    G = nx.Graph()
    G.add_nodes_from([1, 2])
    assert math.isnan(metrics.safe_modularity(G, [{1}, {2}]))


def test_modularity_matches_networkx():
    G = nx.karate_club_graph()
    parts = nx.community.louvain_communities(G, seed=0)
    assert metrics.safe_modularity(G, parts) == pytest.approx(nx.community.modularity(G, parts))
