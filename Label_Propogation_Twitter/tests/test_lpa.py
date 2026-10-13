import networkx as nx

from backend import lpa


def two_triangles() -> nx.Graph:
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (1, 3), (4, 5), (5, 6), (4, 6)], weight=1.0)
    return G


def as_frozensets(communities):
    return {frozenset(c) for c in communities}


def test_two_disconnected_triangles_give_two_communities():
    result = lpa.label_propagation(two_triangles(), seed=0)
    assert as_frozensets(result.communities) == {frozenset({1, 2, 3}), frozenset({4, 5, 6})}


def test_fixed_seed_is_reproducible():
    G = nx.karate_club_graph()
    nx.set_edge_attributes(G, 1.0, "weight")
    a = lpa.label_propagation(G, seed=42)
    b = lpa.label_propagation(G, seed=42)
    assert a.labels == b.labels
    assert a.iterations == b.iterations


def test_every_node_is_assigned_exactly_once():
    G = nx.karate_club_graph()
    result = lpa.label_propagation(G, seed=1)
    members = [n for c in result.communities for n in c]
    assert sorted(members) == sorted(G.nodes)


def test_isolated_node_is_its_own_community():
    G = two_triangles()
    G.add_node(99)
    result = lpa.label_propagation(G, seed=0)
    assert frozenset({99}) in as_frozensets(result.communities)


def test_weights_decide_between_neighbours():
    # Node 0 sits between two groups; the heavier edge should win.
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (1, 3)], weight=1.0)
    G.add_edges_from([(4, 5), (5, 6), (4, 6)], weight=1.0)
    G.add_edge(0, 1, weight=5.0)
    G.add_edge(0, 4, weight=0.1)
    result = lpa.label_propagation(G, seed=3)
    assert result.labels[0] == result.labels[1]


def test_converges_on_stable_partition():
    result = lpa.label_propagation(two_triangles(), seed=0)
    assert result.converged
    assert result.iterations <= 5


def test_trace_records_initial_and_each_iteration():
    result = lpa.label_propagation(two_triangles(), seed=0, trace=True)
    assert result.trace[0] == {n: n for n in range(1, 7)}
    assert len(result.trace) == result.iterations + 1
    assert result.trace[-1] == result.labels


def test_empty_graph():
    result = lpa.label_propagation(nx.Graph(), seed=0)
    assert result.communities == []


def test_best_of_runs_picks_highest_modularity():
    G = nx.karate_club_graph()
    best, runs = lpa.best_of_runs(G, runs=5, seed=10)
    assert len(runs) == 5
    assert best.modularity == max(r.modularity for r in runs)
    assert [r.seed for r in runs] == [10, 11, 12, 13, 14]
