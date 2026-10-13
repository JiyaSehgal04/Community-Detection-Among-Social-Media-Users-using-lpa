import pandas as pd
import pytest

from backend import graphs


@pytest.fixture
def dataset():
    users = pd.DataFrame({
        "user_id": [100, 1, 2, 3, 4],
        "ego_ids": [[100], [100], [100], [100], [100, 200]],
        "n_hashtags": [0, 2, 2, 1, 0],
        "n_mentions": [0, 0, 0, 1, 0],
    })
    interests = pd.DataFrame({
        "user_id": [1, 1, 2, 2, 3, 3],
        "token": ["#a", "#b", "#a", "#b", "#c", "@x"],
        "type": ["hashtag", "hashtag", "hashtag", "hashtag", "hashtag", "mention"],
    })
    follows_by_ego = pd.DataFrame({
        "src": [1, 2, 1, 3, 100, 100, 100, 3],
        "dst": [2, 1, 3, 3, 1, 2, 3, 4],
        "ego_id": [100] * 8,
    })
    egos = pd.DataFrame({"ego_id": [100, 200, 300], "n_nodes": [5, 9, 50],
                         "n_edges": [8, 1, 1], "n_circles": [1, 2, 0],
                         "has_circles": [True, True, False]})
    return graphs.Dataset(users, interests, follows_by_ego, egos, circles={"100": {"0": [1, 2]}})


def test_ego_nodes(dataset):
    assert dataset.ego_nodes(100) == [1, 2, 3, 4, 100]


def test_follow_graph_any_vs_mutual(dataset):
    any_g = graphs.follow_graph(dataset, 100, direction="any")
    mutual_g = graphs.follow_graph(dataset, 100, direction="mutual")
    assert set(any_g.nodes) == {1, 2, 3, 4, 100}
    assert any_g.has_edge(1, 3) and any_g.has_edge(100, 2)
    assert mutual_g.has_edge(1, 2)
    assert not mutual_g.has_edge(1, 3)
    # self-loop 3->3 dropped
    assert not any_g.has_edge(3, 3)
    assert all(d["weight"] == 1.0 for *_, d in any_g.edges(data=True))


def test_interest_graph_links_similar_users(dataset):
    G = graphs.interest_graph(dataset, 100, k=2, min_sim=0.2, tokens="both")
    assert set(G.nodes) == {1, 2, 3, 4, 100}  # users without tokens stay as isolates
    assert G.has_edge(1, 2)
    assert G[1][2]["weight"] == pytest.approx(1.0)
    assert not G.has_edge(1, 3)  # no shared tokens


def test_interest_graph_token_filter(dataset):
    G = graphs.interest_graph(dataset, 100, k=2, min_sim=0.0, tokens="mention")
    assert G.number_of_edges() == 0


def test_hybrid_adds_follow_weight(dataset):
    G = graphs.hybrid_graph(dataset, 100, k=2, min_sim=0.2, tokens="both",
                            direction="any", follow_weight=0.3)
    assert G[1][2]["weight"] == pytest.approx(1.3)
    assert G[1][3]["weight"] == pytest.approx(0.3)


def test_select_egos_top_excludes_egos_without_circles(dataset):
    assert graphs.select_egos(dataset, "top2") == [200, 100]
    assert graphs.select_egos(dataset, "top5") == [200, 100]
    assert graphs.select_egos(dataset, "all") == [100, 200, 300]
    assert graphs.select_egos(dataset, "300,100") == [300, 100]


def test_select_egos_rejects_unknown(dataset):
    with pytest.raises(ValueError):
        graphs.select_egos(dataset, "999")
