import pandas as pd
import pytest

from backend import graphs


@pytest.fixture
def tiny_dataset() -> graphs.Dataset:
    """Ego 100: two follow triangles {1,2,3} and {4,5,6} with shared interests and two circles.
    Ego 200: users 7 and 200, one follow edge (not mutual), no circles."""
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
