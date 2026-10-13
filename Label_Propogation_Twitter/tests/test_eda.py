import pandas as pd
import pytest

from pipeline import eda


@pytest.fixture
def tables():
    users = pd.DataFrame({
        "user_id": [1, 2, 3, 4, 100],
        "ego_ids": [[100], [100], [100], [100], [100]],
        "n_hashtags": [2, 0, 1, 0, 0],
        "n_mentions": [0, 1, 1, 0, 0],
    })
    interests = pd.DataFrame({
        "user_id": [1, 1, 2, 3, 3],
        "token": ["#a", "#b", "@x", "#a", "@x"],
        "type": ["hashtag", "hashtag", "mention", "hashtag", "mention"],
    })
    # 1<->2 mutual, 1->3 one-way, 100->1, 100->2: 5 edges, 2 of them in a mutual pair
    follows = pd.DataFrame({"src": [1, 2, 1, 100, 100], "dst": [2, 1, 3, 1, 2]})
    egos = pd.DataFrame({"ego_id": [100, 200], "n_nodes": [5, 3], "n_edges": [5, 0],
                         "n_circles": [2, 0], "has_circles": [True, False]})
    circles = {"100": {"0": [1, 2, 3], "1": [3, 4]}, "200": {}}
    return eda.Tables(users, interests, follows, egos, circles)


def test_reciprocity(tables):
    assert eda.reciprocity(tables.follows) == pytest.approx(2 / 5)


def test_degree_counts(tables):
    deg = eda.degrees(tables.follows)
    assert deg.loc[1, "in_degree"] == 2
    assert deg.loc[1, "out_degree"] == 2
    assert deg.loc[100, "in_degree"] == 0


def test_circle_stats(tables):
    stats = eda.circle_stats(tables.circles, tables.egos)
    assert stats["n_circles"] == 2
    assert stats["egos_with_circles"] == 1
    # user 3 is in both circles: 1 of 4 circle members overlap
    assert stats["overlap_rate"] == pytest.approx(1 / 4)
    # 4 of ego 100's 5 nodes are in a circle
    assert stats["circle_coverage"] == pytest.approx(4 / 5)


def test_interest_stats(tables):
    stats = eda.interest_stats(tables.users, tables.interests)
    assert stats["share_with_hashtag"] == pytest.approx(2 / 5)
    assert stats["share_with_mention"] == pytest.approx(2 / 5)
    assert stats["share_with_no_tokens"] == pytest.approx(2 / 5)
    assert stats["n_hashtags"] == 2
    assert stats["n_mentions"] == 1


def test_ego_density(tables):
    dens = eda.ego_density(tables.egos)
    assert dens.loc[dens.ego_id == 100, "density"].item() == pytest.approx(5 / 20)
    assert dens.loc[dens.ego_id == 200, "density"].item() == 0


def test_token_filter_stats(tmp_path):
    twitter = tmp_path / "twitter"
    twitter.mkdir()
    (twitter / "1.featnames").write_text("0 #Python\n1 #python\n2 #...\n3 @nasa\n")
    (twitter / "2.featnames").write_text("0 #python\n1 #1\n")
    stats = eda.token_filter_stats(twitter)
    assert stats == {"raw_tokens": 5, "after_lowercase": 4, "after_cleaning": 2}
