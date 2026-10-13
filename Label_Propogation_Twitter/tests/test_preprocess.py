from pathlib import Path

import pytest

from pipeline import preprocess


def _write_ego(twitter_dir: Path, ego_id: int, featnames: list[str], feat: dict[int, list[int]],
               egofeat: list[int], edges: list[tuple[int, int]], circles: str) -> None:
    (twitter_dir / f"{ego_id}.featnames").write_text(
        "".join(f"{i} {name}\n" for i, name in enumerate(featnames)))
    (twitter_dir / f"{ego_id}.feat").write_text(
        "".join(f"{node} {' '.join(map(str, row))}\n" for node, row in feat.items()))
    (twitter_dir / f"{ego_id}.egofeat").write_text(" ".join(map(str, egofeat)) + "\n")
    (twitter_dir / f"{ego_id}.edges").write_text("".join(f"{a} {b}\n" for a, b in edges))
    (twitter_dir / f"{ego_id}.circles").write_text(circles)


@pytest.fixture
def data(tmp_path):
    twitter_dir = tmp_path / "twitter"
    twitter_dir.mkdir()
    # Ego 100: user 1 has "#Python" (index 0 here), user 2 has "@nasa" and junk "#...".
    _write_ego(
        twitter_dir, 100,
        featnames=["#Python", "@nasa", "#...", "#1"],
        feat={1: [1, 0, 1, 1], 2: [0, 1, 1, 0], 3: [0, 0, 0, 0]},
        egofeat=[1, 1, 0, 0],
        edges=[(1, 2), (2, 1)],
        circles="0\t1\t2\n1\t3\n",
    )
    # Ego 200: same hashtag lowercased at a different index; user 1 shared with ego 100.
    _write_ego(
        twitter_dir, 200,
        featnames=["@nasa", "#python", "#rare"],
        feat={1: [0, 1, 0], 4: [1, 1, 1]},
        egofeat=[0, 0, 0],
        edges=[(1, 2), (4, 1)],
        circles="",
    )
    return preprocess.build(twitter_dir, min_token_users=2)


@pytest.mark.parametrize("raw, expected", [
    ("#Python", "#python"),
    ("@NASA", "@nasa"),
    ("#...", None),
    ("#1", None),
    ("#2020", None),
    ("#", None),
    ("python", None),
    ("#café", "#café"),
    # Twitter tags/handles are runs of word characters: trailing punctuation is cut
    ("@mashable:", "@mashable"),
    ("@Foursquare!", "@foursquare"),
    ("@cindyli,", "@cindyli"),
    ("#stdmonth12.", "#stdmonth12"),
    ("@gmail.com", "@gmail"),
    ("#c++", "#c"),
    ("@user_name", "@user_name"),
    ("#_", None),
    ("#مصر", "#مصر"),
    ("#تويتر‌بالعربي", "#تويتر‌بالعربي"),  # zero-width non-joiner is part of the word
])
def test_normalize_token(raw, expected):
    assert preprocess.normalize_token(raw) == expected


def test_same_hashtag_from_two_egos_merges_into_one_token(data):
    interests = data["interests"]
    python_users = set(interests.loc[interests.token == "#python", "user_id"])
    assert python_users == {1, 4, 100}
    assert "#Python" not in set(interests.token)


def test_trailing_punctuation_variants_merge(tmp_path):
    twitter_dir = tmp_path / "twitter"
    twitter_dir.mkdir()
    _write_ego(twitter_dir, 100, featnames=["@mashable", "@mashable:"],
               feat={1: [1, 0], 2: [0, 1]}, egofeat=[0, 0], edges=[(1, 2)], circles="")
    interests = preprocess.build(twitter_dir, min_token_users=2)["interests"]
    assert set(interests.token) == {"@mashable"}
    assert set(interests.user_id) == {1, 2}


def test_junk_and_rare_tokens_are_dropped(data):
    tokens = set(data["interests"].token)
    assert tokens == {"#python", "@nasa"}


def test_interest_types(data):
    types = dict(zip(data["interests"].token, data["interests"].type))
    assert types == {"#python": "hashtag", "@nasa": "mention"}


def test_ego_edges_added_for_every_node_in_edges_file(data):
    by_ego = data["follows_by_ego"]
    ego100 = set(zip(by_ego.loc[by_ego.ego_id == 100, "src"], by_ego.loc[by_ego.ego_id == 100, "dst"]))
    assert {(100, 1), (100, 2)} <= ego100
    # user 3 appears only in .feat, not .edges, so the ego is not assumed to follow them
    assert (100, 3) not in ego100


def test_follows_have_no_duplicate_pairs(data):
    follows = data["follows"]
    assert not follows.duplicated(["src", "dst"]).any()
    # (1, 2) is in both egos' .edges but appears once globally
    assert ((follows.src == 1) & (follows.dst == 2)).sum() == 1
    assert len(data["follows_by_ego"].query("src == 1 and dst == 2")) == 2


def test_users_are_deduplicated_and_include_feature_only_users_and_egos(data):
    users = data["users"].set_index("user_id")
    assert set(users.index) == {1, 2, 3, 4, 100, 200}
    assert sorted(users.loc[1, "ego_ids"]) == [100, 200]
    assert list(users.loc[100, "ego_ids"]) == [100]
    assert users.loc[1, "n_hashtags"] == 1
    assert users.loc[4, "n_mentions"] == 1


def test_empty_circles_marked(data):
    egos = data["egos"].set_index("ego_id")
    assert bool(egos.loc[100, "has_circles"]) is True
    assert bool(egos.loc[200, "has_circles"]) is False
    assert egos.loc[100, "n_circles"] == 2
    assert egos.loc[200, "n_circles"] == 0
    # ego 100: ego + users 1, 2, 3
    assert egos.loc[100, "n_nodes"] == 4
    # 2 edges from file + 2 ego edges
    assert egos.loc[100, "n_edges"] == 4


def test_circles_json_structure(data):
    assert data["circles"] == {"100": {"0": [1, 2], "1": [3]}, "200": {}}
