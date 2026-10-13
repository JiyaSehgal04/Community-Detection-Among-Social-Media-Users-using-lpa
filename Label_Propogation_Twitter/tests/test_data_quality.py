import pandas as pd

from backend import graphs
from pipeline import data_quality


def _dataset() -> graphs.Dataset:
    users = pd.DataFrame({"user_id": [1, 2, 3, 4, 5], "ego_ids": [[100]] * 5,
                          "n_hashtags": [0] * 5, "n_mentions": [0] * 5})
    interests = pd.DataFrame({"user_id": [1, 2], "token": ["#a", "#b"], "type": ["hashtag"] * 2})
    # 1->3 and 3->1 are real follows; 5->5 is a self-loop and must not count
    follows_by_ego = pd.DataFrame({"src": [1, 3, 5], "dst": [3, 1, 5], "ego_id": [100] * 3})
    egos = pd.DataFrame({"ego_id": [100], "n_nodes": [5], "n_edges": [3],
                         "n_circles": [0], "has_circles": [False]})
    return graphs.Dataset(users, interests, follows_by_ego, egos, circles={})


def test_buckets_put_every_user_in_exactly_one_bucket():
    buckets = data_quality.bucket_users(_dataset())
    assert buckets == {"tokens_and_follows": 1, "tokens_only": 1, "follows_only": 1,
                       "neither": 2, "total": 5}
    parts = ("tokens_and_follows", "tokens_only", "follows_only", "neither")
    assert sum(buckets[p] for p in parts) == buckets["total"]
