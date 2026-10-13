# Full-dataset run, outliers, tuning and held-out evaluation: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the pipeline on all 973 egos, add three optional outlier rules, tune hybrid LPA by random search on a stratified ego split, and evaluate the winner once on held-out test egos.

**Architecture:** A seeded split (`pipeline/split.py`) fences the test egos off. Outlier rules live in `backend/outliers.py` and enter graph building through one optional argument. `pipeline/tune.py` random-searches configs on train/val egos with a constrained objective; `pipeline/evaluate.py` scores the winner once on test egos against the defaults and Louvain.

**Tech Stack:** Python 3.14, pandas 3, numpy 2.5, networkx 3.7, scikit-learn, pytest. Venv at `../.venv`.

**Spec:** `docs/superpowers/specs/2026-10-08-full-dataset-tuning-design.md`

All commands run from `/Users/jiyasehgal/Developer/Label_Propogation_Algorithm/Label_Propogation_Twitter`. Shorthand: `PY=../.venv/bin/python`.

## Global Constraints

- `lpa.py`, `metrics.py` and everything in `data/processed/` (except the new `split.json`) are not modified.
- `build_graph(..., outliers=None)` and `OutlierConfig()` must give results identical to today; the 66 existing tests must keep passing after every task.
- Split: 938 egos with circles, 60/20/20, seed 42, stratified by `n_nodes` quartile.
- Tuning: mode `hybrid`, algorithm `lpa`, `runs=10`, `seed=42`; 60 trials; 100-ego train sample; top 5 confirmed on all val egos.
- Objective: mean `interest_purity` + mean `follow_modularity`; feasible = mean `largest_share <= 0.5` and mean `coverage >= 0.8`; ranking key `(feasible, round(objective, 3), circle_f1)` descending, NaN as -inf.
- Purity is always scored with `user_tokens("both")`; follow modularity always on the unscaled `direction="any"` follow graph.
- Tests: only the 7 agreed tests, written first and seen failing. No extra tests.
- Commit messages: `<type>: <description>` (conventional). **Do not commit unless the user has asked**; each task ends with a checkpoint, and the commit step runs only on request.

## Review Focus

Each line is pinned by an assertion inside one of the 7 agreed tests, or (marked "smoke") by a real-data run step, because the user asked for necessary tests only.

1. Every token filtered out, or no interests at all, gives an empty TF-IDF and an interest graph with no edges, not an exception. (Task 3, test 2)
2. An ego with no follow edges, or fewer than 2 users with tokens, gives singleton communities and counted NaN metrics, not a crash, during tuning and evaluation. (smoke: Tasks 5 and 6)
3. A `hub_cap` at or above the maximum degree is an exact no-op; an edge with two hub ends gets both factors multiplied. (Task 3, test 3)
4. Near-equal objectives tie after rounding and fall to `circle_f1`; a NaN objective ranks last. (Task 5, test 5)
5. Evaluating twice must not silently overwrite results: `check_guards` raises without `--force`. (Task 6, test 6b)

## File structure

| File | Responsibility |
|---|---|
| `pipeline/split.py` (new) | `make_split`, `save_split`, `load_split`, `assert_no_test_egos`, CLI |
| `pipeline/data_quality.py` (new) | `bucket_users`, CLI writing `outputs/data_quality.json` |
| `backend/outliers.py` (new) | `OutlierConfig`, `filter_tokens`, `hub_factors` |
| `backend/graphs.py` (modify) | thread `outliers` through `tfidf`, `follow_graph`, `interest_graph`, `hybrid_graph`, `build_graph` |
| `pipeline/experiments.py` (modify) | extract `score_partition` |
| `pipeline/tune.py` (new) | grids, sampling, scoring, ranking, `run_tuning`, CLI |
| `pipeline/evaluate.py` (new) | `check_guards`, `evaluate`, CLI |
| `tests/test_split.py`, `test_data_quality.py`, `test_outliers.py`, `test_tune.py`, `test_evaluate.py` (new) | the 7 agreed tests |

---

### Task 1: Ego split

**Files:**
- Create: `pipeline/split.py`
- Test: `tests/test_split.py`

**Interfaces:**
- Produces: `make_split(egos: pd.DataFrame, seed: int = 42, fractions: tuple = (0.6, 0.2, 0.2)) -> dict[str, list[int]]` with keys `train`, `val`, `test`; `save_split(split, path, seed)`; `load_split(path) -> dict[str, list[int]]`; `assert_no_test_egos(egos: Iterable[int], split) -> None` (raises `ValueError` naming the leaked ids).

- [ ] **Step 1: Write the failing test (agreed test 1)**

```python
# tests/test_split.py
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
```

- [ ] **Step 2: Run it to see it fail**

Run: `$PY -m pytest tests/test_split.py -v`
Expected: FAIL with `ImportError` / `cannot import name 'split'`.

- [ ] **Step 3: Implement `pipeline/split.py`**

```python
"""Seeded, size-stratified train/val/test split of the egos that have circles.

Run from the project root, after pipeline.preprocess:

    python -m pipeline.split

Writes data/processed/split.json. Tuning sees only train and val egos; the test egos are
scored once, by pipeline.evaluate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SPLIT_NAMES = ("train", "val", "test")


def make_split(egos: pd.DataFrame, seed: int = 42,
               fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)) -> dict[str, list[int]]:
    """Split egos with circles into train/val/test, stratified by n_nodes quartile."""
    scored = egos[egos.has_circles].sort_values("ego_id").reset_index(drop=True)
    # Rank first so tied sizes cannot produce duplicate bin edges.
    strata = pd.qcut(scored.n_nodes.rank(method="first"), 4, labels=False)
    rng = np.random.default_rng(seed)
    out: dict[str, list[int]] = {name: [] for name in SPLIT_NAMES}
    for stratum in sorted(strata.unique()):
        ids = scored.ego_id[strata == stratum].astype(int).tolist()
        rng.shuffle(ids)
        n_train = round(fractions[0] * len(ids))
        n_val = round(fractions[1] * len(ids))
        out["train"] += ids[:n_train]
        out["val"] += ids[n_train:n_train + n_val]
        out["test"] += ids[n_train + n_val:]
    return {name: sorted(ids) for name, ids in out.items()}


def save_split(split: dict[str, list[int]], path: Path, seed: int) -> None:
    payload = {"seed": seed, "counts": {k: len(v) for k, v in split.items()}, **split}
    Path(path).write_text(json.dumps(payload, indent=2))


def load_split(path: Path | str) -> dict[str, list[int]]:
    raw = json.loads(Path(path).read_text())
    return {name: [int(e) for e in raw[name]] for name in SPLIT_NAMES}


def assert_no_test_egos(egos: Iterable[int], split: dict[str, list[int]]) -> None:
    """Raise if any ego in `egos` belongs to the held-out test set."""
    leaked = sorted(set(int(e) for e in egos) & set(split["test"]))
    if leaked:
        raise ValueError(f"test egos must not be used for tuning: {leaked[:10]}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    egos = pd.read_parquet(args.processed_dir / "egos.parquet")
    split = make_split(egos, seed=args.seed)
    total = sum(len(v) for v in split.values())
    if total != int(egos.has_circles.sum()):
        print(f"split covers {total} egos, expected {int(egos.has_circles.sum())}", file=sys.stderr)
        return 1
    save_split(split, args.processed_dir / "split.json", args.seed)
    print({k: len(v) for k, v in split.items()}, "-> split.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run it to see it pass**

Run: `$PY -m pytest tests/test_split.py -v`
Expected: PASS.

- [ ] **Step 5: Create the real split and check it**

Run: `$PY -m pipeline.split`
Expected: prints `{'train': 563, 'val': 188, 'test': 187}` (within 1 of each, total 938) and `data/processed/split.json` exists. Run it twice; the file content must be identical.

- [ ] **Step 6: Checkpoint**

Run: `$PY -m pytest -q` (expect 67 passed). Commit only if the user has asked: `git add pipeline/split.py tests/test_split.py && git commit -m "feat: add stratified ego split"`.

---

### Task 2: Data-quality audit and full run

**Files:**
- Create: `pipeline/data_quality.py`
- Test: `tests/test_data_quality.py`

**Interfaces:**
- Consumes: `graphs.Dataset` (fields `users`, `interests`, `follows_by_ego`).
- Produces: `bucket_users(data: graphs.Dataset) -> dict[str, int]` with keys `tokens_and_follows`, `tokens_only`, `follows_only`, `neither`, `total`.

- [ ] **Step 1: Write the failing test (agreed test 7)**

```python
# tests/test_data_quality.py
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
```

- [ ] **Step 2: Run it to see it fail**

Run: `$PY -m pytest tests/test_data_quality.py -v`
Expected: FAIL with `cannot import name 'data_quality'`.

- [ ] **Step 3: Implement `pipeline/data_quality.py`**

```python
"""Audit which users can be placed by follows, by interests, by both, or by neither.

    python -m pipeline.data_quality

Writes outputs/data_quality.json. Users in the "neither" bucket stay singletons; nothing is imputed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backend import graphs

DEFINITIONS = {
    "has_follows": "appears as src or dst of a non-self-loop row in follows_by_ego",
    "has_tokens": "has at least one row in interests",
}


def bucket_users(data: graphs.Dataset) -> dict[str, int]:
    edges = data.follows_by_ego
    edges = edges[edges.src != edges.dst]
    ids = data.users.user_id
    has_follows = ids.isin(set(edges.src) | set(edges.dst))
    has_tokens = ids.isin(set(data.interests.user_id))
    return {
        "tokens_and_follows": int((has_tokens & has_follows).sum()),
        "tokens_only": int((has_tokens & ~has_follows).sum()),
        "follows_only": int((~has_tokens & has_follows).sum()),
        "neither": int((~has_tokens & ~has_follows).sum()),
        "total": len(ids),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--out", type=Path, default=Path("outputs/data_quality.json"))
    args = p.parse_args(argv)

    buckets = bucket_users(graphs.Dataset.load(args.processed_dir))
    parts = ("tokens_and_follows", "tokens_only", "follows_only", "neither")
    if sum(buckets[b] for b in parts) != buckets["total"]:
        print("buckets do not sum to the number of users", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({**buckets, "definitions": DEFINITIONS}, indent=2))
    print(buckets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run it to see it pass**

Run: `$PY -m pytest tests/test_data_quality.py -v`
Expected: PASS.

- [ ] **Step 5: Run the audit on the real data**

Run: `$PY -m pipeline.data_quality`
Expected: `total` is 81306; `neither` is a small number (users with neither tokens nor follows); exit code 0.

- [ ] **Step 6: Run all 973 egos (no code change)**

Run: `$PY -m pipeline.experiments --egos all --runs 10 --seed 42 --out-dir outputs/all --no-figures`
Expected in about 5-15 minutes. Then verify:

```bash
$PY - <<'E'
import pandas as pd
m = pd.read_csv("outputs/all/metrics.csv")
print("egos:", m.ego_id.nunique())                                  # expect 973
print("egos with NaN circle_f1:", m[m.circle_f1.isna()].ego_id.nunique())  # expect 35
E
```

This uses the default `--direction any`, as the spec says; the earlier top20 outputs are untouched because the output goes to `outputs/all/`. (The EDA found `mutual` avoids the giant-community collapse; a second run with `--direction mutual --out-dir outputs/all_mutual` is optional and is not part of this plan. Tuning searches both directions anyway.)

- [ ] **Step 7: Checkpoint**

Run: `$PY -m pytest -q` (expect 68 passed). Commit only if the user has asked: `git commit -m "feat: add data quality audit"` (add `pipeline/data_quality.py tests/test_data_quality.py`).

---

### Task 3: Outlier rules

**Files:**
- Create: `backend/outliers.py`
- Modify: `backend/graphs.py` (`tfidf`, `follow_graph`, `interest_graph`, `hybrid_graph`, `build_graph`)
- Test: `tests/test_outliers.py`

**Interfaces:**
- Produces: `OutlierConfig(token_max_share: float | None = None, hub_cap: int | None = None, max_tokens: int | None = None)` (frozen dataclass) with `.token_key -> tuple`; `filter_tokens(interests: pd.DataFrame, n_users: int, cfg: OutlierConfig) -> pd.DataFrame`; `hub_factors(G: nx.Graph, cap: int) -> dict[node, float]`.
- Produces: `Dataset.tfidf(tokens="both", outliers=None)`; `follow_graph/interest_graph/hybrid_graph/build_graph(..., outliers: OutlierConfig | None = None)` (keyword).

- [ ] **Step 1: Write the failing tests (agreed tests 2, 3 and 4)**

```python
# tests/test_outliers.py
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
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_outliers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.outliers'`.

- [ ] **Step 3: Implement `backend/outliers.py`**

```python
"""Optional outlier rules, applied when graphs are built. All off by default; nothing is
deleted from data/processed/."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import pandas as pd


@dataclass(frozen=True)
class OutlierConfig:
    token_max_share: float | None = None  # drop tokens used by more than this share of all users
    hub_cap: int | None = None            # scale follow edges of users whose degree exceeds this
    max_tokens: int | None = None         # keep each user's N rarest tokens

    @property
    def token_key(self) -> tuple:
        """The settings that change TF-IDF (hub_cap does not)."""
        return (self.token_max_share, self.max_tokens)


def filter_tokens(interests: pd.DataFrame, n_users: int, cfg: OutlierConfig) -> pd.DataFrame:
    """Apply token_max_share, then max_tokens, to an interests frame already filtered by type."""
    if cfg.token_key == (None, None) or interests.empty:
        return interests
    doc_freq = interests.groupby("token").user_id.nunique()
    out = interests
    if cfg.token_max_share is not None:
        too_common = doc_freq.index[doc_freq > cfg.token_max_share * n_users]
        out = out[~out.token.isin(too_common)]
    if cfg.max_tokens is not None:
        ranked = out.assign(_df=out.token.map(doc_freq)).sort_values(["user_id", "_df", "token"])
        out = ranked[ranked.groupby("user_id").cumcount() < cfg.max_tokens].drop(columns="_df")
    return out


def hub_factors(G: nx.Graph, cap: int) -> dict:
    """Scale factor cap/degree for each node whose degree exceeds the cap."""
    return {node: cap / degree for node, degree in G.degree() if degree > cap}
```

- [ ] **Step 4: Modify `backend/graphs.py`**

Add the import below the existing `from sklearn.preprocessing import normalize` line:

```python
from backend.outliers import OutlierConfig, filter_tokens, hub_factors
```

Replace the `tfidf` method (inside `Dataset`) with:

```python
    def tfidf(self, tokens: str = "both",
              outliers: OutlierConfig | None = None) -> tuple[dict[int, int], sparse.csr_matrix]:
        """Row index per user and an L2-normalised TF-IDF matrix (IDF over all users).

        Cached per (tokens, token outlier settings); hub_cap does not affect TF-IDF.
        """
        cfg = outliers or OutlierConfig()
        key = (tokens, cfg.token_key)
        if key not in self._tfidf:
            sub = self.interests[self.interests.type.isin(TOKEN_CHOICES[tokens])]
            sub = filter_tokens(sub, len(self.users), cfg)
            if sub.empty:
                self._tfidf[key] = ({}, sparse.csr_matrix((0, 0)))
                return self._tfidf[key]
            user_codes, user_ids = pd.factorize(sub.user_id)
            tok_codes, _ = pd.factorize(sub.token)
            counts = sparse.csr_matrix(
                (np.ones(len(sub)), (user_codes, tok_codes)),
                shape=(len(user_ids), int(tok_codes.max()) + 1))
            matrix = normalize(TfidfTransformer(norm=None).fit_transform(counts))
            self._tfidf[key] = ({int(u): i for i, u in enumerate(user_ids)}, matrix.tocsr())
        return self._tfidf[key]
```

Replace `follow_graph` with:

```python
def follow_graph(data: Dataset, ego: int, direction: str = "any",
                 outliers: OutlierConfig | None = None) -> nx.Graph:
    """Undirected follow graph. 'any': a follows b or b follows a. 'mutual': both.

    With outliers.hub_cap, an edge's weight is the product of cap/degree for each hub end.
    """
    G = nx.Graph()
    G.add_nodes_from(data.ego_nodes(ego))
    edges = data.ego_edges(ego)
    pairs = set(zip(edges.src.tolist(), edges.dst.tolist()))
    if direction == "mutual":
        pairs = {(a, b) for a, b in pairs if (b, a) in pairs}
    elif direction != "any":
        raise ValueError(f"direction must be 'any' or 'mutual', got {direction!r}")
    G.add_edges_from(pairs, weight=1.0)
    if outliers is not None and outliers.hub_cap is not None:
        factor = hub_factors(G, outliers.hub_cap)
        for a, b, d in G.edges(data=True):
            d["weight"] = factor.get(a, 1.0) * factor.get(b, 1.0)
    return G
```

Change `interest_graph`'s signature and TF-IDF call:

```python
def interest_graph(data: Dataset, ego: int, k: int = 10, min_sim: float = 0.2,
                   tokens: str = "both", outliers: OutlierConfig | None = None) -> nx.Graph:
```

and replace the line `index, matrix = data.tfidf(tokens)` with `index, matrix = data.tfidf(tokens, outliers)`.

Replace `hybrid_graph` and `build_graph` with:

```python
def hybrid_graph(data: Dataset, ego: int, k: int = 10, min_sim: float = 0.2, tokens: str = "both",
                 direction: str = "any", follow_weight: float = 0.3,
                 outliers: OutlierConfig | None = None) -> nx.Graph:
    """Interest graph plus follow edges; a follow edge adds follow_weight * its weight."""
    G = interest_graph(data, ego, k=k, min_sim=min_sim, tokens=tokens, outliers=outliers)
    for a, b, d in follow_graph(data, ego, direction=direction, outliers=outliers).edges(data=True):
        w = follow_weight * d["weight"]
        if G.has_edge(a, b):
            G[a][b]["weight"] += w
        else:
            G.add_edge(a, b, weight=w)
    return G


def build_graph(data: Dataset, ego: int, mode: str, *, direction: str = "any", k: int = 10,
                min_sim: float = 0.2, tokens: str = "both", follow_weight: float = 0.3,
                outliers: OutlierConfig | None = None) -> nx.Graph:
    if mode == "follow":
        return follow_graph(data, ego, direction=direction, outliers=outliers)
    if mode == "interest":
        return interest_graph(data, ego, k=k, min_sim=min_sim, tokens=tokens, outliers=outliers)
    if mode == "hybrid":
        return hybrid_graph(data, ego, k=k, min_sim=min_sim, tokens=tokens, direction=direction,
                            follow_weight=follow_weight, outliers=outliers)
    raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
```

- [ ] **Step 5: Run the new tests and the whole suite**

Run: `$PY -m pytest tests/test_outliers.py -v` then `$PY -m pytest -q`
Expected: new tests PASS; full suite 71 passed (66 + 5 so far: split 1, data quality 1, outliers 3). In particular `tests/test_graphs.py` and `tests/test_experiments.py` still pass, which proves `outliers=None` changed nothing.

- [ ] **Step 6: Checkpoint**

Commit only if the user has asked: `git add backend/outliers.py backend/graphs.py tests/test_outliers.py && git commit -m "feat: add optional outlier rules to graph building"`.

---

### Task 4: Share the scoring code

**Files:**
- Modify: `pipeline/experiments.py:99-109` (the `rows.append({...})` block in `run`)

**Interfaces:**
- Produces: `score_partition(G, comms, follow_G, circles, tokens, f1_baseline, seconds) -> dict` returning the metric columns `n_nodes, n_edges, modularity, follow_modularity, circle_f1, circle_f1_one_community, interest_purity, n_communities, n_singletons, coverage, largest_share, runtime_s` in that order.

No new test: this is a pure extraction, and `tests/test_experiments.py` already pins its output.

- [ ] **Step 1: Add the function** (above `run`, after `_run_rows`)

```python
def score_partition(G: nx.Graph, comms: list[set], follow_G: nx.Graph, circles: dict, tokens: dict,
                    f1_baseline: float, seconds: float) -> dict:
    """Metric columns for one partition of G (shared by experiments, tune and evaluate)."""
    return {
        "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
        "modularity": m.safe_modularity(G, comms),
        "follow_modularity": m.safe_modularity(follow_G, comms),
        "circle_f1": m.circle_f1(comms, circles),
        "circle_f1_one_community": f1_baseline,
        "interest_purity": m.interest_purity(comms, tokens),
        **m.community_stats(comms),
        "runtime_s": seconds,
    }
```

- [ ] **Step 2: Use it in `run`**: replace the whole `rows.append({...})` block with

```python
                rows.append({"ego_id": ego, "mode": mode, "algorithm": algorithm,
                             **score_partition(G, comms, follow_G, circles, tokens,
                                               f1_baseline, seconds)})
```

- [ ] **Step 3: Verify nothing changed**

Run: `$PY -m pytest -q`
Expected: 71 passed. Also run `$PY -m pipeline.experiments --egos top3 --runs 2 --no-figures --out-dir /tmp/lpa_check` and confirm `/tmp/lpa_check/metrics.csv` has the same column order as `outputs/metrics.csv` (`head -1` on both).

- [ ] **Step 4: Checkpoint**

Commit only if the user has asked: `git commit -m "refactor: extract score_partition from experiments"`.

---

### Task 5: Tuning

**Files:**
- Create: `pipeline/tune.py`
- Test: `tests/test_tune.py`

**Interfaces:**
- Consumes: `split.assert_no_test_egos`, `split.load_split`, `graphs.build_graph(..., outliers=)`, `experiments.score_partition`, `lpa.best_of_runs`, `outliers.OutlierConfig`.
- Produces: `DEFAULT_CONFIG: dict`; `CONFIG_KEYS: tuple`; `sample_configs(n, seed) -> list[dict]` (trial 0 is `DEFAULT_CONFIG`); `outlier_config(cfg) -> OutlierConfig`; `score_ego(data, cfg, ego, tokens_both, algorithm="lpa") -> dict`; `score_config(data, cfg, egos, tokens_both) -> dict`; `rank_trials(df) -> pd.DataFrame`; `run_tuning(data, split, train_egos, val_egos, n_trials=60, top_n=5, seed=42) -> tuple[pd.DataFrame, dict]` (all trials, best config); `main(argv) -> int`.

- [ ] **Step 1: Write the failing tests (agreed tests 5 and 6a)**

```python
# tests/test_tune.py
import pandas as pd
import pytest

from pipeline import tune


def test_infeasible_config_ranks_below_any_feasible_one():
    trials = pd.DataFrame({
        "trial": [0, 1, 2, 3],
        "objective": [1.50, 0.40, 0.5001, 0.5004],
        "feasible": [False, True, True, True],
        "circle_f1": [0.90, 0.10, 0.20, 0.60],
    })
    ranked = tune.rank_trials(trials)
    assert ranked.trial.tolist()[-1] == 0                 # infeasible last despite the best raw score
    assert ranked.trial.tolist()[:2] == [3, 2]            # 0.5001 and 0.5004 tie at 3 decimals; higher F1 wins
    nan_obj = pd.DataFrame({"trial": [0, 1], "objective": [float("nan"), 0.1],
                            "feasible": [True, True], "circle_f1": [0.5, 0.5]})
    assert tune.rank_trials(nan_obj).trial.tolist() == [1, 0]  # NaN objective ranks last


def test_tuning_refuses_test_egos():
    split = {"train": [1, 2], "val": [3], "test": [99]}
    with pytest.raises(ValueError, match="test egos"):
        tune.run_tuning(None, split, train_egos=[1, 99], val_egos=[3])  # raises before touching data
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_tune.py -v`
Expected: FAIL with `cannot import name 'tune'`.

- [ ] **Step 3: Implement `pipeline/tune.py`**

```python
"""Random-search tuning of hybrid LPA on train/val egos.

    python -m pipeline.tune

Needs data/processed/split.json (python -m pipeline.split). Test egos are never used here.
Writes outputs/tuning/trials.csv and outputs/tuning/best_config.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import subprocess
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

from backend import graphs, lpa, metrics as m
from backend.outliers import OutlierConfig
from pipeline import experiments, split as split_mod

GRID = {
    "k": [5, 10, 15, 20, 30],
    "min_sim": [0.1, 0.15, 0.2, 0.3, 0.4],
    "follow_weight": [0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
    "direction": ["any", "mutual"],
    "tokens": ["hashtag", "mention", "both"],
    "token_max_share": [None, 0.01, 0.02, 0.05],
    "hub_cap": [None, 10, 25, 50],
    "max_tokens": [None, 10, 20, 50],
}
CONFIG_KEYS = tuple(GRID)
DEFAULT_CONFIG = {"k": 10, "min_sim": 0.2, "follow_weight": 0.3, "direction": "any", "tokens": "both",
                  "token_max_share": None, "hub_cap": None, "max_tokens": None}
RUNS, SEED = 10, 42
MAX_LARGEST_SHARE, MIN_COVERAGE = 0.5, 0.8
AVERAGED = ("interest_purity", "follow_modularity", "circle_f1", "largest_share", "coverage")


def sample_configs(n: int = 60, seed: int = 42) -> list[dict]:
    """Trial 0 is the default config; the rest are distinct random draws from GRID."""
    rng = random.Random(seed)
    configs, seen = [dict(DEFAULT_CONFIG)], {tuple(DEFAULT_CONFIG.values())}
    while len(configs) < n:
        cfg = {key: rng.choice(values) for key, values in GRID.items()}
        if tuple(cfg.values()) not in seen:
            seen.add(tuple(cfg.values()))
            configs.append(cfg)
    return configs


def outlier_config(cfg: dict) -> OutlierConfig:
    return OutlierConfig(cfg["token_max_share"], cfg["hub_cap"], cfg["max_tokens"])


def communities_for(G: nx.Graph, algorithm: str = "lpa") -> list[set]:
    if G.number_of_edges() == 0:
        return [{n} for n in G.nodes]  # nothing to propagate along: everyone stays alone
    if algorithm == "lpa":
        return lpa.best_of_runs(G, runs=RUNS, seed=SEED)[0].communities
    if algorithm == "louvain":
        return [set(c) for c in nx.community.louvain_communities(G, weight="weight", seed=SEED)]
    raise ValueError(algorithm)


def score_ego(data: graphs.Dataset, cfg: dict, ego: int, tokens_both: dict,
              algorithm: str = "lpa") -> dict:
    """Metrics for one ego. Purity uses fixed unfiltered tokens, follow modularity the unscaled graph."""
    G = graphs.build_graph(data, ego, "hybrid", direction=cfg["direction"], k=cfg["k"],
                           min_sim=cfg["min_sim"], tokens=cfg["tokens"],
                           follow_weight=cfg["follow_weight"], outliers=outlier_config(cfg))
    follow_G = graphs.follow_graph(data, ego, direction="any")
    circles = data.ego_circles(ego)
    baseline = m.circle_f1([set(follow_G.nodes)], circles)
    return experiments.score_partition(G, communities_for(G, algorithm), follow_G, circles,
                                       tokens_both, baseline, 0.0)


def score_config(data: graphs.Dataset, cfg: dict, egos: list[int], tokens_both: dict) -> dict:
    df = pd.DataFrame([score_ego(data, cfg, e, tokens_both) for e in egos])
    out = {c: float(df[c].mean()) for c in AVERAGED}  # mean skips NaN
    out["nan_purity"] = int(df.interest_purity.isna().sum())
    out["nan_follow_modularity"] = int(df.follow_modularity.isna().sum())
    out["objective"] = out["interest_purity"] + out["follow_modularity"]
    out["feasible"] = bool(out["largest_share"] <= MAX_LARGEST_SHARE and out["coverage"] >= MIN_COVERAGE)
    return out


def rank_trials(df: pd.DataFrame) -> pd.DataFrame:
    """Feasible first, then objective rounded to 3 decimals, then circle F1; NaN ranks last."""
    keyed = df.assign(_obj=df.objective.round(3).fillna(float("-inf")),
                      _f1=df.circle_f1.fillna(float("-inf")))
    return (keyed.sort_values(["feasible", "_obj", "_f1"], ascending=False, kind="stable")
            .drop(columns=["_obj", "_f1"]).reset_index(drop=True))


def _score_grouped(data, configs: list[tuple[int, dict]], egos, tokens_both, stage: str) -> pd.DataFrame:
    """Score trials grouped by TF-IDF settings and clear the cache between groups (memory)."""
    key = lambda item: (str(item[1]["token_max_share"]), str(item[1]["max_tokens"]))
    rows = []
    for _, group in itertools.groupby(sorted(configs, key=key), key=key):
        data._tfidf.clear()
        for trial, cfg in group:
            rows.append({"stage": stage, "trial": trial, **cfg,
                         **score_config(data, cfg, egos, tokens_both)})
    return rank_trials(pd.DataFrame(rows))


def run_tuning(data: graphs.Dataset | None, split: dict[str, list[int]], train_egos: list[int],
               val_egos: list[int], n_trials: int = 60, top_n: int = 5,
               seed: int = 42) -> tuple[pd.DataFrame, dict]:
    split_mod.assert_no_test_egos([*train_egos, *val_egos], split)  # the leak guard
    tokens_both = data.user_tokens("both")
    configs = sample_configs(n_trials, seed)
    stage1 = _score_grouped(data, list(enumerate(configs)), train_egos, tokens_both, "train")
    finalists = [(int(t), configs[int(t)]) for t in stage1.trial.head(top_n)]
    stage2 = _score_grouped(data, finalists, val_egos, tokens_both, "val")
    return pd.concat([stage1, stage2], ignore_index=True), configs[int(stage2.trial.iloc[0])]


def _git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/tuning"))
    p.add_argument("--n-trials", type=int, default=60)
    p.add_argument("--sample-size", type=int, default=100)
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    data = graphs.Dataset.load(args.processed_dir)
    split = split_mod.load_split(args.processed_dir / "split.json")
    train = sorted(random.Random(args.seed).sample(split["train"], min(args.sample_size, len(split["train"]))))
    trials, best = run_tuning(data, split, train, split["val"], args.n_trials, args.top_n, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trials.to_csv(args.out_dir / "trials.csv", index=False)
    val_best = trials[trials.stage == "val"].iloc[0]
    (args.out_dir / "best_config.json").write_text(json.dumps({
        "config": best,
        "val_metrics": {c: val_best[c] for c in (*AVERAGED, "objective", "feasible")},
        "tuned_on": sorted({*train, *split["val"]}),
        "seed": args.seed, "n_trials": args.n_trials, "git_hash": _git_hash(),
    }, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print("best:", best)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `$PY -m pytest tests/test_tune.py -v` then `$PY -m pytest -q`
Expected: 2 new tests PASS; full suite 73 passed.

- [ ] **Step 5: Smoke run on real data (covers Review Focus item 2)**

Run: `$PY -m pipeline.tune --n-trials 4 --sample-size 6 --top-n 2 --out-dir /tmp/lpa_tune_smoke`
Expected: finishes in under a minute; `trials.csv` has 4 train rows and 2 val rows; `best_config.json` lists `tuned_on` and a `config` with all 8 keys; no exception from egos with sparse graphs.

- [ ] **Step 6: Real tuning run**

Run: `$PY -m pipeline.tune`
Expected in about 10-15 minutes; `outputs/tuning/trials.csv` has 60 train rows and 5 val rows. Open `best_config.json` and check that `val_metrics.feasible` is `true`. If it is `false`, report that to the user instead of proceeding to Task 6.

- [ ] **Step 7: Checkpoint**

Commit only if the user has asked: `git add pipeline/tune.py tests/test_tune.py && git commit -m "feat: add random-search tuning"`.

---

### Task 6: Held-out evaluation

**Files:**
- Create: `pipeline/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `tune.score_ego`, `tune.DEFAULT_CONFIG`, `split.load_split`, `experiments.METRIC_COLS`.
- Produces: `check_guards(best: dict, split: dict, egos: list[int], out_dir: Path, force: bool = False) -> None` (raises `ValueError` for leakage or wrong egos, `FileExistsError` for existing results); `evaluate(data, split, best, out_dir, force=False) -> pd.DataFrame`; `main(argv) -> int`.

- [ ] **Step 1: Write the failing test (agreed test 6b)**

```python
# tests/test_evaluate.py
import pytest

from pipeline import evaluate

SPLIT = {"train": [1, 2], "val": [3], "test": [8, 9]}


def test_guards_block_leakage_wrong_egos_and_overwrites(tmp_path):
    ok = {"tuned_on": [1, 2, 3]}
    evaluate.check_guards(ok, SPLIT, [8, 9], tmp_path)  # passes silently

    with pytest.raises(ValueError, match="tuned on test egos"):
        evaluate.check_guards({"tuned_on": [1, 8]}, SPLIT, [8, 9], tmp_path)
    with pytest.raises(ValueError, match="exactly the test egos"):
        evaluate.check_guards(ok, SPLIT, [8], tmp_path)

    (tmp_path / "metrics.csv").write_text("x")
    with pytest.raises(FileExistsError):
        evaluate.check_guards(ok, SPLIT, [8, 9], tmp_path)
    evaluate.check_guards(ok, SPLIT, [8, 9], tmp_path, force=True)
```

- [ ] **Step 2: Run it to see it fail**

Run: `$PY -m pytest tests/test_evaluate.py -v`
Expected: FAIL with `cannot import name 'evaluate'`.

- [ ] **Step 3: Implement `pipeline/evaluate.py`**

```python
"""Score the tuned config once on the held-out test egos.

    python -m pipeline.evaluate

Compares: the tuned config, the default config, and Louvain on the tuned graph. Refuses to run
if the tuning data touched a test ego, and refuses to overwrite earlier results without --force.
Writes outputs/final/metrics.csv and outputs/final/summary.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from backend import graphs
from pipeline import experiments, split as split_mod, tune


def check_guards(best: dict, split: dict[str, list[int]], egos: list[int], out_dir: Path,
                 force: bool = False) -> None:
    if set(best["tuned_on"]) & set(split["test"]):
        raise ValueError("config was tuned on test egos; re-run tuning")
    if sorted(egos) != sorted(split["test"]):
        raise ValueError("evaluate exactly the test egos, no more and no fewer")
    if (Path(out_dir) / "metrics.csv").exists() and not force:
        raise FileExistsError(f"{out_dir}/metrics.csv exists; the test set is meant to be scored once "
                              "(use --force to overwrite)")


def evaluate(data: graphs.Dataset, split: dict[str, list[int]], best: dict, out_dir: Path,
             force: bool = False) -> pd.DataFrame:
    egos = split["test"]
    check_guards(best, split, egos, out_dir, force)
    tokens_both = data.user_tokens("both")
    variants = (("tuned", best["config"], "lpa"),
                ("default", tune.DEFAULT_CONFIG, "lpa"),
                ("louvain_on_tuned_graph", best["config"], "louvain"))
    rows = []
    for i, ego in enumerate(egos, 1):
        for name, cfg, algorithm in variants:
            rows.append({"ego_id": ego, "variant": name,
                         **tune.score_ego(data, cfg, ego, tokens_both, algorithm)})
        if i % 25 == 0:
            print(f"  [{i}/{len(egos)}]")
    metrics = pd.DataFrame(rows)
    summary = metrics.groupby("variant", sort=False)[experiments.METRIC_COLS].mean().reset_index()
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--best", type=Path, default=Path("outputs/tuning/best_config.json"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/final"))
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    data = graphs.Dataset.load(args.processed_dir)
    split = split_mod.load_split(args.processed_dir / "split.json")
    best = json.loads(args.best.read_text())
    summary = evaluate(data, split, best, args.out_dir, args.force)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Note: `summary.csv` always carries `circle_f1`, `circle_f1_one_community` and `largest_share`, because they are in `experiments.METRIC_COLS`.

- [ ] **Step 4: Run it to see it pass**

Run: `$PY -m pytest tests/test_evaluate.py -v` then `$PY -m pytest -q`
Expected: PASS; full suite 74 passed.

- [ ] **Step 5: Run the real evaluation, once** (covers Review Focus item 2 on the test egos)

Run: `$PY -m pipeline.evaluate`
Expected in about a minute: a three-row table (tuned, default, louvain_on_tuned_graph). Run the command a second time and confirm it refuses with the `FileExistsError` message.

- [ ] **Step 6: Report honestly**

Present `outputs/final/summary.csv` to the user with `circle_f1` beside `circle_f1_one_community` and `largest_share`. Say plainly if the tuned config does not beat the default or Louvain on the test egos.

- [ ] **Step 7: Checkpoint**

Commit only if the user has asked: `git add pipeline/evaluate.py tests/test_evaluate.py && git commit -m "feat: add held-out final evaluation"`.

---

### Task 7: Review and documentation

**Files:**
- Modify: `EXECUTION_STEPS.md`, `CODE_GUIDE.md`

- [ ] **Step 1: Review.** Run the `ml-reviewer` and `python-reviewer` agents on `backend/outliers.py`, `backend/graphs.py`, `pipeline/split.py`, `pipeline/tune.py`, `pipeline/evaluate.py`. Fix CRITICAL and HIGH findings (look especially for leakage, seeding and yardstick consistency).

- [ ] **Step 2: Update `EXECUTION_STEPS.md`.** Add steps for: the full run and data-quality audit (Task 2 commands, expected 973 egos and 35 NaN `circle_f1`), the split, the outlier options, tuning (about 10-15 minutes), and final evaluation (run once, `--force` to redo). Update the test count in step 8 from 66 to 74, extend the "Quick reference: full run order", and add a Troubleshooting row: "`ValueError: test egos must not be used for tuning` means a test ego was passed to `tune`; regenerate with `python -m pipeline.split`."

- [ ] **Step 3: Update `CODE_GUIDE.md`.** Document the new modules, the `outliers` argument and the `(tokens, token_key)` TF-IDF cache key, the new tests, and where the search grids and objective live (`pipeline/tune.py`: `GRID`, `MAX_LARGEST_SHARE`, `MIN_COVERAGE`).

- [ ] **Step 4: Final check.** Run `$PY -m pytest -q` (expect 74 passed). Commit only if the user has asked: `git commit -m "docs: document tuning workflow"`.

---

## Self-review

- **Spec coverage:** split (Task 1), data quality and full run (Task 2), outlier rules and cache key (Task 3), shared scoring (Task 4), tuning with fixed yardsticks, constrained objective, grouped cache clearing, leak guard (Task 5), evaluation with guards and `--force` (Task 6), reviews and docs (Task 7). All 7 agreed tests are present (tests 1, 7, 2-4, 5 and 6a, 6b); no others.
- **Placeholders:** none. Every code step shows the code.
- **Type consistency:** `score_ego(data, cfg, ego, tokens_both, algorithm)` is defined in Task 5 and called with the same arguments in Task 6; `OutlierConfig` fields, `token_key`, `assert_no_test_egos` and `DEFAULT_CONFIG` are used consistently.
- **Test count:** 66 existing + 1 (split) + 1 (data quality) + 3 (outliers) + 2 (tune) + 1 (evaluate) = 74.
