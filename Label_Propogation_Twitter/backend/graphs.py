"""Load the processed tables and build per-ego graphs (follow / interest / hybrid)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

from backend.outliers import OutlierConfig, filter_tokens, hub_factors

MODES = ("follow", "interest", "hybrid")
TOKEN_CHOICES = {"hashtag": ("hashtag",), "mention": ("mention",), "both": ("hashtag", "mention")}


@dataclass
class Dataset:
    users: pd.DataFrame
    interests: pd.DataFrame
    follows_by_ego: pd.DataFrame
    egos: pd.DataFrame
    circles: dict[str, dict[str, list[int]]]
    _tfidf: dict = field(default_factory=dict, repr=False)

    def clear_tfidf_cache(self) -> None:
        self._tfidf.clear()

    @classmethod
    def load(cls, processed_dir: Path | str = "data/processed") -> "Dataset":
        d = Path(processed_dir)
        return cls(
            users=pd.read_parquet(d / "users.parquet"),
            interests=pd.read_parquet(d / "interests.parquet"),
            follows_by_ego=pd.read_parquet(d / "follows_by_ego.parquet"),
            egos=pd.read_parquet(d / "egos.parquet"),
            circles=json.loads((d / "circles.json").read_text()),
        )

    @cached_property
    def _members(self) -> pd.Series:
        exploded = self.users[["user_id", "ego_ids"]].explode("ego_ids")
        return exploded.groupby("ego_ids").user_id.agg(lambda s: sorted(s.astype(np.int64)))

    def ego_nodes(self, ego: int) -> list[int]:
        return list(self._members.get(ego, []))

    def ego_edges(self, ego: int) -> pd.DataFrame:
        edges = self.follows_by_ego[self.follows_by_ego.ego_id == ego]
        return edges[edges.src != edges.dst]

    def ego_circles(self, ego: int) -> dict[str, list[int]]:
        return self.circles.get(str(ego), {})

    def user_tokens(self, tokens: str = "both") -> dict[int, set[str]]:
        sub = self.interests[self.interests.type.isin(TOKEN_CHOICES[tokens])]
        return sub.groupby("user_id").token.agg(set).to_dict()

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


def select_egos(data: Dataset, spec: str) -> list[int]:
    """'topN' = N largest egos with circles; 'all' = every ego; or a comma list of ids."""
    egos = data.egos
    if spec == "all":
        return sorted(egos.ego_id.astype(int))
    if spec.startswith("top") and spec[3:].isdigit():
        ranked = egos[egos.has_circles].sort_values(["n_nodes", "ego_id"], ascending=[False, True])
        return [int(e) for e in ranked.ego_id.head(int(spec[3:]))]
    ids = [int(x) for x in spec.split(",") if x.strip()]
    unknown = set(ids) - set(egos.ego_id.astype(int))
    if unknown:
        raise ValueError(f"unknown ego ids: {sorted(unknown)}")
    return ids


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


def interest_graph(data: Dataset, ego: int, k: int = 10, min_sim: float = 0.2,
                   tokens: str = "both", outliers: OutlierConfig | None = None) -> nx.Graph:
    """kNN graph on TF-IDF cosine similarity. Each user links to its k most similar
    users in the ego network with similarity >= min_sim; edges are made symmetric."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k!r}")
    if not 0 <= min_sim <= 1:
        raise ValueError(f"min_sim must be in [0, 1], got {min_sim!r}")
    G = nx.Graph()
    nodes = data.ego_nodes(ego)
    G.add_nodes_from(nodes)

    index, matrix = data.tfidf(tokens, outliers)
    with_tokens = [n for n in nodes if n in index]
    if len(with_tokens) < 2:
        return G
    sub = matrix[[index[n] for n in with_tokens]]
    sim = (sub @ sub.T).toarray()
    np.fill_diagonal(sim, -1.0)

    kk = min(k, len(with_tokens) - 1)
    neighbours = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
    for i, row in enumerate(neighbours):
        for j in row:
            s = float(sim[i, j])
            if s >= min_sim and s > 0:
                a, b = with_tokens[i], with_tokens[j]
                if not G.has_edge(a, b) or G[a][b]["weight"] < s:
                    G.add_edge(a, b, weight=s)
    return G


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
