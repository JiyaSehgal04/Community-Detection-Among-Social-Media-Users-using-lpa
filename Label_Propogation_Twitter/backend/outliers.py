"""Optional outlier rules, applied when graphs are built. All off by default; nothing is
deleted from data/processed/."""

from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx
import pandas as pd


@dataclass(frozen=True)
class OutlierConfig:
    token_max_share: float | None = None  # drop tokens used by more than this share of all users
    hub_cap: int | None = None            # scale follow edges of users whose degree exceeds this
    max_tokens: int | None = None         # keep each user's N rarest tokens

    def __post_init__(self) -> None:
        share = self.token_max_share
        if share is not None:
            if isinstance(share, bool) or not isinstance(share, (int, float)) \
                    or math.isnan(share) or not 0 < share <= 1:
                raise ValueError(f"token_max_share must be in (0, 1], got {share!r}")
        for name in ("hub_cap", "max_tokens"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be an int >= 1, got {value!r}")

    @property
    def token_key(self) -> tuple[float | None, int | None]:
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
