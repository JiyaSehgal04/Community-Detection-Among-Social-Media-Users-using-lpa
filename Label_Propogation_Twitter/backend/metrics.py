"""Quality metrics for a partition of an ego network."""

from __future__ import annotations

from collections import Counter
from typing import Hashable, Iterable, Mapping

import networkx as nx
import numpy as np

Node = Hashable


def safe_modularity(G: nx.Graph, communities: Iterable[set], weight: str = "weight") -> float:
    """Modularity of the partition on G, or NaN when G has no edges."""
    if G.number_of_edges() == 0 or G.size(weight=weight) == 0:
        return float("nan")
    return float(nx.community.modularity(G, communities, weight=weight))


def community_stats(communities: list[set]) -> dict[str, float]:
    """Counts communities of size >= 2; coverage = share of nodes in such communities."""
    sizes = [len(c) for c in communities]
    total = sum(sizes)
    grouped = [s for s in sizes if s >= 2]
    return {
        "n_communities": len(grouped),
        "n_singletons": len(sizes) - len(grouped),
        "coverage": sum(grouped) / total if total else float("nan"),
        "largest_share": max(sizes) / total if total else float("nan"),
    }


def _f1(a: set, b: set) -> float:
    overlap = len(a & b)
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(b), overlap / len(a)
    return 2 * precision * recall / (precision + recall)


def circle_f1(communities: list[set], circles: Mapping[str, list]) -> float:
    """Symmetric best-match F1 between communities and ground-truth circles.

    Only users in at least one circle are scored. Returns NaN if the ego has no circles.
    Score = mean( avg over circles of best F1 vs any community,
                  avg over communities of best F1 vs any circle ).
    """
    circle_sets = [set(m) for m in circles.values() if m]
    if not circle_sets:
        return float("nan")
    scored = set().union(*circle_sets)
    comms = [c & scored for c in communities]
    comms = [c for c in comms if c]
    if not comms:
        return 0.0
    circle_side = np.mean([max(_f1(circ, c) for c in comms) for circ in circle_sets])
    comm_side = np.mean([max(_f1(circ, c) for circ in circle_sets) for c in comms])
    return float((circle_side + comm_side) / 2)


def interest_purity(communities: list[set], tokens: Mapping[Node, set]) -> float:
    """Size-weighted share of each community (size >= 2) holding its most common token.

    Users without tokens count in the community size, so they lower purity.
    Returns NaN when there are no communities of size >= 2.
    """
    num, den = 0, 0
    for comm in communities:
        if len(comm) < 2:
            continue
        counts = Counter(t for user in comm for t in tokens.get(user, ()))
        num += counts.most_common(1)[0][1] if counts else 0
        den += len(comm)
    return num / den if den else float("nan")
