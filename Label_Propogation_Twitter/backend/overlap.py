"""Overlap-aware comparison of a partition with ground-truth circles (Omega index).

Circles overlap (a user can sit in several) but LPA gives each user one community, so best-match
F1 alone is a blunt yardstick. The Omega index asks, for every pair of users, whether they share
the same NUMBER of groups in both groupings, and corrects for chance agreement.
"""

from __future__ import annotations

from typing import Hashable, Iterable, Mapping

import numpy as np

Node = Hashable


def _pair_counts(cover: Iterable[set], index: dict[Node, int]) -> np.ndarray:
    """For every unordered pair of nodes, the number of groups in `cover` containing both."""
    cover = list(cover)
    membership = np.zeros((len(index), len(cover)))
    for j, group in enumerate(cover):
        for node in group:
            if node in index:
                membership[index[node], j] = 1.0
    co_occurrence = membership @ membership.T
    return co_occurrence[np.triu_indices(len(index), 1)].astype(int)


def omega_index(a: Iterable[set], b: Iterable[set], nodes: Iterable[Node]) -> float:
    """Chance-adjusted agreement of two (possibly overlapping) covers over `nodes`.

    1.0 = identical, about 0.0 = no better than chance. NaN if fewer than two nodes.
    """
    ordered = sorted(set(nodes), key=str)
    if len(ordered) < 2:
        return float("nan")
    index = {node: i for i, node in enumerate(ordered)}
    counts_a, counts_b = _pair_counts(a, index), _pair_counts(b, index)
    observed = float(np.mean(counts_a == counts_b))
    expected = sum(float(np.mean(counts_a == k)) * float(np.mean(counts_b == k))
                   for k in np.union1d(counts_a, counts_b))
    if expected >= 1.0:  # both covers give every pair the same count: nothing to correct
        return 1.0 if observed == 1.0 else float("nan")
    return (observed - expected) / (1.0 - expected)


def circle_omega(communities: list[set], circles: Mapping[str, list]) -> float:
    """Omega index of communities vs circles, scored only on users in at least one circle.

    Returns NaN if the ego has no circles (same rule as circle F1).
    """
    circle_sets = [set(members) for members in circles.values() if members]
    if not circle_sets:
        return float("nan")
    in_graph = set().union(*communities) if communities else set()
    scored = set().union(*circle_sets) & in_graph
    return omega_index([c & scored for c in communities if c & scored],
                       [c & scored for c in circle_sets if c & scored], scored)


def random_partition_like(communities: list[set], rng: np.random.Generator) -> list[set]:
    """A random partition of the same nodes with the same community sizes (chance baseline)."""
    nodes = [node for community in communities for node in sorted(community, key=str)]
    shuffled = [nodes[i] for i in rng.permutation(len(nodes))]
    out, start = [], 0
    for community in communities:
        out.append(set(shuffled[start:start + len(community)]))
        start += len(community)
    return out
