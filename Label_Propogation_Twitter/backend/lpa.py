"""Label Propagation Algorithm (Raghavan, Albert & Kumara, 2007).

Asynchronous, weighted, seeded:
  1. Every node starts with its own label.
  2. Each iteration visits the nodes in a fresh random order. A node takes the label
     with the largest total edge weight among its neighbours. Ties are broken at
     random, but a node keeps its current label if that label is among the tied best.
  3. Stop when an iteration changes no label (every node already holds a label of
     maximum neighbour weight), or after max_iter iterations.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Hashable

import networkx as nx

from backend.metrics import safe_modularity

Node = Hashable


@dataclass
class LPAResult:
    labels: dict[Node, Node]
    iterations: int
    converged: bool
    seed: int | None = None
    trace: list[dict[Node, Node]] = field(default_factory=list)
    modularity: float = float("nan")

    @property
    def communities(self) -> list[set[Node]]:
        groups: dict[Node, set[Node]] = defaultdict(set)
        for node, label in self.labels.items():
            groups[label].add(node)
        return sorted(groups.values(), key=lambda c: (-len(c), min(map(str, c))))


def _sort_key(x: Node) -> tuple:
    # Numbers sort numerically, anything else by its string form, so mixed node types still sort.
    return (0, x, "") if isinstance(x, (int, float)) else (1, 0, str(x))


def label_propagation(G: nx.Graph, seed: int | None = None, max_iter: int = 100,
                      weight: str = "weight", trace: bool = False) -> LPAResult:
    """Run one LPA pass on an undirected graph. Missing edge weights count as 1."""
    rng = random.Random(seed)
    nodes = sorted(G.nodes, key=_sort_key)
    labels: dict[Node, Node] = {n: n for n in nodes}
    snapshots = [dict(labels)] if trace else []

    iterations, converged = 0, False
    while iterations < max_iter:
        order = nodes[:]
        rng.shuffle(order)
        changed = False
        for node in order:
            scores: dict[Node, float] = defaultdict(float)
            for nbr, data in G[node].items():
                if nbr != node:
                    scores[labels[nbr]] += data.get(weight, 1.0)
            if not scores:
                continue  # isolated node keeps its own label
            best = max(scores.values())
            tied = [lab for lab, s in scores.items() if s == best]
            if labels[node] in tied:
                continue
            labels[node] = rng.choice(sorted(tied, key=_sort_key))
            changed = True
        iterations += 1
        if trace:
            snapshots.append(dict(labels))
        if not changed:
            converged = True
            break

    return LPAResult(labels=labels, iterations=iterations, converged=converged,
                     seed=seed, trace=snapshots)


def best_of_runs(G: nx.Graph, runs: int = 10, seed: int = 42, max_iter: int = 100,
                 weight: str = "weight") -> tuple[LPAResult, list[LPAResult]]:
    """Run LPA with seeds seed, seed+1, ... and return the run with the highest modularity."""
    results = []
    for r in range(runs):
        res = label_propagation(G, seed=seed + r, max_iter=max_iter, weight=weight)
        res.modularity = safe_modularity(G, res.communities, weight=weight)
        results.append(res)
    best = max(results, key=lambda r: (r.modularity if r.modularity == r.modularity else float("-inf")))
    return best, results
