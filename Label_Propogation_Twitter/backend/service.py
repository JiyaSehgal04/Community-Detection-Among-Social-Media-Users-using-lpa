"""Logic behind the API: run a detection, shape the payloads, keep the job store.

No FastAPI imports here, so everything is plain functions over a graphs.Dataset.
"""

from __future__ import annotations

import json
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import pandas as pd

from backend import graphs, lpa, metrics as m

MAX_JOBS = 50
COLLAPSE_SHARE = 0.7      # a community holding more than this share of users looks like a collapse
NO_TOKEN_SHARE = 0.3      # warn when more than this share of users have no interest tokens
TOP_TOKENS = 5
SIMILAR_USERS = 5

# data/processed files the API needs, with the columns it reads
REQUIRED_TABLES = {
    "users.parquet": {"user_id", "ego_ids", "n_hashtags", "n_mentions"},
    "interests.parquet": {"user_id", "token", "type"},
    "follows_by_ego.parquet": {"src", "dst", "ego_id"},
    "egos.parquet": {"ego_id", "n_nodes", "n_edges", "n_circles", "has_circles"},
}


class NotFound(Exception):
    """An ego, job or user that does not exist (HTTP 404)."""


class DataError(RuntimeError):
    """The processed data is missing or malformed; the message names the step to run."""


def validate_data_dir(path: Path | str) -> None:
    """Fail early, with the command to run, if the processed tables are missing or empty."""
    d = Path(path)
    fix = "run `python -m pipeline.download` then `python -m pipeline.preprocess`"
    for name, columns in REQUIRED_TABLES.items():
        file = d / name
        if not file.exists():
            raise DataError(f"{file} is missing: {fix}")
        table = pd.read_parquet(file)
        if table.empty:
            raise DataError(f"{file} is empty: {fix}")
        if columns - set(table.columns):
            raise DataError(f"{file} lacks columns {sorted(columns - set(table.columns))}: {fix}")
    if not (d / "circles.json").exists():
        raise DataError(f"{d / 'circles.json'} is missing: {fix}")


def _clean(value):
    """NaN becomes None so it serialises as JSON null (never as 0)."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


# --- jobs ----------------------------------------------------------------------------------

@dataclass
class Job:
    job_id: str
    ego: int
    params: dict
    graph: nx.Graph
    nodes: list[int]                      # sorted node order used by every payload
    labels: dict[int, int]                # node -> raw LPA label (a node id)
    community: dict[int, int]             # node -> community index (0 = largest)
    communities: list[set]
    trace: list[dict[int, int]]           # label snapshots, first = every node on its own label
    metrics: dict
    warnings: list[str] = field(default_factory=list)


class JobStore:
    """Thread-safe in-memory store that keeps the newest MAX_JOBS jobs."""

    def __init__(self, max_jobs: int = MAX_JOBS) -> None:
        self._jobs: OrderedDict[str, object] = OrderedDict()
        self._max = max_jobs
        self._lock = threading.Lock()

    def add(self, job_id: str, job: object) -> None:
        with self._lock:
            self._jobs[job_id] = job
            while len(self._jobs) > self._max:
                self._jobs.popitem(last=False)

    def get(self, job_id: str):
        with self._lock:
            if job_id not in self._jobs:
                raise NotFound(f"unknown or expired job {job_id!r}; run the detection again")
            return self._jobs[job_id]


def _communities(labels: dict[int, int]) -> list[set]:
    groups: dict[int, set] = {}
    for node, label in labels.items():
        groups.setdefault(label, set()).add(node)
    return sorted(groups.values(), key=lambda c: (-len(c), min(map(str, c))))


def _ego_tokens(data: graphs.Dataset, nodes: list[int], kinds: tuple[str, ...]) -> dict[int, set]:
    """Tokens per user, limited to this ego's users (cheap: no groupby over all interests)."""
    sub = data.interests[data.interests.user_id.isin(nodes) & data.interests.type.isin(kinds)]
    return {int(u): set(t) for u, t in sub.groupby("user_id").token}


def compute_warnings(metrics: dict, max_iter: int, no_token_share: float, edgeless: bool,
                     mode: str) -> list[str]:
    """Plain-language flags, so a collapsed or incomplete result is never shown as a success."""
    out: list[str] = []
    if edgeless:
        out.append("The graph has no edges, so every user is in a community of one.")
    if not metrics.get("converged", True):
        out.append(f"LPA did not converge within {max_iter} iterations; the communities may still be shifting.")
    share = metrics.get("largest_share")
    if share is not None and share > COLLAPSE_SHARE:
        out.append(f"The largest community holds {share:.0%} of users: the graph has probably collapsed "
                   "into one community (try direction=mutual or a higher min_sim).")
    if mode in ("interest", "hybrid") and no_token_share > NO_TOKEN_SHARE:
        out.append(f"{no_token_share:.0%} of users have no interest tokens, so they are placed by follows "
                   "only (or left alone).")
    return out


def run_detection(data: graphs.Dataset, ego: int, *, mode: str = "hybrid", direction: str = "mutual",
                  tokens: str = "both", k: int = 10, min_sim: float = 0.2, follow_weight: float = 0.3,
                  runs: int = 10, seed: int = 42, max_iter: int = 100) -> Job:
    if int(ego) not in set(data.egos.ego_id.astype(int)):
        raise NotFound(f"unknown ego {ego}")
    G = graphs.build_graph(data, ego, mode, direction=direction, k=k, min_sim=min_sim,
                           tokens=tokens, follow_weight=follow_weight)
    nodes = [int(n) for n in sorted(G.nodes)]
    edgeless = G.number_of_edges() == 0

    if edgeless:
        labels = {n: n for n in nodes}
        iterations, converged, trace = 0, True, [dict(labels)]
    else:
        best, _ = lpa.best_of_runs(G, runs=runs, seed=seed, max_iter=max_iter)
        labels = {int(n): int(label) for n, label in best.labels.items()}
        iterations, converged = best.iterations, best.converged
        # re-running the best seed with trace=True gives the same labels plus one snapshot per iteration
        replay = lpa.label_propagation(G, seed=best.seed, max_iter=max_iter, trace=True)
        trace = [{int(n): int(label) for n, label in snap.items()} for snap in replay.trace]
    communities = _communities(labels)
    community = {n: i for i, members in enumerate(communities) for n in members}

    follow_G = graphs.follow_graph(data, ego, direction="any")
    circles = data.ego_circles(ego)
    tokens_both = _ego_tokens(data, nodes, graphs.TOKEN_CHOICES["both"])
    result = {
        "modularity": m.safe_modularity(G, communities),
        "follow_modularity": m.safe_modularity(follow_G, communities),
        "circle_f1": m.circle_f1(communities, circles),
        "circle_f1_one_community": m.circle_f1([set(follow_G.nodes)], circles),
        "interest_purity": m.interest_purity(communities, tokens_both),  # always the full token set
        **m.community_stats(communities),
        "iterations": iterations, "converged": converged,
        "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
    }
    result = {key: _clean(value) for key, value in result.items()}
    chosen = _ego_tokens(data, nodes, graphs.TOKEN_CHOICES[tokens])
    no_token_share = 1 - len(chosen) / len(nodes) if nodes else 0.0
    return Job(
        job_id="", ego=int(ego),
        params={"mode": mode, "direction": direction, "tokens": tokens, "k": k, "min_sim": min_sim,
                "follow_weight": follow_weight, "runs": runs, "seed": seed},
        graph=G, nodes=nodes, labels=labels, community=community, communities=communities,
        trace=trace, metrics=result,
        warnings=compute_warnings(result, max_iter, no_token_share, edgeless, mode))


# --- payloads ------------------------------------------------------------------------------

def graph_payload(job: Job) -> dict:
    return {
        "nodes": [{"id": n, "community": job.community[n], "degree": int(job.graph.degree(n))}
                  for n in job.nodes],
        "edges": [{"source": int(a), "target": int(b), "weight": float(d.get("weight", 1.0))}
                  for a, b, d in job.graph.edges(data=True)],
    }


def communities_payload(data: graphs.Dataset, job: Job) -> dict:
    sub = data.interests[data.interests.user_id.isin(job.nodes)].copy()
    sub["community"] = sub.user_id.map(job.community)
    counts = sub.groupby(["community", "type", "token"]).size()
    out = []
    for idx, members in enumerate(job.communities):
        tokens = counts[counts.index.get_level_values(0) == idx] if len(counts) else counts
        by_type = {kind: sorted(((tok, int(c)) for (_, t, tok), c in tokens.items() if t == kind),
                                key=lambda x: (-x[1], x[0]))[:TOP_TOKENS]
                   for kind in ("hashtag", "mention")}
        top_any = tokens.max() if len(tokens) else 0
        purity = float(top_any / len(members)) if len(members) >= 2 else None
        out.append({"id": idx, "size": len(members), "purity": purity,
                    "top_hashtags": [{"token": t, "count": c} for t, c in by_type["hashtag"]],
                    "top_mentions": [{"token": t, "count": c} for t, c in by_type["mention"]]})
    return {"communities": out}


def trace_payload(job: Job) -> dict:
    snapshots = [[snap[n] for n in job.nodes] for snap in job.trace]
    changed = [0] + [sum(a != b for a, b in zip(prev, cur)) for prev, cur in zip(snapshots, snapshots[1:])]
    return {"nodes": job.nodes, "snapshots": snapshots, "changed": changed}


def user_payload(data: graphs.Dataset, user_id: int, job: Job | None) -> dict:
    users = data.users[data.users.user_id == user_id]
    if users.empty:
        raise NotFound(f"unknown user {user_id}")
    rows = data.interests[data.interests.user_id == user_id]
    body = {"user_id": int(user_id),
            "egos": [int(e) for e in users.ego_ids.iloc[0]],
            "tokens": [{"token": t, "type": ty} for t, ty in sorted(zip(rows.token, rows.type))],
            "community": None, "similar_users": []}
    if job is None or user_id not in job.community:
        return body  # similar users and community only make sense inside the run's own ego
    body["community"] = job.community[user_id]
    index, matrix = data.tfidf("both")
    if user_id in index:
        others = [n for n in job.nodes if n != user_id and n in index]
        if others:
            sims = (matrix[index[user_id]] @ matrix[[index[n] for n in others]].T).toarray().ravel()
            ranked = sorted(zip(others, sims), key=lambda x: (-x[1], x[0]))[:SIMILAR_USERS]
            body["similar_users"] = [{"user_id": int(n), "similarity": float(s), "community": job.community[n]}
                                     for n, s in ranked if s > 0]
    return body


def egos_payload(data: graphs.Dataset) -> dict:
    egos = data.egos
    possible = egos.n_nodes * (egos.n_nodes - 1)
    density = (egos.n_edges / possible.where(possible > 0)).fillna(0.0)
    return {"egos": [{"ego_id": int(r.ego_id), "n_nodes": int(r.n_nodes), "n_edges": int(r.n_edges),
                      "density": float(d), "has_circles": bool(r.has_circles), "n_circles": int(r.n_circles)}
                     for r, d in zip(egos.itertuples(), density)]}


def compare_payload(outputs_dir: Path) -> dict:
    """Saved experiment summaries (3 modes x 3 algorithms), one set per follow direction."""
    rows = []
    for direction, path in (("any", outputs_dir / "summary_table.csv"),
                            ("mutual", outputs_dir / "mutual" / "summary_table.csv")):
        if path.exists():
            for record in pd.read_csv(path).to_dict("records"):
                rows.append({"direction": direction, **{k: _clean(v) for k, v in record.items()}})
    return {"rows": rows}


def stats_payload(outputs_dir: Path) -> dict:
    path = outputs_dir / "eda_stats.json"
    if not path.exists():
        raise NotFound(f"{path} is missing; run the EDA notebook or `python -m pipeline.eda`")
    return {"stats": json.loads(path.read_text())}
