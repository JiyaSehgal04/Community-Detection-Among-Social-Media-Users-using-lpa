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
import math
import subprocess
import sys
import warnings
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
assert DEFAULT_CONFIG.keys() == GRID.keys()
RUNS, SEED = 10, 42
MAX_LARGEST_SHARE, MIN_COVERAGE = 0.5, 0.8
AVERAGED = ("interest_purity", "follow_modularity", "circle_f1", "largest_share", "coverage")


def sample_configs(n: int = 60, seed: int = 42) -> list[dict]:
    """Trial 0 is the default config; the rest are distinct random draws from GRID."""
    total = math.prod(len(values) for values in GRID.values())
    if n < 1 or n > total:
        raise ValueError(f"n must be between 1 and {total} (distinct grid combinations), got {n!r}")
    rng = random.Random(seed)
    configs = [dict(DEFAULT_CONFIG)]
    seen = {tuple(DEFAULT_CONFIG[k] for k in CONFIG_KEYS)}
    while len(configs) < n:
        cfg = {key: rng.choice(values) for key, values in GRID.items()}
        key = tuple(cfg[k] for k in CONFIG_KEYS)
        if key not in seen:
            seen.add(key)
            configs.append(cfg)
    return configs


def outlier_config(cfg: dict) -> OutlierConfig:
    return OutlierConfig(cfg["token_max_share"], cfg["hub_cap"], cfg["max_tokens"])


def communities_for(G: nx.Graph, algorithm: str = "lpa") -> list[set]:
    if algorithm not in ("lpa", "louvain"):
        raise ValueError(f"unknown algorithm {algorithm!r}")
    if G.number_of_edges() == 0:
        return [{n} for n in G.nodes]  # nothing to propagate along: everyone stays alone
    if algorithm == "lpa":
        return lpa.best_of_runs(G, runs=RUNS, seed=SEED)[0].communities
    return [set(c) for c in nx.community.louvain_communities(G, weight="weight", seed=SEED)]


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
        data.clear_tfidf_cache()
        for trial, cfg in group:
            rows.append({"stage": stage, "trial": trial, **cfg,
                         **score_config(data, cfg, egos, tokens_both)})
    return rank_trials(pd.DataFrame(rows))


def run_tuning(data: graphs.Dataset | None, split: dict[str, list[int]], train_egos: list[int],
               val_egos: list[int], n_trials: int = 60, top_n: int = 5,
               seed: int = 42) -> tuple[pd.DataFrame, dict]:
    split_mod.assert_no_test_egos([*train_egos, *val_egos], split)  # the leak guard
    if not train_egos or not val_egos:
        raise ValueError("train_egos and val_egos must both be non-empty")
    if top_n < 1 or n_trials < 1:
        raise ValueError(f"top_n and n_trials must be >= 1, got {top_n!r}, {n_trials!r}")
    tokens_both = data.user_tokens("both")
    configs = sample_configs(n_trials, seed)
    stage1 = _score_grouped(data, list(enumerate(configs)), train_egos, tokens_both, "train")
    finalists = [(int(t), configs[int(t)]) for t in stage1.trial.head(top_n)]
    stage2 = _score_grouped(data, finalists, val_egos, tokens_both, "val")
    if not bool(stage2.feasible.iloc[0]):
        warnings.warn("no feasible config was found on val; reporting the best infeasible config's "
                      "val metrics anyway", stacklevel=2)
    return pd.concat([stage1, stage2], ignore_index=True), configs[int(stage2.trial.iloc[0])]


def _git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}")
    return value


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


def _nan_to_none(value):
    v = value.item() if hasattr(value, "item") else value
    return None if isinstance(v, float) and math.isnan(v) else value


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/tuning"))
    p.add_argument("--n-trials", type=positive_int, default=60)
    p.add_argument("--sample-size", type=positive_int, default=100)
    p.add_argument("--top-n", type=positive_int, default=5)
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
        "val_metrics": {c: _nan_to_none(val_best[c]) for c in (*AVERAGED, "objective", "feasible")},
        "tuned_on": sorted({*train, *split["val"]}),
        "seed": args.seed, "n_trials": args.n_trials, "git_hash": _git_hash(),
    }, indent=2, default=_json_default))
    print("best:", best)
    return 0


if __name__ == "__main__":
    sys.exit(main())
