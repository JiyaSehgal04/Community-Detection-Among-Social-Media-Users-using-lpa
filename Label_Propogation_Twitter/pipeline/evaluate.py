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
    existing = [name for name in ("metrics.csv", "summary.csv") if (Path(out_dir) / name).exists()]
    if existing and not force:
        raise FileExistsError(f"{out_dir}/{existing[0]} exists; the test set is meant to be scored once "
                              "(use --force to overwrite)")


def evaluate(data: graphs.Dataset, split: dict[str, list[int]], best: dict, out_dir: Path,
             force: bool = False) -> pd.DataFrame:
    out_dir = Path(out_dir)
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
