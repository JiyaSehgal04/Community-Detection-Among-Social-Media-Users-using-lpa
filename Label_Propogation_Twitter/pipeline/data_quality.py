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
