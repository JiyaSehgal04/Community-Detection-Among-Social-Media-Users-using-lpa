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
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_NAMES = ("train", "val", "test")
N_STRATA = 4


def make_split(egos: pd.DataFrame, seed: int = 42,
               fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)) -> dict[str, list[int]]:
    """Split egos with circles into train/val/test, stratified by n_nodes quartile."""
    if any(f < 0 for f in fractions) or abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions must be non-negative and sum to 1, got {fractions!r}")
    scored = egos[egos.has_circles].sort_values("ego_id").reset_index(drop=True)
    if scored.empty:
        raise ValueError("no egos with circles to split")
    # Rank first so tied sizes cannot produce duplicate bin edges.
    strata = pd.qcut(scored.n_nodes.rank(method="first"), N_STRATA, labels=False)
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
    empty = [name for name, ids in out.items() if not ids]
    if empty:
        raise ValueError(f"split produced empty set(s): {empty}")
    return {name: sorted(ids) for name, ids in out.items()}


def save_split(split: dict[str, list[int]], path: Path, seed: int) -> None:
    payload = {"seed": seed, "counts": {k: len(v) for k, v in split.items()}, **split}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def load_split(path: Path | str) -> dict[str, list[int]]:
    raw = json.loads(Path(path).read_text())
    missing = [name for name in SPLIT_NAMES if name not in raw]
    if missing:
        raise ValueError(f"{path}: split file is missing key(s): {missing}")
    split = {name: [int(e) for e in raw[name]] for name in SPLIT_NAMES}
    sets = {name: set(ids) for name, ids in split.items()}
    for i, a in enumerate(SPLIT_NAMES):
        for b in SPLIT_NAMES[i + 1:]:
            if sets[a] & sets[b]:
                raise ValueError(f"{path}: {a} and {b} lists overlap")
    return split


def assert_no_test_egos(egos: Iterable[int], split: dict[str, list[int]]) -> None:
    """Raise if any ego in `egos` belongs to the held-out test set."""
    leaked = sorted({int(e) for e in egos} & set(split["test"]))
    if leaked:
        raise ValueError(f"test egos must not be used for tuning: {leaked[:10]}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true", help="overwrite an existing split.json")
    args = p.parse_args(argv)

    target = args.processed_dir / "split.json"
    if target.exists() and not args.force:
        print(f"{target} already exists; refusing to overwrite (use --force)", file=sys.stderr)
        return 1
    egos = pd.read_parquet(args.processed_dir / "egos.parquet")
    split = make_split(egos, seed=args.seed)
    total = sum(len(v) for v in split.values())
    if total != int(egos.has_circles.sum()):
        print(f"split covers {total} egos, expected {int(egos.has_circles.sum())}", file=sys.stderr)
        return 1
    save_split(split, target, args.seed)
    print({k: len(v) for k, v in split.items()}, "-> split.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
