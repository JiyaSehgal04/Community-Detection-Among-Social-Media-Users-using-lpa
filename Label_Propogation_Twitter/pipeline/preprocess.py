"""Turn the raw SNAP ego-Twitter files into clean tables.

Run from the project root, after pipeline.download:

    python -m pipeline.preprocess

Reads data/raw/twitter/ and writes to data/processed/:
    users.parquet, interests.parquet, follows_by_ego.parquet, follows.parquet,
    egos.parquet, circles.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TOKEN_TYPES = {"#": "hashtag", "@": "mention"}
# Word characters in any script; zero-width (non-)joiners occur inside Arabic/Persian words.
WORD = re.compile(r"[\w\u200c\u200d]+")

# Values counted from the SNAP files downloaded on 2026-10-01.
EXPECTED = {
    "users": 81_306,
    "follows": 1_762_504,
    "egos": 973,
    "egos_without_circles": 35,
}


def normalize_token(raw: str) -> str | None:
    """Lowercase a #hashtag / @mention and cut it to its word characters, or return None if junk.

    Twitter hashtags and handles are runs of word characters, so trailing punctuation left
    over from tweet text ("@mashable:" from "RT @mashable: ...", "@foursquare!") is cut off
    and the token merges with its clean form.
    Junk: no #/@ prefix, no word characters, digits-only ("#1"), or no letter/digit ("#_").
    """
    token = raw.strip().lower()
    if len(token) < 2 or token[0] not in TOKEN_TYPES:
        return None
    match = WORD.match(token, 1)
    if not match:
        return None
    body = match.group()
    if body.isdigit() or not any(c.isalnum() for c in body):
        return None
    return token[0] + body


def _ego_ids(twitter_dir: Path) -> list[int]:
    return sorted(int(p.stem) for p in twitter_dir.glob("*.edges") if p.stem.isdigit())


def _read_featnames(path: Path) -> np.ndarray:
    """Return the normalized token (or None) for each feature index, in index order."""
    names = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            idx, name = line.split(" ", 1)
            names[int(idx)] = normalize_token(name)
    return np.array([names.get(i) for i in range(max(names) + 1)] if names else [], dtype=object)


def _read_ego_interests(twitter_dir: Path, ego: int) -> pd.DataFrame:
    """(user_id, token) pairs for every user in this ego's .feat plus the ego itself."""
    tokens = _read_featnames(twitter_dir / f"{ego}.featnames")
    frames = []

    feat_path = twitter_dir / f"{ego}.feat"
    if feat_path.stat().st_size:
        feat = pd.read_csv(feat_path, sep=r"\s+", header=None, dtype=np.int64).to_numpy()
        node_ids, matrix = feat[:, 0], feat[:, 1:]
        rows, cols = np.nonzero(matrix[:, : len(tokens)])
        frames.append(pd.DataFrame({"user_id": node_ids[rows], "token": tokens[cols]}))

    egofeat_text = (twitter_dir / f"{ego}.egofeat").read_text().split()
    if egofeat_text:
        cols = np.nonzero(np.array(egofeat_text[: len(tokens)], dtype=np.int64))[0]
        frames.append(pd.DataFrame({"user_id": np.full(len(cols), ego, dtype=np.int64),
                                    "token": tokens[cols]}))

    if not frames:
        return pd.DataFrame({"user_id": pd.Series(dtype=np.int64), "token": pd.Series(dtype=object)})
    out = pd.concat(frames, ignore_index=True)
    return out[out.token.notna()]


def _read_ego_edges(twitter_dir: Path, ego: int) -> pd.DataFrame:
    """Follow edges from .edges plus the implicit ego -> node edges for every node in the file."""
    path = twitter_dir / f"{ego}.edges"
    if path.stat().st_size:
        edges = pd.read_csv(path, sep=r"\s+", header=None, names=["src", "dst"], dtype=np.int64)
    else:
        edges = pd.DataFrame({"src": pd.Series(dtype=np.int64), "dst": pd.Series(dtype=np.int64)})
    nodes = pd.unique(np.concatenate([edges.src.to_numpy(), edges.dst.to_numpy()]))
    ego_edges = pd.DataFrame({"src": np.full(len(nodes), ego, dtype=np.int64), "dst": nodes})
    out = pd.concat([edges, ego_edges], ignore_index=True)
    out["ego_id"] = np.int64(ego)
    return out


def _read_feat_nodes(twitter_dir: Path, ego: int) -> np.ndarray:
    path = twitter_dir / f"{ego}.feat"
    if not path.stat().st_size:
        return np.array([], dtype=np.int64)
    return pd.read_csv(path, sep=r"\s+", header=None, usecols=[0], dtype=np.int64)[0].to_numpy()


def _read_circles(twitter_dir: Path, ego: int) -> dict[str, list[int]]:
    circles: dict[str, set[int]] = {}
    for line in (twitter_dir / f"{ego}.circles").read_text().splitlines():
        parts = [p for p in line.split("\t") if p.strip()]
        if len(parts) < 2:
            continue
        circles.setdefault(parts[0].strip(), set()).update(int(m) for m in parts[1:])
    return {name: sorted(members) for name, members in circles.items()}


def build(twitter_dir: Path, min_token_users: int = 2) -> dict:
    """Read every ego network and return the processed tables (no files written)."""
    interest_frames, edge_frames, membership_frames, ego_rows = [], [], [], []
    circles_out: dict[str, dict[str, list[int]]] = {}

    for ego in _ego_ids(twitter_dir):
        interest_frames.append(_read_ego_interests(twitter_dir, ego))
        edges = _read_ego_edges(twitter_dir, ego).drop_duplicates()
        edge_frames.append(edges)

        nodes = pd.unique(np.concatenate([
            _read_feat_nodes(twitter_dir, ego), edges.src.to_numpy(), edges.dst.to_numpy(),
            np.array([ego], dtype=np.int64),
        ]))
        membership_frames.append(pd.DataFrame({"user_id": nodes, "ego_id": np.int64(ego)}))

        circles = _read_circles(twitter_dir, ego)
        circles_out[str(ego)] = circles
        ego_rows.append({
            "ego_id": ego, "n_nodes": len(nodes), "n_edges": len(edges),
            "n_circles": len(circles), "has_circles": bool(circles),
        })

    # Interests: dedupe per user (merges "#Python"/"#python" and the same token across egos),
    # then drop tokens used by too few distinct users.
    interests = pd.concat(interest_frames, ignore_index=True).drop_duplicates()
    token_users = interests.groupby("token").user_id.transform("size")
    interests = interests[token_users >= min_token_users].copy()
    interests["type"] = interests.token.str[0].map(TOKEN_TYPES)
    interests = interests.sort_values(["user_id", "token"]).reset_index(drop=True)

    follows_by_ego = (pd.concat(edge_frames, ignore_index=True)
                      .sort_values(["ego_id", "src", "dst"]).reset_index(drop=True))
    follows = (follows_by_ego[["src", "dst"]].drop_duplicates()
               .sort_values(["src", "dst"]).reset_index(drop=True))

    membership = pd.concat(membership_frames, ignore_index=True).drop_duplicates()
    users = (membership.sort_values(["user_id", "ego_id"])
             .groupby("user_id").ego_id.agg(list).rename("ego_ids").reset_index())
    counts = interests.groupby(["user_id", "type"]).size().unstack(fill_value=0)
    for kind in TOKEN_TYPES.values():
        col = counts[kind] if kind in counts else pd.Series(dtype=np.int64)
        users[f"n_{kind}s"] = users.user_id.map(col).fillna(0).astype(np.int64)

    return {
        "users": users,
        "interests": interests[["user_id", "token", "type"]],
        "follows_by_ego": follows_by_ego[["src", "dst", "ego_id"]],
        "follows": follows,
        "egos": pd.DataFrame(ego_rows),
        "circles": circles_out,
    }


def write(tables: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("users", "interests", "follows_by_ego", "follows", "egos"):
        tables[name].to_parquet(out_dir / f"{name}.parquet", index=False)
    (out_dir / "circles.json").write_text(json.dumps(tables["circles"]))


def summarize(tables: dict) -> dict[str, int]:
    egos = tables["egos"]
    return {
        "users": len(tables["users"]),
        "follows": len(tables["follows"]),
        "egos": len(egos),
        "egos_without_circles": int((~egos.has_circles).sum()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--min-token-users", type=int, default=2,
                        help="drop tokens used by fewer distinct users than this")
    args = parser.parse_args(argv)

    twitter_dir = args.raw_dir / "twitter"
    if not twitter_dir.is_dir():
        print(f"{twitter_dir} not found. Run `python -m pipeline.download` first.")
        return 1

    print(f"Reading {twitter_dir} ...")
    tables = build(twitter_dir, min_token_users=args.min_token_users)
    write(tables, args.out_dir)

    interests = tables["interests"]
    token_types = interests.drop_duplicates("token").type.value_counts()
    print(f"Wrote {args.out_dir}/")
    print(f"  interests rows          : {len(interests):>10,}")
    print(f"  distinct tokens         : {interests.token.nunique():>10,}"
          f"  ({token_types.get('hashtag', 0):,} hashtags, {token_types.get('mention', 0):,} mentions)")
    print(f"  follows_by_ego rows     : {len(tables['follows_by_ego']):>10,}")

    stats = summarize(tables)
    problems = []
    for key, expected in EXPECTED.items():
        print(f"  {key:<24}: {stats[key]:>10,}  (expected {expected:,})")
        if stats[key] != expected:
            problems.append(f"{key}: got {stats[key]:,}, expected {expected:,}")

    if problems:
        print("\nCheck FAILED:")
        for msg in problems:
            print(f"  - {msg}")
        return 1
    print("\nCheck passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
