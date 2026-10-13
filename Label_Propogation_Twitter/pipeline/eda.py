"""Exploratory data analysis of the processed ego-Twitter data.

Used by notebooks/01_eda.ipynb, and runnable on its own:

    python -m pipeline.eda

Writes outputs/eda_stats.json and outputs/figures/eda_*.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.preprocess import normalize_token


@dataclass(frozen=True)
class Tables:
    users: pd.DataFrame
    interests: pd.DataFrame
    follows: pd.DataFrame
    egos: pd.DataFrame
    circles: dict[str, dict[str, list[int]]]


def load(processed_dir: Path | str = "data/processed") -> Tables:
    d = Path(processed_dir)
    return Tables(
        users=pd.read_parquet(d / "users.parquet"),
        interests=pd.read_parquet(d / "interests.parquet"),
        follows=pd.read_parquet(d / "follows.parquet"),
        egos=pd.read_parquet(d / "egos.parquet"),
        circles=json.loads((d / "circles.json").read_text()),
    )


# --- statistics ----------------------------------------------------------------------------

def reciprocity(follows: pd.DataFrame) -> float:
    """Share of follow edges whose reverse edge also exists."""
    reverse = follows.rename(columns={"src": "dst", "dst": "src"})
    return len(follows.merge(reverse, on=["src", "dst"])) / len(follows)


def degrees(follows: pd.DataFrame) -> pd.DataFrame:
    """In- and out-degree per user (users with no edges in one direction get 0)."""
    deg = pd.DataFrame({"in_degree": follows.dst.value_counts(),
                        "out_degree": follows.src.value_counts()})
    return deg.fillna(0).astype(int)


def circle_stats(circles: dict, egos: pd.DataFrame) -> dict[str, float]:
    """Circle counts, how often users sit in several circles, and how much of each ego they cover."""
    n_nodes = egos.set_index("ego_id").n_nodes
    sizes, overlap, coverage = [], [], []
    for ego, ego_circles in circles.items():
        if not ego_circles:
            continue
        sizes += [len(m) for m in ego_circles.values()]
        counts = Counter(u for members in ego_circles.values() for u in members)
        overlap.append(sum(c > 1 for c in counts.values()) / len(counts))
        coverage.append(len(counts) / n_nodes[int(ego)])
    return {
        "n_circles": len(sizes),
        "egos_with_circles": len(overlap),
        "circle_size_median": float(np.median(sizes)) if sizes else float("nan"),
        "circle_size_max": max(sizes, default=0),
        "single_member_circles": sum(s == 1 for s in sizes),
        "overlap_rate": float(np.mean(overlap)) if overlap else float("nan"),
        "circle_coverage": float(np.mean(coverage)) if coverage else float("nan"),
    }


def circle_sizes(circles: dict) -> list[int]:
    return [len(m) for ego_circles in circles.values() for m in ego_circles.values()]


def interest_stats(users: pd.DataFrame, interests: pd.DataFrame) -> dict[str, float]:
    per_token = interests.groupby("token").user_id.nunique()
    per_user = interests.groupby("user_id").size()
    types = interests.drop_duplicates("token").type.value_counts()
    return {
        "interest_rows": len(interests),
        "n_tokens": int(per_token.size),
        "n_hashtags": int(types.get("hashtag", 0)),
        "n_mentions": int(types.get("mention", 0)),
        "share_with_hashtag": float((users.n_hashtags > 0).mean()),
        "share_with_mention": float((users.n_mentions > 0).mean()),
        "share_with_no_tokens": float(((users.n_hashtags + users.n_mentions) == 0).mean()),
        "tokens_per_user_median": float(per_user.median()),
        "users_per_token_median": float(per_token.median()),
        "users_per_token_min": int(per_token.min()),
    }


def ego_density(egos: pd.DataFrame) -> pd.DataFrame:
    """Directed density of each ego network: edges / (n * (n - 1))."""
    possible = egos.n_nodes * (egos.n_nodes - 1)
    density = (egos.n_edges / possible.where(possible > 0)).fillna(0.0)
    return egos.assign(density=density)


def token_filter_stats(twitter_dir: Path) -> dict[str, int]:
    """Distinct feature names: raw, lowercased, and after cleaning (normalize_token)."""
    raw: set[str] = set()
    for path in twitter_dir.glob("*.featnames"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                raw.add(line.split(" ", 1)[1].strip())
    return {
        "raw_tokens": len(raw),
        "after_lowercase": len({t.lower() for t in raw}),
        "after_cleaning": len({normalize_token(t) for t in raw} - {None}),
    }


def dataset_stats(t: Tables) -> dict[str, float]:
    deg = degrees(t.follows)
    in_follows = set(t.follows.src) | set(t.follows.dst)
    dens = ego_density(t.egos)
    return {
        "users": len(t.users),
        "follow_edges": len(t.follows),
        "egos": len(t.egos),
        "egos_without_circles": int((~t.egos.has_circles).sum()),
        "users_without_follow_edges": int((~t.users.user_id.isin(in_follows)).sum()),
        "reciprocity": reciprocity(t.follows),
        "in_degree_median": float(deg.in_degree.median()),
        "in_degree_max": int(deg.in_degree.max()),
        "out_degree_median": float(deg.out_degree.median()),
        "out_degree_max": int(deg.out_degree.max()),
        "ego_nodes_median": float(t.egos.n_nodes.median()),
        "ego_nodes_max": int(t.egos.n_nodes.max()),
        "ego_density_median": float(dens.density.median()),
        **circle_stats(t.circles, t.egos),
        **interest_stats(t.users, t.interests),
    }


# --- figures -------------------------------------------------------------------------------
# Reference palette (dataviz skill), light mode. One series -> slot 1; two -> slots 1-2.
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dc"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _style(ax, grid_axis: str = "y") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _title(ax, text: str, sub: str = "") -> None:
    ax.set_title(text, color=INK, fontsize=11, loc="left", pad=18 if sub else 6)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, color=INK_2, fontsize=9)


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    return path


def plot_degree_distribution(t: Tables, path: Path):
    plt = _plt()
    deg = degrees(t.follows)
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    _style(ax, "both")
    for col, colour, name in (("in_degree", BLUE, "followers (in-degree)"),
                              ("out_degree", ORANGE, "following (out-degree)")):
        counts = deg[col][deg[col] > 0].value_counts().sort_index()
        ax.scatter(counts.index, counts.values, s=10, color=colour, label=name, linewidths=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("degree k", color=INK_2)
    ax.set_ylabel("number of users with degree k", color=INK_2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    _title(ax, "Degree distribution of the follow graph",
           "heavy-tailed: most users have few links, a few have thousands")
    return _save(fig, path)


def plot_top_tokens(t: Tables, path: Path, n: int = 30):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.5), facecolor=SURFACE)
    for ax, kind in zip(axes, ("hashtag", "mention")):
        _style(ax, "x")
        top = (t.interests[t.interests.type == kind].token.value_counts().head(n).iloc[::-1])
        ax.barh(top.index, top.values, color=BLUE, height=0.7)
        ax.set_xlabel("users", color=INK_2)
        ax.tick_params(axis="y", labelsize=8)
        _title(ax, f"Top {n} {kind}s", "users holding the token")
    return _save(fig, path)


def plot_interest_sparsity(t: Tables, path: Path):
    plt = _plt()
    per_user = (t.users.n_hashtags + t.users.n_mentions)
    per_token = t.interests.groupby("token").user_id.nunique()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURFACE)

    ax = axes[0]
    _style(ax)
    # Log x-axis cannot show 0, so users without tokens are reported in the subtitle instead.
    has = per_user[per_user > 0]
    ax.hist(has, bins=np.unique(np.logspace(0, np.log10(has.max() + 1), 30).astype(int)),
            color=BLUE, rwidth=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("tokens per user (hashtags + mentions)", color=INK_2)
    ax.set_ylabel("users", color=INK_2)
    none = int((per_user == 0).sum())
    _title(ax, "Tokens per user", f"median {has.median():.0f} among users with tokens; "
           f"{none:,} users ({none / len(per_user):.0%}) have none")

    ax = axes[1]
    _style(ax, "both")
    counts = per_token.value_counts().sort_index()
    ax.scatter(counts.index, counts.values, s=10, color=BLUE, linewidths=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("users sharing the token", color=INK_2)
    ax.set_ylabel("number of tokens", color=INK_2)
    _title(ax, "Users per token", f"median {per_token.median():.0f}; "
           f"{(per_token == 2).mean():.0%} of tokens are shared by only 2 users")
    return _save(fig, path)


def plot_circles(t: Tables, path: Path):
    plt = _plt()
    sizes = np.array(circle_sizes(t.circles))
    per_ego = t.egos[t.egos.has_circles].n_circles
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURFACE)

    ax = axes[0]
    _style(ax)
    ax.hist(sizes, bins=np.unique(np.logspace(0, np.log10(sizes.max() + 1), 25).astype(int)),
            color=BLUE, rwidth=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("circle size (members)", color=INK_2)
    ax.set_ylabel("circles", color=INK_2)
    _title(ax, "Circle size", f"median {np.median(sizes):.0f} members, max {sizes.max()}")

    ax = axes[1]
    _style(ax)
    cap = 20
    ax.hist(per_ego.clip(upper=cap), bins=np.arange(1, cap + 2) - 0.5, color=BLUE, rwidth=0.9)
    ax.set_xticks([1, 5, 10, 15, cap], ["1", "5", "10", "15", f"{cap}+"])
    ax.set_xlabel("circles per ego", color=INK_2)
    ax.set_ylabel("egos", color=INK_2)
    _title(ax, "Circles per ego", f"{len(per_ego)} egos with circles, "
           f"{int((~t.egos.has_circles).sum())} without")
    return _save(fig, path)


def plot_ego_density(t: Tables, path: Path):
    plt = _plt()
    dens = ego_density(t.egos)
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    _style(ax, "both")
    ax.scatter(dens.n_nodes, dens.density, s=12, color=BLUE, alpha=0.6, linewidths=0)
    med = dens.density.median()
    ax.axhline(med, color=INK_2, linewidth=1, linestyle="--")
    ax.text(1.0, med, f" median\n {med:.2f}", transform=ax.get_yaxis_transform(),
            va="center", ha="left", color=INK_2, fontsize=9)
    ax.set_xlabel("users in ego network", color=INK_2)
    ax.set_ylabel("follow density", color=INK_2)
    _title(ax, "Ego network size vs follow density",
           "dense ego networks push LPA towards one giant community")
    return _save(fig, path)


FIGURES = {
    "eda_degree_distribution.png": plot_degree_distribution,
    "eda_top_tokens.png": plot_top_tokens,
    "eda_interest_sparsity.png": plot_interest_sparsity,
    "eda_circles.png": plot_circles,
    "eda_ego_density.png": plot_ego_density,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)

    t = load(args.processed_dir)
    stats = {**dataset_stats(t), **token_filter_stats(args.raw_dir / "twitter")}
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "eda_stats.json").write_text(json.dumps(stats, indent=2))

    plt = _plt()
    for name, plot in FIGURES.items():
        plot(t, args.out_dir / "figures" / name)
        plt.close("all")

    width = max(map(len, stats))
    for key, value in stats.items():
        shown = f"{value:.3f}" if isinstance(value, float) else f"{value:,}"
        print(f"  {key:<{width}} : {shown}")
    print(f"\nWrote {args.out_dir}/eda_stats.json and {len(FIGURES)} figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
