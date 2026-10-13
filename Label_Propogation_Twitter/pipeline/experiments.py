"""Compare LPA, NetworkX LPA and Louvain on follow / interest / hybrid ego graphs.

Run from the project root, after pipeline.preprocess:

    python -m pipeline.experiments --egos top20 --runs 10 --seed 42

Writes outputs/metrics.csv, outputs/summary_table.csv, outputs/lpa_runs.csv and
figures in outputs/figures/.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import networkx as nx
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from backend import graphs, lpa, metrics as m

ALGORITHMS = ("lpa", "nx_lpa", "louvain")
# circle_f1_one_community: F1 if every user were put in one community. Circle F1 favours
# large communities, so an algorithm only beats chance if it beats this baseline.
METRIC_COLS = ["modularity", "follow_modularity", "circle_f1", "circle_f1_one_community",
               "interest_purity",
               "n_communities", "n_singletons", "coverage", "largest_share", "runtime_s"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--egos", default="top20", help="topN | all | comma-separated ego ids")
    p.add_argument("--mode", default="all", choices=[*graphs.MODES, "all"])
    p.add_argument("--direction", default="any", choices=["any", "mutual"])
    p.add_argument("--tokens", default="both", choices=list(graphs.TOKEN_CHOICES))
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--min-sim", type=float, default=0.2)
    p.add_argument("--follow-weight", type=float, default=0.3)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    p.add_argument("--no-figures", action="store_true")
    return p.parse_args(argv)


def _detect(G: nx.Graph, algorithm: str, args: argparse.Namespace):
    """Return (communities, seconds per run, LPA run list or None)."""
    if G.number_of_edges() == 0:
        return [{n} for n in G.nodes], 0.0, None
    start = time.perf_counter()
    if algorithm == "lpa":
        best, runs = lpa.best_of_runs(G, runs=args.runs, seed=args.seed)
        return best.communities, (time.perf_counter() - start) / args.runs, (best, runs)
    if algorithm == "nx_lpa":
        comms = [set(c) for c in nx.community.asyn_lpa_communities(G, weight="weight", seed=args.seed)]
    elif algorithm == "louvain":
        comms = [set(c) for c in nx.community.louvain_communities(G, weight="weight", seed=args.seed)]
    else:
        raise ValueError(algorithm)
    return comms, time.perf_counter() - start, None


def _run_rows(ego: int, mode: str, G: nx.Graph, best: lpa.LPAResult,
              runs: list[lpa.LPAResult]) -> list[dict]:
    nodes = list(G.nodes)
    best_vec = [best.labels[n] for n in nodes]
    return [{
        "ego_id": ego, "mode": mode, "run": i, "seed": r.seed,
        "modularity": r.modularity,
        "n_communities": m.community_stats(r.communities)["n_communities"],
        "iterations": r.iterations, "converged": r.converged,
        "ari_vs_best": adjusted_rand_score(best_vec, [r.labels[n] for n in nodes]),
        "is_best": r is best,
    } for i, r in enumerate(runs)]


def score_partition(G: nx.Graph, comms: list[set], follow_G: nx.Graph, circles: dict, tokens: dict,
                    f1_baseline: float, seconds: float) -> dict:
    """Metric columns for one partition of G (shared by experiments, tune and evaluate)."""
    return {
        "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
        "modularity": m.safe_modularity(G, comms),
        "follow_modularity": m.safe_modularity(follow_G, comms),
        "circle_f1": m.circle_f1(comms, circles),
        "circle_f1_one_community": f1_baseline,
        "interest_purity": m.interest_purity(comms, tokens),
        **m.community_stats(comms),
        "runtime_s": seconds,
    }


def run(data: graphs.Dataset, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    egos = graphs.select_egos(data, args.egos)
    modes = graphs.MODES if args.mode == "all" else (args.mode,)
    tokens = data.user_tokens(args.tokens)
    rows, run_rows = [], []

    for idx, ego in enumerate(egos, 1):
        circles = data.ego_circles(ego)
        follow_G = graphs.follow_graph(data, ego, direction="any")
        f1_baseline = m.circle_f1([set(follow_G.nodes)], circles)
        for mode in modes:
            G = graphs.build_graph(data, ego, mode, direction=args.direction, k=args.k,
                                   min_sim=args.min_sim, tokens=args.tokens,
                                   follow_weight=args.follow_weight)
            for algorithm in ALGORITHMS:
                comms, seconds, lpa_out = _detect(G, algorithm, args)
                if lpa_out:
                    run_rows += _run_rows(ego, mode, G, *lpa_out)
                rows.append({"ego_id": ego, "mode": mode, "algorithm": algorithm,
                             **score_partition(G, comms, follow_G, circles, tokens,
                                               f1_baseline, seconds)})
        print(f"  [{idx}/{len(egos)}] ego {ego}: {follow_G.number_of_nodes()} nodes")

    return pd.DataFrame(rows), pd.DataFrame(run_rows)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    summary = metrics.groupby(["mode", "algorithm"], sort=False)[METRIC_COLS].mean()
    summary["n_egos"] = metrics.groupby(["mode", "algorithm"], sort=False).ego_id.nunique()
    summary["n_egos_with_circles"] = (metrics.dropna(subset=["circle_f1"])
                                      .groupby(["mode", "algorithm"], sort=False).ego_id.nunique())
    summary["n_egos_with_circles"] = summary["n_egos_with_circles"].fillna(0).astype(int)
    return summary.reset_index()


# --- figures -------------------------------------------------------------------------------
# Reference palette (dataviz skill), light mode: categorical slots 1-3 validate all-pairs.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
OTHER = "#c3c2b7"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dc"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_metric_comparison(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("circle_f1", "Circle F1"), ("interest_purity", "Interest purity"),
              ("follow_modularity", "Modularity on follow graph"), ("coverage", "Coverage")]
    modes = [mo for mo in graphs.MODES if mo in set(summary["mode"])]
    fig, axes = plt.subplots(1, len(panels), figsize=(14, 3.6), facecolor=SURFACE)
    width = 0.26
    for ax, (col, title) in zip(axes, panels):
        _style(ax)
        for j, algorithm in enumerate(ALGORITHMS):
            vals = [summary.loc[(summary["mode"] == mo) & (summary.algorithm == algorithm), col]
                    .squeeze() for mo in modes]
            xs = [i + (j - 1) * (width + 0.02) for i in range(len(modes))]
            ax.bar(xs, vals, width=width, color=SERIES[j], label=algorithm, zorder=2)
        ax.set_xticks(range(len(modes)), modes)
        ax.set_ylim(0, 1)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, fontsize=9,
               labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_ego_communities(data: graphs.Dataset, ego: int, args: argparse.Namespace,
                         path: Path) -> None:
    """One panel per mode: our LPA's 3 largest communities coloured, the rest grey."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    tokens = data.user_tokens(args.tokens)
    layout_G = graphs.build_graph(data, ego, "hybrid", direction=args.direction, k=args.k,
                                  min_sim=args.min_sim, tokens=args.tokens,
                                  follow_weight=args.follow_weight)
    # Lay out only users connected in the hybrid graph; far-off isolates would squash the plot.
    shown = [n for n in layout_G.nodes if layout_G.degree(n) > 0]
    hidden = layout_G.number_of_nodes() - len(shown)
    pos = nx.spring_layout(layout_G.subgraph(shown), seed=args.seed, weight="weight")
    fig, axes = plt.subplots(1, len(graphs.MODES), figsize=(15, 6), facecolor=SURFACE)
    for ax, mode in zip(axes, graphs.MODES):
        ax.set_facecolor(SURFACE)
        ax.axis("off")
        G = graphs.build_graph(data, ego, mode, direction=args.direction, k=args.k,
                               min_sim=args.min_sim, tokens=args.tokens,
                               follow_weight=args.follow_weight)
        comms, _, _ = _detect(G, "lpa", args)
        top = [c for c in comms if len(c) >= 2][:3]
        colour = {n: OTHER for n in G.nodes}
        handles = []
        for i, c in enumerate(top):
            for n in c:
                colour[n] = SERIES[i]
            tok = Counter(t for u in c for t in tokens.get(u, ())).most_common(1)
            label = f"{len(c)} users" + (f", top {tok[0][0]}" if tok else "")
            handles.append(Line2D([], [], marker="o", linestyle="", color=SERIES[i], label=label))
        handles.append(Line2D([], [], marker="o", linestyle="", color=OTHER, label="other"))
        H = G.subgraph(shown)
        nx.draw_networkx_edges(H, pos, ax=ax, edge_color=GRID, width=0.4, alpha=0.6)
        nx.draw_networkx_nodes(H, pos, ax=ax, node_size=22, linewidths=0.6, edgecolors=SURFACE,
                               node_color=[colour[n] for n in H.nodes])
        ax.set_title(f"{mode}  (modularity {m.safe_modularity(G, comms):.2f})",
                     color=INK, fontsize=11, loc="left")
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, 0), frameon=False,
                  fontsize=8, labelcolor=INK_2)
    note = f" ({hidden} unconnected user{'s' if hidden != 1 else ''} not shown)" if hidden else ""
    fig.suptitle(f"Ego {ego}: LPA communities by graph mode{note}", color=INK, fontsize=12,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"Loading {args.processed_dir} ...")
    data = graphs.Dataset.load(args.processed_dir)
    print(f"Running on egos={args.egos} mode={args.mode} runs={args.runs} seed={args.seed}")
    metrics, runs = run(data, args)
    summary = summarize(metrics)

    out = args.out_dir
    (out / "figures").mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "metrics.csv", index=False)
    summary.to_csv(out / "summary_table.csv", index=False)
    runs.to_csv(out / "lpa_runs.csv", index=False)

    if not args.no_figures:
        plot_metric_comparison(summary, out / "figures" / "metric_comparison.png")
        first = int(metrics.ego_id.iloc[0])
        plot_ego_communities(data, first, args, out / "figures" / f"communities_ego_{first}.png")

    with pd.option_context("display.width", 140, "display.float_format", "{:.3f}".format):
        print("\n" + summary[["mode", "algorithm", "circle_f1", "circle_f1_one_community",
                              "interest_purity", "modularity", "follow_modularity",
                              "n_communities", "coverage", "largest_share",
                              "runtime_s"]].to_string(index=False))
    if not runs.empty:
        print(f"\nLPA stability: mean ARI vs best run = {runs.ari_vs_best.mean():.3f}, "
              f"converged {runs.converged.mean():.0%} of runs")
    print(f"\nWrote {out}/metrics.csv, summary_table.csv, lpa_runs.csv"
          + ("" if args.no_figures else ", figures/"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
