"""Extra analyses that explain the tuned config. None of them changes a setting.

    python -m pipeline.analyze --which all

Validation egos only (they explain the model, so they stay away from the test set):
  ablation     tuned config vs. follow-only / interest-only / each rule off / each setting reverted
  sensitivity  one-at-a-time sweeps of k, min_sim and follow_weight around the tuned config
Test egos, for the one frozen tuned config (read-only, nothing is tuned on them):
  test         Omega index (+ chance baselines) for tuned / default / Louvain, then paired bootstrap
               intervals from those scores and the saved outputs/final/metrics.csv
  stability    agreement (ARI) between LPA runs with different seeds

Writes CSVs and two figures to outputs/analysis/.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from backend import graphs, lpa, metrics as m, overlap
from pipeline import experiments, split as split_mod, tune
from pipeline.eda import BLUE, INK_2, ORANGE, _plt, _save, _style, _title

N_BOOT = 2000
STABILITY_SEEDS = range(100, 110)  # distinct from the tuning seeds (42-51)
SWEEPS = {name: tune.GRID[name] for name in ("k", "min_sim", "follow_weight")}
SCORE_COLS = ("interest_purity", "follow_modularity", "largest_share", "coverage",
              "circle_f1", "omega")
CONTRASTS = (("tuned", "default"), ("tuned", "louvain_on_tuned_graph"))
TEST_VARIANTS = ("tuned", "default", "louvain_on_tuned_graph")


# --- scoring -------------------------------------------------------------------------------

def score_variant(data: graphs.Dataset, cfg: dict, ego: int, tokens_both: dict,
                  mode: str = "hybrid", algorithm: str = "lpa") -> dict:
    """tune.score_ego plus the Omega index and its two chance baselines, for any graph mode."""
    G = graphs.build_graph(data, ego, mode, direction=cfg["direction"], k=cfg["k"],
                           min_sim=cfg["min_sim"], tokens=cfg["tokens"],
                           follow_weight=cfg["follow_weight"], outliers=tune.outlier_config(cfg))
    comms = tune.communities_for(G, algorithm)
    follow_G = graphs.follow_graph(data, ego, direction="any")
    circles = data.ego_circles(ego)
    baseline = m.circle_f1([set(follow_G.nodes)], circles)
    row = experiments.score_partition(G, comms, follow_G, circles, tokens_both, baseline, 0.0)
    row["omega"] = overlap.circle_omega(comms, circles)
    row["omega_one_community"] = overlap.circle_omega([set(G.nodes)], circles)
    row["omega_random"] = overlap.circle_omega(
        overlap.random_partition_like(comms, np.random.default_rng(int(ego))), circles)
    return row


def _mean_scores(data: graphs.Dataset, cfg: dict, egos: list[int], tokens_both: dict,
                 mode: str = "hybrid") -> dict:
    df = pd.DataFrame([score_variant(data, cfg, e, tokens_both, mode) for e in egos])
    out = {c: float(df[c].mean()) for c in SCORE_COLS}  # mean skips NaN
    out["feasible"] = bool(out["largest_share"] <= tune.MAX_LARGEST_SHARE
                           and out["coverage"] >= tune.MIN_COVERAGE)
    return out


# --- ablation and sensitivity (validation egos) ---------------------------------------------

def ablation_variants(best: dict) -> dict[str, tuple[str, dict]]:
    """name -> (graph mode, config). Variants equal to the tuned config are dropped."""
    variants: dict[str, tuple[str, dict]] = {"tuned (full)": ("hybrid", dict(best)),
                                             "default": ("hybrid", dict(tune.DEFAULT_CONFIG))}
    variants["follow-only graph"] = ("follow", dict(best))
    variants["interest-only graph"] = ("interest", dict(best))
    for rule in ("token_max_share", "hub_cap", "max_tokens"):
        if best[rule] is not None:
            variants[f"without {rule}"] = ("hybrid", {**best, rule: None})
    for name in ("k", "min_sim", "follow_weight", "direction", "tokens"):
        if best[name] != tune.DEFAULT_CONFIG[name]:
            variants[f"{name} back to default"] = ("hybrid", {**best, name: tune.DEFAULT_CONFIG[name]})
    return variants


def run_ablation(data: graphs.Dataset | None, split: dict[str, list[int]], egos: list[int],
                 best: dict) -> pd.DataFrame:
    split_mod.assert_no_test_egos(egos, split)  # validation egos only
    tokens_both = data.user_tokens("both")
    rows = []
    for name, (mode, cfg) in ablation_variants(best).items():
        print(f"  ablation: {name}")
        rows.append({"variant": name, "mode": mode, **_mean_scores(data, cfg, egos, tokens_both, mode)})
    return pd.DataFrame(rows)


def run_sensitivity(data: graphs.Dataset | None, split: dict[str, list[int]], egos: list[int],
                    best: dict) -> pd.DataFrame:
    split_mod.assert_no_test_egos(egos, split)  # validation egos only
    tokens_both = data.user_tokens("both")
    rows = []
    for param, values in SWEEPS.items():
        for value in values:
            print(f"  sensitivity: {param}={value}")
            rows.append({"param": param, "value": value, "is_tuned": value == best[param],
                         **_mean_scores(data, {**best, param: value}, egos, tokens_both)})
    return pd.DataFrame(rows)


# --- test egos: Omega, bootstrap, stability (frozen tuned config) --------------------------------

def _check_frozen(best: dict, split: dict[str, list[int]]) -> None:
    if set(best["tuned_on"]) & set(split["test"]):
        raise ValueError("config was tuned on test egos; re-run tuning")


def run_test_scores(data: graphs.Dataset, split: dict[str, list[int]], best: dict) -> pd.DataFrame:
    """Per-ego scores incl. Omega for tuned / default / Louvain-on-tuned-graph on the test egos."""
    _check_frozen(best, split)
    tokens_both = data.user_tokens("both")
    configs = {"tuned": (best["config"], "lpa"), "default": (tune.DEFAULT_CONFIG, "lpa"),
               "louvain_on_tuned_graph": (best["config"], "louvain")}
    rows = []
    for i, ego in enumerate(split["test"], 1):
        for variant, (cfg, algorithm) in configs.items():
            rows.append({"ego_id": ego, "variant": variant,
                         **score_variant(data, cfg, ego, tokens_both, "hybrid", algorithm)})
        if i % 25 == 0:
            print(f"  test scores [{i}/{len(split['test'])}]")
    return pd.DataFrame(rows)


def paired_bootstrap(a, b, n_boot: int = N_BOOT, seed: int = 42) -> dict:
    """Mean of a-b over egos with a 95% interval from resampling EGOS with replacement."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = (a - b)[~(np.isnan(a) | np.isnan(b))]
    if diff.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, diff.size, size=(n_boot, diff.size))].mean(axis=1)
    return {"mean": float(diff.mean()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)), "n": int(diff.size)}


def bootstrap_table(test_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in SCORE_COLS:
        wide = test_scores.pivot(index="ego_id", columns="variant", values=metric)
        for left, right in CONTRASTS:
            rows.append({"metric": metric, "contrast": f"{left} - {right}",
                         **paired_bootstrap(wide[left], wide[right])})
    return pd.DataFrame(rows)


def run_stability(data: graphs.Dataset, split: dict[str, list[int]], best: dict) -> pd.DataFrame:
    """Mean pairwise ARI between single LPA runs with different seeds, per test ego."""
    _check_frozen(best, split)
    cfg = best["config"]
    rows = []
    for ego in split["test"]:
        G = graphs.build_graph(data, ego, "hybrid", direction=cfg["direction"], k=cfg["k"],
                               min_sim=cfg["min_sim"], tokens=cfg["tokens"],
                               follow_weight=cfg["follow_weight"], outliers=tune.outlier_config(cfg))
        if G.number_of_edges() == 0:
            continue
        nodes = sorted(G.nodes)
        runs = [lpa.label_propagation(G, seed=s).labels for s in STABILITY_SEEDS]
        vectors = [[labels[n] for n in nodes] for labels in runs]
        pair_ari = [adjusted_rand_score(x, y) for x, y in itertools.combinations(vectors, 2)]
        rows.append({"ego_id": ego, "mean_ari": float(np.mean(pair_ari))})
    return pd.DataFrame(rows)


# --- figures -------------------------------------------------------------------------------

def plot_sensitivity(df: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, len(SWEEPS), figsize=(11, 3.8), sharey=True)
    for ax, (param, _) in zip(np.atleast_1d(axes), SWEEPS.items()):
        part = df[df.param == param]
        _style(ax, "y")
        ax.plot(part.value, part.interest_purity, marker="o", color=BLUE, label="interest purity")
        ax.plot(part.value, part.largest_share, marker="o", color=ORANGE, label="largest community share")
        ax.axhline(tune.MAX_LARGEST_SHARE, color=INK_2, linestyle="--", linewidth=0.8)
        tuned = part[part.is_tuned]
        if len(tuned):
            ax.axvline(float(tuned.value.iloc[0]), color=INK_2, linewidth=0.8, alpha=0.5)
        ax.set_xlabel(param, color=INK_2)
    _title(np.atleast_1d(axes)[0], "Sensitivity around the tuned config (validation egos)",
           "dashed line: largest-share limit; vertical line: tuned value")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8, labelcolor=INK_2)
    _save(fig, path)


def plot_ablation(df: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(df) + 1.6))
    _style(ax, "x")
    y = np.arange(len(df))
    ax.barh(y - 0.2, df.interest_purity, height=0.38, color=BLUE, label="interest purity")
    ax.barh(y + 0.2, df.largest_share, height=0.38, color=ORANGE, label="largest community share")
    ax.set_yticks(y)
    ax.set_yticklabels(df.variant, color=INK_2, fontsize=8)
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    _title(ax, "Ablation (validation egos)", "purity should rise without the share approaching 1")
    _save(fig, path)


# --- CLI -----------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--which", default="all",
                   choices=["all", "ablation", "sensitivity", "test", "stability"])
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--best", type=Path, default=Path("outputs/tuning/best_config.json"))
    p.add_argument("--final", type=Path, default=Path("outputs/final/metrics.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/analysis"))
    args = p.parse_args(argv)

    data = graphs.Dataset.load(args.processed_dir)
    split = split_mod.load_split(args.processed_dir / "split.json")
    best = json.loads(args.best.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    todo = {"ablation", "sensitivity", "test", "stability"} if args.which == "all" else {args.which}

    if "ablation" in todo:
        df = run_ablation(data, split, split["val"], best["config"])
        df.to_csv(args.out_dir / "ablation.csv", index=False)
        plot_ablation(df, args.out_dir / "ablation.png")
    if "sensitivity" in todo:
        df = run_sensitivity(data, split, split["val"], best["config"])
        df.to_csv(args.out_dir / "sensitivity.csv", index=False)
        plot_sensitivity(df, args.out_dir / "sensitivity.png")
    if "test" in todo:
        scores = run_test_scores(data, split, best)
        scores.to_csv(args.out_dir / "test_scores.csv", index=False)
        table = bootstrap_table(scores)
        table.to_csv(args.out_dir / "bootstrap.csv", index=False)
        if args.final.exists():  # the recomputed scores must reproduce the saved final metrics
            saved = pd.read_csv(args.final).set_index(["ego_id", "variant"]).interest_purity
            fresh = scores.set_index(["ego_id", "variant"]).interest_purity
            print("max |purity difference| vs saved final metrics:",
                  float((saved - fresh.reindex(saved.index)).abs().max()))
    if "stability" in todo:
        df = run_stability(data, split, best)
        df.to_csv(args.out_dir / "stability.csv", index=False)
        print(f"stability: median per-ego ARI {df.mean_ari.median():.3f} over {len(df)} egos")
    print("wrote", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
