# Full-dataset run, outlier handling, tuning and held-out evaluation (design)

Date: 2026-10-08 · Status: draft for review
Supersedes the priority order of `2026-10-07-lpa-api-frontend-design.md` (API and frontend are deferred, not cancelled).

## 1. Goal and scope

Priorities, in order: (1) dataset completion, (2) outlier handling, (3) model training.
LPA is unsupervised, so "training" means tuning graph and outlier settings, then measuring the
winner on egos that tuning never saw.

**In scope:** a seeded ego split; a full run on all 973 egos; a data-quality audit; three optional
outlier rules; random-search tuning; one held-out evaluation; the 7 tests in section 8; doc updates.
**Out of scope:** the API and frontend, imputing missing data, circle-outlier rules (circle scoring
is only reported), Optuna, graph mode or algorithm in the search, CI, extra tests.

**Success criteria**
- `split.json` is deterministic; train/val/test are disjoint and cover all 938 egos with circles.
- `data_quality.json` places all 81,306 users in exactly one bucket.
- `outputs/all/metrics.csv` covers 973 egos; `circle_f1` is NaN for exactly the 35 egos without circles.
- `build_graph(..., outliers=None)` and `OutlierConfig()` give results identical to today; the 66
  existing tests still pass.
- Test egos are touched only by `evaluate.py`, once, with all comparators and baselines reported.

## 2. Components

| File | Job |
|---|---|
| `pipeline/split.py` | seeded, stratified split; `load_split`; `assert_no_test_egos` |
| `pipeline/data_quality.py` | user buckets, writes `outputs/data_quality.json` |
| `backend/outliers.py` | `OutlierConfig` and pure rule functions |
| `backend/graphs.py` | optional `outliers` argument (minimal edits) |
| `pipeline/experiments.py` | extract the per-partition scoring into `score_partition(...)` (no behaviour change) |
| `pipeline/tune.py` | random search; writes `outputs/tuning/` |
| `pipeline/evaluate.py` | one-time test evaluation; writes `outputs/final/` |

`lpa.py`, `metrics.py` and `data/processed/*` are unchanged. The outlier code lives in `backend/`
(not `pipeline/`) because `graphs.py` imports it and `backend` must not depend on `pipeline`.

## 3. Split

`make_split(egos, seed=42, fractions=(0.6, 0.2, 0.2))`:
- Keep egos with `has_circles`, sorted by `ego_id` so row order cannot matter.
- Strata: `pd.qcut(n_nodes.rank(method="first"), 4, labels=False)` (rank first, so ties cannot
  break the bins).
- Shuffle each stratum with `np.random.default_rng(seed)`; train = round(0.6n), val = round(0.2n),
  test = the rest (about 563 / 188 / 187).
- `main()` writes `data/processed/split.json`: `{seed, strata, train, val, test, counts}` and checks
  the total is 938.

## 4. Dataset completion

- **Full run:** `python -m pipeline.experiments --egos all --runs 10 --seed 42 --out-dir outputs/all`.
  No code change; the existing top20 outputs are not overwritten. Expect 5-15 minutes.
- **Data quality** (`outputs/data_quality.json`, computed with outlier rules OFF): every user is in
  exactly one bucket: tokens+follows, tokens only, follows only, neither. "Has follows" means the
  user appears as `src` or `dst` of a non-self-loop row in `follows_by_ego`; "has tokens" means at
  least one row in `interests`. "Neither" users stay singletons and are never imputed. The script
  exits 1 if the buckets do not sum to 81,306. A user can still be isolated inside one ego or under
  `mutual`; that is not a bucket.

## 5. Outlier rules

`OutlierConfig` is a frozen, hashable dataclass; every field defaults to `None` (off).

| Field | Rule |
|---|---|
| `token_max_share` | drop tokens used by more than this share of all users (denominator `len(users)` = 81,306) |
| `hub_cap` | for a user with follow-degree `d` in the ego above the cap, multiply that user's follow-edge weights by `cap / d`; if both ends are hubs, multiply both factors |
| `max_tokens` | keep each user's `max_tokens` tokens with the lowest document frequency (highest IDF); ties alphabetical |

- **Token pipeline order:** type filter (`tokens`) -> `token_max_share` -> `max_tokens` -> TF-IDF refit.
- **Hub rule detail:** `d` is the undirected degree in the follow graph for the chosen `direction`.
  In a hybrid edge only the follow part is scaled (`follow_weight * factor`); interest similarity is
  untouched. The ego node follows everyone, so it is always the biggest hub; `hub_cap` therefore
  mostly acts on the ego. A `hub_cap` at or above the maximum degree is an exact no-op.
- **`graphs.py` edits:** `Dataset.tfidf(tokens, outliers=None)` with cache key
  `(tokens, token_max_share, max_tokens)` (`hub_cap` does not affect TF-IDF); with both token rules
  `None` the existing code path runs unchanged. `outliers` is threaded through `interest_graph`,
  `hybrid_graph`, `follow_graph` and `build_graph` as a keyword argument.
  `user_tokens` is not touched: it stays the fixed evaluation yardstick.
- **Known effects:** fractional weights remove many exact ties in LPA, so the random stream shifts;
  results stay reproducible per seed but are not comparable seed-for-seed with unscaled runs.
  Dropping the most common tokens keeps the rarest ones, which can remove kNN edges.

## 6. Tuning (`tune.py`)

**Fixed during tuning:** `mode=hybrid`, `algorithm=lpa`, `runs=10`, `seed=42`.

**Search space** (60 random configs; the default config is always trial 0):

| Parameter | Values |
|---|---|
| k | 5, 10, 15, 20, 30 |
| min_sim | 0.1, 0.15, 0.2, 0.3, 0.4 |
| follow_weight | 0.05, 0.1, 0.2, 0.3, 0.5, 1.0 |
| direction | any, mutual |
| tokens | hashtag, mention, both |
| token_max_share | None, 0.01, 0.02, 0.05 |
| hub_cap | None, 10, 25, 50 |
| max_tokens | None, 10, 20, 50 |

**Protocol:** stage 1 scores all 60 on a seeded 100-ego sample of `split.train`; stage 2 takes the
top 5 and scores them on all of `split.val`; the best by val objective wins.

**Fixed yardsticks (so trials are comparable):** `interest_purity` always uses the unfiltered
`user_tokens("both")`; `follow_modularity` always uses the unscaled `direction="any"` follow graph.

**Objective:** mean `interest_purity` + mean `follow_modularity` over the sample egos (NaN dropped
with nanmean; the NaN count is recorded per trial). Feasible means mean `largest_share <= 0.5` and
mean `coverage >= 0.8` (constraints on the mean, not per ego). Ranking key:
`(feasible, round(objective, 3), circle_f1)`, all descending, NaN as -inf; an infeasible config ranks
below every feasible one.

**Outputs:** `outputs/tuning/trials.csv` (stage, config, metrics, NaN counts) and
`best_config.json` (config, val metrics, `tuned_on` ego ids, seed, git hash).

**Cost:** about 6,000 (config, ego) pairs at roughly 60-120 ms each, plus up to 48 TF-IDF rebuilds:
about 10-15 minutes in total. Trials are grouped by TF-IDF key and `data._tfidf` is cleared between
groups (each entry is about 15 MB).

## 7. Final evaluation (`evaluate.py`)

- Guards: refuse if `best_config.json`'s `tuned_on` overlaps `split.test`, or if the egos to
  evaluate are not exactly `split.test`. Refuse to overwrite `outputs/final/` without `--force`
  (this enforces "run once").
- Per test ego, three rows: winner LPA; defaults LPA (hybrid, any, k=10, min_sim=0.2,
  follow_weight=0.3, tokens both, no outlier rules); Louvain on the winner's graph.
- Output `outputs/final/metrics.csv` and `summary.csv`; both always include `circle_f1`,
  `circle_f1_one_community` and `largest_share`.
- `tune.py` raises `ValueError` if any ego it is given is in `split.test`
  (`assert_no_test_egos` in `split.py`).

## 8. Tests (necessary only, written first)

1. Split: deterministic; train/val/test disjoint; union equals exactly the egos with circles.
2. Token rule drops tokens above the threshold and keeps the rest; `OutlierConfig()` equals `None`.
3. Hub rule rescales only hub-incident follow edges (exact weights); interest-only edges unchanged;
   off is a no-op.
4. Heavy-user rule keeps exactly `min(n, max_tokens)` tokens per user, deterministically.
5. An infeasible config ranks below any feasible one regardless of raw score.
6. `tune.run` raises on a test ego; `evaluate.run` raises when `tuned_on` overlaps test or the egos
   are not exactly the test set.
7. Data-quality buckets sum to the user count and each user lands in exactly one.

The 66 existing tests must still pass.

## 9. Order and docs

Phase 1 split (first) -> Phase 2 data quality and full run (independent) -> Phase 3 outliers ->
Phase 4 tuning -> Phase 5 final evaluation. After Phases 3-5, run the ml-reviewer and
python-reviewer agents. Update `EXECUTION_STEPS.md` (new steps for full run, split, outlier
options, tuning, final evaluation; test count; run order; a "test ego leaked" troubleshooting row)
and `CODE_GUIDE.md` (new modules, the `outliers` argument and cache key, the tests, the search grids
and objective) after each phase.
