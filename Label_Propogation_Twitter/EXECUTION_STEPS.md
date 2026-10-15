# Execution Steps

**Project:** Community Detection Among Social Media Users Based on Common Interests Using the Label Propagation Algorithm
**Dataset:** SNAP ego-Twitter (McAuley & Leskovec, NIPS 2012), https://snap.stanford.edu/data/ego-Twitter.html

This file lists every step to run the project from a fresh machine, in order.
Each step shows the command, what it produces, and how to check it worked.

Status legend: ✅ ready to run · 🛠 to be built (planned in this file)

What each file does is explained in [`CODE_GUIDE.md`](CODE_GUIDE.md); the data findings are in
[`report/EDA_REPORT.md`](report/EDA_REPORT.md).

---

## 0. Prerequisites

| Tool | Version | Check |
|---|---|---|
| Python | 3.10+ | `python --version` |
| Node.js | 18+ (for the web frontend) | `node --version` |
| Git | any | `git --version` |
| Disk space | ~1 GB free | |

---

## 1. Set up the project ✅

```bash
git clone <your-repo-url> lpa-twitter-communities
cd lpa-twitter-communities

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` should contain:

```
pandas>=2.0
numpy>=1.24
networkx>=3.2
scikit-learn>=1.3
matplotlib>=3.7
pyarrow>=14.0
fastapi>=0.110
uvicorn>=0.29
pytest>=8.0
jupyter>=1.0
```

Add a `pyproject.toml` at the project root so `pytest` can import `backend` and `pipeline`:

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

**Check:** `python -c "import networkx, sklearn, fastapi, pytest; print('ok')"` prints `ok`.

Expected folder layout (create missing folders now):

```
lpa-twitter-communities/
├── data/raw/          downloaded SNAP files
├── data/processed/    cleaned files used by everything else
├── pipeline/          __init__.py, download.py, preprocess.py, experiments.py
├── backend/           __init__.py, FastAPI app + algorithm modules
├── frontend/          React app
├── notebooks/         exploratory analysis
├── tests/             pytest tests
├── outputs/           figures, tables, metrics
├── report/
└── pyproject.toml
```

```bash
mkdir -p data/raw data/processed pipeline backend frontend notebooks tests outputs/figures report
touch pipeline/__init__.py backend/__init__.py
```

`pipeline/` and `backend/` are packages, so pipeline scripts are run as modules
(`python -m pipeline.<name>`) from the project root. Running them by path
(`python pipeline/x.py`) breaks `from backend ... import ...`.

---

## 2. Download the dataset ✅

From the project root, run:

```bash
python -m pipeline.download
```

**Produces:**

```
data/raw/
├── twitter.tar.gz
├── twitter_combined.txt.gz
├── readme-Ego.txt
├── twitter/                  973 ego networks, 5 files each
│   ├── <id>.edges            "a follows b" (directed)
│   ├── <id>.feat             0/1 features per user
│   ├── <id>.egofeat          the ego user's own features
│   ├── <id>.featnames        index -> #hashtag or @mention
│   └── <id>.circles          user-made lists (ground truth)
└── twitter_combined.txt      all follow edges
```

**Check:** the script's verify step should print:

```
Ego networks found     : 973        (expected 973)
Egos missing files     : 0
Egos with empty circles: 35         (expected 35)
Combined lines         : 2,420,766  (expected 2,420,766)
Combined unique edges  : 1,768,149  (expected 1,768,149)
Combined unique nodes  : 81,306     (expected 81,306)
```

`twitter_combined.txt` repeats edges that appear in more than one ego network,
so the line count (2,420,766) is larger than the unique edge count SNAP publishes
(1,768,149). Verify both.

**Manual fallback:** download `twitter.tar.gz` and `twitter_combined.txt.gz` from the SNAP page into `data/raw/`, then:

```bash
cd data/raw && tar -xzf twitter.tar.gz && gunzip -k twitter_combined.txt.gz && cd ../..
```

Read `data/raw/readme-Ego.txt` once before moving on; it defines every file format.

**Known data quirks** (verified against the downloaded files):

| Quirk | Effect |
|---|---|
| 35 egos have an empty `.circles` file | No ground truth for them; skip in circle F1 |
| Circle names are numbers (`0`, `1`, ...), not real list names | Use them as IDs only |
| `.featnames` contains junk tokens (`#...`, `#1`, pure numbers) | Filter in preprocessing |
| Same edge appears in several egos' `.edges` (2,286,909 lines, 1,667,885 unique, before ego edges) | Deduplicate before building graphs |

---

## 3. Preprocess the data ✅

```bash
python -m pipeline.preprocess
```

**What it must do:**

1. Loop over all 973 egos in `data/raw/twitter/`.
2. Read `<id>.featnames` and map each feature index to its **text** (`#python`, `@nasa`).
   Indices differ between egos, so always merge by text, never by index.
   Lowercase tokens so `#Python` and `#python` merge.
3. Cut each token to its leading word characters, as Twitter does for tags and handles:
   `@mashable:` (from *"RT @mashable: …"*) → `@mashable`, `@foursquare!` → `@foursquare`.
   About 30% of raw tokens carry such stray punctuation. Then drop junk: no word characters
   (`#...`), digits only (`#1`), or no letter/digit (`#_`). Also drop tokens used by fewer
   than 2 users (`--min-token-users 2`). SNAP already did this, so it removes nothing here.
4. Read `<id>.feat` and `<id>.egofeat` -> list of hashtags/mentions per user.
5. Read `<id>.edges` and add the implicit edges **ego -> every node in the file**.
6. Read `<id>.circles` -> list of member sets per ego. Keep egos with an empty
   circles file, but mark them `has_circles = False`.
7. Deduplicate users that appear in several egos (union their features).
8. Write follow edges in two forms:
   - `follows_by_ego` keeps one row per (src, dst, ego_id), used to build per-ego graphs.
   - `follows` keeps one row per unique (src, dst), used for global stats.

**Produces:**

| File | Columns |
|---|---|
| `data/processed/users.parquet` | user_id, ego_ids, n_hashtags, n_mentions |
| `data/processed/interests.parquet` | user_id, token, type (`hashtag` / `mention`) |
| `data/processed/follows_by_ego.parquet` | src, dst, ego_id |
| `data/processed/follows.parquet` | src, dst (unique) |
| `data/processed/circles.json` | {ego_id: {circle_id: [user_ids]}} |
| `data/processed/egos.parquet` | ego_id, n_nodes, n_edges, n_circles, has_circles |

**Check:**

```bash
python -c "import pandas as pd; print(pd.read_parquet('data/processed/users.parquet').shape)"
python -c "import pandas as pd; print(len(pd.read_parquet('data/processed/follows.parquet')))"
python -c "import pandas as pd; e=pd.read_parquet('data/processed/egos.parquet'); print(len(e), (~e.has_circles).sum())"
```

Expected:

- `users.parquet`: 81,306 rows. Take users from `.feat` and the egos, not only from
  `.edges`: 5,037 users have features but no edges, and `.edges` + egos alone give 76,269.
- `follows.parquet`: 1,762,504 unique edges (1,667,885 from `.edges` plus the added
  ego -> node edges). This is slightly below SNAP's 1,768,149 because
  `twitter_combined.txt` contains some edges not rebuilt from the per-ego files.
  Do not treat the difference as a bug.
- `egos.parquet`: 973 rows, 35 with `has_circles = False`

These numbers were checked against the raw SNAP files. Token filtering does not change them.

With the default `--min-token-users 2` the script also prints (for reference, not checked):
1,084,064 interest rows and 155,454 distinct tokens (24,032 hashtags, 131,422 mentions),
down from 216,817 raw feature names. 64.2% of users have at least one hashtag,
83.2% at least one mention, and 13.4% have neither.
Non-Latin hashtags (Arabic, Japanese, Hebrew, ...) are real tokens and are kept.
The script exits with code 1 if any checked count is wrong. It takes about 25 s.

---

## 4. Exploratory data analysis ✅

```bash
python -m pipeline.eda                     # numbers + figures, about 3 s
jupyter notebook notebooks/01_eda.ipynb    # same analysis with explanations
```

The notebook calls `pipeline/eda.py` (tested in `tests/test_eda.py`), so both give the same numbers.
The written findings are in [`report/EDA_REPORT.md`](report/EDA_REPORT.md).

**Produces:**

| File | Content |
|---|---|
| `outputs/eda_stats.json` | every number below |
| `outputs/figures/eda_degree_distribution.png` | followers / following, log-log |
| `outputs/figures/eda_top_tokens.png` | top 30 hashtags and top 30 mentions |
| `outputs/figures/eda_interest_sparsity.png` | tokens per user, users per token |
| `outputs/figures/eda_circles.png` | circle sizes, circles per ego |
| `outputs/figures/eda_ego_density.png` | ego size vs follow density |

**Key numbers:**

| Measure | Value |
|---|---|
| Users / follow edges / egos / circles | 81,306 / 1,762,504 / 973 / 4,065 |
| Reciprocity (mutual follows) | 48.3% |
| Median ego size / follow density | 141 users / 0.11 |
| Users with no follow edges | 5,037 |
| Users with ≥1 hashtag / ≥1 mention / no tokens | 64.2% / 83.2% / 13.4% |
| Distinct tokens: raw → lowercased → cleaned | 216,817 → 198,458 → 155,454 |
| Median users per token | 3 |
| Circle size (median / max); single-member circles | 7 / 207; 520 |
| Circle members in >1 circle; ego users covered by circles | 18.1%; 27.5% |

**Check:** `outputs/eda_stats.json` has `users = 81306` and `after_cleaning = 155454`, and the
5 `eda_*.png` figures exist.

---

## 5. Run the algorithm experiments ✅

```bash
python -m pipeline.experiments --egos top20 --runs 10 --seed 42
```

**Ego selection:** `topN` means the N egos **with circles** (`has_circles = True`)
that have the most nodes, ties broken by ego_id. This makes the set reproducible
and guarantees every selected ego has ground truth.

**Graph modes compared:**

| Mode | Edges | Weight |
|---|---|---|
| `follow` | follows, as undirected (`--direction any` or `mutual`) | 1 |
| `interest` | kNN on TF-IDF cosine similarity of hashtags **and** mentions | similarity |
| `hybrid` | interest graph + follow edges | similarity + follow weight |

Use `--tokens hashtag | mention | both` (default `both`) to compare which signal
works best. Mentions (shared accounts followed or talked to) are a strong interest
signal, so leaving them out needs a stated reason in the report.

**Algorithms:**

| Name | Implementation | Notes |
|---|---|---|
| `lpa` | our LPA, `backend/lpa.py` | asynchronous, random order, seeded; best of `--runs` by modularity |
| `nx_lpa` | `nx.community.asyn_lpa_communities(G, weight="weight", seed=s)` | seeded and weighted. Do **not** use `label_propagation_communities`: it takes no seed and ignores weights |
| `louvain` | `nx.community.louvain_communities(G, weight="weight", seed=s)` | baseline, built into NetworkX (no extra package) |

**Useful options:**

```bash
--egos top20 | all | 12831,14203,...   which ego networks (topN = largest egos with circles)
--mode follow | interest | hybrid | all
--direction any | mutual
--tokens hashtag | mention | both       which interest tokens feed TF-IDF
--k 10 --min-sim 0.2                    interest-graph parameters
--follow-weight 0.3                     hybrid weight
--runs 10 --seed 42                     LPA repetitions and reproducibility
```

**Produces** in `outputs/`:

- `metrics.csv` — one row per (ego, mode, algorithm)
- `summary_table.csv` — averages for the report
- `lpa_runs.csv` — every LPA run: seed, modularity, iterations, converged, ARI vs the best run
- `figures/metric_comparison.png` — circle F1, purity, follow modularity, coverage per mode × algorithm
- `figures/communities_ego_<id>.png` — the largest selected ego, LPA communities in each mode

Use `--out-dir outputs/<name>` to keep variants apart (e.g. `outputs/mutual`), and
`--no-figures` for quick runs. Takes about 18 s for top20 with 10 runs.

**Code:** `backend/lpa.py` (our LPA), `backend/graphs.py` (data loading and the three
graph modes), `backend/metrics.py`, `pipeline/experiments.py` (CLI and figures).

**Metrics reported:**

| Column | Meaning |
|---|---|
| `modularity` | modularity of the partition on the graph it was found on (not comparable across modes) |
| `follow_modularity` | modularity of the same partition on the follow graph, so modes can be compared |
| `circle_f1` | symmetric best-match F1 vs circles (rules below) |
| `circle_f1_one_community` | **baseline**: circle F1 if every user were put in one community |
| `interest_purity` | size-weighted share of each community (size ≥ 2) holding its most common token |
| `n_communities` / `n_singletons` | communities of size ≥ 2 / users left alone |
| `coverage` | share of users in a community of size ≥ 2 |
| `largest_share` | share of users in the biggest community (flags the "one giant community" failure) |
| `runtime_s` | seconds per single run (our LPA: total time / `--runs`) |

Our LPA follows Raghavan et al. (2007): asynchronous updates in a fresh random order each
iteration; ties are broken at random, but a node keeps its label if it is among the tied best;
it stops when an iteration changes no label (max 100 iterations).

**Circle F1 rules:**

- Skip egos with `has_circles = False` (35 of them). Write `NaN` for F1, not 0.
- Score only users who belong to at least one circle. Circles cover a subset of
  each ego network, so counting the other users as errors would unfairly lower F1.
- Circles overlap but LPA communities do not. Use the symmetric best-match F1
  (average of circle -> best community and community -> best circle).

- **Circle F1 favours large communities.** On the top 20 egos, putting everyone in one
  community scores 0.689, higher than every algorithm. Always report `circle_f1` next to
  `circle_f1_one_community` and `largest_share`. A high F1 together with
  `largest_share` ≈ 0.8–0.9 means the graph collapsed, not that it found the circles.

**Check:** `summary_table.csv` has rows for all 3 modes × 3 algorithms, and
`metrics.csv` has no F1 values for egos without circles.

**Results** (top20, `--runs 10 --seed 42`, averages, with the cleaned tokens from step 3):

| Mode (direction) | Algorithm | Circle F1 | Purity | Follow modularity | Communities | Largest share |
|---|---|---|---|---|---|---|
| follow (any) | lpa | 0.672 | 0.286 | 0.120 | 2.5 | 0.82 |
| follow (any) | louvain | 0.609 | 0.335 | 0.289 | 4.2 | 0.37 |
| interest | lpa | 0.420 | 0.839 | 0.050 | 32.7 | 0.11 |
| interest | louvain | 0.443 | 0.756 | 0.058 | 24.4 | 0.10 |
| hybrid (any) | lpa | 0.677 | 0.301 | 0.137 | 3.0 | 0.80 |
| hybrid (mutual) | lpa | 0.570 | 0.469 | 0.153 | 14.5 | 0.47 |
| hybrid (mutual) | louvain | 0.570 | 0.443 | 0.199 | 9.8 | 0.23 |

What this shows (the EDA in step 4 explains why):

- With `--direction any` the follow graphs are dense (median density 0.11, about 3,800 edges
  per 250-user ego), and LPA collapses about 80% of users into one community. Follow edges
  also swamp the interest signal in hybrid mode.
- `--direction mutual` breaks the giant community and gives a real hybrid result
  (about 15 communities, purity up from 0.30 to 0.47). It is the better setting to present.
- Interest mode gives the purest communities (0.84), but only 61% of users end up in a
  group, because 13% have no tokens and others have no neighbour above `--min-sim`.
- LPA stability across seeds: mean ARI vs the best run is 0.77 (any) / 0.82 (mutual),
  and 100% of runs converged.
- Cleaning the tokens (step 3) raised interest coverage from 55% to 61% and hybrid-mutual
  purity from 0.44 to 0.47.

---

## 6. Full run of all 973 egos and data quality ✅

```bash
python -m pipeline.experiments --egos all --runs 10 --seed 42 --out-dir outputs/all --no-figures   # 10-15 min
python -m pipeline.data_quality                                                                    # a few seconds
```

The first command is step 5 on every ego. It writes `outputs/all/metrics.csv`, `summary_table.csv` and
`lpa_runs.csv` (no figures). `metrics.csv` has 8,757 rows; `circle_f1` is `NaN` for exactly the 35 egos without circles.

`pipeline.data_quality` writes `outputs/data_quality.json`. It puts each of the 81,306 users in exactly one bucket:

| Bucket | Users |
|---|---|
| tokens + follows | 67,816 |
| tokens only | 2,572 |
| follows only | 8,453 |
| neither | 2,465 |

Users in the "neither" bucket stay singletons in every graph mode. Nothing is imputed for them.

**Check:** `outputs/all/metrics.csv` has 8,757 rows, and the four buckets in `data_quality.json` add up to 81,306.

---

## 7. Split the egos into train / validation / test ✅

```bash
python -m pipeline.split [--seed 42] [--force]
```

Takes the 938 egos that have circles and splits them 60/20/20 with a seeded shuffle, stratified by ego size
into 4 strata so each part has small and large egos. **Produces** `data/processed/split.json`
(562 train / 188 val / 188 test).

The script refuses to overwrite an existing `split.json` unless you pass `--force`. A different split
changes which egos are test egos, so any tuning done before it is no longer valid.

**Check:** the three lists in `split.json` have 562, 188 and 188 egos and do not overlap.

---

## 8. Outlier options ✅

No command to run: these are options in `backend/outliers.py` that the tuning step (step 9) switches on or off.
`OutlierConfig(token_max_share, hub_cap, max_tokens)` is passed to the graph builders as `outliers=`.
All three are off (`None`) by default, and with `outliers=None` the graphs are identical to before.

| Option | Rule |
|---|---|
| `token_max_share` | Drop tokens used by more than this share of all users (e.g. `0.05`) |
| `hub_cap` | A hub's follow-edge weights are scaled by `cap / degree`. If both ends are hubs the factors multiply. Only the follow part of a hybrid edge is scaled |
| `max_tokens` | Keep only each user's rarest `N` tokens |

Invalid values raise `ValueError`. The TF-IDF cache is keyed on `(tokens, (token_max_share, max_tokens))`;
`hub_cap` does not affect TF-IDF.

---

## 9. Tune the hybrid LPA configuration ✅

```bash
python -m pipeline.tune [--n-trials 60] [--sample-size 100] [--top-n 5] [--seed 42]    # about 10 min
```

Needs `data/processed/split.json` (step 7). Random search over `GRID` (`k`, `min_sim`, `follow_weight`,
`direction`, `tokens`, `token_max_share`, `hub_cap`, `max_tokens`), using hybrid mode and our LPA with
`runs=10`, `seed 42`. Trial 0 is always the default config.

- **Stage 1:** every trial is scored on a 100-ego sample of the train egos.
- **Stage 2:** the top 5 are re-scored on all 188 validation egos; the best of those wins.
- **Objective:** mean `interest_purity` + mean `follow_modularity`. A trial is *feasible* if mean `largest_share` <= 0.5
  and mean `coverage` >= 0.8. Ranking: feasible first, then objective (rounded to 3 decimals), then `circle_f1`.
- **Fair scoring:** purity always uses the unfiltered tokens (`both`), and follow modularity always uses the
  unscaled `direction=any` follow graph, so trials with different settings are scored on the same footing.
- **Leak guard:** the script raises an error if a test ego is used.
- **Do not re-run:** do not re-run `pipeline.tune` after step 10 has been run: it overwrites outputs/tuning/best_config.json and leaves it out of step with outputs/final/. A new search needs a fresh split (the current test egos have been scored).
- **`git_hash`:** `best_config.json`'s `git_hash` records the commit at the time and does not identify uncommitted code.

**Produces** `outputs/tuning/trials.csv` (all trials) and `outputs/tuning/best_config.json`.

**Result:** 27 of 60 train trials were feasible. The winner (trial 12):

| Setting | Value |
|---|---|
| `k` / `min_sim` | 30 / 0.2 |
| `follow_weight` / `direction` | 0.1 / mutual |
| `tokens` | mention |
| `token_max_share` / `hub_cap` / `max_tokens` | 0.05 / 10 / 20 |

On the 188 validation egos: purity 0.646, follow modularity 0.106, largest share 0.299, coverage 0.866, circle F1 0.534. These were selected on the validation egos, so they lean optimistic; the test numbers in step 10 are the unbiased estimate.

---

## 10. Final evaluation on the test egos (run once) ✅

```bash
python -m pipeline.evaluate [--force]
```

Scores three variants on the 188 test egos, once: the tuned config (LPA), the default config (LPA), and
Louvain on the tuned graph. It refuses to run if `tuned_on` overlaps the test egos, and refuses to overwrite
`outputs/final/metrics.csv` or `summary.csv` unless you pass `--force`.
**Produces** `outputs/final/metrics.csv` and `outputs/final/summary.csv`.
`runtime_s` is 0.0 in this flow because runtime is not measured here.

**Results** (188 test egos, per-variant means):

| Variant | Circle F1 | Circle F1, one-community baseline | Purity | Follow modularity | Largest share | Coverage | Communities |
|---|---|---|---|---|---|---|---|
| tuned (LPA) | 0.534 | 0.678 | 0.623 | 0.106 | 0.316 | 0.863 | 16.9 |
| default (LPA) | 0.692 | 0.678 | 0.362 | 0.106 | 0.809 | 0.973 | 2.9 |
| Louvain on tuned graph | 0.541 | 0.678 | 0.552 | 0.126 | 0.251 | 0.863 | 9.6 |

### How to read these results / limitations

1. The tuned config gains interest purity (0.623 vs 0.362) and avoids the giant community (largest share 0.32 vs 0.81),
   but its circle F1 (0.534) is **lower** than the default's (0.692) and below the one-community baseline (0.678).
   The default's F1 comes from one giant community. Read per ego, the tuned config beats the one-community baseline's circle F1 on only 20.7% of the test egos, the default on 24.5% and Louvain on the tuned graph on 22.3%, so circle F1 favours none of them on most egos.
2. Purity ignores singletons and rewards finer partitions, so the purity gain is partly granularity, not proof of
   better communities. The objective is effectively purity, because follow modularity varies much less than purity across trials
   (standard deviation 0.025 vs 0.093 over the 60 train trials; the objective correlates 0.97 with purity over all 60 trials and 0.89 among the 27 feasible ones). (The tuned and default test means both being 0.106 is a coincidence; their
   per-ego values differ.)
3. Louvain was run once (LPA is the best of 10 runs) and only on the tuned graph, so the comparison is not fully like-for-like. Louvain on the tuned graph is slightly ahead of tuned LPA on circle F1 (0.541 vs 0.534, a gap that is not significant, see step 10b) and clearly ahead on follow modularity (0.126 vs 0.106); tuned LPA wins only on interest purity (0.623 vs 0.552), by producing more and smaller communities (16.9 vs 9.5 on average).
4. The default config is infeasible under the tuning constraints on the train sample (largest share 0.775), so it is
   a plain baseline.
5. The tuned config was evaluated on the test egos once, and no setting was chosen from test numbers. The test egos were seen earlier at default settings (the all-egos run in outputs/all, and 4 of the 20 largest egos used in step 5), and the evaluation was re-run once into a scratch folder to confirm it reproduces (results identical). Changing the objective or constraints now would make the test result tuned-on-test, so a new search would need a fresh split.
6. About 8,300 of the 21,265 users in the test egos also appear in train/val egos. IDF and token-share statistics
   use all users (no labels involved).

---

## 10b. Extra analyses: ablation, sensitivity, error bars, stability ✅

```bash
python -m pipeline.analyze --which all     # about 10 min; or ablation | sensitivity | test | stability
```

Nothing here changes a setting. Ablation and sensitivity use the **validation** egos only (they refuse test
egos). Omega, the bootstrap and stability describe the one frozen tuned config on the test egos (the Omega pass
scores the test egos once more; the script checks it reproduces the saved final purity, difference about 1e-16).

**Produces** in `outputs/analysis/`: `ablation.csv` / `ablation.png`, `sensitivity.csv` / `sensitivity.png`,
`test_scores.csv` (per-ego scores incl. Omega), `bootstrap.csv`, `stability.csv`.

**What it shows** (188 egos each; 95% intervals from 2,000 bootstrap resamples of egos):

1. **Omega index** (overlap-aware agreement with circles; 1 = identical, 0 = chance). On the test egos: tuned 0.193,
   default 0.222, Louvain 0.208, against 0.067 for the one-community baseline and 0.009 to 0.033 for a random partition
   with the same community sizes. All three recover circle structure beyond chance, and unlike circle F1 Omega does not
   reward one giant community. The differences between them are **not significant** (tuned minus default -0.028
   [-0.073, 0.016]; tuned minus Louvain -0.015 [-0.042, 0.012]; 180 egos have scorable circles).
2. **Error bars, tuned minus default / minus Louvain:** interest purity +0.261 [0.244, 0.277] / +0.071 [0.059, 0.082];
   largest share -0.494 / +0.065 (the tuned biggest community is larger than Louvain's); follow modularity -0.000
   [-0.014, 0.014] / -0.021 [-0.025, -0.016]; circle F1 -0.158 [-0.181, -0.132] / -0.007 [-0.022, 0.009]. So the tuned
   config clearly beats the default on purity and giant-community share, is not distinguishable from Louvain on circle F1
   (this corrects step 10's "slightly ahead"), and is clearly behind Louvain on follow modularity.
3. **Ablation** (validation egos). Settings that matter: `hub_cap` (without it purity 0.578, largest share 0.438 vs
   0.646 / 0.299), `follow_weight` (0.3 instead of 0.1: purity 0.582) and `direction` (any instead of mutual: purity 0.556).
   `token_max_share`, `max_tokens`, `k` and `tokens` change purity by 0.01 or less. Interest-only has the highest purity
   (0.821) but places only 58.5% of users, follow-only gives purity 0.550 and coverage 0.779: both break the tuning
   constraints, and the hybrid is what satisfies them.
4. **Sensitivity** (validation egos). Purity is flat in `k` (0.646 to 0.655 for k = 5 to 30) and sensitive to `min_sim`
   (0.555 at 0.1, 0.646 at 0.2 to 0.3) and `follow_weight`. As `follow_weight` rises from 0.05 to 1.0, purity falls
   (0.666 to 0.560) while Omega rises (0.167 to 0.270): interest purity and agreement with circles pull in opposite
   directions. The objective is mostly purity, so it chose 0.1; a larger weight would be preferred if circle agreement
   mattered more. Every swept configuration stays feasible.
5. **Stability:** the median per-ego ARI between single LPA runs with different seeds is 0.779 (mean 0.760). Results
   are reproducible for a seed but not identical across seeds; reported results use the best of 10 runs by modularity.

---

## 11. Start the backend API ✅

```bash
uvicorn backend.main:app --reload --port 8000
```

Startup loads `data/processed/` once (about 3 s, override the folder with `LPA_DATA_DIR`; saved results are read from
`outputs/`, override with `LPA_OUTPUTS_DIR`). If a table is missing or empty it stops and names the command to run.
A run takes about 0.1 to 0.4 s even on the largest ego (251 nodes).

**Check:** open http://localhost:8000/docs (Swagger UI) and try each endpoint:

| Endpoint | What it does |
|---|---|
| `GET /api/egos` | 973 egos: id, nodes, edges, density, circles |
| `GET /api/stats` | the EDA numbers from `outputs/eda_stats.json` |
| `POST /api/detect` | body `{"ego": 170729553, "mode": "hybrid", "direction": "mutual", "runs": 5}`. One ego per run. Returns a `job_id`, the metrics (always with `circle_f1_one_community` and `largest_share`; `circle_f1` is `null` for egos without circles) and `warnings` |
| `GET /api/graph/{job_id}` | nodes (id, community, degree) and edges (source, target, weight) |
| `GET /api/communities/{job_id}` | per community: size, purity, top hashtags and mentions |
| `GET /api/trace/{job_id}` | one label per node for every iteration, plus how many nodes changed |
| `GET /api/compare` | the saved top-20 summaries: 3 modes x 3 algorithms, for `any` and `mutual` follows |
| `GET /api/users/{user_id}?job_id=` | the user's tokens; with a `job_id` also their community and the most similar users **in that run's ego** |

Defaults: `direction=mutual` (the EDA found `any` collapses into one giant community), `runs=10`, `seed=42`.
Limits: `runs` and `k` 1 to 50, `min_sim` 0 to 1; a bad value is a 422. An unknown ego, job or user is a 404, and
every error has the shape `{"error": ..., "detail": ...}`. The newest 50 runs are kept in memory; an older `job_id`
returns 404 ("run the detection again"). `warnings` flags a graph with no edges, a run that did not converge, a
largest community above 70% of users, and more than 30% of users without tokens.
Interest purity in the response always uses the full token set (hashtags and mentions), whatever `tokens` the run used.

---

## 12. Start the frontend ✅

In a second terminal (the API from step 11 must be running):

```bash
cd frontend
npm install        # first time only
npm run dev
```

Open http://localhost:5173. It must be port 5173: the API only allows that origin (and `127.0.0.1:5173`).
The frontend expects the API at `http://localhost:8000` (copy `frontend/.env.example` to `frontend/.env` and set
`VITE_API_URL` to change it).

`npm run typecheck` checks the types and `npm run build` makes a production build. The API types in
`frontend/src/api/schema.d.ts` are **generated** from the backend: after any change to `backend/schemas.py`,
run `npm run gen:api` (it needs `data/processed/` and the project venv at `../.venv`) and fix the type errors it reveals.

**Check, page by page** (there are no frontend unit tests, so this list is the test):

1. Explorer: dataset stats load, egos sort by column, a row click selects an ego and opens Run
2. Run: changing the graph mode and pressing Run returns metric cards; warnings show in a yellow box; an ego without circles shows Circle F1 "n/a"
3. Graph: users coloured by community (12 colours, grey "other"); hovering a user shows their interests
4. Communities: one card per community with its top hashtags and mentions; a checkbox reveals users who are alone
5. Step-through: Play animates label propagation; the slider jumps to an iteration
6. Compare: chart and table show all graph modes and algorithms, with the one-community baseline marked
7. User lookup: a user id shows their interests, community in the selected run and similar users

Failure cases to try: stop the API (a red banner appears within about 10 s and screens say "The API is not reachable");
restart the API and reload `/graph?job=<old id>` ("Not found … run the detection again" with a link).

**Limitations:**

- The Run screen has no tuned preset, because `POST /api/detect` does not accept the three outlier rules the tuned configuration uses.
- Compare shows the saved top-20 experiment summaries, not the held-out final evaluation (`outputs/final/`), which has no endpoint yet.

---

## 13. Run the tests ✅

```bash
pytest -q          # 91 tests, about 2 s; no dataset needed
```

All of these are covered (see the test table in [`CODE_GUIDE.md`](CODE_GUIDE.md)):

- LPA on two disconnected triangles returns exactly 2 communities
- LPA with a fixed seed gives identical output twice
- preprocess merges the same hashtag from two egos into one token
- ego -> node edges are added for every node in `.edges`
- circle F1 = 1.0 when communities equal circles
- junk tokens (`#...`, `#1`) are removed and `#Python` / `#python` merge
- an ego with an empty `.circles` file gets F1 = `NaN`, not 0
- `follows.parquet` has no duplicate (src, dst) pairs
- `@mashable:` and `@mashable` merge into one token
- the train/val/test split is deterministic, disjoint and covers every ego with circles
- every user falls into exactly one data-quality bucket
- outlier rules drop common tokens, scale only hub edges, keep each user's rarest tokens, and are no-ops when off
- an infeasible config ranks below any feasible one; tuning refuses test egos
- evaluate refuses leaked configs, the wrong ego set, and overwriting earlier results

`pyproject.toml` (step 1) sets `pythonpath = ["."]`, so tests can `import backend` and `import pipeline`.

---

## 14. Deploy (optional) 🛠

| Part | Option |
|---|---|
| Backend | Render or Railway: start command `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` |
| Frontend | Vercel: build `npm run build`, output `dist/`, set `VITE_API_URL` to the backend URL |
| Local demo | `docker compose up` (backend + frontend together). Needs `backend/Dockerfile`, `frontend/Dockerfile` and `docker-compose.yml`, which are not written yet. |

Ship `data/processed/` with the backend (not `data/raw/`) to keep the deploy small.

---

## 15. Demo script (for viva / presentation)

1. Explorer: show dataset size and why Twitter + hashtags fit the topic.
2. Run hybrid mode on 5 egos.
3. Graph view: point out one clear community and its hashtags.
4. Step-through: play LPA from unique labels to stable communities.
5. Compare: follow vs interest vs hybrid, and LPA vs Louvain.
6. User lookup: pick a user, show their community and similar users.

---

## Quick reference: full run order

```bash
source venv/bin/activate
python -m pipeline.download                      # step 2
python -m pipeline.preprocess                    # step 3
python -m pipeline.experiments --egos top20 --seed 42   # step 5
python -m pipeline.experiments --egos all --runs 10 --seed 42 --out-dir outputs/all --no-figures   # step 6 (10-15 min)
python -m pipeline.data_quality                  # step 6
python -m pipeline.split                         # step 7 (once; refuses to overwrite)
python -m pipeline.tune                          # step 9 (about 10 min)
python -m pipeline.evaluate                      # step 10 (once; scores the test egos)
python -m pipeline.analyze --which all           # step 10b (about 10 min)
uvicorn backend.main:app --reload --port 8000    # step 11 (terminal 1)
cd frontend && npm run dev                       # step 12 (terminal 2)
pytest -q                                        # step 13
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Download fails or stalls | Re-run the script (it skips finished files) or use the manual fallback |
| `Ego networks found` is not 973 | Delete `data/raw/twitter/` and re-run so it re-extracts |
| LPA returns one giant community | Use `--direction mutual`, lower `--k`, or raise `--min-sim` |
| LPA returns hundreds of tiny communities | Raise `--k`, lower `--min-sim`, or use hybrid mode |
| Browser freezes on the graph page | Select fewer egos (start with 1–5) |
| Frontend shows "network error" | Check the backend is running and `VITE_API_URL` is correct |
| Results change every run | Pass `--seed 42`; LPA is randomised by design. Use `asyn_lpa_communities`, not `label_propagation_communities` (no seed) |
| `ModuleNotFoundError: No module named 'backend'` | Run from the project root with `python -m pipeline.<name>`, not `python pipeline/<name>.py`; check `__init__.py` files exist |
| Combined count shows 2,420,766, not 1,768,149 | Expected: that is the line count with duplicates. Unique edges should be 1,768,149 |
| `follows.parquet` has 1,762,504 edges, not 1,768,149 | Expected: per-ego files rebuild slightly fewer edges than `twitter_combined.txt` |
| Circle F1 is 0 or missing for some egos | 35 egos have no circles; they should show `NaN` and be excluded from averages |
| `ValueError: test egos must not be used for tuning` | A test ego was passed to the tuning code. Fix the ego list; do not regenerate the split casually, because a new split invalidates the tuning already done. `split.py` refuses to overwrite `split.json` without `--force` |
| `FileExistsError ... scored once` | `pipeline.evaluate` refuses to overwrite `outputs/final/metrics.csv` or `summary.csv`: the test set is meant to be scored once. Use `--force` only if you accept that the result is no longer a clean single test |
| Top tokens show `#...`, `#1` or `@name:` variants | Token cleaning (`normalize_token`) was not applied; re-run `python -m pipeline.preprocess` |

---

## Neovim (AstroNvim) shortcuts for working through these steps

| Shortcut | Use |
|---|---|
| `<Space>ff` | Open a file by name (e.g. `preprocess.py`) |
| `<Space>fw` | Search text across the repo (e.g. `featnames`) |
| `<Space>e` | Toggle the file explorer |
| `<Space>tf` | Floating terminal to run the commands above |
| `gd` / `<C-o>` | Jump to a function's definition / jump back |
| `K` | Show docs for the symbol under the cursor |
| `]d` / `[d` | Next / previous error from the language server |
| `gx` | Open the URL under the cursor in the browser |

---

## Citation

J. McAuley and J. Leskovec. *Learning to Discover Social Circles in Ego Networks.* NIPS, 2012.
Raghavan, U. N., Albert, R., & Kumara, S. (2007). Near linear time algorithm to detect community structures in large-scale networks. *Physical Review E*, 76(3), 036106.
