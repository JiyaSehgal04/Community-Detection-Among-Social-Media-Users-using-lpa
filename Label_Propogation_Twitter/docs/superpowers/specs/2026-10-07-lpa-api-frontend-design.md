# LPA Twitter Communities: backend API and frontend (design)

Date: 2026-10-07 · Status: draft for review · Scope: steps 11 and 12 of `EXECUTION_STEPS.md`

Note: the work in 2026-10-08-full-dataset-tuning-design.md was done first; EXECUTION_STEPS.md steps for the API and frontend are now 11 and 12.

## 1. Goal and scope

Add a FastAPI service over the existing `backend/` code and a web UI that shows how Label
Propagation finds communities in the SNAP ego-Twitter data. It is run locally for a viva demo.

**In scope:** backend API, 7-screen frontend, the tests listed in section 7.
**Out of scope:** deploy, Docker, auth, rate limiting, a database, a job queue, CI, end-to-end
(Playwright) tests, frontend unit tests, the final written report.

**Success criteria**
- `uvicorn backend.main:app` and `npm run dev` start the demo; every endpoint in section 3 works.
- A run on any ego returns in well under a second and renders in all screens.
- The UI never presents a collapsed or non-converged result as a success (section 5).
- `pytest -q` passes (existing 74 tests plus the new ones) with no dataset needed.

**Assumptions (not stated by the user):** a viva/demo audience; the UI works on one ego network
per run (a multi-ego graph is unreadable); `direction` defaults to `mutual`, as the EDA concluded.

## 2. Architecture

```
frontend/ (Vite + React + TS)  --HTTP/JSON-->  backend/main.py (FastAPI routes only)
                                                  |
                                       backend/service.py (pure functions, no FastAPI)
                                                  |
                          backend/graphs.py · lpa.py · metrics.py  (existing, unchanged)
```

New files: `backend/main.py`, `backend/service.py`, `backend/schemas.py`, `tests/conftest.py`,
`tests/test_service.py`, `tests/test_api.py`, and `frontend/`.
No changes to `lpa.py`, `graphs.py`, `metrics.py`. `lpa.label_propagation(trace=True)` already
records one label snapshot per iteration.

- `main.py` loads `graphs.Dataset` once at startup from `data/processed/` (override with
  `LPA_DATA_DIR`) and allows CORS from `http://localhost:5173`.
- `service.py` holds the logic: running a detection, shaping graph/community/trace payloads,
  computing warnings, the job store, the user lookup. It imports no FastAPI.
- `schemas.py` holds the pydantic request/response models. They define the OpenAPI schema.

**Jobs.** `POST /api/detect` runs synchronously (an ego has at most 251 nodes). The result is
stored in an in-memory dict under a `job_id`: the graph, the best-run labels, the trace, the
metrics. At most 50 jobs are kept; the oldest is evicted. The best run is chosen by
`best_of_runs`; the trace comes from re-running the best seed with `trace=True`
(deterministic).

## 3. API

| Endpoint | Returns |
|---|---|
| `GET /api/egos` | all egos: id, n_nodes, n_edges, density, has_circles |
| `GET /api/stats` | the EDA numbers from `outputs/eda_stats.json` |
| `POST /api/detect` | body `{ego, mode, direction="mutual", tokens="both", k=10, min_sim=0.2, follow_weight=0.3, runs=10, seed=42}`. Returns `job_id`, `metrics` (modularity, follow_modularity, circle_f1, circle_f1_one_community, interest_purity, largest_share, coverage, n_communities, iterations, converged) and `warnings` |
| `GET /api/graph/{job_id}` | nodes (id, community, degree) and edges (source, target, weight) |
| `GET /api/communities/{job_id}` | per community: size, top hashtags and mentions, purity |
| `GET /api/trace/{job_id}` | one label array per iteration (indexed by node order) and the number of nodes changed per iteration |
| `GET /api/compare` | rows from the saved `summary_table.csv` for `direction` any and mutual: 3 modes x 3 algorithms |
| `GET /api/users/{user_id}?job_id=` | the user's tokens, their community in that job, and the most similar users (TF-IDF cosine) **in the same ego** |

`circle_f1` is `null` (not 0) for egos without circles. `circle_f1_one_community` is always returned.

## 4. Frontend

- **Stack:** Vite, React, TypeScript, React Router, in `frontend/`. API types are generated from
  the backend's OpenAPI schema with `openapi-typescript`. `VITE_API_URL` sets the API base
  (default `http://localhost:8000`).
- **Fetching:** a small typed `useApi` hook around `fetch` with loading, error and empty states.
  No state library.
- **Graph:** Cytoscape.js; 12 distinct community colours plus grey for "other"; hover shows top tokens.
- **Charts:** Recharts, on Compare only.
- **Screens**
  1. **Explorer:** EDA numbers and a sortable ego table; choosing an ego selects it.
  2. **Run:** form (mode, direction default `mutual`, tokens, k, min_sim, follow_weight, runs); metric
     cards and the warnings banner.
  3. **Graph:** the coloured network.
  4. **Communities:** a card per community with size and top hashtags.
  5. **Step-through:** play/pause and a slider that recolours the graph through the trace snapshots.
  6. **Compare:** circle F1 bars with the one-community baseline marked, plus `largest_share`.
  7. **User lookup:** tokens, community and similar users.

## 5. Reliability

| Risk | Guard |
|---|---|
| Missing/stale/invalid data | At startup, check each parquet file exists, is non-empty and has the expected columns; fail naming the step to run (`python -m pipeline.preprocess`) |
| Bad requests | Pydantic bounds: `runs` 1-50, `k` 1-50, `min_sim` in [0, 1], `mode` and `direction` as enums. One error shape: `{error, detail}`; invalid input is 422, unknown ego/job/user is 404 |
| Algorithm edge cases | `warnings` in the detect response: not converged within `max_iter`; `largest_share > 0.7` (likely collapsed; try `mutual`); more than 30% of users have no tokens (interest/hybrid modes). An ego with no edges or no tokens returns a valid result (every user alone), not a 500 |
| Memory | Job store capped at 50, oldest evicted |
| Frontend/backend drift | Generated TypeScript types: a changed field fails the frontend build |
| UI failure states | Every screen has loading, error and empty states; an error boundary and an "API unreachable" banner |
| Expired job | 404 from the API; the UI shows "this run expired, run again" with a link to Run |

## 6. Data flow

1. Explorer: pick an ego.
2. Run: `POST /api/detect` returns a `job_id`, kept in app state and the URL (`?job=...`).
3. Graph, Communities, Step-through and User lookup fetch by `job_id`.
4. Compare needs no job; it reads the saved summary results.

## 7. Tests (necessary only, written first)

Shared fixture: the tiny `Dataset` from `tests/test_graphs.py`, moved to `tests/conftest.py`.

`tests/test_service.py`
1. Warnings: collapsed (`largest_share > 0.7`) warns; `max_iter=1` gives "not converged"; a normal run has none.
2. An ego with no edges or tokens returns a valid result, no exception.
3. The 51st job evicts the oldest; the newest stays reachable.
4. Trace: the first snapshot has every node under its own label; the last equals the final labels.

`tests/test_api.py` (FastAPI `TestClient`)
5. `POST /api/detect` returns a `job_id` and metrics including `circle_f1_one_community` and `largest_share`.
6. `/api/graph`: every ego node appears once with exactly one community; every edge endpoint is a node.
7. `/api/communities`: community sizes sum to the node count.
8. Bad input (`mode="foo"`, `runs=0`, `runs=999`) returns 422.
9. Unknown ego, job or user returns 404 with the `{error, detail}` shape.
10. `/api/users/{id}`: community and similar users, all from the same ego.
11. Startup with a missing or empty data directory fails with a message naming the step to run.
12. `/api/egos` returns one row per ego; `/api/compare` returns rows for each mode and algorithm.

No frontend unit tests: type generation and the build guard schema drift; the screens are checked
manually with the step 12 checklist in `EXECUTION_STEPS.md`.

## 8. Documentation updates

Update `EXECUTION_STEPS.md` (steps 11 and 12 from 🛠 to ✅, `/api/detect` takes one `ego`) and
`CODE_GUIDE.md` (new files) when the work lands. Add `frontend/` build artifacts to `.gitignore`.
