import pytest

from backend import service

PLANTED = dict(mode="hybrid", direction="any", k=3, min_sim=0.1, runs=5)


def test_warnings_for_collapse_non_convergence_and_clean_run(tiny_dataset):
    collapsed = service.compute_warnings({"largest_share": 0.8, "converged": True}, max_iter=100,
                                         no_token_share=0.0, edgeless=False, mode="hybrid")
    assert any("collapsed" in w for w in collapsed)

    stopped = service.run_detection(tiny_dataset, 100, max_iter=1, **PLANTED)
    assert any("converge" in w for w in stopped.warnings)

    clean = service.run_detection(tiny_dataset, 100, **PLANTED)
    assert clean.warnings == []


def test_ego_without_edges_or_tokens_gives_a_valid_result(tiny_dataset):
    # ego 200 under mutual follows has no edges; it also has no circles
    job = service.run_detection(tiny_dataset, 200, mode="follow", direction="mutual")
    assert len(job.labels) == 2 and all(len(c) == 1 for c in job.communities)
    assert job.metrics["circle_f1"] is None  # no circles: null, not 0
    assert any("no edges" in w for w in job.warnings)
    # interest mode on the same ego: fewer than two users have tokens, so no edges either
    assert service.run_detection(tiny_dataset, 200, mode="interest").metrics["n_communities"] == 0


def test_job_store_evicts_the_oldest_job_at_the_cap():
    store = service.JobStore()
    for i in range(service.MAX_JOBS + 1):
        store.add(f"j{i}", object())
    with pytest.raises(service.NotFound):
        store.get("j0")
    assert store.get(f"j{service.MAX_JOBS}") is not None
    assert store.get("j1") is not None


def test_trace_starts_with_unique_labels_and_ends_with_the_final_labels(tiny_dataset):
    job = service.run_detection(tiny_dataset, 100, **PLANTED)
    payload = service.trace_payload(job)
    assert payload["snapshots"][0] == payload["nodes"]  # every node under its own label
    assert payload["snapshots"][-1] == [job.labels[n] for n in payload["nodes"]]
