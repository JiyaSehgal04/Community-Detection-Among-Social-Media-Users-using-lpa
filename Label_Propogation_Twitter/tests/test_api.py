import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend import service
from backend.app import create_app

DETECT = {"ego": 100, "mode": "hybrid", "direction": "any", "k": 3, "min_sim": 0.1, "runs": 5}
MODES, ALGORITHMS = ("follow", "interest", "hybrid"), ("lpa", "nx_lpa", "louvain")


@pytest.fixture
def client(tiny_dataset, tmp_path):
    rows = [{"mode": m, "algorithm": a, "circle_f1": 0.5, "interest_purity": 0.5, "largest_share": 0.4}
            for m in MODES for a in ALGORITHMS]
    (tmp_path / "mutual").mkdir()
    pd.DataFrame(rows).to_csv(tmp_path / "summary_table.csv", index=False)
    pd.DataFrame(rows).to_csv(tmp_path / "mutual" / "summary_table.csv", index=False)
    (tmp_path / "eda_stats.json").write_text(json.dumps({"users": 9}))
    return TestClient(create_app(data=tiny_dataset, outputs_dir=tmp_path))


def _detect(client) -> dict:
    response = client.post("/api/detect", json=DETECT)
    assert response.status_code == 200
    return response.json()


def test_detect_returns_a_job_and_the_honest_metrics(client):
    body = _detect(client)
    assert isinstance(body["job_id"], str)
    assert {"circle_f1_one_community", "largest_share"} <= set(body["metrics"])
    assert body["metrics"]["circle_f1_one_community"] == pytest.approx(2 / 3)
    assert body["warnings"] == []


def test_graph_has_every_ego_node_once_with_one_community_and_valid_edges(client):
    graph = client.get(f"/api/graph/{_detect(client)['job_id']}").json()
    ids = [n["id"] for n in graph["nodes"]]
    assert sorted(ids) == [1, 2, 3, 4, 5, 6, 100]
    assert all(isinstance(n["community"], int) for n in graph["nodes"])
    assert all(e["source"] in ids and e["target"] in ids for e in graph["edges"])


def test_community_sizes_sum_to_the_node_count(client):
    communities = client.get(f"/api/communities/{_detect(client)['job_id']}").json()["communities"]
    assert sum(c["size"] for c in communities) == 7


@pytest.mark.parametrize("patch", [{"mode": "foo"}, {"runs": 0}, {"runs": 999}])
def test_bad_input_is_a_422(client, patch):
    response = client.post("/api/detect", json={**DETECT, **patch})
    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_unknown_ego_job_and_user_are_404_with_the_standard_shape(client):
    for response in (client.post("/api/detect", json={**DETECT, "ego": 9999}),
                     client.get("/api/graph/nope"), client.get("/api/users/99999")):
        assert response.status_code == 404
        assert set(response.json()) == {"error", "detail"}


def test_user_lookup_stays_inside_the_runs_ego(client):
    job_id = _detect(client)["job_id"]
    body = client.get(f"/api/users/1?job_id={job_id}").json()
    assert isinstance(body["community"], int)
    assert {s["user_id"] for s in body["similar_users"]} <= {2, 3, 4, 5, 6, 100}
    assert 2 in {s["user_id"] for s in body["similar_users"]}  # shares #a with user 1
    outsider = client.get(f"/api/users/7?job_id={job_id}").json()  # user 7 belongs to ego 200
    assert outsider["community"] is None and outsider["similar_users"] == []


def test_startup_fails_with_a_message_naming_the_step_to_run(tmp_path):
    with pytest.raises(service.DataError, match="pipeline.preprocess"):
        create_app(data_dir=tmp_path)  # empty directory


def test_egos_and_compare_endpoints(client):
    egos = client.get("/api/egos").json()["egos"]
    assert sorted(e["ego_id"] for e in egos) == [100, 200]
    rows = client.get("/api/compare").json()["rows"]
    assert {(r["mode"], r["algorithm"]) for r in rows} == {(m, a) for m in MODES for a in ALGORITHMS}
    assert {r["direction"] for r in rows} == {"any", "mutual"}
