"""FastAPI app: routes only. The logic lives in backend/service.py."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend import graphs, schemas, service

ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
NOT_FOUND = {404: {"model": schemas.ErrorResponse}}
ROOT = Path(__file__).resolve().parent.parent  # the project folder, wherever the server is started from


def create_app(data: graphs.Dataset | None = None, data_dir: Path | str | None = None,
               outputs_dir: Path | str | None = None) -> FastAPI:
    """Build the API. Loads data/processed/ once (override with LPA_DATA_DIR) unless `data` is given."""
    if data is None:
        directory = Path(data_dir or os.environ.get("LPA_DATA_DIR", ROOT / "data" / "processed"))
        service.validate_data_dir(directory)  # raises DataError naming the step to run
        data = graphs.Dataset.load(directory)
    outputs = Path(outputs_dir or os.environ.get("LPA_OUTPUTS_DIR", ROOT / "outputs"))
    data.tfidf("both")  # warm the TF-IDF cache so the first request is not slow

    app = FastAPI(title="LPA Twitter communities", version="0.1.0")
    app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["*"], allow_headers=["*"])
    jobs = service.JobStore()

    @app.exception_handler(service.NotFound)
    async def _not_found(_: Request, exc: service.NotFound) -> JSONResponse:
        return JSONResponse(status_code=404, content={"error": "not_found", "detail": str(exc)})

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, exc: RequestValidationError) -> JSONResponse:
        detail = [{"field": ".".join(str(p) for p in e["loc"][1:]), "message": e["msg"]} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"error": "validation_error", "detail": detail})

    @app.get("/api/egos", response_model=schemas.EgosResponse)
    def egos():
        return service.egos_payload(data)

    @app.get("/api/stats", response_model=schemas.StatsResponse, responses=NOT_FOUND)
    def stats():
        return service.stats_payload(outputs)

    @app.post("/api/detect", response_model=schemas.DetectResponse, responses=NOT_FOUND)
    def detect(request: schemas.DetectRequest):
        job = service.run_detection(data, request.ego, mode=request.mode, direction=request.direction,
                                    tokens=request.tokens, k=request.k, min_sim=request.min_sim,
                                    follow_weight=request.follow_weight, runs=request.runs, seed=request.seed)
        job.job_id = uuid.uuid4().hex
        jobs.add(job.job_id, job)
        return {"job_id": job.job_id, "ego": job.ego, "params": job.params,
                "metrics": job.metrics, "warnings": job.warnings}

    @app.get("/api/graph/{job_id}", response_model=schemas.GraphResponse, responses=NOT_FOUND)
    def graph(job_id: str):
        return service.graph_payload(jobs.get(job_id))

    @app.get("/api/communities/{job_id}", response_model=schemas.CommunitiesResponse, responses=NOT_FOUND)
    def communities(job_id: str):
        return service.communities_payload(data, jobs.get(job_id))

    @app.get("/api/trace/{job_id}", response_model=schemas.TraceResponse, responses=NOT_FOUND)
    def trace(job_id: str):
        return service.trace_payload(jobs.get(job_id))

    @app.get("/api/compare", response_model=schemas.CompareResponse)
    def compare():
        return service.compare_payload(outputs)

    @app.get("/api/users/{user_id}", response_model=schemas.UserResponse, responses=NOT_FOUND)
    def user(user_id: int, job_id: str | None = None):
        job = jobs.get(job_id) if job_id else None
        return service.user_payload(data, user_id, job)

    return app
