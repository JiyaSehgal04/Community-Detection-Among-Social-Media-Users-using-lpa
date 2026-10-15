"""Request and response models. They define the OpenAPI schema the frontend types come from."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DetectRequest(BaseModel):
    ego: int
    mode: Literal["follow", "interest", "hybrid"] = "hybrid"
    direction: Literal["any", "mutual"] = "mutual"
    tokens: Literal["hashtag", "mention", "both"] = "both"
    k: int = Field(10, ge=1, le=50)
    min_sim: float = Field(0.2, ge=0.0, le=1.0)
    follow_weight: float = Field(0.3, ge=0.0, le=10.0)
    runs: int = Field(10, ge=1, le=50)
    seed: int = 42


class Metrics(BaseModel):
    modularity: float | None
    follow_modularity: float | None
    circle_f1: float | None  # null (not 0) for egos without circles
    circle_f1_one_community: float | None  # always read circle F1 next to this baseline
    interest_purity: float | None
    n_communities: int
    n_singletons: int
    coverage: float | None
    largest_share: float | None
    iterations: int
    converged: bool
    n_nodes: int
    n_edges: int


class DetectResponse(BaseModel):
    job_id: str
    ego: int
    params: dict[str, Any]
    metrics: Metrics
    warnings: list[str]


class GraphNode(BaseModel):
    id: int
    community: int
    degree: int


class GraphEdge(BaseModel):
    source: int
    target: int
    weight: float


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class TokenCount(BaseModel):
    token: str
    count: int


class CommunityCard(BaseModel):
    id: int
    size: int
    purity: float | None
    top_hashtags: list[TokenCount]
    top_mentions: list[TokenCount]


class CommunitiesResponse(BaseModel):
    communities: list[CommunityCard]


class TraceResponse(BaseModel):
    nodes: list[int]
    snapshots: list[list[int]]  # one label per node, in `nodes` order, per iteration
    changed: list[int]          # nodes whose label changed in each iteration


class UserToken(BaseModel):
    token: str
    type: str


class SimilarUser(BaseModel):
    user_id: int
    similarity: float
    community: int


class UserResponse(BaseModel):
    user_id: int
    egos: list[int]
    tokens: list[UserToken]
    community: int | None
    similar_users: list[SimilarUser]


class EgoRow(BaseModel):
    ego_id: int
    n_nodes: int
    n_edges: int
    density: float
    has_circles: bool
    n_circles: int


class EgosResponse(BaseModel):
    egos: list[EgoRow]


class CompareResponse(BaseModel):
    rows: list[dict[str, Any]]


class StatsResponse(BaseModel):
    stats: dict[str, Any]


class ErrorResponse(BaseModel):
    error: str
    detail: Any
