import type { components } from "./schema";
import type { DetectRequest, DetectResponse, GraphResponse, TraceResponse, UserResponse } from "./types";

export const API_URL: string = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

/** Any failed call. status 0 means the API could not be reached at all. */
export class ApiError extends Error {
  constructor(public status: number, public kind: string, public detail: unknown) {
    super(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(API_URL + path, init);
  } catch {
    throw new ApiError(0, "unreachable", `Cannot reach the API at ${API_URL}. Is it running?`);
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new ApiError(res.status, body?.error ?? "error", body?.detail ?? res.statusText);
  }
  return (await res.json()) as T;
}

const userCache = new Map<string, UserResponse>();

export const api = {
  egos: () => request<components["schemas"]["EgosResponse"]>("/api/egos"),
  stats: () => request<components["schemas"]["StatsResponse"]>("/api/stats"),
  detect: (body: DetectRequest) =>
    request<DetectResponse>("/api/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  graph: (job: string) => request<GraphResponse>(`/api/graph/${job}`),
  communities: (job: string) => request<components["schemas"]["CommunitiesResponse"]>(`/api/communities/${job}`),
  trace: (job: string) => request<TraceResponse>(`/api/trace/${job}`),
  compare: () => request<components["schemas"]["CompareResponse"]>("/api/compare"),
  user: (id: number, job?: string | null) =>
    request<UserResponse>(`/api/users/${id}` + (job ? `?job_id=${job}` : "")),
  /** Same as user(), remembered for the session (used by graph hover). */
  async userCached(id: number, job?: string | null): Promise<UserResponse> {
    const key = `${id}:${job ?? ""}`;
    const hit = userCache.get(key);
    if (hit) return hit;
    const fresh = await api.user(id, job);
    userCache.set(key, fresh);
    return fresh;
  },
  /** True when the API answers at all (used by the "API unreachable" banner). */
  async ping(): Promise<boolean> {
    try {
      await fetch(API_URL + "/openapi.json");
      return true;
    } catch {
      return false;
    }
  },
};
