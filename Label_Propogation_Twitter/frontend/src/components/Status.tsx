import type { ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";
import type { ApiError } from "../api/client";
import type { ApiResult } from "../api/useApi";

export function ErrorBox({ error, onRetry }: { error: ApiError; onRetry?: () => void }) {
  const { search } = useLocation();
  const unreachable = error.status === 0;
  return (
    <div className="box error" role="alert">
      <strong>{unreachable ? "The API is not reachable" : error.status === 404 ? "Not found" : "Something went wrong"}</strong>
      <p>{error.message}</p>
      {unreachable && <p className="muted">Start it with: <code>uvicorn backend.main:app --port 8000</code></p>}
      {error.status === 404 && String(error.detail).includes("expired job") && (
        <p>
          The server keeps only the newest 50 runs. <Link to={{ pathname: "/run", search }}>Run the detection again</Link>.
        </p>
      )}
      {onRetry && <button onClick={onRetry}>Try again</button>}
    </div>
  );
}

/** Renders loading and error states for an API call, then the content. */
export function Query<T>({ result, children }: { result: ApiResult<T>; children: (data: T) => ReactNode }) {
  if (result.error) return <ErrorBox error={result.error} onRetry={result.reload} />;
  if (result.loading || !result.data) return <p className="muted">Loading…</p>;
  return <>{children(result.data)}</>;
}

export function NeedJob() {
  const { search } = useLocation();
  return (
    <div className="box">
      <p>No run selected yet. <Link to={{ pathname: "/run", search }}>Run the detection</Link> on an ego network first.</p>
    </div>
  );
}
