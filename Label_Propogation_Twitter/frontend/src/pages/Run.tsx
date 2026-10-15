import { useEffect, useRef, useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { ApiError, api } from "../api/client";
import type { DetectRequest, DetectResponse } from "../api/types";
import { ErrorBox } from "../components/Status";
import { MetricCards } from "../components/MetricCards";
import { useSelection } from "../lib/selection";

export function Run() {
  const { ego: selectedEgo, job, update } = useSelection();
  const [ego, setEgo] = useState<string>(selectedEgo ? String(selectedEgo) : "");
  const [mode, setMode] = useState<DetectRequest["mode"]>("hybrid");
  const [direction, setDirection] = useState<DetectRequest["direction"]>("mutual");
  const [tokens, setTokens] = useState<DetectRequest["tokens"]>("both");
  const [k, setK] = useState(10);
  const [minSim, setMinSim] = useState(0.2);
  const [followWeight, setFollowWeight] = useState(0.3);
  const [runs, setRuns] = useState(10);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [result, setResult] = useState<DetectResponse | null>(null);
  // If the user leaves this screen while a run is in flight, do not pull them back by rewriting the URL.
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const res = await api.detect({
        ego: Number(ego), mode, direction, tokens, k, min_sim: minSim, follow_weight: followWeight, runs, seed: 42,
      });
      if (!mounted.current) return;
      setResult(res);
      update({ ego: res.ego, job: res.job_id });
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(0, "unknown", String(err)));
    } finally {
      setBusy(false);
    }
  }

  const search = result ? `?ego=${result.ego}&job=${result.job_id}` : "";
  return (
    <>
      <h2>Run community detection</h2>
      <form onSubmit={submit} className="form">
        <label>Ego id <input value={ego} onChange={(e) => setEgo(e.target.value)} inputMode="numeric" required pattern="[0-9]+" /></label>
        <label>Graph
          <select value={mode} onChange={(e) => setMode(e.target.value as DetectRequest["mode"])}>
            <option value="hybrid">hybrid (interests + follows)</option>
            <option value="interest">interest only</option>
            <option value="follow">follow only</option>
          </select>
        </label>
        <label>Follows
          <select value={direction} onChange={(e) => setDirection(e.target.value as DetectRequest["direction"])}>
            <option value="mutual">mutual only (recommended)</option>
            <option value="any">either direction</option>
          </select>
        </label>
        <label>Interest tokens
          <select value={tokens} onChange={(e) => setTokens(e.target.value as DetectRequest["tokens"])}>
            <option value="both">hashtags and mentions</option>
            <option value="mention">mentions only</option>
            <option value="hashtag">hashtags only</option>
          </select>
        </label>
        <label>Neighbours k <input type="number" min={1} max={50} value={k} onChange={(e) => setK(Number(e.target.value))} /></label>
        <label>Min similarity <input type="number" min={0} max={1} step={0.05} value={minSim} onChange={(e) => setMinSim(Number(e.target.value))} /></label>
        <label>Follow weight <input type="number" min={0} max={10} step={0.05} value={followWeight} onChange={(e) => setFollowWeight(Number(e.target.value))} /></label>
        <label>Runs <input type="number" min={1} max={50} value={runs} onChange={(e) => setRuns(Number(e.target.value))} /></label>
        <button type="submit" disabled={busy}>{busy ? "Running…" : "Run"}</button>
      </form>

      {error && <ErrorBox error={error} />}
      {result && (
        <>
          {result.warnings.length > 0 && (
            <div className="box warn" role="status">
              <strong>Check this result</strong>
              <ul>{result.warnings.map((w) => <li key={w}>{w}</li>)}</ul>
            </div>
          )}
          <MetricCards m={result.metrics} />
          <p>
            See it: <Link to={{ pathname: "/graph", search }}>Graph</Link> · <Link to={{ pathname: "/communities", search }}>Communities</Link> ·{" "}
            <Link to={{ pathname: "/steps", search }}>Step-through</Link> · <Link to={{ pathname: "/users", search }}>User lookup</Link>
          </p>
        </>
      )}
      {!result && job && <p className="muted">A run is selected (see the other screens). Press Run to see its metrics again.</p>}
    </>
  );
}
