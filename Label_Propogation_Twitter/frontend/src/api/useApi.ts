import { useEffect, useState } from "react";
import { ApiError } from "./client";

export interface ApiResult<T> {
  data: T | null;
  error: ApiError | null;
  loading: boolean;
  reload: () => void;
}

/** Run `load` whenever `deps` change; tracks loading and error state. */
export function useApi<T>(load: () => Promise<T>, deps: unknown[]): ApiResult<T> {
  const [state, setState] = useState<{ data: T | null; error: ApiError | null; loading: boolean }>({
    data: null,
    error: null,
    loading: true,
  });
  const [tick, setTick] = useState(0);

  useEffect(() => {
    let alive = true;
    setState((s) => ({ ...s, loading: true, error: null }));
    load().then(
      (data) => alive && setState({ data, error: null, loading: false }),
      (e) =>
        alive &&
        setState({ data: null, error: e instanceof ApiError ? e : new ApiError(0, "unknown", String(e)), loading: false }),
    );
    return () => {
      alive = false;
    };
  }, [...deps, tick]); // eslint-disable-line react-hooks/exhaustive-deps

  return { ...state, reload: () => setTick((t) => t + 1) };
}
