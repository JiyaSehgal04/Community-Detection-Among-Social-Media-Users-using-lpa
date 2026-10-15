import { useSearchParams } from "react-router-dom";

/** The chosen ego and the latest run live in the URL (?ego=..&job=..), so a refresh or a shared link keeps them. */
export function useSelection() {
  const [params, setParams] = useSearchParams();
  const rawEgo = Number(params.get("ego"));
  const ego = params.get("ego") && Number.isFinite(rawEgo) ? rawEgo : null;
  const job = params.get("job");

  function update(next: { ego?: number | null; job?: string | null }) {
    const p = new URLSearchParams(params);
    for (const [key, value] of Object.entries(next)) {
      if (value === null || value === undefined) p.delete(key);
      else p.set(key, String(value));
    }
    setParams(p);
  }
  return { ego, job, update };
}
