import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api/client";
import type { EgoRow } from "../api/types";
import { useApi } from "../api/useApi";
import { Query } from "../components/Status";
import { int, num, pct } from "../lib/format";
import { useSelection } from "../lib/selection";

const STAT_CARDS: [key: string, label: string, kind: "int" | "pct"][] = [
  ["users", "Users", "int"],
  ["follow_edges", "Follow edges", "int"],
  ["egos", "Ego networks", "int"],
  ["n_circles", "Circles", "int"],
  ["reciprocity", "Follows that are mutual", "pct"],
  ["share_with_hashtag", "Users with a hashtag", "pct"],
  ["share_with_mention", "Users with a mention", "pct"],
  ["share_with_no_tokens", "Users with no tokens", "pct"],
];

type SortKey = "ego_id" | "n_nodes" | "n_edges" | "density" | "n_circles";
const COLUMNS: [SortKey, string][] = [
  ["ego_id", "Ego"],
  ["n_nodes", "Users"],
  ["n_edges", "Follows"],
  ["density", "Density"],
  ["n_circles", "Circles"],
];
const PAGE = 100;

function EgoTable({ egos }: { egos: EgoRow[] }) {
  const navigate = useNavigate();
  const { ego: selected } = useSelection();
  const [sortKey, setSortKey] = useState<SortKey>("n_nodes");
  const [desc, setDesc] = useState(true);
  const [onlyCircles, setOnlyCircles] = useState(false);
  const [limit, setLimit] = useState(PAGE);

  const rows = useMemo(() => {
    const filtered = onlyCircles ? egos.filter((e) => e.has_circles) : egos;
    return [...filtered].sort((a, b) => (desc ? b[sortKey] - a[sortKey] : a[sortKey] - b[sortKey]));
  }, [egos, sortKey, desc, onlyCircles]);

  function sortBy(key: SortKey) {
    if (key === sortKey) setDesc(!desc);
    else {
      setSortKey(key);
      setDesc(true);
    }
  }

  return (
    <>
      <label className="small">
        <input type="checkbox" checked={onlyCircles} onChange={(e) => setOnlyCircles(e.target.checked)} /> only egos with circles
      </label>
      <table>
        <thead>
          <tr>
            {COLUMNS.map(([key, label]) => (
              <th key={key} onClick={() => sortBy(key)} className="sortable">
                {label} {key === sortKey ? (desc ? "▼" : "▲") : ""}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, limit).map((e) => (
            <tr
              key={e.ego_id}
              className={e.ego_id === selected ? "selected" : ""}
              onClick={() => navigate({ pathname: "/run", search: `?ego=${e.ego_id}` })}
              title="Click to run community detection on this ego"
            >
              <td>{e.ego_id}</td>
              <td>{int(e.n_nodes)}</td>
              <td>{int(e.n_edges)}</td>
              <td>{num(e.density)}</td>
              <td>{e.has_circles ? e.n_circles : "none"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {limit < rows.length && <button onClick={() => setLimit(limit + PAGE)}>Show more ({rows.length - limit} left)</button>}
    </>
  );
}

export function Explorer() {
  const stats = useApi(api.stats, []);
  const egos = useApi(api.egos, []);
  return (
    <>
      <h2>Dataset explorer</h2>
      <Query result={stats}>
        {({ stats: s }) => (
          <div className="cards">
            {STAT_CARDS.map(([key, label, kind]) => {
              const v = typeof s[key] === "number" ? (s[key] as number) : null;
              return (
                <div className="card" key={key}>
                  <div className="muted small">{label}</div>
                  <div className="big">{kind === "pct" ? pct(v) : int(v)}</div>
                </div>
              );
            })}
          </div>
        )}
      </Query>
      <h3>Ego networks</h3>
      <p className="muted small">Click an ego to run community detection on it.</p>
      <Query result={egos}>{({ egos: rows }) => <EgoTable egos={rows} />}</Query>
    </>
  );
}
