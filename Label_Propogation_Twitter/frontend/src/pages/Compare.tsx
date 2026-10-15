import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api/client";
import { useApi } from "../api/useApi";
import { Query } from "../components/Status";
import { int, num } from "../lib/format";

type Row = Record<string, unknown>;
const MODES = ["follow", "interest", "hybrid"] as const;
const ALGOS = [["lpa", "Our LPA"], ["nx_lpa", "NetworkX LPA"], ["louvain", "Louvain"]] as const;
const SERIES = ["#2a78d6", "#eb6834", "#1baf7a"];
const METRICS = [
  ["circle_f1", "Circle F1"],
  ["interest_purity", "Interest purity"],
  ["follow_modularity", "Follow modularity"],
  ["largest_share", "Largest community share"],
  ["coverage", "Coverage"],
] as const;

const val = (r: Row | undefined, key: string): number | null => (typeof r?.[key] === "number" ? (r[key] as number) : null);

function Body({ rows }: { rows: Row[] }) {
  const [direction, setDirection] = useState<"mutual" | "any">("mutual");
  const [metric, setMetric] = useState<(typeof METRICS)[number][0]>("circle_f1");
  const mine = rows.filter((r) => r.direction === direction);
  const find = (mode: string, algo: string) => mine.find((r) => r.mode === mode && r.algorithm === algo);
  const data = MODES.map((mode) => ({
    mode,
    ...Object.fromEntries(ALGOS.map(([algo]) => [algo, val(find(mode, algo), metric)])),
  }));
  const baseline = val(mine[0], "circle_f1_one_community");
  const egos = val(mine[0], "n_egos");

  if (rows.length === 0) return <p className="muted">No saved experiment results. Run <code>python -m pipeline.experiments</code> first.</p>;
  return (
    <>
      <div className="controls">
        <label>Follows
          <select value={direction} onChange={(e) => setDirection(e.target.value as "mutual" | "any")}>
            <option value="mutual">mutual only</option>
            <option value="any">either direction</option>
          </select>
        </label>
        <label>Metric
          <select value={metric} onChange={(e) => setMetric(e.target.value as typeof metric)}>
            {METRICS.map(([key, label]) => <option key={key} value={key}>{label}</option>)}
          </select>
        </label>
      </div>
      <p className="muted small">Averages over the {int(egos)} largest egos with circles. Interest mode leaves about 40% of users alone, so its coverage is lower.</p>
      <ResponsiveContainer width="100%" height={340}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="mode" />
          <YAxis domain={[0, 1]} />
          <Tooltip formatter={(v) => num(typeof v === "number" ? v : null)} />
          <Legend />
          {ALGOS.map(([algo, name], i) => <Bar key={algo} dataKey={algo} name={name} fill={SERIES[i]} />)}
          {metric === "circle_f1" && baseline !== null && (
            <ReferenceLine y={baseline} stroke="#0b0b0b" strokeDasharray="6 4" label={{ value: `one community ${num(baseline)}`, position: "insideTopRight", fontSize: 11 }} />
          )}
        </BarChart>
      </ResponsiveContainer>
      {metric === "circle_f1" && <p className="small">Circle F1 favours one big community. A bar above the dashed line only counts if the largest-community share is not also high.</p>}
      <table>
        <thead><tr><th>Graph</th><th>Algorithm</th><th>Circle F1</th><th>One-community</th><th>Purity</th><th>Follow modularity</th><th>Largest share</th><th>Communities</th></tr></thead>
        <tbody>
          {mine.map((r) => (
            <tr key={`${r.mode}-${r.algorithm}`}>
              <td>{String(r.mode)}</td><td>{String(r.algorithm)}</td>
              <td>{num(val(r, "circle_f1"))}</td><td>{num(val(r, "circle_f1_one_community"))}</td>
              <td>{num(val(r, "interest_purity"))}</td><td>{num(val(r, "follow_modularity"))}</td>
              <td>{num(val(r, "largest_share"))}</td><td>{num(val(r, "n_communities"), 1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

export function ComparePage() {
  const result = useApi(api.compare, []);
  return (
    <>
      <h2>Compare</h2>
      <Query result={result}>{({ rows }) => <Body rows={rows} />}</Query>
    </>
  );
}
