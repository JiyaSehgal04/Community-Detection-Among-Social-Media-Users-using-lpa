import type { Metrics } from "../api/types";
import { int, num, pct } from "../lib/format";

function Card({ title, value, note }: { title: string; value: string; note?: string }) {
  return (
    <div className="card">
      <div className="muted small">{title}</div>
      <div className="big">{value}</div>
      {note && <div className="small">{note}</div>}
    </div>
  );
}

export function MetricCards({ m }: { m: Metrics }) {
  const noCircles = m.circle_f1 === null;
  const belowBaseline = !noCircles && m.circle_f1_one_community !== null && m.circle_f1! < m.circle_f1_one_community;
  return (
    <>
      <div className="cards">
        <Card
          title="Circle F1"
          value={noCircles ? "n/a" : num(m.circle_f1)}
          note={noCircles ? "this ego has no circles" : `one-community baseline ${num(m.circle_f1_one_community)}`}
        />
        <Card title="Interest purity" value={num(m.interest_purity)} />
        <Card title="Largest community" value={pct(m.largest_share)} note="share of users" />
        <Card title="Coverage" value={pct(m.coverage)} note="users in a community of 2 or more" />
        <Card title="Communities" value={int(m.n_communities)} note={`${int(m.n_singletons)} users alone`} />
        <Card title="Modularity" value={num(m.modularity)} note={`on follows: ${num(m.follow_modularity)}`} />
        <Card title="Iterations" value={int(m.iterations)} note={m.converged ? "converged" : "did not converge"} />
        <Card title="Graph" value={`${int(m.n_nodes)} nodes`} note={`${int(m.n_edges)} edges`} />
      </div>
      <p className="muted small">
        Circle F1 favours one big community, so always read it next to the baseline and the largest-community share.
        {belowBaseline && " This run scores below the one-community baseline."}
      </p>
    </>
  );
}
