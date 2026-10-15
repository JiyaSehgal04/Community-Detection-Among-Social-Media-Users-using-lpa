import { useCallback, useMemo, useState } from "react";
import { api } from "../api/client";
import type { GraphResponse } from "../api/types";
import { useApi } from "../api/useApi";
import { GraphView } from "../components/GraphView";
import { Legend } from "../components/Legend";
import { NeedJob, Query } from "../components/Status";
import { OTHER, PALETTE, countBy, rankColors } from "../lib/colors";
import { useSelection } from "../lib/selection";

function HoverPanel({ id, job, community }: { id: number; job: string; community: number | undefined }) {
  const user = useApi(() => api.userCached(id, job), [id, job]);
  return (
    <div className="box">
      <strong>User {id}</strong> · community {community ?? "?"}
      <Query result={user}>
        {(u) => (
          <p className="small">
            {u.tokens.length === 0 ? "No hashtags or mentions." : u.tokens.slice(0, 10).map((t) => t.token).join("  ")}
          </p>
        )}
      </Query>
    </div>
  );
}

function GraphBody({ graph, job }: { graph: GraphResponse; job: string }) {
  const [hovered, setHovered] = useState<number | null>(null);
  const communityOf = useMemo(() => new Map(graph.nodes.map((n) => [n.id, n.community])), [graph]);
  const sizes = useMemo(() => countBy(graph.nodes.map((n) => n.community)), [graph]);
  const colours = useMemo(() => rankColors(sizes), [sizes]);
  const colorOf = useCallback((id: number) => colours.get(communityOf.get(id) ?? -1) ?? OTHER, [colours, communityOf]);

  const ranked = [...sizes.entries()].sort((a, b) => b[1] - a[1] || a[0] - b[0]);
  const entries = ranked.slice(0, PALETTE.length).map(([, size], i) => ({ color: PALETTE[i], label: `#${i + 1} (${size})` }));

  return (
    <>
      <p className="muted small">{graph.nodes.length} users, {graph.edges.length} links, {sizes.size} communities. Hover a user to see their interests.</p>
      <GraphView nodes={graph.nodes} edges={graph.edges} colorOf={colorOf} onHover={setHovered} />
      <Legend entries={entries} otherCount={Math.max(0, ranked.length - PALETTE.length)} />
      {hovered !== null ? <HoverPanel id={hovered} job={job} community={communityOf.get(hovered)} /> : <p className="muted small">Hover over a user.</p>}
    </>
  );
}

export function GraphPage() {
  const { job } = useSelection();
  if (!job) return <NeedJob />;
  return <GraphScreen job={job} />;
}

function GraphScreen({ job }: { job: string }) {
  const graph = useApi(() => api.graph(job), [job]);
  return (
    <>
      <h2>Graph</h2>
      <Query result={graph}>{(g) => <GraphBody graph={g} job={job} />}</Query>
    </>
  );
}
