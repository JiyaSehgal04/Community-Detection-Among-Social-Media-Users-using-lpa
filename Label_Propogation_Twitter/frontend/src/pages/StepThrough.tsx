import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { GraphResponse, TraceResponse } from "../api/types";
import { useApi } from "../api/useApi";
import { GraphView } from "../components/GraphView";
import { NeedJob, Query } from "../components/Status";
import { OTHER, countBy, rankColors } from "../lib/colors";
import { int } from "../lib/format";
import { useSelection } from "../lib/selection";

function Player({ graph, trace }: { graph: GraphResponse; trace: TraceResponse }) {
  const last = trace.snapshots.length - 1;
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);

  const index = useMemo(() => new Map(trace.nodes.map((id, i) => [id, i])), [trace]);
  // Colour a label by its rank in the FINAL partition, so colours stay put while the animation runs.
  const labelColours = useMemo(() => rankColors(countBy(trace.snapshots[last] ?? [])), [trace, last]);
  const colorOf = useCallback(
    (id: number) => {
      const i = index.get(id);
      return i === undefined ? OTHER : (labelColours.get(trace.snapshots[step][i]) ?? OTHER);
    },
    [index, labelColours, trace, step],
  );

  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(() => {
      setStep((s) => {
        if (s >= last) {
          setPlaying(false);
          return s;
        }
        return s + 1;
      });
    }, 700);
    return () => clearInterval(timer);
  }, [playing, last]);

  function togglePlay() {
    if (!playing && step >= last) setStep(0);
    setPlaying(!playing);
  }

  const labels = new Set(trace.snapshots[step]).size;
  return (
    <>
      <div className="controls">
        <button onClick={togglePlay} disabled={last === 0}>{playing ? "Pause" : step >= last && last > 0 ? "Replay" : "Play"}</button>
        <input
          type="range" min={0} max={last} value={step} disabled={last === 0}
          onChange={(e) => { setPlaying(false); setStep(Number(e.target.value)); }}
          aria-label="Iteration"
        />
        <span>Iteration {step} of {last}</span>
      </div>
      <p className="muted small">
        {step === 0 ? "Start: every user has their own label." : `${int(trace.changed[step] ?? 0)} users changed label.`}{" "}
        {int(labels)} distinct labels now. Coloured users hold a label that survives to the end; grey users hold one that does not (or a small community).
      </p>
      <GraphView nodes={graph.nodes} edges={graph.edges} colorOf={colorOf} />
    </>
  );
}

export function StepThroughPage() {
  const { job } = useSelection();
  if (!job) return <NeedJob />;
  return <StepScreen job={job} />;
}

function StepScreen({ job }: { job: string }) {
  const graph = useApi(() => api.graph(job), [job]);
  const trace = useApi(() => api.trace(job), [job]);
  return (
    <>
      <h2>Step-through</h2>
      <Query result={graph}>{(g) => <Query result={trace}>{(t) => <Player graph={g} trace={t} />}</Query>}</Query>
    </>
  );
}
