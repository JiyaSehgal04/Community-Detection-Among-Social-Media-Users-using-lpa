import { useMemo, useState } from "react";
import { api } from "../api/client";
import type { CommunityCard } from "../api/types";
import { useApi } from "../api/useApi";
import { NeedJob, Query } from "../components/Status";
import { num } from "../lib/format";
import { rankColors } from "../lib/colors";
import { useSelection } from "../lib/selection";

function Cards({ cards }: { cards: CommunityCard[] }) {
  const [showSingles, setShowSingles] = useState(false);
  const sorted = useMemo(() => [...cards].sort((a, b) => b.size - a.size || a.id - b.id), [cards]);
  const colours = useMemo(() => rankColors(new Map(cards.map((c) => [c.id, c.size]))), [cards]);
  const shown = sorted.filter((c) => showSingles || c.size >= 2);

  return (
    <>
      <label className="small">
        <input type="checkbox" checked={showSingles} onChange={(e) => setShowSingles(e.target.checked)} /> also show users who are alone
        ({sorted.filter((c) => c.size < 2).length})
      </label>
      {shown.length === 0 && <p className="muted">No communities of two or more users in this run.</p>}
      <div className="cards wide">
        {shown.map((c) => (
          <div className="card" key={c.id}>
            <div>
              <span className="dot" style={{ background: colours.get(c.id) }} /> <strong>{c.size} users</strong>
              <span className="muted small"> · purity {num(c.purity, 2)}</span>
            </div>
            <div className="chips">
              {c.top_hashtags.map((t) => <span className="chip" key={t.token}>{t.token} <em>{t.count}</em></span>)}
              {c.top_mentions.map((t) => <span className="chip" key={t.token}>{t.token} <em>{t.count}</em></span>)}
              {c.top_hashtags.length + c.top_mentions.length === 0 && <span className="muted small">no shared interests</span>}
            </div>
          </div>
        ))}
      </div>
    </>
  );
}

export function CommunitiesPage() {
  const { job } = useSelection();
  if (!job) return <NeedJob />;
  return <CommunitiesScreen job={job} />;
}

function CommunitiesScreen({ job }: { job: string }) {
  const result = useApi(() => api.communities(job), [job]);
  return (
    <>
      <h2>Communities</h2>
      <Query result={result}>{({ communities }) => <Cards cards={communities} />}</Query>
    </>
  );
}
