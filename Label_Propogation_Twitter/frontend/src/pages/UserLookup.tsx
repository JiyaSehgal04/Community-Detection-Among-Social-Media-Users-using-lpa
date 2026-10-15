import { useState, type FormEvent } from "react";
import { Link, useLocation } from "react-router-dom";
import { api } from "../api/client";
import { useApi } from "../api/useApi";
import { Query } from "../components/Status";
import { num } from "../lib/format";
import { useSelection } from "../lib/selection";

function Result({ userId, job }: { userId: number; job: string | null }) {
  const { search } = useLocation();
  const user = useApi(() => api.user(userId, job), [userId, job]);
  return (
    <Query result={user}>
      {(u) => (
        <>
          <p>
            User <strong>{u.user_id}</strong> appears in ego network{u.egos.length === 1 ? "" : "s"} {u.egos.join(", ")}.{" "}
            {job
              ? u.community !== null ? <>In the selected run they are in community <strong>{u.community}</strong>.</> : "They are not in the selected run's ego network."
              : <>Select a run on the <Link to={{ pathname: "/run", search }}>Run</Link> screen to see their community.</>}
          </p>
          <h3>Interests</h3>
          <p>{u.tokens.length === 0 ? <span className="muted">No hashtags or mentions.</span> : u.tokens.map((t) => <span className="chip" key={t.token}>{t.token}</span>)}</p>
          {job && u.similar_users.length > 0 && (
            <>
              <h3>Most similar users in the same ego network</h3>
              <table>
                <thead><tr><th>User</th><th>Similarity</th><th>Community</th></tr></thead>
                <tbody>{u.similar_users.map((s) => <tr key={s.user_id}><td>{s.user_id}</td><td>{num(s.similarity)}</td><td>{s.community}</td></tr>)}</tbody>
              </table>
            </>
          )}
        </>
      )}
    </Query>
  );
}

export function UserLookupPage() {
  const { job } = useSelection();
  const [text, setText] = useState("");
  const [userId, setUserId] = useState<number | null>(null);

  function submit(e: FormEvent) {
    e.preventDefault();
    const id = Number(text.trim());
    setUserId(Number.isInteger(id) && text.trim() !== "" ? id : null);
  }
  return (
    <>
      <h2>User lookup</h2>
      <form onSubmit={submit} className="form inline">
        <label>User id <input value={text} onChange={(e) => setText(e.target.value)} inputMode="numeric" required pattern="[0-9]+" /></label>
        <button type="submit">Look up</button>
      </form>
      {userId !== null && <Result userId={userId} job={job} />}
    </>
  );
}
