import { useEffect, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { API_URL, api } from "../api/client";
import { ErrorBoundary } from "./ErrorBoundary";
import { useSelection } from "../lib/selection";

const LINKS = [
  ["/explorer", "Explorer"],
  ["/run", "Run"],
  ["/graph", "Graph"],
  ["/communities", "Communities"],
  ["/steps", "Step-through"],
  ["/compare", "Compare"],
  ["/users", "Users"],
] as const;

function ApiBanner() {
  const [up, setUp] = useState(true);
  useEffect(() => {
    let alive = true;
    const check = () => api.ping().then((ok) => alive && setUp(ok));
    check();
    const timer = setInterval(check, 10000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);
  if (up) return null;
  return (
    <div className="banner" role="alert">
      API unreachable at {API_URL}. Start it with <code>uvicorn backend.main:app --port 8000</code>.
    </div>
  );
}

export function Layout() {
  const { pathname, search } = useLocation();
  const { ego, job } = useSelection();
  return (
    <>
      <ApiBanner />
      <header>
        <h1>LPA communities</h1>
        <nav>
          {LINKS.map(([to, label]) => (
            <NavLink key={to} to={{ pathname: to, search }} className={({ isActive }) => (isActive ? "active" : "")}>
              {label}
            </NavLink>
          ))}
        </nav>
        <span className="muted chip">
          {ego ? `ego ${ego}` : "no ego"} · {job ? "run ready" : "no run"}
        </span>
      </header>
      <main>
        <ErrorBoundary key={pathname}>
          <Outlet />
        </ErrorBoundary>
      </main>
    </>
  );
}
