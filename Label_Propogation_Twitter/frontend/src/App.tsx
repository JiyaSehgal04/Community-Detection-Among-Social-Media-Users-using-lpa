import { Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { CommunitiesPage } from "./pages/Communities";
import { ComparePage } from "./pages/Compare";
import { Explorer } from "./pages/Explorer";
import { GraphPage } from "./pages/Graph";
import { Run } from "./pages/Run";
import { StepThroughPage } from "./pages/StepThrough";
import { UserLookupPage } from "./pages/UserLookup";

export function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Navigate to="/explorer" replace />} />
        <Route path="/explorer" element={<Explorer />} />
        <Route path="/run" element={<Run />} />
        <Route path="/graph" element={<GraphPage />} />
        <Route path="/communities" element={<CommunitiesPage />} />
        <Route path="/steps" element={<StepThroughPage />} />
        <Route path="/compare" element={<ComparePage />} />
        <Route path="/users" element={<UserLookupPage />} />
        <Route path="*" element={<p>Page not found.</p>} />
      </Route>
    </Routes>
  );
}
