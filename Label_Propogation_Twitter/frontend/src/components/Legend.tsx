import { OTHER } from "../lib/colors";

export function Legend({ entries, otherCount }: { entries: { color: string; label: string }[]; otherCount: number }) {
  return (
    <ul className="legend">
      {entries.map((e) => (
        <li key={e.label}>
          <span className="dot" style={{ background: e.color }} /> {e.label}
        </li>
      ))}
      {otherCount > 0 && (
        <li>
          <span className="dot" style={{ background: OTHER }} /> other ({otherCount})
        </li>
      )}
    </ul>
  );
}
