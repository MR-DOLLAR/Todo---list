export const SEVERITY_COLOR = { none: 'var(--good)', mild: 'var(--warn)', moderate: 'var(--orange)', severe: 'var(--bad)' };
export const URGENCY_COLOR = { low: 'var(--good)', medium: 'var(--warn)', high: 'var(--orange)', critical: 'var(--bad)' };

export function Badge({ children, color }) {
  return (
    <span className="badge" style={{ '--badge': color }}>
      {children}
    </span>
  );
}

export function Meter({ value, color, label }) {
  return (
    <div className="meter" aria-label={label}>
      <div className="meter-fill" style={{ width: `${Math.max(0, Math.min(100, value))}%`, background: color }} />
    </div>
  );
}

export function pct(x) {
  return `${Math.round(x * 100)}%`;
}

export function List({ items, empty = '—' }) {
  if (!items?.length) return <p className="muted">{empty}</p>;
  return (
    <ul className="list">
      {items.map((i) => (
        <li key={i}>{i}</li>
      ))}
    </ul>
  );
}
