import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api.js';
import { Badge, pct, SEVERITY_COLOR, URGENCY_COLOR } from '../components/ui.jsx';

export default function History() {
  const [items, setItems] = useState(null);
  const [stats, setStats] = useState(null);
  const [error, setError] = useState('');

  const load = () =>
    Promise.all([api.history(), api.stats()])
      .then(([h, s]) => {
        setItems(h);
        setStats(s);
      })
      .catch((e) => setError(e.message));

  useEffect(() => {
    load();
  }, []);

  async function remove(id) {
    if (!confirm('Delete this diagnosis?')) return;
    await api.remove(id).catch((e) => setError(e.message));
    load();
  }

  if (error) return <div className="error">{error}</div>;
  if (!items) return <p className="muted">Loading…</p>;

  return (
    <div className="stack">
      <h1>Diagnosis history</h1>
      {stats && stats.total > 0 && (
        <div className="stats">
          <Stat label="Total scans" value={stats.total} />
          <Stat label="Healthy" value={stats.healthy} color="var(--good)" />
          <Stat label="Diseased" value={stats.diseased} color="var(--bad)" />
          <Stat label="Severe cases" value={stats.bySeverity.severe} color="var(--orange)" />
          {stats.uncertain > 0 && <Stat label="Not sure" value={stats.uncertain} color="var(--warn)" />}
          {stats.topDiseases.length > 0 && (
            <div className="card stat wide">
              <div className="muted small">Most common</div>
              {stats.topDiseases.map((d) => (
                <div key={d.name} className="row between small">
                  <span>{d.name}</span>
                  <strong>{d.count}</strong>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {items.length === 0 ? (
        <p className="muted">
          No diagnoses yet. <Link to="/">Analyse your first leaf →</Link>
        </p>
      ) : (
        <div className="history">
          {items.map((r) => (
            <div key={r.id} className="card history-item">
              <Link to={`/result/${r.id}`}>
                <img src={r.imageUrl} alt="" />
              </Link>
              <div className="grow">
                <Link to={`/result/${r.id}`}>
                  <strong>
                    {r.status === 'no_leaf'
                      ? 'No leaf found'
                      : r.status === 'uncertain'
                        ? `Possibly ${r.prediction.healthy ? 'healthy' : r.prediction.disease}`
                        : r.prediction.healthy ? 'Healthy' : r.prediction.disease}
                  </strong>
                </Link>
                <div className="muted small">
                  {r.status === 'no_leaf'
                    ? new Date(r.createdAt).toLocaleDateString()
                    : `${r.prediction.crop} · ${pct(r.prediction.confidence)} · ${new Date(r.createdAt).toLocaleDateString()}`}
                </div>
                <div className="row wrap">
                  {r.status === 'no_leaf' && <Badge color="var(--muted)">no leaf</Badge>}
                  {r.status === 'uncertain' && <Badge color="var(--warn)">not sure</Badge>}
                  {(r.status || 'confident') === 'confident' && (
                    <>
                      <Badge color={SEVERITY_COLOR[r.severity.level]}>{r.severity.level}</Badge>
                      <Badge color={URGENCY_COLOR[r.urgency]}>{r.urgency}</Badge>
                    </>
                  )}
                </div>
              </div>
              <button className="btn btn-ghost small" onClick={() => remove(r.id)} aria-label="Delete">
                🗑
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function Stat({ label, value, color }) {
  return (
    <div className="card stat">
      <div className="muted small">{label}</div>
      <div className="big" style={{ color }}>{value}</div>
    </div>
  );
}
