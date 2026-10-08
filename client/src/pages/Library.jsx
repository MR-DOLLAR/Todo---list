import { useEffect, useState } from 'react';
import { api } from '../api.js';
import { Badge, List } from '../components/ui.jsx';

const RISK_COLOR = { none: 'var(--good)', low: 'var(--good)', medium: 'var(--warn)', high: 'var(--bad)' };

export default function Library() {
  const [crops, setCrops] = useState([]);
  const [crop, setCrop] = useState('');
  const [q, setQ] = useState('');
  const [items, setItems] = useState(null);
  const [open, setOpen] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    api.crops().then(setCrops).catch(() => {});
  }, []);

  useEffect(() => {
    const t = setTimeout(() => {
      const params = {};
      if (crop) params.crop = crop;
      if (q) params.q = q;
      api.diseases(params).then(setItems).catch((e) => setError(e.message));
    }, 250);
    return () => clearTimeout(t);
  }, [crop, q]);

  return (
    <div className="stack">
      <h1>Disease library</h1>
      <div className="row wrap">
        <input className="grow" placeholder="Search diseases, symptoms, pathogens…" value={q} onChange={(e) => setQ(e.target.value)} />
        <select value={crop} onChange={(e) => setCrop(e.target.value)}>
          <option value="">All crops</option>
          {crops.map((c) => (
            <option key={c}>{c}</option>
          ))}
        </select>
      </div>
      {error && <div className="error">{error}</div>}
      {!items ? (
        <p className="muted">Loading…</p>
      ) : items.length === 0 ? (
        <p className="muted">No matching diseases.</p>
      ) : (
        <div className="library">
          {items.filter((d) => !d.healthy).map((d) => (
            <div key={d.id} className="card">
              <button className="lib-head" onClick={() => setOpen(open === d.id ? null : d.id)}>
                <div>
                  <div className="muted small">{d.crop}</div>
                  <strong>{d.name}</strong>
                </div>
                <div className="row">
                  <Badge color="var(--accent)">{d.type}</Badge>
                  <Badge color={RISK_COLOR[d.spread_risk]}>{d.spread_risk} spread</Badge>
                </div>
              </button>
              {open === d.id && (
                <div className="stack lib-body">
                  <p>{d.description}</p>
                  {d.pathogen && <p className="muted"><em>{d.pathogen}</em></p>}
                  <h4>Symptoms</h4>
                  <List items={d.symptoms} />
                  <h4>Organic treatment</h4>
                  <List items={d.treatment.organic} />
                  <h4>Chemical treatment</h4>
                  <List items={d.treatment.chemical} />
                  <h4>Cultural practices</h4>
                  <List items={d.treatment.cultural} />
                  <h4>Prevention</h4>
                  <List items={d.prevention} />
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
