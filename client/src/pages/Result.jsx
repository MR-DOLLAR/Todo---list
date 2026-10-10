import { useEffect, useState } from 'react';
import { Link, useLocation, useParams } from 'react-router-dom';
import { api } from '../api.js';
import { Badge, List, Meter, pct, SEVERITY_COLOR, URGENCY_COLOR } from '../components/ui.jsx';

const TABS = ['organic', 'chemical', 'cultural'];

function PhotoTips({ tips }) {
  if (!tips.length) return null;
  return (
    <div className="warning small">
      <strong>Photo tips</strong>
      <ul className="list">
        {tips.map((t) => (
          <li key={t}>{t}</li>
        ))}
      </ul>
    </div>
  );
}

export default function Result() {
  const { id } = useParams();
  const { state } = useLocation();
  const [record, setRecord] = useState(state?.id === id ? state : null);
  const [error, setError] = useState('');
  const [tab, setTab] = useState('organic');

  useEffect(() => {
    if (record?.id === id) return;
    api.record(id).then(setRecord).catch((e) => setError(e.message));
  }, [id, record]);

  if (error) return <div className="error">{error}</div>;
  if (!record) return <p className="muted">Loading…</p>;

  const { prediction, severity, disease, treatment_plan: plan, alternatives, mode } = record.result;
  // Records saved before statuses existed count as confident.
  const status = record.result.status || 'confident';
  const uncertain = status === 'uncertain';
  const threshold = prediction.threshold || 0.6;
  const confColor = prediction.confidence >= Math.max(threshold, 0.8)
    ? 'var(--good)'
    : prediction.confidence >= threshold ? 'var(--warn)' : 'var(--bad)';
  const photoTips = record.result.photo_warnings || [];
  const showSeverity = severity.measurable !== false;
  const title = prediction.healthy
    ? uncertain ? 'Probably healthy' : '✅ Healthy leaf'
    : uncertain ? `Possibly ${prediction.disease}` : prediction.disease;

  if (status === 'no_leaf') {
    return (
      <div className="stack">
        <Link to="/" className="muted">← New diagnosis</Link>
        <div className="grid-2">
          <div className="card">
            <img src={record.imageUrl} alt="Uploaded" className="result-img" />
          </div>
          <div className="card stack">
            <h1 className="result-title">No leaf found</h1>
            <p>{record.result.message}</p>
            <PhotoTips tips={photoTips} />
            <Link to="/" className="btn" style={{ alignSelf: 'flex-start' }}>Try another photo</Link>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="stack">
      <Link to="/" className="muted">← New diagnosis</Link>

      <div className="grid-2">
        <div className="card">
          <img src={record.imageUrl} alt="Analysed leaf" className="result-img" />
          <p className="muted small">
            {new Date(record.createdAt).toLocaleString()} · {mode === 'model' ? 'CNN model' : 'heuristic mode'}
            {record.notes && ` · ${record.notes}`}
          </p>
        </div>

        <div className="card stack">
          <div>
            <div className="muted small">{prediction.crop}</div>
            <h1 className="result-title">{title}</h1>
            {disease.pathogen && <div className="muted"><em>{disease.pathogen}</em></div>}
          </div>

          {uncertain && (
            <div className="banner">
              <strong>Not sure.</strong> {record.result.message}
              {record.result.supported_crops?.length > 0 && (
                <div className="small" style={{ marginTop: 6 }}>
                  Supported crops: {record.result.supported_crops.join(', ')}.
                </div>
              )}
            </div>
          )}
          <PhotoTips tips={photoTips} />

          <div className="row wrap">
            {!prediction.healthy && <Badge color="var(--accent)">{disease.type}</Badge>}
            {!uncertain && showSeverity && (
              <Badge color={SEVERITY_COLOR[severity.level]}>severity: {severity.level}</Badge>
            )}
            {!uncertain && <Badge color={URGENCY_COLOR[plan.urgency]}>urgency: {plan.urgency}</Badge>}
            {!prediction.healthy && (
              <Badge color={plan.curable ? 'var(--good)' : 'var(--bad)'}>{plan.curable ? 'treatable' : 'not curable'}</Badge>
            )}
          </div>

          <div>
            <div className="row between small"><span>Confidence</span><strong>{pct(prediction.confidence)}</strong></div>
            <Meter value={prediction.confidence * 100} color={confColor} label="confidence" />
          </div>
          {!prediction.healthy && !uncertain && showSeverity && (
            <div>
              <div className="row between small"><span>Affected leaf area (estimate)</span><strong>{severity.affected_area_pct}%</strong></div>
              <Meter value={severity.affected_area_pct} color={SEVERITY_COLOR[severity.level]} label="affected area" />
            </div>
          )}

          <p>{disease.description}</p>

          {alternatives?.length > 0 && (
            <div>
              <div className={uncertain ? 'small' : 'small muted'}>
                {uncertain ? <strong>Other likely matches</strong> : 'Other possibilities'}
              </div>
              {alternatives.map((a) => (
                <div key={a.label} className="row between small">
                  <span>{a.crop && a.crop !== 'Unknown' ? `${a.crop} – ` : ''}{a.name}</span>
                  <span className="muted">{pct(a.confidence)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="card stack">
        <h2>{uncertain && !prediction.healthy ? `If it is ${prediction.disease}` : 'Treatment plan'}</h2>
        <p className="lead">{plan.summary}</p>
        {plan.warnings.map((w) => (
          <div key={w} className="warning">⚠️ {w}</div>
        ))}

        {!uncertain && <div className="prognosis">
          <div>
            <div className="muted small">Recovery chance</div>
            <div className="big">
              {plan.prognosis.recovery_chance == null ? 'N/A' : pct(plan.prognosis.recovery_chance)}
            </div>
          </div>
          <div>
            <div className="muted small">Estimated recovery time</div>
            <div className="big">
              {plan.prognosis.estimated_recovery_days == null
                ? 'N/A'
                : plan.prognosis.estimated_recovery_days === 0
                  ? '—'
                  : `~${plan.prognosis.estimated_recovery_days} days`}
            </div>
          </div>
          <div>
            <div className="muted small">Approach</div>
            <div className="big cap">{plan.approach}</div>
          </div>
          <div>
            <div className="muted small">Spread risk</div>
            <div className="big cap">{disease.spread_risk}</div>
          </div>
        </div>}

        <ol className="timeline">
          {plan.steps.map((s) => (
            <li key={s.phase}>
              <div className="phase">{s.phase}</div>
              <List items={s.actions} />
            </li>
          ))}
        </ol>
      </div>

      {!prediction.healthy && (
        <div className="grid-2">
          <div className="card">
            <h3>Symptoms</h3>
            <List items={disease.symptoms} />
            <h3>Causes</h3>
            <List items={disease.causes} />
          </div>
          <div className="card">
            <h3>All treatment options</h3>
            <div className="tabs">
              {TABS.map((t) => (
                <button key={t} className={tab === t ? 'active' : ''} onClick={() => setTab(t)}>
                  {t}
                </button>
              ))}
            </div>
            <List items={disease.treatment[tab]} />
            <h3>Prevention</h3>
            <List items={disease.prevention} />
          </div>
        </div>
      )}
    </div>
  );
}
