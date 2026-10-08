import { useEffect, useState } from 'react';
import { NavLink, Route, Routes } from 'react-router-dom';
import { api } from './api.js';
import Diagnose from './pages/Diagnose.jsx';
import Result from './pages/Result.jsx';
import History from './pages/History.jsx';
import Library from './pages/Library.jsx';

export default function App() {
  const [health, setHealth] = useState(null);

  useEffect(() => {
    // Poll so the status pill recovers on its own once a service is started.
    const check = () => api.health().then(setHealth).catch(() => setHealth({ status: 'api-down' }));
    check();
    const timer = setInterval(check, 10000);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        <NavLink to="/" className="brand">🌿 LeafCare</NavLink>
        <nav>
          <NavLink to="/" end>Diagnose</NavLink>
          <NavLink to="/history">History</NavLink>
          <NavLink to="/library">Disease Library</NavLink>
        </nav>
        <StatusPill health={health} />
      </header>
      {health?.ml?.mode === 'heuristic' && (
        <div className="banner">
          Demo mode: the trained model is not loaded, so the ML service is using a colour-based heuristic that only
          recognises general symptom groups.{' '}
          {health.ml.fallback_reason || 'Train the CNN (see README) for crop-specific diagnoses.'}
        </div>
      )}
      {health?.status === 'api-down' && (
        <div className="banner banner-bad">
          Cannot reach the LeafCare API server. Start it with <code>npm run dev</code> in{' '}
          <code>server/</code>, or run <code>npm run dev</code> from the repo root to start everything.
        </div>
      )}
      {health?.status === 'degraded' && (
        <div className="banner banner-bad">
          The API server is running but cannot reach the ML service. Start it with <code>python app.py</code> in{' '}
          <code>ml-service/</code> (inside its virtual environment), or run <code>npm run dev</code> from the repo root.
        </div>
      )}
      <main>
        <Routes>
          <Route path="/" element={<Diagnose />} />
          <Route path="/result/:id" element={<Result />} />
          <Route path="/history" element={<History />} />
          <Route path="/library" element={<Library />} />
          <Route path="*" element={<p className="muted">Page not found.</p>} />
        </Routes>
      </main>
      <footer className="muted">
        AI-assisted guidance only. Confirm serious cases with a local agricultural extension service.
      </footer>
    </div>
  );
}

function StatusPill({ health }) {
  if (!health) return <span className="pill">checking…</span>;
  if (health.status === 'api-down') return <span className="pill pill-bad">API offline</span>;
  if (health.status !== 'ok') return <span className="pill pill-bad">ML offline</span>;
  return (
    <span className={`pill ${health.ml.mode === 'model' ? 'pill-good' : 'pill-warn'}`}>
      {health.ml.mode === 'model' ? `CNN · ${health.ml.classes} classes` : 'Heuristic mode'}
    </span>
  );
}
