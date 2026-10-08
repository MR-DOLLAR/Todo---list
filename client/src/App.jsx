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
    api.health().then(setHealth).catch(() => setHealth({ status: 'down' }));
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
          Demo mode: no trained model found, so the ML service is using a colour-based heuristic that only
          recognises general symptom groups. Train the CNN (see README) for crop-specific diagnoses.
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
  if (health.status !== 'ok') return <span className="pill pill-bad">ML offline</span>;
  return (
    <span className={`pill ${health.ml.mode === 'model' ? 'pill-good' : 'pill-warn'}`}>
      {health.ml.mode === 'model' ? `CNN · ${health.ml.classes} classes` : 'Heuristic mode'}
    </span>
  );
}
