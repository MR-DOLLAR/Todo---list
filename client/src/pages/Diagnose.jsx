import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api.js';

export default function Diagnose() {
  const navigate = useNavigate();
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [crops, setCrops] = useState([]);
  const [crop, setCrop] = useState('');
  const [preference, setPreference] = useState('integrated');
  const [notes, setNotes] = useState('');
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    api.crops().then(setCrops).catch(() => {});
  }, []);

  useEffect(() => {
    if (!file) return setPreview(null);
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function pick(f) {
    setError('');
    if (!f) return;
    if (!f.type.startsWith('image/')) return setError('Please choose an image file.');
    if (f.size > 10 * 1024 * 1024) return setError('Image must be under 10 MB.');
    setFile(f);
  }

  async function submit(e) {
    e.preventDefault();
    if (!file) return setError('Please add a leaf photo first.');
    setLoading(true);
    setError('');
    try {
      const record = await api.diagnose({ file, crop, preference, notes });
      navigate(`/result/${record.id}`, { state: record });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid-2">
      <section>
        <h1>Diagnose a leaf</h1>
        <p className="muted">
          Upload or snap a photo of a single leaf. The AI identifies the disease, estimates how much of the leaf
          is affected and builds a step-by-step treatment plan.
        </p>

        <form onSubmit={submit} className="card stack">
          <div
            className={`dropzone ${dragging ? 'dragging' : ''}`}
            onClick={() => inputRef.current.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              pick(e.dataTransfer.files[0]);
            }}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => e.key === 'Enter' && inputRef.current.click()}
          >
            {preview ? (
              <img src={preview} alt="Selected leaf" className="preview" />
            ) : (
              <div>
                <div className="drop-icon">📷</div>
                <strong>Drop a leaf photo here</strong>
                <div className="muted">or click to browse / use your camera</div>
              </div>
            )}
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              capture="environment"
              hidden
              onChange={(e) => pick(e.target.files[0])}
            />
          </div>

          <label>
            Crop <span className="muted">(optional – improves accuracy)</span>
            <select value={crop} onChange={(e) => setCrop(e.target.value)}>
              <option value="">Auto-detect</option>
              {crops.map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
          </label>

          <label>
            Treatment preference
            <select value={preference} onChange={(e) => setPreference(e.target.value)}>
              <option value="integrated">Integrated (organic first, chemical if needed)</option>
              <option value="organic">Organic only</option>
              <option value="chemical">Chemical (fastest control)</option>
            </select>
          </label>

          <label>
            Notes <span className="muted">(optional)</span>
            <input value={notes} maxLength={500} onChange={(e) => setNotes(e.target.value)} placeholder="e.g. Back garden, bed 2" />
          </label>

          {error && <div className="error">{error}</div>}
          <div className="row">
            <button type="submit" className="btn" disabled={loading || !file}>
              {loading ? 'Analysing…' : 'Analyse leaf'}
            </button>
            {file && (
              <button type="button" className="btn btn-ghost" onClick={() => setFile(null)}>
                Clear
              </button>
            )}
          </div>
        </form>
      </section>

      <aside className="card tips">
        <h3>Tips for an accurate diagnosis</h3>
        <ol>
          <li>Photograph one leaf at a time, filling most of the frame.</li>
          <li>Use natural daylight; avoid harsh shadows and flash glare.</li>
          <li>Place the leaf on a plain background if possible.</li>
          <li>Capture the side with the clearest symptoms (spots, powder, discolouration).</li>
          <li>Select the crop if you know it – it narrows down the possibilities.</li>
        </ol>
      </aside>
    </div>
  );
}
