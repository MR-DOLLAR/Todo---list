const API_DOWN =
  'Cannot reach the LeafCare API server (port 4000). Is it running? Start it with "npm run dev" in server/ (or "npm run dev" from the repo root).';

async function request(url, options) {
  let res;
  try {
    res = await fetch(url, options);
  } catch {
    throw new Error(API_DOWN);
  }
  if (res.status === 204) return null;
  const isJson = (res.headers.get('content-type') || '').includes('application/json');
  const body = isJson ? await res.json().catch(() => ({})) : {};
  if (!res.ok) {
    // The Vite dev proxy answers with a non-JSON 5xx when the API server is down.
    if (!isJson && res.status >= 500) throw new Error(API_DOWN);
    throw new Error(body.error || `Request failed (${res.status})`);
  }
  return body;
}

export const api = {
  health: async () => {
    const res = await fetch('/api/health').catch(() => null);
    const isJson = res && (res.headers.get('content-type') || '').includes('application/json');
    if (!isJson) return { status: 'api-down' };
    return res.json();
  },
  diagnose: ({ file, crop, preference, notes }) => {
    const form = new FormData();
    form.append('image', file);
    if (crop) form.append('crop', crop);
    if (preference) form.append('preference', preference);
    if (notes) form.append('notes', notes);
    return request('/api/diagnose', { method: 'POST', body: form });
  },
  history: () => request('/api/history'),
  record: (id) => request(`/api/history/${id}`),
  remove: (id) => request(`/api/history/${id}`, { method: 'DELETE' }),
  stats: () => request('/api/stats'),
  diseases: (params = {}) => request(`/api/diseases?${new URLSearchParams(params)}`),
  crops: () => request('/api/crops'),
};
