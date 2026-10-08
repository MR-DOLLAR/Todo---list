async function request(url, options) {
  const res = await fetch(url, options);
  if (res.status === 204) return null;
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `Request failed (${res.status})`);
  return body;
}

export const api = {
  health: () => request('/api/health'),
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
