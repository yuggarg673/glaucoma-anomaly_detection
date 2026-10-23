// All backend calls live here. Errors always become an Error with a readable message.
export const BASE = (import.meta.env.VITE_API_BASE_URL || "http://localhost:5000").replace(/\/$/, "");

async function request(path, options) {
  let res;
  try {
    res = await fetch(`${BASE}${path}`, options);
  } catch {
    throw new Error(
      `Cannot reach the analysis server at ${BASE}. Check that the backend is running (python -m backend.app), then reload.`
    );
  }
  let body = null;
  try {
    body = await res.json();
  } catch {
    /* not JSON */
  }
  if (!res.ok) {
    const msg = body?.error?.message || `The server answered with status ${res.status}.`;
    throw new Error(msg);
  }
  return body;
}

export const getHealth = () => request("/health");
export const getExplain = () => request("/explain");
export const getHistory = (limit = 50) => request(`/history?limit=${limit}`);
export const predict = (file) => {
  const form = new FormData();
  form.append("image", file);
  return request("/predict", { method: "POST", body: form });
};
