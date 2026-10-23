import { useEffect, useState } from "react";
import { getHealth } from "./api.js";
import Home from "./pages/Home.jsx";
import Analyze from "./pages/Analyze.jsx";
import History from "./pages/History.jsx";
import Guide from "./pages/Guide.jsx";

const ROUTES = {
  "": Home,
  analyze: Analyze,
  history: History,
  guide: Guide,
};

const NAV = [
  ["", "Home"],
  ["analyze", "Analyze"],
  ["guide", "How to read results"],
  ["history", "History"],
];

function useRoute() {
  const get = () => window.location.hash.replace(/^#\/?/, "");
  const [route, setRoute] = useState(get());
  useEffect(() => {
    const onChange = () => setRoute(get());
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return route;
}

function StatusPill() {
  const [state, setState] = useState({ kind: "checking" });
  useEffect(() => {
    let alive = true;
    const check = async () => {
      try {
        const h = await getHealth();
        if (alive) setState({ kind: h.models_loaded ? "ok" : "degraded", detail: h.detail });
      } catch {
        if (alive) setState({ kind: "down" });
      }
    };
    check();
    const id = setInterval(check, 15000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);
  const styles = {
    checking: ["bg-gray-100 text-gray-700", "bg-gray-400", "Checking server"],
    ok: ["bg-green-100 text-green-900", "bg-green-600", "Server online, models loaded"],
    degraded: ["bg-amber-100 text-amber-900", "bg-amber-600", "Server online, models not loaded"],
    down: ["bg-red-100 text-red-900", "bg-red-600", "Server unreachable"],
  }[state.kind];
  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-sm font-medium ${styles[0]}`}
      title={state.detail || ""}
    >
      <span className={`h-2 w-2 rounded-full ${styles[1]}`} />
      {styles[2]}
    </span>
  );
}

export default function App() {
  const route = useRoute();
  const Page = ROUTES[route] || Home;
  return (
    <div className="min-h-screen">
      <header className="border-b border-line bg-white">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-4 px-6 py-4">
          <a href="#/" className="flex items-center gap-3 no-underline text-ink">
            <svg width="34" height="34" viewBox="0 0 36 36" fill="none" aria-hidden="true">
              <circle cx="18" cy="18" r="16" stroke="#1B6F6A" strokeWidth="2" />
              <circle cx="18" cy="18" r="8" stroke="#1B6F6A" strokeWidth="2" />
              <circle cx="18" cy="18" r="3" fill="#1B6F6A" />
            </svg>
            <span className="font-serif text-xl font-semibold">Glaucoma Screening</span>
          </a>
          <nav className="flex flex-wrap items-center gap-5">
            {NAV.map(([path, label]) => (
              <a
                key={path}
                href={`#/${path}`}
                className={`border-b-2 py-2 text-[15px] no-underline text-ink ${
                  route === path ? "border-brand font-semibold" : "border-transparent"
                }`}
              >
                {label}
              </a>
            ))}
            <StatusPill />
          </nav>
        </div>
      </header>
      <main>
        <Page />
      </main>
      <footer className="mt-12 border-t border-line bg-white">
        <div className="mx-auto max-w-6xl px-6 py-6 text-sm text-ink-soft">
          Student research project. A screening aid, not a medical diagnosis. See an eye doctor for any concern about
          your vision.
        </div>
      </footer>
    </div>
  );
}
