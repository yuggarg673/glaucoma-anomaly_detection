import { useEffect, useState } from "react";
import { getHistory } from "../api.js";

const LABELS = { glaucoma_suspected: "Glaucoma suspected", normal: "Normal-like" };

export default function History() {
  const [items, setItems] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    getHistory().then((d) => setItems(d.items)).catch((e) => setError(e.message));
  }, []);
  return (
    <div className="mx-auto max-w-6xl space-y-6 px-6 py-10">
      <h1 className="m-0 font-serif text-4xl font-semibold">Recent analyses</h1>
      <p className="m-0 text-ink-soft">Images are not stored. Only the file name and scores are saved.</p>
      {error && (
        <div role="alert" className="rounded-xl border border-red-300 bg-red-50 px-5 py-4 text-red-950">
          {error}
        </div>
      )}
      {items && items.length === 0 && <p className="m-0">Nothing here yet. Analyze an image first.</p>}
      {items && items.length > 0 && (
        <div className="overflow-x-auto rounded-xl border border-line bg-white">
          <table className="w-full min-w-[640px] border-collapse text-[15px]">
            <thead>
              <tr className="text-left text-sm text-ink-soft">
                {["Time (UTC)", "File", "Label", "Anomaly", "CDR", "Fused", "Confidence"].map((h) => (
                  <th key={h} className="border-b border-line px-4 py-3 font-medium">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {items.map((r) => (
                <tr key={r.id}>
                  <td className="border-b border-gray-100 px-4 py-3">{r.created_at.replace("T", " ").replace("Z", "")}</td>
                  <td className="border-b border-gray-100 px-4 py-3">{r.filename}</td>
                  <td className="border-b border-gray-100 px-4 py-3">{LABELS[r.label] || r.label}</td>
                  <td className="border-b border-gray-100 px-4 py-3">{r.anomaly_score?.toFixed(4)}</td>
                  <td className="border-b border-gray-100 px-4 py-3">{r.cdr === null ? "unavailable" : r.cdr.toFixed(2)}</td>
                  <td className="border-b border-gray-100 px-4 py-3">{r.fused_score?.toFixed(3)}</td>
                  <td className="border-b border-gray-100 px-4 py-3">{r.confidence_level}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
