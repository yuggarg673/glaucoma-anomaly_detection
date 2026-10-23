// One glossary entry (what it is / how to read it / caution) from GET /explain.
export default function Explain({ entry }) {
  if (!entry) return null;
  return (
    <div className="space-y-2 text-[15px] leading-relaxed text-ink">
      <p>
        <span className="font-semibold">What it is. </span>
        {entry.what}
      </p>
      <p>
        <span className="font-semibold">How to read it. </span>
        {entry.read}
      </p>
      {entry.caution && (
        <p className="rounded-lg bg-amber-50 px-3 py-2 text-amber-950">
          <span className="font-semibold">Be careful. </span>
          {entry.caution}
        </p>
      )}
    </div>
  );
}
