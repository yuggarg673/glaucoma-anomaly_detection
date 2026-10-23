import Disclaimer from "../components/Disclaimer.jsx";
import Explain from "../components/Explain.jsx";
import useExplain from "../hooks/useExplain.js";

const ORDER = ["label", "fused_score", "anomaly_score", "cdr", "confidence", "signals_agree", "regions"];

export default function Guide() {
  const { data, error } = useExplain();
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-6 py-10">
      <h1 className="m-0 font-serif text-4xl font-semibold">How to use this tool and read the results</h1>
      <Disclaimer text={data?.disclaimer} />
      {error && (
        <div role="alert" className="rounded-xl border border-red-300 bg-red-50 px-5 py-4 text-red-950">
          The explanations are loaded from the server and it could not be reached. {error}
        </div>
      )}
      {data && (
        <>
          <section className="space-y-3">
            <h2 className="m-0 font-serif text-2xl font-semibold">Using the tool</h2>
            <ol className="m-0 space-y-2 pl-6 text-[16px] leading-relaxed">
              {data.how_to_use.map((s) => (
                <li key={s}>{s}</li>
              ))}
            </ol>
          </section>

          <section className="space-y-4">
            <h2 className="m-0 font-serif text-2xl font-semibold">What each number means</h2>
            {ORDER.filter((k) => data.glossary[k]).map((k) => (
              <article key={k} className="space-y-2 rounded-xl border border-line bg-white p-5">
                <h3 className="m-0 text-lg font-semibold">{data.glossary[k].title}</h3>
                <Explain entry={data.glossary[k]} />
              </article>
            ))}
          </section>

          <section className="space-y-3">
            <h2 className="m-0 font-serif text-2xl font-semibold">Limits you should know about</h2>
            <ul className="m-0 space-y-2 pl-6 text-[16px] leading-relaxed">
              {data.limits.map((s) => (
                <li key={s}>{s}</li>
              ))}
            </ul>
          </section>
        </>
      )}
    </div>
  );
}
