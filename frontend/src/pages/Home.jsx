import Disclaimer from "../components/Disclaimer.jsx";

const STEPS = [
  ["Upload a fundus image", "Drag in a JPG or PNG photo of the back of the eye."],
  [
    "Compare with healthy eyes",
    "A model trained only on healthy images tries to redraw the image. Areas it redraws poorly get a higher anomaly score.",
  ],
  [
    "Measure the cup-to-disc ratio",
    "The optic cup and disc are outlined and their ratio is calculated, a number eye doctors already use.",
  ],
  ["Get a combined result", "See the label, scores, confidence and an annotated image showing where the image differs."],
];

export default function Home() {
  return (
    <div>
      <section className="mx-auto flex max-w-6xl flex-wrap items-center gap-12 px-6 py-16">
        <div className="flex-[1.2] basis-96 space-y-6">
          <span className="inline-block rounded-full bg-green-100 px-4 py-1.5 text-sm font-medium text-green-900">
            Medical anomaly detection for glaucoma
          </span>
          <h1 className="m-0 font-serif text-5xl font-semibold leading-tight">
            Spot early signs of glaucoma from a single eye photo
          </h1>
          <p className="m-0 max-w-xl text-lg leading-relaxed text-ink-soft">
            Upload a retinal fundus image. The system compares it with what healthy optic discs look like, measures the
            cup-to-disc ratio, and marks where the image differs.
          </p>
          <div className="flex flex-wrap gap-3">
            <a
              href="#/analyze"
              className="inline-flex min-h-[52px] items-center rounded-xl bg-brand px-7 text-[17px] font-semibold text-white no-underline"
            >
              Analyze an image
            </a>
            <a
              href="#/guide"
              className="inline-flex min-h-[52px] items-center rounded-xl border border-gray-300 bg-white px-7 text-[17px] font-medium text-ink no-underline"
            >
              How to read results
            </a>
          </div>
        </div>
      </section>

      <section className="border-y border-line bg-white">
        <div className="mx-auto max-w-6xl space-y-8 px-6 py-14">
          <h2 className="m-0 font-serif text-3xl font-semibold">How it works</h2>
          <div className="flex flex-wrap gap-6">
            {STEPS.map(([title, text], i) => (
              <div key={title} className="flex-1 basis-56 space-y-2 rounded-2xl border border-line p-6">
                <span className="flex h-9 w-9 items-center justify-center rounded-full bg-brand font-semibold text-white">
                  {i + 1}
                </span>
                <h3 className="m-0 text-lg font-semibold">{title}</h3>
                <p className="m-0 text-[15px] leading-relaxed text-ink-soft">{text}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-6xl space-y-4 px-6 py-14">
        <h2 className="m-0 font-serif text-3xl font-semibold">Why early screening matters</h2>
        <p className="m-0 max-w-3xl text-[17px] leading-relaxed text-ink-soft">
          Glaucoma usually causes no symptoms until the optic nerve is permanently damaged. Projections estimate about
          112 million people will be affected by 2040 (Tham et al., Ophthalmology, 2014). Manual screening is slow and
          hard to reach in rural areas, so an automated first check can help.
        </p>
        <Disclaimer text="This is a student research project. It has not been clinically validated and does not replace an eye examination." />
      </section>
    </div>
  );
}
