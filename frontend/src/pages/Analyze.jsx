import { useEffect, useRef, useState } from "react";
import { predict } from "../api.js";
import Disclaimer from "../components/Disclaimer.jsx";
import ResultCard from "../components/ResultCard.jsx";
import useExplain from "../hooks/useExplain.js";

const MAX_MB = 10;

export default function Analyze() {
  const { data: explain } = useExplain();
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  useEffect(() => {
    if (!file) return undefined;
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function choose(f) {
    setResult(null);
    setError(null);
    if (!f) return;
    if (!/\.(jpe?g|png)$/i.test(f.name)) {
      setFile(null);
      setPreviewUrl(null);
      setError("Please choose a JPG or PNG image.");
      return;
    }
    if (f.size > MAX_MB * 1024 * 1024) {
      setFile(null);
      setPreviewUrl(null);
      setError(`That file is larger than ${MAX_MB} MB.`);
      return;
    }
    setFile(f);
  }

  async function analyze() {
    if (!file) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      setResult(await predict(file));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  function clear() {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  return (
    <div className="mx-auto max-w-6xl space-y-8 px-6 py-10">
      <div className="space-y-2">
        <h1 className="m-0 font-serif text-4xl font-semibold">Check a retinal fundus image</h1>
        <p className="m-0 max-w-3xl text-lg text-ink-soft">
          Upload a colour photo of the back of the eye, centred on the optic disc. New to this? Read{" "}
          <a href="#/guide" className="text-brand">how to read the results</a> first.
        </p>
      </div>
      <Disclaimer text={explain?.disclaimer} />

      <div className="flex flex-wrap items-start gap-8">
        <section className="flex-1 basis-96 space-y-4 rounded-2xl border border-line bg-white p-6">
          <h2 className="m-0 font-serif text-2xl font-semibold">1. Your image</h2>
          <label
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              choose(e.dataTransfer.files?.[0]);
            }}
            className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed px-6 py-10 text-center ${
              dragging ? "border-brand bg-teal-50" : "border-gray-300 bg-gray-50"
            }`}
          >
            <span className="text-lg font-medium">Drag and drop an image here</span>
            <span className="text-sm text-ink-soft">or click to choose a file (JPG or PNG, up to {MAX_MB} MB)</span>
            <input
              ref={inputRef}
              type="file"
              accept=".jpg,.jpeg,.png,image/jpeg,image/png"
              className="sr-only"
              onChange={(e) => choose(e.target.files?.[0])}
            />
          </label>
          {previewUrl && (
            <img src={previewUrl} alt="Preview of the chosen image" className="max-h-96 w-full rounded-xl bg-black object-contain" />
          )}
          <div className="flex flex-wrap gap-3">
            <button
              type="button"
              onClick={analyze}
              disabled={!file || busy}
              className="min-h-[48px] flex-1 basis-40 rounded-lg bg-brand px-6 text-base font-semibold text-white disabled:cursor-not-allowed disabled:opacity-50"
            >
              {busy ? "Analyzing..." : "Analyze image"}
            </button>
            <button
              type="button"
              onClick={clear}
              className="min-h-[48px] rounded-lg border border-gray-300 bg-white px-5 text-base font-medium"
            >
              Clear
            </button>
          </div>
          <p className="m-0 text-sm text-ink-soft">The image is analysed in memory and is not stored.</p>
        </section>

        <section className="flex-[1.5] basis-[32rem] space-y-4 rounded-2xl border border-line bg-white p-6">
          <h2 className="m-0 font-serif text-2xl font-semibold">2. Result</h2>
          {busy && (
            <p role="status" className="m-0 text-ink-soft">
              Analyzing the image. This usually takes a few seconds.
            </p>
          )}
          {error && (
            <div role="alert" className="rounded-xl border border-red-300 bg-red-50 px-5 py-4 text-red-950">
              <strong>Something went wrong.</strong> {error}
            </div>
          )}
          {!busy && !error && !result && (
            <p className="m-0 text-ink-soft">Your result will appear here after you analyze an image.</p>
          )}
          {result && <ResultCard result={result} previewUrl={previewUrl} glossary={explain?.glossary} />}
        </section>
      </div>
    </div>
  );
}
