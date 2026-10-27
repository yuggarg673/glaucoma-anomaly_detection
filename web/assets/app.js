/* Glaucoma Screening website.

   Plain JavaScript: no build step, no libraries, nothing to install. The backend serves this file
   (python -m backend.app, then open http://localhost:5000).

   Rules this file follows:
   - Every URL is relative ("predict", not "/predict"), so the site also works behind a proxy such as Colab's.
   - Text that comes from the server or from the user (file names, messages) is only ever added with
     textContent / text nodes, never innerHTML.
   - Every network call is in api() below, so there is one place to look when something fails.
*/
"use strict";

(function () {
  // ------------------------------------------------------------------ small helpers
  /** Build an element: h("div", {class: "x", onclick: fn}, "text", childElement, ...). */
  function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "style") for (const [p, val] of Object.entries(v)) el.style.setProperty(p, val);
      else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
      else if (v === true) el.setAttribute(k, "");
      else el.setAttribute(k, String(v));
    }
    for (const kid of kids.flat(Infinity)) {
      if (kid === null || kid === undefined || kid === false) continue;
      el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    }
    return el;
  }

  const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(Number(v)) ? "n/a" : Number(v).toFixed(d));
  const clamp01 = (v) => Math.min(Math.max(Number(v) || 0, 0), 1);

  /** data:image/png;base64,... -> Blob (so the picture can be opened and downloaded; browsers block data: links). */
  function dataUrlToBlob(dataUrl) {
    const comma = dataUrl.indexOf(",");
    const mime = (/^data:([^;,]+)/.exec(dataUrl) || [])[1] || "image/png";
    const bin = atob(dataUrl.slice(comma + 1));
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Blob([bytes], { type: mime });
  }

  /** Words for where a region box sits, as the picture is displayed (left/right are the screen's, not the patient's). */
  function whereIs(box, size) {
    if (!box || !size || !size.width || !size.height) return "n/a";
    const cx = (box[0] + box[2] / 2) / size.width;
    const cy = (box[1] + box[3] / 2) / size.height;
    const v = cy < 0.34 ? "upper" : cy > 0.66 ? "lower" : "middle";
    const hz = cx < 0.34 ? "left" : cx > 0.66 ? "right" : "centre";
    if (v === "middle" && hz === "centre") return "centre of the image";
    if (v === "middle") return `${hz} side`;
    if (hz === "centre") return `${v} part`;
    return `${v} ${hz}`;
  }

  // ------------------------------------------------------------------ talking to the server
  class ApiError extends Error {}

  /** A request the page itself stopped (Clear, another file, leaving the page, the time limit): err.aborted is true. */
  function failure(e) {
    if (e && e.name === "AbortError") {
      const stopped = new ApiError("The request was stopped.");
      stopped.aborted = true;
      return stopped;
    }
    return new ApiError("Cannot reach the analysis server. Check that it is still running (python -m backend.app), then reload this page.");
  }

  async function api(path, options) {
    let res;
    try {
      res = await fetch(path, options);
    } catch (e) {
      throw failure(e);
    }
    let body = null;
    try {
      body = await res.json();
    } catch (e) {
      if (e && e.name === "AbortError") throw failure(e);   // stopped while the answer was still arriving
      /* the answer was not JSON */
    }
    if (!res.ok) {
      const message = (body && body.error && body.error.message) || `The server answered with status ${res.status}.`;
      const ref = body && body.request_id ? ` (reference ${body.request_id}, the same id is in the server log)` : "";
      throw new ApiError(message + ref);
    }
    if (body === null) throw new ApiError("The server sent an answer this page cannot read. Look at the server log.");
    return body;
  }

  const getHistory = (limit) => api(`history?limit=${limit || 50}`);
  const predict = (file, signal) => {
    const form = new FormData();
    form.append("image", file);
    return api("predict", { method: "POST", body: form, signal });
  };
  let explainPromise = null;   // the plain-language texts are fetched once
  function getExplain() {
    if (!explainPromise) explainPromise = api("explain").catch((e) => { explainPromise = null; throw e; });
    return explainPromise;
  }

  // ------------------------------------------------------------------ server status (the pill in the header)
  // kind: checking | ok | test (models trained on synthetic data, meaningless) | degraded (not loaded) | down
  const TRAIN_HINT = "Train them with `python scripts/run_pipeline.py` (or the Colab notebook), then restart the server.";
  const health = { kind: "checking", data: null };
  const PILL_TEXT = {
    checking: "Checking server",
    ok: "Server online, models loaded",
    test: "Server online, TEST models only",
    degraded: "Server online, models not loaded",
    down: "Server unreachable",
  };

  function paintPill() {
    const pill = document.getElementById("status-pill");
    pill.className = "pill pill-" + health.kind;
    pill.replaceChildren(h("span", { class: "dot" }), PILL_TEXT[health.kind]);
    pill.title = (health.data && health.data.detail) || "";
  }

  async function checkHealth() {
    let kind;
    let data = null;
    try {
      data = await api("health");
      kind = !data.models_loaded ? "degraded" : data.test_models ? "test" : "ok";
    } catch (e) {
      kind = "down";
    }
    const changed = kind !== health.kind;
    health.kind = kind;
    health.data = data;
    if (changed) {
      paintPill();
      document.dispatchEvent(new CustomEvent("healthchange"));
    }
  }

  // ------------------------------------------------------------------ pieces used by several pages
  /** "run `python x.py` now" -> text with the backticked part in <code> (built from text nodes, never markup). */
  function withCode(text) {
    return String(text || "").split("`").map((part, i) => (i % 2 === 1 ? h("code", {}, part) : part));
  }

  function disclaimerBox(text) {
    return h("div", { class: "note note-amber", role: "note" },
      h("strong", {}, "Screening aid only."), " ",
      text || "This tool does not give a medical diagnosis. Results should be reviewed by an ophthalmologist.");
  }

  function errorBox(message) {
    return h("div", { class: "note note-red", role: "alert" }, h("strong", {}, "Something went wrong. "), message);
  }

  /** One glossary entry (what it is / how to read it / caution) from GET /explain. */
  function explainBlock(entry) {
    if (!entry) return null;
    return h("div", { class: "explain" },
      h("p", {}, h("strong", {}, "What it is. "), entry.what),
      h("p", {}, h("strong", {}, "How to read it. "), entry.read),
      entry.caution ? h("p", { class: "careful" }, h("strong", {}, "Be careful. "), entry.caution) : null);
  }

  // ------------------------------------------------------------------ Home
  const STEPS = [
    ["Upload a fundus image", "Drag in a JPG or PNG photo of the back of the eye, cropped close around the optic disc."],
    ["Compare with healthy eyes", "A model trained only on healthy images tries to redraw the image. Areas it redraws poorly get a higher anomaly score."],
    ["Measure the cup-to-disc ratio", "The optic cup and disc are outlined and their ratio is calculated, a number eye doctors already use."],
    ["Get a combined result", "See the label, scores, confidence and an annotated image showing where the image differs."],
  ];

  function renderHome(main) {
    main.replaceChildren(
      h("section", { class: "container hero" },
        h("div", { class: "hero-text" },
          h("span", { class: "badge" }, "Medical anomaly detection for glaucoma"),
          h("h1", {}, "Spot early signs of glaucoma from a single eye photo"),
          h("p", { class: "lead" },
            "Upload a retinal fundus image. The system compares it with what healthy optic discs look like, measures the cup-to-disc ratio, and marks where the image differs."),
          h("div", { class: "row" },
            h("a", { class: "btn btn-primary btn-lg", href: "#/analyze" }, "Analyze an image"),
            h("a", { class: "btn btn-secondary btn-lg", href: "#/guide" }, "How to read results"))),
        h("div", { class: "hero-art" }, h("img", { src: "assets/hero.svg", alt: "", width: 420, height: 420 }))),
      h("section", { class: "band" },
        h("div", { class: "container" },
          h("h2", {}, "How it works"),
          h("div", { class: "steps" },
            STEPS.map(([title, text], i) =>
              h("div", { class: "step" }, h("span", { class: "step-num" }, i + 1), h("h3", {}, title), h("p", {}, text)))))),
      h("section", { class: "container section" },
        h("h2", {}, "Why early screening matters"),
        h("p", { class: "lead" },
          "Glaucoma usually causes no symptoms until the optic nerve is permanently damaged. Projections estimate about 112 million people will be affected by 2040 (Tham et al., Ophthalmology, 2014). Manual screening is slow and hard to reach in rural areas, so an automated first check can help."),
        disclaimerBox("This is a student research project. It has not been clinically validated and does not replace an eye examination.")));
  }

  // ------------------------------------------------------------------ Analyze
  function tile(title, value, sub, warn) {
    return h("div", { class: "tile" },
      h("span", { class: "tile-title" }, title),
      h("span", { class: "tile-value" }, value),
      h("span", { class: "tile-sub" + (warn ? " warn" : "") }, sub));
  }

  /** Full-width "what does this show?" panels under the three number tiles (a narrow tile made them unreadable). */
  function helpPanels(g) {
    const items = [
      ["What does the anomaly score show?", g.anomaly_score],
      ["What does the cup-to-disc ratio show?", g.cdr],
      ["What does the confidence show?", g.confidence],
    ].filter(([, entry]) => entry);
    if (items.length === 0) return null;
    return h("div", { class: "help" }, items.map(([question, entry]) =>
      h("details", {}, h("summary", {}, question), explainBlock(entry))));
  }

  function legendLine(color, dashed, text) {
    return h("p", {}, h("span", { class: "swatch" + (dashed ? " dashed" : ""), style: { "border-top-color": color } }), text);
  }

  function resultCard(r, previewUrl, annotatedUrl, g) {
    g = g || {};
    const suspected = r.label === "glaucoma_suspected";
    const anomHigh = r.anomaly_score >= r.anomaly_threshold;
    const cdrHigh = r.cdr !== null && r.cdr !== undefined && r.cdr >= r.cdr_reference;
    const noCdr = r.cdr === null || r.cdr === undefined;
    const regions = r.regions || [];
    const size = r.annotated_size || r.image_size;
    const versions = Object.entries(r.model_versions || {}).map(([k, v]) => `${k} ${v === null || v === undefined ? "n/a" : v}`).join(" · ");

    const trainedOn = r.trained_on
      ? Object.entries(r.trained_on).map(([k, v]) => `${k} ${v}`).join(", ")
      : null;

    return h("div", { class: "result" },
      r.test_models
        ? h("p", { class: "note note-red", role: "note" },
          h("strong", {}, "Test models: ignore this result. "),
          "The models behind it were not trained on real eye photos, so the label and numbers mean nothing.")
        : null,
      h("div", { class: "verdict", id: "result-top", tabindex: "-1" },
        h("span", { class: "verdict-pill " + (suspected ? "verdict-suspected" : "verdict-normal") },
          h("span", { class: "dot" }), suspected ? "Glaucoma suspected" : "Normal-like"),
        h("span", { class: "muted" },
          (suspected ? "The combined score is above the cut-off." : "The combined score is below the cut-off.") + " This is not a diagnosis.")),

      h("div", {},
        h("div", { class: "scorebar-head" },
          h("span", {}, h("strong", {}, `Combined score: ${fmt(r.fused_score)}`)),
          h("span", { class: "muted" }, `Cut-off ${fmt(r.threshold, 2)} (chosen on validation images)`)),
        h("div", { class: "scorebar", role: "img", "aria-label": `Combined score ${fmt(r.fused_score, 2)} on a scale from 0 to 1, cut-off ${fmt(r.threshold, 2)}` },
          h("div", { class: "cut", style: { left: clamp01(r.threshold) * 100 + "%" } }),
          h("div", { class: "knob", style: { left: clamp01(r.fused_score) * 100 + "%" } })),
        h("div", { class: "scorebar-scale" }, h("span", {}, "0 · normal-like"), h("span", {}, "1 · more anomalous"))),

      h("div", { class: "tiles" },
        tile("Anomaly score", fmt(r.anomaly_score, 4),
          `${anomHigh ? "Above" : "Below"} typical cut-off ${fmt(r.anomaly_threshold, 4)}`, anomHigh),
        tile("Cup-to-disc ratio", noCdr ? "Unavailable" : fmt(r.cdr, 2),
          noCdr ? (r.cdr_reason || "Could not be measured")
            : `${cdrHigh ? "At or above" : "Below"} the ${fmt(r.cdr_reference, 2)} reference`, cdrHigh),
        tile("Confidence", r.confidence_level,
          r.signals_agree === null || r.signals_agree === undefined ? "Based on the anomaly score only"
            : r.signals_agree ? "Anomaly score and CDR agree" : "Anomaly score and CDR disagree",
          r.signals_agree === false)),

      helpPanels(g),

      r.confidence_level === "Low"
        ? h("p", { class: "note note-amber" },
          "Confidence is low: the score is close to the cut-off, so treat this result as unclear and have the image reviewed.")
        : null,

      h("section", { class: "divider" },
        h("div", { class: "sec-head" },
          h("h3", {}, "Annotated image generated by the model"),
          h("div", { class: "row" },
            h("a", { class: "btn btn-secondary", href: annotatedUrl, target: "_blank", rel: "noopener" }, "Open full size"),
            h("a", { class: "btn btn-secondary", href: annotatedUrl, download: "annotated_fundus.png" }, "Download annotated image"))),
        h("div", { class: "fig-row" },
          previewUrl
            ? h("figure", { class: "fig" }, h("img", { src: previewUrl, alt: "The image you uploaded" }), h("figcaption", {}, "Original image"))
            : null,
          h("figure", { class: "fig" },
            h("img", { src: annotatedUrl, alt: "The same image with the optic disc, cup and unusual regions outlined and labelled" }),
            h("figcaption", {}, "Annotated image")),
          h("div", { class: "legend" },
            h("p", {}, h("strong", {}, "Legend")),
            legendLine("#FF5A4E", true, "A1, A2 ...: high difference from healthy eyes"),
            legendLine("#FFB020", true, "Moderate difference"),
            legendLine("#4FD1C5", false, "Optic disc outline"),
            legendLine("#9CA3AF", false, "Optic cup outline (white on the image)"),
            g.regions ? h("p", { class: "muted small" }, g.regions.read) : null,
            regions.length === 0 ? h("p", { class: "muted small" }, "No region stood out above the healthy-eye range.") : null)),
        regions.length > 0
          ? h("div", { class: "table-wrap regions" },
            h("table", {},
              h("caption", { class: "sr-only" }, "Regions marked on the annotated image"),
              h("thead", {}, h("tr", {}, h("th", { scope: "col" }, "Label"), h("th", { scope: "col" }, "Difference"), h("th", { scope: "col" }, "Where (as shown)"))),
              h("tbody", {}, regions.map((x) =>
                h("tr", {}, h("td", {}, x.id), h("td", {}, x.severity === "high" ? "High" : "Moderate"), h("td", {}, whereIs(x.box, size)))))))
          : null),

      h("p", { class: "fine" }, `Models: ${versions}`, trainedOn ? ` · trained on: ${trainedOn}` : ""));
  }

  function renderAnalyze(main) {
    let file = null;
    let previewUrl = null;
    let annotatedUrl = null;
    let controller = null;
    let glossary = null;
    const maxMb = () => (health.data && health.data.max_upload_mb) || 10;
    // A non-breaking space keeps "10 MB" together when the line wraps.
    const dropHint = () => `or click to choose a file (JPG or PNG, up to ${maxMb()} MB)`;
    let busy = false;
    let hasError = false;
    let hasResult = false;

    const modelNote = h("div", { class: "note note-amber", role: "status", hidden: true });
    const disclaimerSlot = h("div", {}, disclaimerBox());
    const fileInput = h("input", { type: "file", id: "file", class: "sr-only", accept: ".jpg,.jpeg,.png,image/jpeg,image/png" });
    const dropText = h("span", { class: "muted" }, dropHint());
    const dropzone = h("label", { class: "dropzone", for: "file" }, h("span", { class: "dz-title" }, "Drag and drop an image here"), dropText, fileInput);
    const preview = h("img", { class: "preview", alt: "Preview of the chosen image", hidden: true });
    const analyzeBtn = h("button", { type: "button", class: "btn btn-primary grow", disabled: true }, "Analyze image");
    const clearBtn = h("button", { type: "button", class: "btn btn-secondary" }, "Clear");
    const busyBox = h("div", { role: "status", "aria-live": "polite" });   // "Analyzing..." (kept apart from errors)
    const errorSlot = h("div", {});                                       // the error box has role="alert" itself
    const resultBox = h("div", { "aria-live": "polite" });
    const placeholder = h("p", { class: "result-empty" }, "Your result will appear here after you analyze an image.");

    function paintModelNote() {
      if (health.kind === "degraded") {
        modelNote.hidden = false;
        modelNote.className = "note note-amber";
        modelNote.replaceChildren(
          h("strong", {}, "The trained models are not loaded yet, so images cannot be analysed. "),
          health.data && health.data.detail ? `Reason: ${health.data.detail} ` : "",
          ...withCode((health.data && health.data.fix) || TRAIN_HINT));   // replaceChildren needs nodes or strings, not an array
      } else if (health.kind === "test") {
        modelNote.hidden = false;
        modelNote.className = "note note-red";
        modelNote.replaceChildren(
          h("strong", {}, "These are test models. "),
          "They were trained on synthetic drawings as a code check, not on eye photos, so every result is meaningless. Train the real models with ",
          h("code", {}, "python scripts/run_pipeline.py"), " (or the Colab notebook), then restart the server.");
      } else if (health.kind === "down") {
        modelNote.hidden = false;
        modelNote.className = "note note-red";
        modelNote.replaceChildren(
          h("strong", {}, "Cannot reach the analysis server. "),
          location.protocol === "file:"
            ? "This page was opened as a file. Start the server with python -m backend.app and open http://localhost:5000 instead."
            : "Check that it is running (python -m backend.app), then reload this page.");
      } else {
        modelNote.hidden = true;
      }
      dropText.textContent = dropHint();
    }

    // The "Your result will appear here" line only shows while nothing else is in the result panel.
    function syncPlaceholder() {
      placeholder.hidden = busy || hasError || hasResult;
    }

    function setBusy(on) {
      busy = on;
      analyzeBtn.disabled = on || !file;
      analyzeBtn.textContent = on ? "Analyzing..." : "Analyze image";
      busyBox.replaceChildren(on ? h("div", { class: "busy" }, h("span", { class: "spinner", "aria-hidden": "true" }), "Analyzing the image. This usually takes a few seconds.") : "");
      syncPlaceholder();
    }

    function showError(message) {
      hasError = Boolean(message);
      errorSlot.replaceChildren(message ? errorBox(message) : "");
      syncPlaceholder();
    }

    function dropResult() {
      if (annotatedUrl) URL.revokeObjectURL(annotatedUrl);
      annotatedUrl = null;
      hasResult = false;
      resultBox.replaceChildren();
      syncPlaceholder();
    }

    // A file the browser cannot draw (a renamed text file, a damaged JPG) would show a broken-image icon.
    // Hide the preview; the server still decides, and says so clearly if the file is unreadable.
    preview.addEventListener("error", () => { preview.hidden = true; });

    function setFile(f) {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      file = f;
      previewUrl = f ? URL.createObjectURL(f) : null;
      if (previewUrl) preview.src = previewUrl; else preview.removeAttribute("src");
      preview.hidden = !f;
      analyzeBtn.disabled = !f;
    }

    function choose(f) {
      stopRequest();      // a result for the previous file must never appear next to the new picture
      dropResult();
      showError(null);
      if (!f) return;
      if (!/\.(jpe?g|png)$/i.test(f.name)) { setFile(null); fileInput.value = ""; showError("Please choose a JPG or PNG image."); return; }
      if (f.size > maxMb() * 1024 * 1024) { setFile(null); fileInput.value = ""; showError(`That file is larger than ${maxMb()} MB.`); return; }
      setFile(f);
    }

    function stopRequest() {
      if (controller) controller.abort();
    }

    async function analyze() {
      if (!file || controller) return;
      const mine = new AbortController();
      controller = mine;
      let timedOut = false;
      const timer = setTimeout(() => { timedOut = true; mine.abort(); }, 180000);
      dropResult();
      showError(null);
      setBusy(true);
      try {
        const res = await predict(file, mine.signal);
        annotatedUrl = URL.createObjectURL(dataUrlToBlob(res.annotated_image));
        hasResult = true;
        syncPlaceholder();
        resultBox.replaceChildren(resultCard(res, previewUrl, annotatedUrl, glossary));
        const top = document.getElementById("result-top");
        if (top) top.focus({ preventScroll: false });
      } catch (e) {
        // Clear, another file, or leaving the page stops the request on purpose: show nothing for that.
        if (e.aborted && !timedOut) return;
        showError(timedOut
          ? "The analysis took too long and was stopped. Try a smaller image, or look at the server log."
          : e.message);
      } finally {
        clearTimeout(timer);
        if (controller === mine) controller = null;
        setBusy(false);
      }
    }

    function clearAll() {
      stopRequest();
      setFile(null);
      fileInput.value = "";
      dropResult();
      showError(null);
    }

    fileInput.addEventListener("change", () => choose(fileInput.files && fileInput.files[0]));
    analyzeBtn.addEventListener("click", analyze);
    clearBtn.addEventListener("click", clearAll);
    ["dragenter", "dragover"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("dragging"); }));
    ["dragleave", "drop"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("dragging"); }));
    dropzone.addEventListener("drop", (e) => choose(e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]));
    // A file dropped outside the box must not make the browser leave the page and open the picture.
    const stopStray = (e) => e.preventDefault();
    window.addEventListener("dragover", stopStray);
    window.addEventListener("drop", stopStray);
    document.addEventListener("healthchange", paintModelNote);

    getExplain().then((d) => {
      glossary = d.glossary;
      disclaimerSlot.replaceChildren(disclaimerBox(d.disclaimer));
    }).catch(() => { /* the page works without the explanations */ });

    main.replaceChildren(
      h("div", { class: "container page" },
        h("div", { class: "page-head" },
          h("h1", {}, "Check a retinal fundus image"),
          h("p", { class: "lead" }, "Upload a colour photo of the back of the eye, cropped close around the optic disc. New to this? Read ",
            h("a", { href: "#/guide" }, "how to read the results"), " first.")),
        disclaimerSlot,
        modelNote,
        h("div", { class: "cols" },
          h("section", { class: "card card-pad col-input", "aria-labelledby": "step1" },
            h("h2", { id: "step1" }, "1. Your image"),
            dropzone, preview,
            h("div", { class: "row" }, analyzeBtn, clearBtn),
            h("p", { class: "muted small" }, "The image is analysed in memory and is not stored.")),
          h("section", { class: "card card-pad col-result", "aria-labelledby": "step2" },
            h("h2", { id: "step2" }, "2. Result"),
            busyBox, errorSlot, placeholder, resultBox))));
    paintModelNote();

    return function cleanup() {
      if (controller) controller.abort();
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      if (annotatedUrl) URL.revokeObjectURL(annotatedUrl);
      window.removeEventListener("dragover", stopStray);
      window.removeEventListener("drop", stopStray);
      document.removeEventListener("healthchange", paintModelNote);
    };
  }

  // ------------------------------------------------------------------ Guide
  const GUIDE_ORDER = ["label", "fused_score", "anomaly_score", "cdr", "confidence", "signals_agree", "regions"];

  function renderGuide(main) {
    const body = h("div", { class: "stack-lg" }, h("p", { class: "muted" }, "Loading the explanations..."));
    main.replaceChildren(
      h("div", { class: "container page page-narrow" },
        h("h1", {}, "How to use this tool and read the results"),
        h("div", { class: "disclaimer-slot" }, disclaimerBox()),
        body));
    let alive = true;
    getExplain().then((d) => {
      if (!alive) return;
      main.querySelector(".disclaimer-slot").replaceChildren(disclaimerBox(d.disclaimer));
      body.replaceChildren(
        h("section", { class: "stack" }, h("h2", {}, "Using the tool"),
          h("ol", { class: "guide-list" }, d.how_to_use.map((s) => h("li", {}, s)))),
        h("section", { class: "stack" }, h("h2", {}, "What each number means"),
          GUIDE_ORDER.filter((k) => d.glossary[k]).map((k) =>
            h("article", { class: "guide-item" }, h("h3", {}, d.glossary[k].title), explainBlock(d.glossary[k])))),
        h("section", { class: "stack" }, h("h2", {}, "Limits you should know about"),
          h("ul", { class: "guide-list" }, d.limits.map((s) => h("li", {}, s)))));
    }).catch((e) => {
      if (alive) body.replaceChildren(errorBox("The explanations are loaded from the server and it could not be reached. " + e.message));
    });
    return () => { alive = false; };
  }

  // ------------------------------------------------------------------ History
  const LABELS = { glaucoma_suspected: "Glaucoma suspected", normal: "Normal-like" };

  function renderHistory(main) {
    const holder = h("div", {}, h("p", { class: "muted" }, "Loading..."));
    let alive = true;

    function load() {
      holder.replaceChildren(h("p", { class: "muted" }, "Loading..."));
      getHistory(50).then((d) => {
        if (!alive) return;
        const items = d.items || [];
        if (items.length === 0) { holder.replaceChildren(h("p", {}, "Nothing here yet. Analyze an image first.")); return; }
        holder.replaceChildren(h("div", { class: "table-wrap" },
          h("table", { style: { "min-width": "640px" } },
            h("thead", {}, h("tr", {}, ["Time", "File", "Label", "Anomaly", "CDR", "Fused", "Confidence"].map((t) => h("th", { scope: "col" }, t)))),
            h("tbody", {}, items.map((r) => {
              const when = new Date(r.created_at);
              return h("tr", {},
                h("td", { title: r.created_at + " (UTC)" }, Number.isNaN(when.getTime()) ? r.created_at : when.toLocaleString()),
                h("td", {}, r.filename),
                h("td", {}, LABELS[r.label] || r.label),
                h("td", {}, r.anomaly_score === null ? "n/a" : fmt(r.anomaly_score, 4)),
                h("td", {}, r.cdr === null || r.cdr === undefined ? "unavailable" : fmt(r.cdr, 2)),
                h("td", {}, r.fused_score === null ? "n/a" : fmt(r.fused_score, 3)),
                h("td", {}, r.confidence_level));
            })))));
      }).catch((e) => { if (alive) holder.replaceChildren(errorBox(e.message)); });
    }

    main.replaceChildren(
      h("div", { class: "container page" },
        h("div", { class: "page-head" },
          h("h1", {}, "Recent analyses"),
          h("p", { class: "muted" }, "Images are not stored. Only the file name and scores are saved on the server.")),
        h("div", {}, h("button", { type: "button", class: "btn btn-secondary", onclick: load }, "Refresh")),
        holder));
    load();
    return () => { alive = false; };
  }

  function renderNotFound(main) {
    main.replaceChildren(h("div", { class: "container page" },
      h("h1", {}, "Page not found"),
      h("p", { class: "lead" }, "There is no page at this address. ", h("a", { href: "#/" }, "Go to the home page"), ".")));
  }

  // ------------------------------------------------------------------ router
  const ROUTES = {
    "": ["Home", renderHome],
    analyze: ["Analyze", renderAnalyze],
    guide: ["How to read results", renderGuide],
    history: ["History", renderHistory],
  };
  let cleanup = null;

  function route(initial) {
    if (cleanup) { try { cleanup(); } catch (e) { /* ignore */ } cleanup = null; }
    const name = location.hash.replace(/^#\/?/, "").split("?")[0];
    const entry = ROUTES[name];
    const main = document.getElementById("main");
    document.title = (entry ? entry[0] : "Page not found") + " - Glaucoma Screening";
    document.querySelectorAll(".nav-link").forEach((a) => {
      if (entry && a.dataset.route === name) a.setAttribute("aria-current", "page");
      else a.removeAttribute("aria-current");
    });
    try {
      cleanup = (entry ? entry[1] : renderNotFound)(main) || null;
    } catch (e) {
      main.replaceChildren(h("div", { class: "container page" }, errorBox("This page could not be shown: " + e.message)));
    }
    if (!initial) { window.scrollTo(0, 0); main.focus({ preventScroll: true }); }
  }

  window.addEventListener("hashchange", () => route(false));
  document.addEventListener("visibilitychange", () => { if (!document.hidden) checkHealth(); });
  setInterval(() => { if (!document.hidden) checkHealth(); }, 15000);
  paintPill();
  route(true);
  checkHealth();
})();
