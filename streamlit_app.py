"""Streamlit version of the glaucoma screening page.

Run it on your computer:   streamlit run streamlit_app.py
Put it online:             Streamlit Community Cloud, main file path `streamlit_app.py` (steps: docs/DEPLOY.md)

This file is only the screen. The work is done by the same code as the Flask website:
  backend/service.py   checks the upload and runs the trained models
  backend/present.py   turns a result into words and numbers
  backend/advice.py    says what to do when the models cannot be loaded
  backend/glossary.py  the plain-language explanations
It never shows a result it did not compute: without trained models it says so and stops.
It saves no image and keeps no history.

Status: written without being able to install Streamlit, so it has only been run against a recording stand-in for the
`streamlit` module (tests/test_streamlit_page.py). The first real run is the real test.
"""
from __future__ import annotations

import hashlib
import io
import logging
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for _p in (ROOT / "src", ROOT):       # Streamlit runs this file directly, so make the project importable
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from backend import present  # noqa: E402
from backend.glossary import DISCLAIMER, GLOSSARY, HOW_TO_USE, LIMITS  # noqa: E402
from backend.service import UploadProblem, check_upload, load_analyzer, run_analysis  # noqa: E402
from glaucoma.config import load_config  # noqa: E402

log = logging.getLogger("streamlit_app")

SHORT_NOTE = "A student research project. Not a medical device and not a diagnosis."
PRIVACY = ("This page does not save your image and keeps no history. It is still a public web page when deployed: "
           "do not upload identifiable patient images to it.")
TEST_MODELS_TEXT = ("**These are test models.** They were trained on synthetic drawings as a code check, not on eye "
                    "photos, so every result is meaningless. Train the real models (`python scripts/run_pipeline.py` or "
                    "the Colab notebook), add them to the repository and redeploy: see docs/DEPLOY.md.")
STREAMLIT_FIX = ("On Streamlit Community Cloud the trained files must be in your GitHub repository: add them with "
                 "`git add -f checkpoints/...`, push, then reboot the app (docs/DEPLOY.md, part 3).")
ANNOTATED_CAPTION = ("Made by the model. Teal outline = optic disc, white outline = cup. Dashed boxes mark where the "
                     "redrawn image differs most: red = high difference, amber = moderate. They are not confirmed "
                     "lesions. Left and right are the screen's, not the patient's.")
GLOSSARY_ORDER = ("label", "fused_score", "anomaly_score", "cdr", "confidence", "signals_agree", "regions")
PREVIEW_SIDE = 640   # the preview is shrunk to this many pixels on its longest side; the analysis uses the full picture


@st.cache_resource(show_spinner="Loading the trained models ...")
def get_models():
    """Loaded once per server and shared by all visitors (the models hold no personal data).
    Returns (analyzer, problem, fix). Never raises."""
    return load_analyzer(load_config())


@st.cache_resource
def get_lock():
    """One analysis at a time: keeps memory use predictable on a small free server."""
    return threading.Lock()


def glossary_text(key: str) -> str:
    g = GLOSSARY[key]
    return f"{g['what']} {g['read']}"


def side_panel():
    with st.sidebar:
        st.header("How to use")
        st.markdown("\n".join(f"{i}. {step}" for i, step in enumerate(present.how_to_steps(HOW_TO_USE), 1)))
        st.header("Limits")
        st.markdown("\n".join(f"- {item}" for item in present.limits_without_history(LIMITS)))
        st.caption(PRIVACY)


def explanations():
    st.subheader("What the numbers mean")
    for key in GLOSSARY_ORDER:
        g = GLOSSARY[key]
        with st.expander(g["title"]):
            st.markdown(f"**What it is.** {g['what']}")
            st.markdown(f"**How to read it.** {g['read']}")
            if g.get("caution"):
                st.markdown(f"**Be careful.** {g['caution']}")


def preview_of(rgb):
    """A small copy of the picture for the screen."""
    h, w = rgb.shape[:2]
    scale = PREVIEW_SIDE / max(h, w)
    if scale < 1:
        rgb = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    return rgb


def show_result(r: dict, preview):
    s = present.summarize(r)
    if r.get("test_models"):
        st.error(TEST_MODELS_TEXT)
    if s["suspected"]:
        st.warning(f"**{s['verdict']}**  \n{s['verdict_note']}")
    else:
        st.info(f"**{s['verdict']}**  \n{s['verdict_note']}")
    if s["low_confidence"]:
        st.caption("Confidence is Low: the score is close to the cut-off, so treat this result as unclear.")

    cols = st.columns(4)
    with cols[0]:
        st.metric("Combined score", s["fused"], help=glossary_text("fused_score"))
        st.caption(f"Cut-off {s['cutoff']}")
    with cols[1]:
        st.metric("Anomaly score", s["anomaly"], help=glossary_text("anomaly_score"))
        st.caption(s["anomaly_sub"])
    with cols[2]:
        st.metric("Cup-to-disc ratio", s["cdr"], help=glossary_text("cdr"))
        st.caption(s["cdr_sub"])
    with cols[3]:
        st.metric("Confidence", s["confidence"], help=glossary_text("confidence"))
        st.caption(s["confidence_sub"])
    st.progress(s["fused_value"])
    st.caption(f"Combined score {s['fused']} on a 0 to 1 scale. The label changes at the cut-off {s['cutoff']}. "
               "Confidence is not the probability of disease.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Your image**")
        st.image(preview)
    with right:
        st.markdown("**Annotated image**")
        try:
            png = present.png_from_data_url(r["annotated_image"])
        except (KeyError, ValueError):
            st.warning("The annotated image could not be shown.")
        else:
            st.image(io.BytesIO(png))
            st.caption(ANNOTATED_CAPTION)
            st.download_button("Download the annotated image (PNG)", data=png, file_name="annotated_fundus.png",
                               mime="image/png")

    st.markdown("**Marked regions**")
    if s["regions"]:
        st.table(pd.DataFrame(s["regions"]).set_index("Label"))
    else:
        st.caption("No regions were marked.")
    st.caption(s["models"])


def main():
    st.set_page_config(page_title="Glaucoma Screening", page_icon="👁️", layout="wide")
    st.title("Glaucoma Screening")
    st.caption(SHORT_NOTE)
    side_panel()

    cfg = load_config()
    analyzer, problem, fix = get_models()
    if analyzer is None:
        st.error(f"The trained models are not loaded, so images cannot be analysed. Reason: {problem}")
        st.info(f"{fix}\n\n{STREAMLIT_FIX}")
        if st.button("Try loading the models again"):
            get_models.clear()
            st.rerun()
        st.stop()
    if getattr(analyzer, "test_models", False):
        st.error(TEST_MODELS_TEXT)

    max_mb = int(cfg["api"]["max_upload_mb"])
    uploaded = st.file_uploader("Choose a retinal photo (JPG or PNG)", type=["jpg", "jpeg", "png"],
                                help=f"Up to {max_mb} MB. A colour fundus photo cropped around the optic disc works best.")
    if uploaded is None:
        st.session_state.pop("analysis", None)
        st.info("Choose an image above to begin. The sidebar explains which pictures work best.")
        explanations()
        st.caption(DISCLAIMER)
        return

    data = uploaded.getvalue()
    key = f"{uploaded.name}|{len(data)}|{hashlib.sha256(data).hexdigest()[:16]}"
    try:
        rgb = check_upload(uploaded.name, data, cfg)
    except UploadProblem as exc:
        st.error(exc.message)
        st.session_state.pop("analysis", None)
        st.stop()
    preview = preview_of(rgb)
    st.text(f"File: {uploaded.name}   ({rgb.shape[1]} x {rgb.shape[0]} pixels)")

    if st.button("Analyze image", type="primary"):
        with st.spinner("Analysing ..."):
            try:
                st.session_state["analysis"] = {"key": key, "result": run_analysis(analyzer, rgb, lock=get_lock())}
            except UploadProblem as exc:
                st.session_state["analysis"] = {"key": key, "error": exc.message}

    analysis = st.session_state.get("analysis")
    if analysis and analysis.get("key") == key:
        if "error" in analysis:
            st.error(analysis["error"])
        else:
            try:
                show_result(analysis["result"], preview)
            except Exception:  # noqa: BLE001  never show a traceback to a visitor; the log has it
                log.exception("could not display a result")
                st.error("The result could not be displayed. The details are in the app log.")
    else:
        st.image(preview)
        st.caption("Press **Analyze image** to run the models.")
    explanations()
    st.caption(DISCLAIMER)


main()
