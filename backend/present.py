"""Turns an analysis result into the words a page shows. Pure functions: no web framework is imported, so the Streamlit
page (streamlit_app.py) and the tests can use it. The Flask website's JavaScript (web/assets/app.js) has its own copy of
the same wording; change both together."""
from __future__ import annotations

import base64
import math

VERDICT_TEXT = {"glaucoma_suspected": "Glaucoma suspected", "normal": "Normal-like"}


def fmt(value, digits: int = 3) -> str:
    """A number as text; 'n/a' when it is missing."""
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if math.isnan(number) else f"{number:.{digits}f}"


def clamp01(value) -> float:
    """A number forced into 0..1 (a progress bar refuses anything else). Missing, NaN or unreadable -> 0."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(number) else min(max(number, 0.0), 1.0)


def verdict_text(label: str) -> str:
    return VERDICT_TEXT.get(label, str(label))


def where_is(box, size) -> str:
    """Where a region box [x, y, width, height] sits in the picture as displayed. 'left' and 'right' are the screen's,
    not the patient's."""
    if not box or not size or not size.get("width") or not size.get("height"):
        return "n/a"
    cx = (box[0] + box[2] / 2) / size["width"]
    cy = (box[1] + box[3] / 2) / size["height"]
    vertical = "upper" if cy < 0.34 else "lower" if cy > 0.66 else "middle"
    horizontal = "left" if cx < 0.34 else "right" if cx > 0.66 else "centre"
    if vertical == "middle" and horizontal == "centre":
        return "centre of the image"
    if vertical == "middle":
        return f"{horizontal} side"
    if horizontal == "centre":
        return f"{vertical} part"
    return f"{vertical} {horizontal}"


def region_rows(regions, size) -> list[dict]:
    return [{"Label": r.get("id", "?"), "Difference": "High" if r.get("severity") == "high" else "Moderate",
             "Where (as shown)": where_is(r.get("box"), size)} for r in (regions or [])]


def png_from_data_url(data_url: str) -> bytes:
    """'data:image/png;base64,....' -> the picture's bytes."""
    head, _, payload = str(data_url).partition(",")
    if not head.startswith("data:image/") or ";base64" not in head:
        raise ValueError("Not a base64 image data URL.")
    return base64.b64decode(payload, validate=True)   # binascii.Error (a ValueError) on anything that is not base64


def models_line(r: dict) -> str:
    versions = " · ".join(f"{k} {'n/a' if v is None else v}" for k, v in (r.get("model_versions") or {}).items())
    trained = ", ".join(f"{k} {v}" for k, v in (r.get("trained_on") or {}).items())
    return f"Models: {versions}" + (f" · trained on: {trained}" if trained else "")


def summarize(r: dict) -> dict:
    """Everything a page shows about one result, as plain text and flags."""
    suspected = r["label"] == "glaucoma_suspected"
    anomaly_high = r["anomaly_score"] >= r["anomaly_threshold"]
    cdr = r.get("cdr")
    cdr_high = cdr is not None and cdr >= r["cdr_reference"]
    agree = r.get("signals_agree")
    return {
        "suspected": suspected,
        "verdict": verdict_text(r["label"]),
        "verdict_note": ("The combined score is above the cut-off." if suspected
                         else "The combined score is below the cut-off.") + " This is not a diagnosis.",
        "fused": fmt(r["fused_score"]),
        "fused_value": clamp01(r["fused_score"]),
        "cutoff": fmt(r["threshold"], 2),
        "anomaly": fmt(r["anomaly_score"], 4),
        "anomaly_sub": f"{'Above' if anomaly_high else 'Below'} typical cut-off {fmt(r['anomaly_threshold'], 4)}",
        "cdr": "Unavailable" if cdr is None else fmt(cdr, 2),
        "cdr_sub": (r.get("cdr_reason") or "Could not be measured") if cdr is None
        else f"{'At or above' if cdr_high else 'Below'} the {fmt(r['cdr_reference'], 2)} reference",
        "confidence": r["confidence_level"],
        "confidence_sub": "Based on the anomaly score only" if agree is None
        else "Anomaly score and CDR agree" if agree else "Anomaly score and CDR disagree",
        "low_confidence": r["confidence_level"] == "Low",
        "regions": region_rows(r.get("regions"), r.get("annotated_size") or r.get("image_size")),
        "models": models_line(r),
    }


def how_to_steps(steps) -> list[str]:
    """The website's how-to list, reworded for a page that has no separate 'Analyze page'."""
    return [str(s).replace(" on the Analyze page", " here") for s in steps]


def limits_without_history(limits) -> list[str]:
    """The limits list minus the line about the website's history table (pages that keep no history)."""
    return [str(x) for x in limits if "history" not in str(x).lower()]
