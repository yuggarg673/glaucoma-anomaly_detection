"""From an uploaded file to an analysis result. Shared by the Flask website (backend/app.py) and the Streamlit page
(streamlit_app.py), so both accept and refuse exactly the same files. Imports no web framework."""
from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path

from glaucoma.preprocessing import MAX_UPLOAD_PIXELS, ImageDecodeError, decode_image_bytes

from .advice import load_failure_advice
from .glossary import DISCLAIMER

log = logging.getLogger("api")
ALLOWED_EXT = {".jpg", ".jpeg", ".png"}


class UploadProblem(Exception):
    """Something the person can fix (wrong file type, unreadable image) or a failure on our side.
    `message` is safe to show: it never carries internals. `code` and `status` are what the JSON API reports."""

    def __init__(self, code: str, message: str, status: int):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def load_analyzer(cfg: dict):
    """Load the trained models. Returns (analyzer, problem, fix). Never raises: when the models cannot be loaded,
    analyzer is None and `problem` / `fix` say what went wrong and what to do next."""
    try:
        from glaucoma.inference import Analyzer
        return Analyzer(cfg), None, None
    except Exception as exc:  # noqa: BLE001  the caller must be able to keep running and show the reason
        problem, fix = load_failure_advice(exc)
        return None, problem, fix


def check_upload(filename: str, data: bytes, cfg: dict, max_pixels: int | None = None):
    """Refuse files we cannot analyse, otherwise return the decoded picture (an RGB uint8 array).

    max_pixels: pictures with more pixels are refused before decoding (default: api.max_image_pixels from the config).
    Raises UploadProblem."""
    if Path(filename or "").suffix.lower() not in ALLOWED_EXT:
        raise UploadProblem("wrong_type", "Only JPG or PNG images are accepted.", 415)
    if max_pixels is None:
        max_pixels = int(cfg["api"].get("max_image_pixels", MAX_UPLOAD_PIXELS))
    try:
        rgb = decode_image_bytes(data, max_pixels)
    except ImageDecodeError as exc:
        raise UploadProblem("unreadable_image", str(exc), 422) from exc
    h, w = rgb.shape[:2]
    min_side = cfg["api"]["min_image_side"]
    if min(h, w) < min_side:
        raise UploadProblem("image_too_small", f"The image is {w}x{h}. Each side must be at least {min_side} pixels.", 422)
    return rgb


def run_analysis(analyzer, rgb, lock=None, context: str = "") -> dict:
    """Run the trained models on a decoded picture and return the result dict (with the disclaimer added).

    lock: optional threading.Lock, so only one analysis runs at a time (keeps memory use predictable).
    context: text added to the log line if the analysis crashes (the website passes its request id).
    Raises UploadProblem (code analysis_failed) without leaking internals; the traceback goes to the log."""
    try:
        with (lock if lock is not None else nullcontext()):
            result = analyzer.analyze(rgb)
    except Exception as exc:  # noqa: BLE001
        log.exception("analysis failed %s", context)
        raise UploadProblem("analysis_failed", f"The analysis failed ({type(exc).__name__}). "
                                               "The details are in the server log.", 500) from exc
    result["disclaimer"] = DISCLAIMER
    return result


def analyze_upload(analyzer, filename: str, data: bytes, cfg: dict, lock=None, max_pixels: int | None = None,
                   context: str = "") -> dict:
    """check_upload, then run_analysis. Raises UploadProblem."""
    return run_analysis(analyzer, check_upload(filename, data, cfg, max_pixels), lock=lock, context=context)
