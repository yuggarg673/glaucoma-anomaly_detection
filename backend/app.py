"""Flask API.

Run:  python -m backend.app            (from the repo root)
Endpoints: GET /health, GET /explain, POST /predict (multipart field 'image'), GET /history

No fake results: if the trained models are missing, /predict answers 503 and says how to train them.
"""
from __future__ import annotations

import argparse
import logging
import time
import uuid
from pathlib import Path

from flask import Flask, g, jsonify, request

from glaucoma.config import REPO_ROOT, load_config, resolve
from glaucoma.preprocessing import ImageDecodeError, decode_image_bytes

from . import db
from .glossary import DISCLAIMER, GLOSSARY, HOW_TO_USE, LIMITS

log = logging.getLogger("api")
ALLOWED_EXT = {".jpg", ".jpeg", ".png"}


def create_app(config_path: str | None = None, analyzer=None, db_path: Path | None = None,
               load_models: bool = True) -> Flask:
    """analyzer: pass an object with .analyze(rgb)->dict to inject one (used by tests).
    Otherwise real checkpoints are loaded, unless load_models=False (tests of the 'models missing' path)."""
    cfg = load_config(config_path)
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(cfg["api"]["max_upload_mb"]) * 1024 * 1024
    db_file = Path(db_path) if db_path else resolve(cfg, "db")
    db.init_db(db_file)

    state = {"analyzer": analyzer, "error": None}
    if analyzer is None and not load_models:
        state["error"] = "Model loading was disabled."
    elif analyzer is None:
        try:
            from glaucoma.inference import Analyzer
            state["analyzer"] = Analyzer(cfg)
            log.info("Models loaded.")
        except Exception as exc:  # noqa: BLE001  keep the API up and report the reason in /health
            state["error"] = f"{type(exc).__name__}: {exc}"
            log.warning("Models NOT loaded: %s", state["error"])

    @app.before_request
    def _start():
        g.rid, g.t0 = uuid.uuid4().hex[:8], time.time()

    @app.after_request
    def _cors_and_log(resp):
        origin = request.headers.get("Origin")
        if origin == cfg["api"]["frontend_origin"]:
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        log.info("rid=%s %s %s -> %s in %.0f ms", getattr(g, "rid", "-"), request.method, request.path,
                 resp.status_code, 1000 * (time.time() - getattr(g, "t0", time.time())))
        return resp

    def err(code: str, message: str, status: int):
        return jsonify({"error": {"code": code, "message": message}, "request_id": getattr(g, "rid", None)}), status

    @app.errorhandler(413)
    def _too_big(_):
        return err("file_too_large", f"The file is larger than {cfg['api']['max_upload_mb']} MB.", 413)

    @app.get("/health")
    def health():
        ok = state["analyzer"] is not None
        body = {"status": "ok" if ok else "degraded", "models_loaded": ok}
        if not ok:
            body["detail"] = state["error"]
        return jsonify(body)

    @app.get("/explain")
    def explain():
        return jsonify({"disclaimer": DISCLAIMER, "glossary": GLOSSARY, "how_to_use": HOW_TO_USE, "limits": LIMITS})

    @app.post("/predict")
    def predict():
        if state["analyzer"] is None:
            return err("models_not_loaded", state["error"] or "Models are not loaded. Train them first.", 503)
        f = request.files.get("image")
        if f is None or not f.filename:
            return err("no_file", "Send the image as multipart form field 'image'.", 400)
        if Path(f.filename).suffix.lower() not in ALLOWED_EXT:
            return err("wrong_type", "Only JPG or PNG images are accepted.", 415)
        try:
            rgb = decode_image_bytes(f.read())
        except ImageDecodeError as exc:
            return err("unreadable_image", str(exc), 422)
        h, w = rgb.shape[:2]
        if min(h, w) < cfg["api"]["min_image_side"]:
            return err("image_too_small", f"The image is {w}x{h}. Each side must be at least {cfg['api']['min_image_side']} pixels.", 422)
        try:
            result = state["analyzer"].analyze(rgb)
        except Exception as exc:  # noqa: BLE001
            log.exception("analysis failed rid=%s", g.rid)
            return err("analysis_failed", f"The analysis failed: {exc}", 500)
        result["disclaimer"] = DISCLAIMER
        result["id"] = db.add_prediction(db_file, f.filename, result)
        return jsonify(result)

    @app.get("/history")
    def history():
        return jsonify({"items": db.recent(db_file, min(int(request.args.get("limit", 50)), 200))})

    return app


def main():
    p = argparse.ArgumentParser(description="Run the API")
    p.add_argument("--config", default=None)
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--host", default="127.0.0.1")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    create_app(a.config).run(host=a.host, port=a.port)


if __name__ == "__main__":
    main()
