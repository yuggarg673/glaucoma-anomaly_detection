"""Flask app: the JSON API and the website, on one address.

Run:  python -m backend.app            (from the repo root), then open http://localhost:5000
Website:  GET /                 the single-page site in web/ (no Node, no build step)
API:      GET /health, GET /explain, POST /predict (multipart field 'image'), GET /history

No fake results: if the trained models are missing, /predict answers 503 and says how to train them.
This is Flask's development server: fine for a demo or a class presentation, not for hosting on the open internet.
"""
from __future__ import annotations

import argparse
import logging
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, g, jsonify, request, send_from_directory

from glaucoma.config import REPO_ROOT, load_config, resolve
from glaucoma.preprocessing import MAX_UPLOAD_PIXELS, ImageDecodeError, decode_image_bytes

from . import db
from .glossary import DISCLAIMER, GLOSSARY, HOW_TO_USE, LIMITS

log = logging.getLogger("api")
ALLOWED_EXT = {".jpg", ".jpeg", ".png"}
WEB_DIR = REPO_ROOT / "web"

# The site loads its own script and stylesheet, Google Fonts (optional, it falls back to system fonts) and talks only to
# itself. It sets no frame-ancestors rule on purpose, so it can be shown inside a Colab output frame.
CSP = ("default-src 'self'; img-src 'self' data: blob:; style-src 'self' https://fonts.googleapis.com; "
       "font-src https://fonts.gstatic.com; script-src 'self'; connect-src 'self'; object-src 'none'; "
       "base-uri 'self'; form-action 'self'")


TRAIN_FIX = ("Train them with `python scripts/run_pipeline.py` (or the Colab notebook), copy `checkpoints/` next to "
             "`backend/`, then restart the server.")


def load_failure_advice(exc: Exception) -> tuple[str, str]:
    """(what went wrong, what to do next) when the models cannot be loaded. Shown on the website and in the console."""
    detail = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, ModuleNotFoundError):
        return (f"A required Python package is missing ({exc.name}).",
                "Install the requirements with `pip install -r requirements.txt`, then restart the server.")
    if type(exc).__name__ == "ModelsNotReady":
        return str(exc), TRAIN_FIX
    return (detail, "Look for the line starting 'Models NOT loaded' in the server log. If the checkpoints are damaged "
                    "or from an older version, train again with `python scripts/run_pipeline.py`.")


def create_app(config_path: str | None = None, analyzer=None, db_path: Path | None = None,
               load_models: bool = True, web_dir: Path | None = None) -> Flask:
    """analyzer: pass an object with .analyze(rgb)->dict to inject one (used by tests).
    Otherwise real checkpoints are loaded, unless load_models=False (tests of the 'models missing' path).
    web_dir: folder holding index.html and assets/ (default: web/ in the repo)."""
    cfg = load_config(config_path)
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = int(cfg["api"]["max_upload_mb"]) * 1024 * 1024
    app.config["MAX_IMAGE_PIXELS"] = int(cfg["api"].get("max_image_pixels", MAX_UPLOAD_PIXELS))
    db_file = Path(db_path) if db_path else resolve(cfg, "db")
    db.init_db(db_file)
    site = Path(web_dir) if web_dir else WEB_DIR

    state = {"analyzer": analyzer, "error": None, "fix": None}
    if analyzer is None and not load_models:
        state["error"] = "Model loading was disabled."
    elif analyzer is None:
        try:
            from glaucoma.inference import Analyzer
            state["analyzer"] = Analyzer(cfg)
            log.info("Models loaded.")
        except Exception as exc:  # noqa: BLE001  keep the API up and report the reason in /health
            state["error"], state["fix"] = load_failure_advice(exc)
            log.warning("Models NOT loaded: %s", state["error"])
    app.config["MODELS_LOADED"] = state["analyzer"] is not None
    app.config["MODELS_ERROR"] = state["error"]
    app.config["MODELS_FIX"] = state["fix"]
    analysis_lock = threading.Lock()   # one analysis at a time: keeps memory use predictable on small machines

    @app.before_request
    def _start():
        g.rid, g.t0 = uuid.uuid4().hex[:8], time.time()

    @app.after_request
    def _headers_and_log(resp):
        origin = request.headers.get("Origin")
        if origin == cfg["api"]["frontend_origin"]:   # only needed by the optional React dev server
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        if resp.mimetype == "text/html":
            resp.headers["Content-Security-Policy"] = CSP
        if request.path != "/" and not request.path.startswith("/assets/"):
            resp.headers["Cache-Control"] = "no-store"   # results and history are never cached by the browser
        log.info("rid=%s %s %s -> %s in %.0f ms", getattr(g, "rid", "-"), request.method, request.path,
                 resp.status_code, 1000 * (time.time() - getattr(g, "t0", time.time())))
        return resp

    def err(code: str, message: str, status: int):
        return jsonify({"error": {"code": code, "message": message}, "request_id": getattr(g, "rid", None)}), status

    @app.errorhandler(413)
    def _too_big(_):
        return err("file_too_large", f"The file is larger than {cfg['api']['max_upload_mb']} MB.", 413)

    @app.errorhandler(404)
    def _not_found(_):
        return err("not_found", "That address does not exist on this server.", 404)

    @app.errorhandler(405)
    def _bad_method(_):
        return err("method_not_allowed", "That address does not accept this kind of request.", 405)

    @app.errorhandler(500)
    def _server_error(_):
        # Flask has already logged the traceback. The person gets the reference id, never the internals.
        return err("server_error", "The server hit an unexpected problem. The details are in the server log.", 500)

    # ------------------------------------------------------------------ website
    @app.get("/")
    def index():
        if not (site / "index.html").exists():
            return err("website_missing", f"The website files were not found ({site / 'index.html'}). "
                                          "The API still works; see README.md.", 404)
        return send_from_directory(site, "index.html")

    @app.get("/assets/<path:name>")
    def assets(name):
        return send_from_directory(site / "assets", name)   # refuses paths that escape the folder

    # ---------------------------------------------------------------------- API
    @app.get("/health")
    def health():
        ok = state["analyzer"] is not None
        body = {"status": "ok" if ok else "degraded", "models_loaded": ok,
                "test_models": bool(getattr(state["analyzer"], "test_models", False)),   # trained on synthetic data
                "max_upload_mb": int(cfg["api"]["max_upload_mb"])}
        if not ok:
            body["detail"] = state["error"]
            if state["fix"]:
                body["fix"] = state["fix"]
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
            rgb = decode_image_bytes(f.read(), app.config["MAX_IMAGE_PIXELS"])
        except ImageDecodeError as exc:
            return err("unreadable_image", str(exc), 422)
        h, w = rgb.shape[:2]
        if min(h, w) < cfg["api"]["min_image_side"]:
            return err("image_too_small", f"The image is {w}x{h}. Each side must be at least {cfg['api']['min_image_side']} pixels.", 422)
        try:
            with analysis_lock:
                result = state["analyzer"].analyze(rgb)
        except Exception as exc:  # noqa: BLE001
            log.exception("analysis failed rid=%s", g.rid)
            return err("analysis_failed", f"The analysis failed ({type(exc).__name__}). The details are in the server log.", 500)
        result["disclaimer"] = DISCLAIMER
        result["id"] = db.add_prediction(db_file, f.filename, result)
        return jsonify(result)

    @app.get("/history")
    def history():
        try:
            limit = int(request.args.get("limit", 50))
        except ValueError:
            limit = 50
        return jsonify({"items": db.recent(db_file, max(1, min(limit, 200)))})

    return app


def main():
    p = argparse.ArgumentParser(description="Run the website and API")
    p.add_argument("--config", default=None)
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to reach it from other devices (e.g. in Colab)")
    p.add_argument("--open", action="store_true", help="open the website in your browser")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = create_app(a.config)
    url = f"http://{'localhost' if a.host in ('127.0.0.1', '0.0.0.0') else a.host}:{a.port}/"
    print(f"\n  Website: {url}\n")
    if not app.config["MODELS_LOADED"]:
        print("  NOTE: the trained models are not loaded, so the site opens but cannot analyse images yet.\n"
              f"        Reason: {app.config['MODELS_ERROR']}\n"
              f"        {(app.config['MODELS_FIX'] or '').replace('`', '')}\n")
    if a.open:
        import webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=a.host, port=a.port)


if __name__ == "__main__":
    main()
