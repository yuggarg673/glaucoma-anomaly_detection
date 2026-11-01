"""What to tell the person when the trained models cannot be loaded.
Shared by the Flask website (backend/app.py) and the Streamlit page (streamlit_app.py); imports no web framework."""
from __future__ import annotations

TRAIN_FIX = ("Train them with `python scripts/run_pipeline.py` (or the Colab notebook), copy `checkpoints/` next to "
             "`backend/`, then restart the server.")


def load_failure_advice(exc: Exception) -> tuple[str, str]:
    """(what went wrong, what to do next) when the models cannot be loaded."""
    detail = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, ModuleNotFoundError):
        return (f"A required Python package is missing ({exc.name}).",
                "Install the requirements with `pip install -r requirements.txt`, then restart the server.")
    if type(exc).__name__ == "ModelsNotReady":
        return str(exc), TRAIN_FIX
    return (detail, "Look for the line starting 'Models NOT loaded' in the server log. If the checkpoints are damaged "
                    "or from an older version, train again with `python scripts/run_pipeline.py`.")
