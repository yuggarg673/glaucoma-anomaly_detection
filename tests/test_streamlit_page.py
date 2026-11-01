"""Tests for streamlit_app.py.

Streamlit could not be installed where this was written (no network), so the page runs here against FakeStreamlit: a
recording stand-in that implements ONLY the Streamlit calls the page is meant to use. Any other call raises, so the page
cannot quietly depend on something nobody has checked.

What this proves: the page's own logic (what it shows when the models are missing, when the file is wrong, when the
analysis crashes, after a rerun, when the person switches file) and that it only uses the calls listed in MODELLED.
What it cannot prove: that those calls behave in real Streamlit the way the stand-in assumes. The first real run is that test.

Run:  python -m unittest tests.test_streamlit_page -v"""
import ast
import io
import re
import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from backend.advice import TRAIN_FIX  # noqa: E402
from backend.glossary import DISCLAIMER, GLOSSARY  # noqa: E402
from glaucoma.config import load_config  # noqa: E402
from tests.standin import StandInAnalyzer  # noqa: E402

PAGE = ROOT / "streamlit_app.py"
RETRY = "Try loading the models again"
ANALYZE = "Analyze image"


class Stop(Exception):
    """What st.stop() does: end this run of the script."""


class Rerun(Exception):
    """What st.rerun() does: end this run and start another."""


class FakeUpload:
    def __init__(self, name, data):
        self.name, self._data, self.size = name, data, len(data)

    def getvalue(self):
        return self._data


class Where:
    """`with st.sidebar:`, `with column:`, `with st.expander(...)`: events drawn inside are tagged with the place."""

    def __init__(self, fake, label):
        self.fake, self.label = fake, label

    def __enter__(self):
        self.fake.where.append(self.label)
        return self

    def __exit__(self, *exc):
        self.fake.where.pop()
        return False


class FakeStreamlit(types.ModuleType):
    # Every name the page may use on `st`. test_page_uses_only_modelled_calls checks the page's source against this.
    MODELLED = {"set_page_config", "title", "header", "subheader", "markdown", "caption", "text", "info", "warning",
                "error", "image", "columns", "metric", "progress", "table", "expander", "file_uploader", "button",
                "download_button", "spinner", "stop", "rerun", "cache_resource", "session_state", "sidebar"}

    def __init__(self, upload=None, clicks=(), session=None, caches=None):
        super().__init__("streamlit")
        self.events = []                 # (kind, where, text)
        self.where = ["main"]
        self.session_state = {} if session is None else session
        self.upload, self.clicks = upload, set(clicks)
        self.caches = {} if caches is None else caches
        self.sidebar = Where(self, "sidebar")
        self.stopped = self.rerun_called = False
        self.cache_cleared, self.downloads, self.images, self.uploader_kwargs = [], [], [], {}

    def __getattr__(self, name):         # only reached for names that are not defined below
        raise AttributeError(f"The page used st.{name}, which FakeStreamlit does not model. If it exists in real "
                             "Streamlit (>= 1.36) add it to FakeStreamlit and to MODELLED; if not, fix the page.")

    def _add(self, kind, text=""):
        self.events.append((kind, self.where[-1], str(text)))

    # --- text
    def set_page_config(self, **kw):
        self._add("page_config", kw.get("page_title"))

    def title(self, text):
        self._add("title", text)

    def header(self, text):
        self._add("header", text)

    def subheader(self, text):
        self._add("subheader", text)

    def markdown(self, text):
        self._add("markdown", text)

    def caption(self, text):
        self._add("caption", text)

    def text(self, text):
        self._add("text", text)

    def info(self, text):
        self._add("info", text)

    def warning(self, text):
        self._add("warning", text)

    def error(self, text):
        self._add("error", text)

    # --- pictures, numbers, tables
    def image(self, image, **kw):
        shape = getattr(image, "shape", None)
        self.images.append(shape if shape is not None else ("bytes", len(image.getvalue())))
        self._add("image", shape if shape is not None else "bytes")

    def metric(self, label, value, **kw):
        self._add("metric", f"{label}={value}")

    def progress(self, value):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
            raise ValueError(f"st.progress needs a number from 0 to 1, got {value!r}")   # real Streamlit refuses too
        self._add("progress", value)

    def table(self, data):
        self._add("table", data.to_string())

    # --- layout
    def columns(self, spec):
        n = spec if isinstance(spec, int) else len(spec)
        return [Where(self, f"col{i}") for i in range(n)]

    def expander(self, label, **kw):
        self._add("expander", label)
        return Where(self, f"expander:{label}")

    def spinner(self, text=""):
        self._add("spinner", text)
        return Where(self, "spinner")

    # --- widgets
    def file_uploader(self, label, **kw):
        self.uploader_kwargs = kw
        self._add("file_uploader", label)
        return self.upload

    def button(self, label, **kw):
        self._add("button", label)
        return label in self.clicks

    def download_button(self, label, data=None, file_name=None, mime=None, **kw):
        self.downloads.append((label, data, file_name, mime))
        self._add("download_button", label)
        return False

    # --- control and caching
    def stop(self):
        self.stopped = True
        raise Stop()

    def rerun(self):
        self.rerun_called = True
        raise Rerun()

    def cache_resource(self, func=None, **kw):
        def wrap(f):
            name = f.__qualname__

            def inner(*a, **k):
                if name not in self.caches:
                    self.caches[name] = f(*a, **k)
                return self.caches[name]

            def clear():
                self.caches.pop(name, None)
                self.cache_cleared.append(name)

            inner.clear = clear
            return inner

        return wrap(func) if callable(func) else wrap

    # --- reading what was drawn
    def texts(self, kind=None, where=None):
        return [t for k, w, t in self.events if (kind is None or k == kind) and (where is None or w.startswith(where))]

    def all_text(self):
        return "\n".join(t for _, _, t in self.events)


class CountingStandIn(StandInAnalyzer):
    """The test stand-in for the trained networks (see tests/standin.py), counting its uses."""

    def __init__(self, test_models=False):
        super().__init__()
        self.test_models = test_models
        self.calls, self.last_shape = 0, None

    def analyze(self, rgb):
        self.calls += 1
        self.last_shape = rgb.shape
        return super().analyze(rgb)


def png_bytes(w=300, h=260, value=150):
    ok, buf = cv2.imencode(".png", np.full((h, w, 3), value, np.uint8))
    assert ok
    return buf.tobytes()


def loader(analyzer=None, problem=None, fix=None):
    return mock.Mock(return_value=(analyzer, problem, fix))


def run_page(fake, load):
    """One run of the page script: what Streamlit does after every click."""
    with mock.patch.dict(sys.modules, {"streamlit": fake}), mock.patch("backend.service.load_analyzer", load):
        try:
            runpy.run_path(str(PAGE), run_name="__main__")
        except (Stop, Rerun):
            pass
    return fake


def good_upload(name="eye.png", **kw):
    return FakeUpload(name, png_bytes(**kw))


class NoUploadTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = CountingStandIn()
        self.fake = run_page(FakeStreamlit(), loader(self.analyzer))

    def test_page_config_comes_first_and_the_title_is_set(self):
        self.assertEqual(self.fake.events[0][0], "page_config")
        self.assertEqual(self.fake.events[0][2], "Glaucoma Screening")
        self.assertEqual(self.fake.texts("title"), ["Glaucoma Screening"])

    def test_invites_the_person_to_choose_an_image(self):
        self.assertTrue(any("Choose an image above" in t for t in self.fake.texts("info")))
        self.assertEqual(self.fake.texts("file_uploader"), ["Choose a retinal photo (JPG or PNG)"])
        self.assertEqual(self.fake.uploader_kwargs["type"], ["jpg", "jpeg", "png"])
        self.assertIn(f"{load_config()['api']['max_upload_mb']} MB", self.fake.uploader_kwargs["help"])
        self.assertEqual(self.fake.texts("button"), [])
        self.assertEqual(self.analyzer.calls, 0)

    def test_sidebar_explains_use_and_limits_without_website_only_wording(self):
        side = self.fake.texts(where="sidebar")
        self.assertIn("How to use", side)
        self.assertIn("Limits", side)
        joined = "\n".join(side)
        self.assertNotIn("Analyze page", joined)
        self.assertNotIn("history", joined.replace("keeps no history", ""))   # only our own privacy sentence may say it
        self.assertIn("do not upload identifiable patient images", joined)

    def test_every_number_is_explained_and_the_disclaimer_is_shown(self):
        self.assertEqual(self.fake.texts("expander"), [GLOSSARY[k]["title"] for k in
                                                        ("label", "fused_score", "anomaly_score", "cdr", "confidence",
                                                         "signals_agree", "regions")])
        self.assertIn(DISCLAIMER, self.fake.texts("caption"))

    def test_no_errors_no_results(self):
        self.assertEqual(self.fake.texts("error"), [])
        self.assertEqual(self.fake.texts("metric"), [])
        self.assertFalse(self.fake.stopped)

    def test_real_models_show_no_test_banner(self):
        self.assertNotIn("test models", self.fake.all_text().lower())


class ModelsMissingTests(unittest.TestCase):
    PROBLEM = "Missing checkpoints: autoencoder (autoencoder.pt)."

    def test_says_so_gives_the_reason_and_the_fix_and_stops(self):
        fake = run_page(FakeStreamlit(upload=good_upload()), loader(None, self.PROBLEM, TRAIN_FIX))
        errors = fake.texts("error")
        self.assertEqual(len(errors), 1)
        self.assertIn("not loaded", errors[0])
        self.assertIn(self.PROBLEM, errors[0])
        self.assertTrue(any("Train them with" in t and "git add -f" in t for t in fake.texts("info")))
        self.assertTrue(fake.stopped)
        self.assertEqual(fake.texts("file_uploader"), [], "it must not offer an upload it cannot analyse")
        self.assertEqual(fake.texts("metric"), [])
        self.assertEqual(fake.texts("button"), [RETRY])

    def test_retry_clears_the_cache_and_reruns(self):
        fake = run_page(FakeStreamlit(clicks=[RETRY]), loader(None, self.PROBLEM, TRAIN_FIX))
        self.assertEqual(fake.cache_cleared, ["get_models"])
        self.assertTrue(fake.rerun_called)

    def test_models_are_loaded_once_per_server_not_once_per_click(self):
        load = loader(CountingStandIn())
        first = run_page(FakeStreamlit(), load)
        run_page(FakeStreamlit(caches=first.caches), load)
        self.assertEqual(load.call_count, 1)


class TestModelsTests(unittest.TestCase):
    def test_banner_comes_before_the_upload(self):
        fake = run_page(FakeStreamlit(), loader(CountingStandIn(test_models=True)))
        kinds = [k for k, _, _ in fake.events]
        banner = next(i for i, (k, _, t) in enumerate(fake.events) if k == "error" and "These are test models" in t)
        self.assertLess(banner, kinds.index("file_uploader"))


class UploadTests(unittest.TestCase):
    def test_picture_chosen_but_not_analysed_yet(self):
        analyzer = CountingStandIn()
        session = {}
        fake = run_page(FakeStreamlit(upload=good_upload(w=300, h=260), session=session), loader(analyzer))
        self.assertIn("File: eye.png   (300 x 260 pixels)", fake.texts("text"))
        self.assertEqual(fake.texts("button"), [ANALYZE])
        self.assertTrue(any("Press **Analyze image**" in t for t in fake.texts("caption")))
        self.assertEqual(len(fake.images), 1)
        self.assertEqual(fake.texts("metric"), [])
        self.assertEqual(analyzer.calls, 0)
        self.assertNotIn("analysis", session)

    def test_wrong_file_type_is_refused_with_a_clear_message(self):
        session = {"analysis": {"key": "old"}}
        fake = run_page(FakeStreamlit(upload=FakeUpload("notes.txt", b"hello"), session=session), loader(CountingStandIn()))
        self.assertEqual(fake.texts("error"), ["Only JPG or PNG images are accepted."])
        self.assertTrue(fake.stopped)
        self.assertEqual(fake.texts("button"), [])
        self.assertNotIn("analysis", session)

    def test_unreadable_picture_is_refused_with_a_clear_message(self):
        from backend.service import UploadProblem, check_upload
        with self.assertRaises(UploadProblem) as expected:        # the page must show exactly what the checker says
            check_upload("x.png", b"this is not a picture", load_config())
        fake = run_page(FakeStreamlit(upload=FakeUpload("x.png", b"this is not a picture")), loader(CountingStandIn()))
        self.assertEqual(fake.texts("error"), [expected.exception.message])
        self.assertIn("readable", expected.exception.message)
        self.assertTrue(fake.stopped)
        self.assertEqual(fake.texts("button"), [])

    def test_removing_the_upload_clears_the_old_result(self):
        analyzer = CountingStandIn()
        session = {}
        run_page(FakeStreamlit(upload=good_upload(), clicks=[ANALYZE], session=session), loader(analyzer))
        self.assertIn("analysis", session)
        run_page(FakeStreamlit(upload=None, session=session), loader(analyzer))
        self.assertNotIn("analysis", session)

    def test_preview_is_small_but_the_models_see_the_whole_picture(self):
        analyzer = CountingStandIn()
        fake = run_page(FakeStreamlit(upload=FakeUpload("big.png", png_bytes(1500, 1000)), clicks=[ANALYZE]),
                        loader(analyzer))
        self.assertEqual(analyzer.last_shape, (1000, 1500, 3))
        previews = [s for s in fake.images if isinstance(s, tuple) and len(s) == 3]
        self.assertTrue(previews)
        self.assertLessEqual(max(previews[0][:2]), 640)


class ResultTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = CountingStandIn()
        self.session = {}
        self.fake = run_page(FakeStreamlit(upload=good_upload(), clicks=[ANALYZE], session=self.session),
                             loader(self.analyzer))

    def test_one_analysis_and_the_four_numbers(self):
        self.assertEqual(self.analyzer.calls, 1)
        labels = [t.split("=")[0] for t in self.fake.texts("metric")]
        self.assertEqual(labels, ["Combined score", "Anomaly score", "Cup-to-disc ratio", "Confidence"])

    def test_a_label_that_says_it_is_not_a_diagnosis(self):
        verdicts = [t for k, _, t in self.fake.events if k in ("warning", "info") and t.startswith("**")]
        self.assertEqual(len(verdicts), 1)
        self.assertRegex(verdicts[0], r"^\*\*(Glaucoma suspected|Normal-like)\*\*")
        self.assertIn("not a diagnosis", verdicts[0])

    def test_annotated_picture_and_download(self):
        self.assertEqual(len(self.fake.images), 2)            # the person's picture and the annotated one
        (label, data, name, mime), = self.fake.downloads
        self.assertTrue(data.startswith(b"\x89PNG"))
        self.assertEqual((name, mime), ("annotated_fundus.png", "image/png"))
        self.assertTrue(any("Teal outline" in t for t in self.fake.texts("caption")))

    def test_regions_table_and_models_line(self):
        table = "\n".join(self.fake.texts("table"))
        self.assertIn("A1", table)
        self.assertIn("Where (as shown)", table)
        self.assertTrue(any(t.startswith("Models: ") for t in self.fake.texts("caption")))

    def test_the_result_is_kept_for_this_visitor_only(self):
        self.assertEqual(list(self.session), ["analysis"])
        self.assertTrue(self.session["analysis"]["key"].startswith("eye.png|"))

    def test_nothing_that_looks_like_a_programming_leftover(self):
        text = self.fake.all_text()
        for pattern in (r"\bNone\b", r"\bnan\b", r"\bNaN\b", r"\[object", r"Traceback", r"undefined"):
            self.assertIsNone(re.search(pattern, text), pattern)

    def test_a_rerun_after_the_download_button_keeps_the_result_without_analysing_again(self):
        again = run_page(FakeStreamlit(upload=good_upload(), session=self.session, caches=self.fake.caches),
                         loader(self.analyzer))
        self.assertEqual(self.analyzer.calls, 1)
        self.assertEqual(len(again.texts("metric")), 4)

    def test_choosing_another_picture_hides_the_old_result(self):
        other = run_page(FakeStreamlit(upload=FakeUpload("other.png", png_bytes(310, 270, 90)), session=self.session,
                                       caches=self.fake.caches), loader(self.analyzer))
        self.assertEqual(other.texts("metric"), [])
        self.assertTrue(any("Press **Analyze image**" in t for t in other.texts("caption")))
        self.assertEqual(self.analyzer.calls, 1)

    def test_same_file_name_but_a_different_picture_hides_the_old_result(self):
        other = run_page(FakeStreamlit(upload=FakeUpload("eye.png", png_bytes(300, 260, 60)), session=self.session,
                                       caches=self.fake.caches), loader(self.analyzer))
        self.assertEqual(other.texts("metric"), [])
        self.assertTrue(any("Press **Analyze image**" in t for t in other.texts("caption")))

    def test_test_models_are_flagged_on_the_page_and_on_the_result(self):
        flagged = run_page(FakeStreamlit(upload=good_upload(), clicks=[ANALYZE]), loader(CountingStandIn(test_models=True)))
        banners = [t for t in flagged.texts("error") if "These are test models" in t]
        self.assertEqual(len(banners), 2)
        self.assertEqual([t for t in self.fake.texts("error")], [], "real models must not carry the banner")


class Forced(CountingStandIn):
    """The stand-in with the label (and optionally the confidence) forced, so both wordings can be checked."""

    def __init__(self, label, confidence=None):
        super().__init__()
        self.label, self.confidence = label, confidence

    def analyze(self, rgb):
        r = super().analyze(rgb)
        r["label"] = self.label
        if self.confidence:
            r["confidence_level"] = self.confidence
        return r


class VerdictWordingTests(unittest.TestCase):
    def run_with(self, analyzer):
        return run_page(FakeStreamlit(upload=good_upload(), clicks=[ANALYZE]), loader(analyzer))

    def test_suspected_is_a_warning_and_says_it_is_not_a_diagnosis(self):
        fake = self.run_with(Forced("glaucoma_suspected"))
        verdict = [t for t in fake.texts("warning") if t.startswith("**")]
        self.assertEqual(len(verdict), 1)
        self.assertIn("**Glaucoma suspected**", verdict[0])
        self.assertIn("above the cut-off", verdict[0])
        self.assertIn("not a diagnosis", verdict[0])

    def test_normal_like_is_neutral_not_green_and_also_says_it_is_not_a_diagnosis(self):
        fake = self.run_with(Forced("normal"))
        verdict = [t for t in fake.texts("info") if t.startswith("**")]
        self.assertEqual(len(verdict), 1)
        self.assertIn("**Normal-like**", verdict[0])
        self.assertIn("below the cut-off", verdict[0])
        self.assertIn("not a diagnosis", verdict[0])
        self.assertEqual([t for t in fake.texts("warning") if t.startswith("**")], [])

    def test_low_confidence_is_called_unclear(self):
        low = self.run_with(Forced("normal", confidence="Low"))
        self.assertTrue(any("Confidence is Low" in t and "unclear" in t for t in low.texts("caption")))
        high = self.run_with(Forced("normal", confidence="High"))
        self.assertFalse(any("Confidence is Low" in t for t in high.texts("caption")))

    def test_confidence_is_never_described_as_a_probability_of_disease(self):
        fake = self.run_with(Forced("glaucoma_suspected"))
        self.assertTrue(any("not the probability of disease" in t for t in fake.texts("caption")))


class FailureTests(unittest.TestCase):
    def test_a_crash_in_the_models_shows_a_safe_message(self):
        class Boom(CountingStandIn):
            def analyze(self, rgb):
                raise RuntimeError("secret path /srv/private/model.pt")

        session = {}
        with self.assertLogs("api", level="ERROR"):
            fake = run_page(FakeStreamlit(upload=good_upload(), clicks=[ANALYZE], session=session), loader(Boom()))
        self.assertEqual(fake.texts("error"), ["The analysis failed (RuntimeError). The details are in the server log."])
        self.assertNotIn("secret", fake.all_text())
        self.assertNotIn("/srv", fake.all_text())
        self.assertEqual(fake.texts("metric"), [])
        self.assertIn("error", session["analysis"])

    def test_a_result_that_cannot_be_drawn_never_shows_a_traceback(self):
        with mock.patch("backend.present.summarize", side_effect=KeyError("boom")), \
                self.assertLogs("streamlit_app", level="ERROR"):
            fake = run_page(FakeStreamlit(upload=good_upload(), clicks=[ANALYZE]), loader(CountingStandIn()))
        self.assertEqual(fake.texts("error"), ["The result could not be displayed. The details are in the app log."])
        self.assertNotIn("boom", fake.all_text())

    def test_a_broken_annotated_picture_is_a_warning_not_a_crash(self):
        class NoPicture(CountingStandIn):
            def analyze(self, rgb):
                r = super().analyze(rgb)
                r["annotated_image"] = "data:image/png;base64,@@@@"
                return r

        fake = run_page(FakeStreamlit(upload=good_upload(), clicks=[ANALYZE]), loader(NoPicture()))
        self.assertIn("The annotated image could not be shown.", fake.texts("warning"))
        self.assertEqual(len(fake.texts("metric")), 4)        # the numbers still show
        self.assertEqual(fake.downloads, [])


class SourceTests(unittest.TestCase):
    """Static checks of the page and its settings: they cover branches the runs above do not reach."""

    source = PAGE.read_text(encoding="utf-8")

    def test_page_uses_only_modelled_calls(self):
        used = {n.attr for n in ast.walk(ast.parse(self.source))
                if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "st"}
        self.assertTrue(used, "the check found no st.<call> at all")
        self.assertEqual(used - FakeStreamlit.MODELLED, set())
        for call in ("set_page_config", "file_uploader", "button", "cache_resource", "session_state"):
            self.assertIn(call, used)

    def test_no_risky_or_deprecated_features(self):
        for banned, why in (("unsafe_allow_html", "would let file names or text inject HTML"),
                            ("cache_data", "would share patients' pictures between visitors"),
                            ("experimental_", "experimental calls get removed"),
                            ("use_container_width", "deprecated in newer Streamlit"),
                            ("use_column_width", "deprecated in newer Streamlit"),
                            ("sqlite", "this page keeps no history"),
                            ("open(", "this page writes no files"), ("write_bytes", "this page writes no files"),
                            ("write_text", "this page writes no files"), (".save(", "this page writes no files")):
            self.assertNotIn(banned, self.source, why)

    def test_the_page_calls_main_exactly_once_at_the_end(self):
        tree = ast.parse(self.source)
        calls = [n for n in tree.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                 and getattr(n.value.func, "id", None) == "main"]
        self.assertEqual(len(calls), 1)
        self.assertIs(tree.body[-1], calls[0])

    def test_upload_limit_in_streamlit_settings_matches_the_website(self):
        try:
            import tomllib
        except ModuleNotFoundError:          # Python 3.10
            self.skipTest("tomllib needs Python 3.11+")
        settings = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
        self.assertEqual(settings["server"]["maxUploadSize"], load_config()["api"]["max_upload_mb"])
        self.assertFalse(settings["browser"]["gatherUsageStats"])
        css = (ROOT / "web" / "assets" / "styles.css").read_text(encoding="utf-8")
        brand = re.search(r"--brand:\s*(#[0-9A-Fa-f]{6})", css).group(1)
        self.assertEqual(settings["theme"]["primaryColor"].lower(), brand.lower())

    def test_requirements_ask_for_cpu_pytorch_and_streamlit(self):
        lines = [x.strip() for x in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
                 if x.strip() and not x.lstrip().startswith("#")]
        self.assertIn("--extra-index-url https://download.pytorch.org/whl/cpu", lines)
        self.assertTrue(any(x.startswith("streamlit") for x in lines))
        names = [re.split(r"[<>=~! ]", x)[0].lower() for x in lines if not x.startswith("-")]
        for needed in ("torch", "torchvision", "numpy", "pandas", "scikit-learn", "opencv-python-headless", "pillow",
                       "pyyaml", "flask", "streamlit"):
            self.assertIn(needed, names)
        for x in lines:
            self.assertTrue(x.startswith("--extra-index-url ") or re.fullmatch(r"[A-Za-z0-9_.-]+(>=[0-9.]+)?", x), x)
        self.assertLess(lines.index("--extra-index-url https://download.pytorch.org/whl/cpu"),
                        next(i for i, x in enumerate(lines) if x.startswith("torch")))

    def test_the_fake_really_refuses_unmodelled_calls(self):
        with self.assertRaises(AttributeError):
            FakeStreamlit().experimental_rerun()
        with self.assertRaises(ValueError):
            FakeStreamlit().progress(float("nan"))
        with self.assertRaises(ValueError):
            FakeStreamlit().progress(1.5)

    def test_io_is_used_for_image_bytes(self):          # st.image is given a BytesIO, which Streamlit documents
        self.assertIn("io.BytesIO(png)", self.source)
        self.assertTrue(io.BytesIO)


if __name__ == "__main__":
    unittest.main()
