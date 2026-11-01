"""Tests for the code that the Flask website and the Streamlit page share:
backend/advice.py, backend/present.py and backend/service.py. None of them needs PyTorch, Flask or Streamlit.
Run:  python -m unittest discover -s tests -v"""
import base64
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from backend import advice, present, service  # noqa: E402
from backend.glossary import DISCLAIMER, HOW_TO_USE, LIMITS  # noqa: E402
from glaucoma.config import load_config  # noqa: E402
from glaucoma.inference import ModelsNotReady  # noqa: E402
from tests.standin import StandInAnalyzer  # noqa: E402


def png_bytes(w=128, h=128, value=100):
    ok, buf = cv2.imencode(".png", np.full((h, w, 3), value, np.uint8))
    assert ok
    return buf.tobytes()


class FormatTests(unittest.TestCase):
    def test_fmt_shows_n_a_instead_of_none_or_nan(self):
        for bad in (None, float("nan"), "abc", [], {}):
            self.assertEqual(present.fmt(bad), "n/a", bad)

    def test_fmt_rounds(self):
        self.assertEqual(present.fmt(0.12341), "0.123")
        self.assertEqual(present.fmt(0.12341, 4), "0.1234")
        self.assertEqual(present.fmt(1, 2), "1.00")
        self.assertEqual(present.fmt("0.5", 1), "0.5")

    def test_clamp01_always_gives_something_a_progress_bar_accepts(self):
        self.assertEqual(present.clamp01(1.7), 1.0)
        self.assertEqual(present.clamp01(-3), 0.0)
        self.assertEqual(present.clamp01(0.4), 0.4)
        for bad in (None, "abc", float("nan")):
            self.assertEqual(present.clamp01(bad), 0.0, bad)

    def test_verdict_text(self):
        self.assertEqual(present.verdict_text("glaucoma_suspected"), "Glaucoma suspected")
        self.assertEqual(present.verdict_text("normal"), "Normal-like")
        self.assertEqual(present.verdict_text("something_new"), "something_new")   # never crashes on an unknown label


class WhereIsTests(unittest.TestCase):
    SIZE = {"width": 100, "height": 100}

    def test_positions(self):
        cases = {
            (0, 0, 10, 10): "upper left",
            (80, 0, 15, 10): "upper right",
            (40, 0, 20, 10): "upper part",
            (0, 40, 10, 20): "left side",
            (80, 40, 15, 20): "right side",
            (40, 40, 20, 20): "centre of the image",
            (40, 80, 20, 15): "lower part",
            (0, 80, 10, 15): "lower left",
            (85, 85, 10, 10): "lower right",
        }
        for box, expected in cases.items():
            self.assertEqual(present.where_is(list(box), self.SIZE), expected, box)

    def test_missing_information_gives_n_a(self):
        for box, size in ((None, self.SIZE), ([], self.SIZE), ([1, 2, 3, 4], None), ([1, 2, 3, 4], {}),
                          ([1, 2, 3, 4], {"width": 0, "height": 10})):
            self.assertEqual(present.where_is(box, size), "n/a")

    def test_region_rows(self):
        rows = present.region_rows([{"id": "A1", "severity": "high", "box": [0, 0, 10, 10]},
                                    {"id": "A2", "severity": "moderate", "box": [40, 40, 20, 20]}], self.SIZE)
        self.assertEqual(rows, [{"Label": "A1", "Difference": "High", "Where (as shown)": "upper left"},
                                {"Label": "A2", "Difference": "Moderate", "Where (as shown)": "centre of the image"}])
        self.assertEqual(present.region_rows(None, self.SIZE), [])


class DataUrlTests(unittest.TestCase):
    def test_round_trip(self):
        data = png_bytes()
        url = "data:image/png;base64," + base64.b64encode(data).decode()
        self.assertEqual(present.png_from_data_url(url), data)

    def test_rejects_anything_that_is_not_a_base64_image(self):
        for bad in ("data:text/plain;base64,AAAA", "not a url", "data:image/png,AAAA", "data:image/png;base64,@@@@", ""):
            with self.assertRaises(ValueError, msg=bad):
                present.png_from_data_url(bad)


class SummarizeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analyzer = StandInAnalyzer()

    def result_for(self, value):
        rng = np.random.default_rng(1)
        rgb = np.clip(rng.normal(value, 8, (200, 200, 3)), 0, 255).astype(np.uint8)
        return self.analyzer.analyze(rgb)

    def all_text(self, summary):
        parts = []
        for v in summary.values():
            if isinstance(v, list):
                parts += [str(x) for row in v for x in row.values()]
            else:
                parts.append(str(v))
        return " | ".join(parts)

    def test_fields_for_a_normal_result(self):
        r = self.result_for(150)
        s = present.summarize(r)
        self.assertIn(s["verdict"], ("Glaucoma suspected", "Normal-like"))
        self.assertEqual(s["suspected"], r["label"] == "glaucoma_suspected")
        self.assertTrue(0.0 <= s["fused_value"] <= 1.0)
        self.assertRegex(s["cdr"], r"^\d\.\d\d$")
        self.assertIn("reference", s["cdr_sub"])
        self.assertIn(s["confidence"], ("High", "Medium", "Low"))
        self.assertTrue(s["regions"])
        self.assertIn("stand-in (test only)", s["models"])
        self.assertIn("not a diagnosis", s["verdict_note"])

    def test_cdr_unavailable_is_said_plainly(self):
        r = self.result_for(5)          # a nearly black picture: the stand-in finds no cup
        self.assertIsNone(r["cdr"])
        s = present.summarize(r)
        self.assertEqual(s["cdr"], "Unavailable")
        self.assertTrue(s["cdr_sub"])
        self.assertEqual(s["confidence_sub"], "Based on the anomaly score only")

    def test_no_programming_leftovers_in_any_text(self):
        for value in (5, 60, 150, 230):
            text = self.all_text(present.summarize(self.result_for(value)))
            for bad in ("None", "nan", "NaN", "undefined", "[object", "{", "}"):
                self.assertNotIn(bad, text, f"{bad!r} in {text!r}")

    def test_the_two_wordings_for_the_label(self):
        base = self.result_for(150)
        base.update(label="glaucoma_suspected")
        self.assertTrue(present.summarize(base)["verdict_note"].startswith("The combined score is above"))
        base.update(label="normal")
        self.assertTrue(present.summarize(base)["verdict_note"].startswith("The combined score is below"))

    def test_low_confidence_is_flagged_and_signals_wording(self):
        r = self.result_for(150)
        r.update(confidence_level="Low", signals_agree=False)
        s = present.summarize(r)
        self.assertTrue(s["low_confidence"])
        self.assertEqual(s["confidence_sub"], "Anomaly score and CDR disagree")


class WordingHelperTests(unittest.TestCase):
    def test_how_to_steps_do_not_mention_a_page_that_does_not_exist(self):
        steps = present.how_to_steps(HOW_TO_USE)
        self.assertEqual(len(steps), len(HOW_TO_USE))
        self.assertTrue(any(" on the Analyze page" in s for s in HOW_TO_USE), "glossary changed: review how_to_steps")
        self.assertFalse(any("Analyze page" in s for s in steps))

    def test_limits_without_the_history_line(self):
        limits = present.limits_without_history(LIMITS)
        self.assertLess(len(limits), len(LIMITS))
        self.assertFalse(any("history" in x.lower() for x in limits))


class UploadCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()

    def problem(self, *args, **kw):
        with self.assertRaises(service.UploadProblem) as cm:
            service.check_upload(*args, **kw)
        return cm.exception

    def test_wrong_type(self):
        for name in ("notes.txt", "scan.gif", "scan", "", None):
            e = self.problem(name, png_bytes(), self.cfg)
            self.assertEqual((e.code, e.status), ("wrong_type", 415), name)

    def test_good_picture_returns_an_rgb_array(self):
        for name in ("scan.png", "SCAN.PNG", "scan.JPG.png"):
            rgb = service.check_upload(name, png_bytes(100, 80), self.cfg)
            self.assertEqual(rgb.shape, (80, 100, 3))
            self.assertEqual(rgb.dtype, np.uint8)

    def test_garbage_is_unreadable(self):
        e = self.problem("a.png", b"this is not a picture", self.cfg)
        self.assertEqual((e.code, e.status), ("unreadable_image", 422))

    def test_too_small(self):
        e = self.problem("a.png", png_bytes(32, 32), self.cfg)
        self.assertEqual((e.code, e.status), ("image_too_small", 422))
        self.assertIn("32x32", e.message)
        self.assertIn(str(self.cfg["api"]["min_image_side"]), e.message)

    def test_too_many_pixels_is_refused_before_decoding(self):
        e = self.problem("a.png", png_bytes(128, 128), self.cfg, max_pixels=1000)
        self.assertEqual((e.code, e.status), ("unreadable_image", 422))


class RunAnalysisTests(unittest.TestCase):
    def test_success_adds_the_disclaimer(self):
        r = service.run_analysis(StandInAnalyzer(), np.full((120, 120, 3), 150, np.uint8))
        self.assertEqual(r["disclaimer"], DISCLAIMER)
        self.assertIn("label", r)

    def test_a_crash_is_reported_without_internals_and_logged(self):
        class Boom:
            def analyze(self, rgb):
                raise RuntimeError("secret path /srv/private/model.pt")

        with self.assertLogs("api", level="ERROR") as logs:
            with self.assertRaises(service.UploadProblem) as cm:
                service.run_analysis(Boom(), np.zeros((100, 100, 3), np.uint8), context="rid=abc123")
        e = cm.exception
        self.assertEqual((e.code, e.status), ("analysis_failed", 500))
        self.assertIn("RuntimeError", e.message)
        self.assertNotIn("secret", e.message)
        self.assertNotIn("/srv", e.message)
        self.assertIn("rid=abc123", logs.output[0])         # the person's reference finds the traceback
        self.assertIn("secret path", "\n".join(logs.output))  # the full details are in the log

    def test_the_lock_is_held_during_the_analysis(self):
        events = []

        class Lock:
            def __enter__(self):
                events.append("lock")

            def __exit__(self, *a):
                events.append("unlock")

        class Spy:
            def analyze(self, rgb):
                events.append("analyze")
                return {}

        service.run_analysis(Spy(), np.zeros((100, 100, 3), np.uint8), lock=Lock())
        self.assertEqual(events, ["lock", "analyze", "unlock"])

    def test_analyze_upload_is_check_then_run(self):
        cfg = load_config()
        r = service.analyze_upload(StandInAnalyzer(), "a.png", png_bytes(128, 128, 150), cfg)
        self.assertIn("anomaly_score", r)
        with self.assertRaises(service.UploadProblem):
            service.analyze_upload(StandInAnalyzer(), "a.txt", b"x", cfg)


class LoadAnalyzerTests(unittest.TestCase):
    def test_missing_checkpoints(self):
        with mock.patch("glaucoma.inference.Analyzer", side_effect=ModelsNotReady("Missing checkpoints: segmenter (segmenter.pt).")):
            analyzer, problem, fix = service.load_analyzer(load_config())
        self.assertIsNone(analyzer)
        self.assertIn("segmenter.pt", problem)
        self.assertEqual(fix, advice.TRAIN_FIX)

    def test_missing_package(self):
        with mock.patch("glaucoma.inference.Analyzer", side_effect=ModuleNotFoundError("No module named 'torch'", name="torch")):
            analyzer, problem, fix = service.load_analyzer(load_config())
        self.assertIsNone(analyzer)
        self.assertIn("torch", problem)
        self.assertIn("pip install -r requirements.txt", fix)

    def test_unexpected_error_still_does_not_raise(self):
        with mock.patch("glaucoma.inference.Analyzer", side_effect=ValueError("bad file")):
            analyzer, problem, fix = service.load_analyzer(load_config())
        self.assertIsNone(analyzer)
        self.assertIn("ValueError: bad file", problem)
        self.assertIn("Models NOT loaded", fix)

    def test_success(self):
        fake = object()
        with mock.patch("glaucoma.inference.Analyzer", return_value=fake):
            self.assertEqual(service.load_analyzer(load_config()), (fake, None, None))


class StructureTests(unittest.TestCase):
    def test_the_old_import_paths_still_work(self):
        import backend.app as app_module
        self.assertIs(app_module.TRAIN_FIX, advice.TRAIN_FIX)
        self.assertIs(app_module.load_failure_advice, advice.load_failure_advice)
        self.assertEqual(app_module.ALLOWED_EXT, service.ALLOWED_EXT)

    def test_shared_helpers_need_no_web_framework(self):
        code = ("import sys; sys.path[:0] = [%r, %r]\n"
                "import backend.advice, backend.present, backend.service, backend.glossary\n"
                "bad = [m for m in ('flask', 'streamlit', 'torch') if m in sys.modules]\n"
                "assert not bad, bad\n") % (str(ROOT / "src"), str(ROOT))
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 0, done.stderr)


if __name__ == "__main__":
    unittest.main()
