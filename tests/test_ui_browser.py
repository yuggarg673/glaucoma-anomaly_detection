"""Tests the website in a real browser (headless Chromium through Playwright).

They are skipped, not failed, when Playwright or its browser is not installed:
    pip install playwright && python -m playwright install chromium
    python -m unittest tests.test_ui_browser -v

The server behind the page is the real Flask app with the real post-processing, but the neural networks are replaced by
tests/standin.py (invented outputs), because PyTorch and trained models are not needed to test a web page. So these tests
check the page and the way it talks to the server. They say nothing about how good the models are.
"""
from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from playwright.sync_api import sync_playwright
    HAVE_PLAYWRIGHT = True
except ImportError:                                   # pragma: no cover
    HAVE_PLAYWRIGHT = False

from werkzeug.serving import make_server  # noqa: E402

from backend.app import create_app  # noqa: E402
from tests.standin import StandInAnalyzer  # noqa: E402


class _LooksLikeRealModels(StandInAnalyzer):
    """Test only: the stand-in without the 'test models' flag, to check how the page looks when models are real."""

    def __init__(self):
        super().__init__()
        self.test_models = False


class _SlowAnalyzer(_LooksLikeRealModels):
    """Test only: takes a moment, so the page can be tested while a request is still in flight."""

    def _forward(self, rgb):
        time.sleep(1.5)
        return super()._forward(rgb)


def _serve(app):
    server = make_server("127.0.0.1", 0, app, threaded=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


def _fundus_png(path: Path, w=640, h=480, level=140) -> str:
    img = np.full((h, w, 3), (level // 2, level - 20, level), np.uint8)
    cv2.circle(img, (w // 2, h // 2), min(w, h) // 3, (60, 150, 230), -1)
    cv2.circle(img, (w // 2 + 8, h // 2), min(w, h) // 8, (200, 235, 250), -1)
    cv2.imwrite(str(path), img)
    return str(path)


@unittest.skipUnless(HAVE_PLAYWRIGHT, "Playwright is not installed (pip install playwright)")
class WebsiteInBrowser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        cls.servers = []
        for name, kwargs in (("test_models", {"analyzer": StandInAnalyzer()}),
                             ("real_looking", {"analyzer": _LooksLikeRealModels()}),
                             ("slow", {"analyzer": _SlowAnalyzer()}),
                             ("no_models", {"analyzer": None, "load_models": False})):
            server, url = _serve(create_app(db_path=tmp / f"{name}.sqlite3", **kwargs))
            cls.servers.append(server)
            setattr(cls, "url_" + name, url)
        cls.image = _fundus_png(tmp / "Im0001_SYN.png")
        cls.playwright = sync_playwright().start()
        try:
            cls.browser = cls.playwright.chromium.launch()
        except Exception as exc:                       # the browser binary is missing: skip, do not fail
            cls.playwright.stop()
            cls.tmp.cleanup()
            for s in cls.servers:
                s.shutdown()
            raise unittest.SkipTest(f"Chromium could not start ({type(exc).__name__}). Run: python -m playwright install chromium")

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()
        for s in cls.servers:
            s.shutdown()
        cls.tmp.cleanup()

    def setUp(self):
        self.context = self.browser.new_context(viewport={"width": 1280, "height": 900})
        # Fonts come from Google when the machine is online. Answer them locally so the tests do not depend on the network.
        self.context.route("https://fonts.googleapis.com/**", lambda r: r.fulfill(status=200, content_type="text/css", body=""))
        self.context.route("https://fonts.gstatic.com/**", lambda r: r.fulfill(status=200, body=""))
        self.page = self.context.new_page()
        self.page.set_default_timeout(15000)
        self.js_errors, self.csp_messages = [], []
        self.page.on("pageerror", lambda e: self.js_errors.append(str(e)))
        self.page.on("console", lambda m: self.csp_messages.append(m.text)
                     if "Content Security Policy" in m.text or "Refused to" in m.text else None)

    def tearDown(self):
        try:                                           # whatever the test did, the page must not show programming leftovers
            visible = self.page.inner_text("body")
        except Exception:                              # the page is already gone (a test closed it on purpose)
            visible = ""
        self.context.close()
        self.assertNotRegex(visible, r"\[object|\bundefined\b|\bNaN\b|\bnull\b", "the page shows a programming leftover")
        self.assertEqual(self.js_errors, [], "uncaught JavaScript errors")
        self.assertEqual(self.csp_messages, [], "the content security policy blocked something")

    # ------------------------------------------------------------------ helpers
    def open(self, base, route=""):
        self.page.goto(f"{base}/#/{route}")
        self.page.wait_for_selector("#status-pill:not(.pill-checking)")

    def analyze(self, path=None):
        self.page.set_input_files("#file", path or self.image)
        self.page.wait_for_selector("button:has-text('Analyze image'):not([disabled])")   # a file was accepted
        self.page.click("button:has-text('Analyze image')")

    def no_sideways_scroll(self):
        return self.page.evaluate("document.documentElement.scrollWidth - window.innerWidth")

    # ------------------------------------------------------------------ the main flow
    def test_home_page_loads_with_its_content_and_status(self):
        self.open(self.url_real_looking)
        self.assertIn("Spot early signs of glaucoma", self.page.inner_text("h1"))
        self.assertIn("models loaded", self.page.inner_text("#status-pill"))
        self.assertEqual(self.page.title(), "Home - Glaucoma Screening")
        self.assertTrue(self.page.evaluate("document.querySelector('.hero-art img').naturalWidth > 0"), "hero picture did not load")
        self.assertEqual(self.page.locator(".step").count(), 4)

    def test_upload_then_result_with_annotated_image(self):
        self.open(self.url_real_looking, "analyze")
        self.assertTrue(self.page.is_disabled("button:has-text('Analyze image')"))      # nothing chosen yet
        self.analyze()
        self.page.wait_for_selector("#result-top")
        self.assertIn(self.page.inner_text(".verdict-pill").strip(), ("Normal-like", "Glaucoma suspected"))
        self.assertEqual(self.page.locator(".tile").count(), 3)
        self.assertIn("Anomaly score", self.page.inner_text(".tiles"))
        self.assertIn("Cup-to-disc ratio", self.page.inner_text(".tiles"))
        self.assertIn("Confidence", self.page.inner_text(".tiles"))
        # both pictures really loaded (a broken data/blob URL would leave naturalWidth at 0)
        sizes = self.page.evaluate("[...document.querySelectorAll('.fig img')].map(i => i.naturalWidth)")
        self.assertEqual(len(sizes), 2)
        self.assertTrue(all(w > 0 for w in sizes), sizes)
        # keyboard and screen-reader users land on the result
        self.assertEqual(self.page.evaluate("document.activeElement.id"), "result-top")
        # the annotated image can be opened and downloaded
        href = self.page.get_attribute("a:has-text('Download annotated image')", "href")
        self.assertTrue(href.startswith("blob:"), href)
        self.assertEqual(self.page.get_attribute("a:has-text('Download annotated image')", "download"), "annotated_fundus.png")
        # no test-model warnings when the models are not test models
        self.assertEqual(self.page.locator("text=Test models").count(), 0)
        self.assertIn("models loaded", self.page.inner_text("#status-pill"))

    def test_result_table_lists_the_marked_regions_with_plain_words(self):
        self.open(self.url_real_looking, "analyze")
        self.analyze()
        self.page.wait_for_selector(".regions table")
        rows = self.page.locator(".regions tbody tr")
        self.assertGreaterEqual(rows.count(), 1)
        text = self.page.inner_text(".regions")
        self.assertIn("A1", text)
        self.assertRegex(text, r"(High|Moderate)")
        self.assertRegex(text, r"(left|right|centre|upper|lower|middle)")

    def test_help_panels_open_and_show_what_it_is_and_how_to_read_it(self):
        self.open(self.url_real_looking, "analyze")
        self.analyze()
        self.page.wait_for_selector(".help details")
        panels = self.page.locator(".help details")
        self.assertEqual(panels.count(), 3)
        for i in range(3):
            panels.nth(i).locator("summary").click()
        text = self.page.inner_text(".help")
        self.assertIn("What it is.", text)
        self.assertIn("How to read it.", text)
        self.assertIn("NOT the probability", text)       # the confidence warning is shown, not hidden
        self.assertEqual(self.page.evaluate("document.querySelectorAll('.help details[open]').length"), 3)

    def test_test_models_are_flagged_everywhere(self):
        self.open(self.url_test_models, "analyze")
        self.assertIn("TEST models", self.page.inner_text("#status-pill"))
        self.assertIn("These are test models", self.page.inner_text("main"))
        self.analyze()
        self.page.wait_for_selector("#result-top")
        self.assertIn("Test models: ignore this result", self.page.inner_text(".result"))
        self.assertIn("stand-in (test only)", self.page.inner_text(".fine"))

    # ------------------------------------------------------------------ errors and edge cases
    def test_unreadable_image_gives_a_clear_error_with_a_reference(self):
        bad = Path(self.tmp.name) / "broken.jpg"
        bad.write_bytes(bytes(range(256)) * 20)
        self.open(self.url_real_looking, "analyze")
        self.analyze(str(bad))
        self.page.wait_for_selector("[role=alert]")
        text = self.page.inner_text("[role=alert]")
        self.assertIn("not a readable JPG or PNG", text)
        self.assertRegex(text, r"reference [0-9a-f]{8}")
        self.assertEqual(self.page.locator(".result").count(), 0)
        self.assertTrue(self.page.is_hidden("img.preview"))                             # no broken-image icon is left behind
        self.assertFalse(self.page.is_disabled("button:has-text('Analyze image')"))     # the person can simply try again

    def test_wrong_file_type_is_refused_in_the_page(self):
        note = Path(self.tmp.name) / "notes.txt"
        note.write_text("hello")
        self.open(self.url_real_looking, "analyze")
        self.page.set_input_files("#file", str(note))
        self.page.wait_for_selector("[role=alert]")
        self.assertIn("Please choose a JPG or PNG", self.page.inner_text("[role=alert]"))
        self.assertTrue(self.page.is_disabled("button:has-text('Analyze image')"))

    def test_models_not_loaded_is_explained_not_hidden(self):
        self.open(self.url_no_models, "analyze")
        self.assertIn("models not loaded", self.page.inner_text("#status-pill"))
        self.assertIn("trained models are not loaded", self.page.inner_text("main"))
        self.assertIn("scripts/run_pipeline.py", self.page.inner_text("main"))
        self.analyze()
        self.page.wait_for_selector("[role=alert]")
        self.assertIn("reference", self.page.inner_text("[role=alert]"))
        self.assertEqual(self.page.locator(".verdict-pill").count(), 0)          # never a made-up result

    def test_clear_button_resets_the_page(self):
        self.open(self.url_real_looking, "analyze")
        self.analyze()
        self.page.wait_for_selector("#result-top")
        self.page.click("button:has-text('Clear')")
        self.assertEqual(self.page.locator(".result").count(), 0)
        self.assertTrue(self.page.is_hidden("img.preview"))
        self.assertTrue(self.page.is_disabled("button:has-text('Analyze image')"))
        self.assertIn("Your result will appear here", self.page.inner_text("main"))

    def test_clear_while_analysing_leaves_a_clean_page_not_a_false_error(self):
        self.open(self.url_slow, "analyze")
        self.analyze()
        self.page.wait_for_selector(".busy")
        self.page.click("button:has-text('Clear')")
        self.page.wait_for_timeout(2500)                  # long enough for the abandoned request to finish on the server
        self.assertEqual(self.page.locator("[role=alert]").count(), 0, "a cancelled analysis must not show an error")
        self.assertEqual(self.page.locator(".result").count(), 0)
        self.assertEqual(self.page.locator(".busy").count(), 0)
        self.assertIn("Your result will appear here", self.page.inner_text("main"))
        self.assertTrue(self.page.is_disabled("button:has-text('Analyze image')"))

    def test_choosing_another_file_while_analysing_never_pairs_it_with_the_old_result(self):
        second = _fundus_png(Path(self.tmp.name) / "Im0002_SYN.png", level=90)
        self.open(self.url_slow, "analyze")
        self.analyze()
        self.page.wait_for_selector(".busy")
        self.page.set_input_files("#file", second)        # changes its mind while the first picture is being analysed
        self.page.wait_for_timeout(2500)
        self.assertEqual(self.page.locator(".result").count(), 0, "the old file's result appeared")
        self.assertEqual(self.page.locator("[role=alert]").count(), 0)
        self.page.click("button:has-text('Analyze image')")
        self.page.wait_for_selector("#result-top")
        same = self.page.evaluate("document.querySelector('.fig img').src === document.querySelector('img.preview').src")
        self.assertTrue(same, "the 'original image' beside the result is not the file that was analysed")

    def test_server_going_away_gives_a_helpful_message(self):
        server, url = _serve(create_app(analyzer=_LooksLikeRealModels(), db_path=Path(self.tmp.name) / "gone.sqlite3"))
        self.open(url, "analyze")
        server.shutdown()
        server.server_close()
        self.analyze()
        self.page.wait_for_selector("[role=alert]")
        self.assertIn("Cannot reach the analysis server", self.page.inner_text("[role=alert]"))

    # ------------------------------------------------------------------ other pages
    def test_history_shows_the_analysis_and_shows_a_file_name_as_text_only(self):
        evil = Path(self.tmp.name) / "<img src=x onerror=window.__xss=1>.png"
        _fundus_png(evil)
        self.open(self.url_real_looking, "analyze")
        self.analyze(str(evil))
        self.page.wait_for_selector("#result-top")
        self.page.click("a.nav-link[data-route=history]")
        self.page.wait_for_selector("table tbody tr")
        self.assertIn("<img src=x onerror=window.__xss=1>.png", self.page.inner_text("table"))
        self.assertEqual(self.page.locator("table img").count(), 0)
        self.assertIsNone(self.page.evaluate("window.__xss"))

    def test_guide_page_explains_the_three_key_numbers(self):
        self.open(self.url_real_looking, "guide")
        self.page.wait_for_selector(".guide-item")
        text = self.page.inner_text("main")
        for phrase in ("Anomaly score", "Cup-to-disc ratio (CDR)", "Confidence", "NOT the probability"):
            self.assertIn(phrase, text)
        self.assertGreaterEqual(self.page.locator(".guide-item").count(), 5)

    def test_unknown_page_says_so(self):
        self.open(self.url_real_looking, "nope")
        self.assertIn("Page not found", self.page.inner_text("h1"))

    def test_navigation_updates_title_and_current_link(self):
        self.open(self.url_real_looking)
        self.page.click("a.nav-link[data-route=guide]")
        self.page.wait_for_selector(".guide-item")
        self.assertEqual(self.page.title(), "How to read results - Glaucoma Screening")
        self.assertEqual(self.page.get_attribute("a.nav-link[aria-current=page]", "data-route"), "guide")

    # ------------------------------------------------------------------ phones
    def test_no_sideways_scrolling_on_a_phone(self):
        self.context.close()
        self.context = self.browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
        self.context.route("https://fonts.googleapis.com/**", lambda r: r.fulfill(status=200, content_type="text/css", body=""))
        self.context.route("https://fonts.gstatic.com/**", lambda r: r.fulfill(status=200, body=""))
        self.page = self.context.new_page()
        self.page.set_default_timeout(15000)
        self.page.on("pageerror", lambda e: self.js_errors.append(str(e)))
        for route in ("", "guide", "history"):
            self.open(self.url_real_looking, route)
            self.assertEqual(self.no_sideways_scroll(), 0, f"page '{route}' scrolls sideways")
        self.open(self.url_real_looking, "analyze")
        self.analyze()
        self.page.wait_for_selector("#result-top")
        self.assertEqual(self.no_sideways_scroll(), 0, "the result scrolls sideways")


if __name__ == "__main__":
    unittest.main()
