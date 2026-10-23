"""Tests that need no PyTorch. Run:  python -m unittest discover -s tests -v   (or pytest -q)."""
import io
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from glaucoma.cdr import compute_cdr, masks_from_refuge  # noqa: E402
from glaucoma.config import load_config  # noqa: E402
from glaucoma.data.splits import assert_no_leakage, make_all_splits  # noqa: E402
from glaucoma.data.synthetic import generate  # noqa: E402
from glaucoma.eval.annotate import annotate  # noqa: E402
from glaucoma.eval.fusion import FusionModel  # noqa: E402
from glaucoma.eval.metrics import bootstrap_auc_ci, compute_metrics, youden_threshold  # noqa: E402


def disc_cup(disc_h, cup_h, size=200):
    disc = np.zeros((size, size), np.uint8)
    cup = np.zeros((size, size), np.uint8)
    cy = size // 2
    disc[cy - disc_h // 2: cy - disc_h // 2 + disc_h, 60:140] = 1
    cup[cy - cup_h // 2: cy - cup_h // 2 + cup_h, 80:120] = 1
    return disc, cup


class CDRTests(unittest.TestCase):
    def test_known_ratio(self):
        r = compute_cdr(*disc_cup(100, 40))
        self.assertEqual(r.value, 0.4)
        self.assertIsNone(r.reason)

    def test_empty_masks_give_none_with_reason(self):
        z = np.zeros((50, 50), np.uint8)
        r = compute_cdr(z, z)
        self.assertIsNone(r.value)
        self.assertEqual(r.reason, "optic disc not found")

    def test_cup_not_found(self):
        d, _ = disc_cup(100, 40)
        r = compute_cdr(d, np.zeros_like(d))
        self.assertIsNone(r.value)
        self.assertEqual(r.reason, "optic cup not found")

    def test_cup_taller_than_disc_is_rejected(self):
        r = compute_cdr(*disc_cup(60, 90))
        self.assertIsNone(r.value)

    def test_largest_component_ignores_specks(self):
        d, c = disc_cup(100, 40)
        d[5, 5] = 1  # a stray pixel far from the disc must not change the height
        self.assertEqual(compute_cdr(d, c).value, 0.4)

    def test_refuge_mask_decoding(self):
        m = np.full((10, 10), 255, np.uint8)
        m[2:8, 2:8] = 128
        m[4:6, 4:6] = 0
        disc, cup = masks_from_refuge(m)
        self.assertEqual(int(disc.sum()), 36)
        self.assertEqual(int(cup.sum()), 4)


class MetricsFusionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 200
        self.y = np.array([0, 1] * (n // 2))
        self.s = np.where(self.y == 1, rng.normal(0.06, 0.01, n), rng.normal(0.03, 0.01, n))
        self.c = np.where(self.y == 1, rng.normal(0.7, 0.08, n), rng.normal(0.4, 0.08, n))

    def test_metrics_on_separable_data(self):
        thr = youden_threshold(self.y, self.s)
        m = compute_metrics(self.y, self.s, thr)
        self.assertGreater(m["auc"], 0.95)
        self.assertGreater(m["sensitivity"], 0.8)
        self.assertEqual(sum(m["confusion"].values()), len(self.y))

    def test_single_class_fails_loudly(self):
        with self.assertRaises(ValueError):
            youden_threshold(np.zeros(10), np.arange(10))

    def test_bootstrap_ci_brackets_auc(self):
        thr = youden_threshold(self.y, self.s)
        auc = compute_metrics(self.y, self.s, thr)["auc"]
        lo, hi = bootstrap_auc_ci(self.y, self.s, 200, 1)
        self.assertLessEqual(lo, auc + 1e-9)
        self.assertGreaterEqual(hi, auc - 1e-9)

    def test_fusion_predicts_and_roundtrips(self):
        fm = FusionModel.fit(self.s, self.c, self.y)
        hi = fm.predict(0.07, 0.8)
        lo = fm.predict(0.02, 0.3)
        self.assertEqual(hi["label"], "glaucoma_suspected")
        self.assertEqual(lo["label"], "normal")
        self.assertGreater(hi["fused_score"], lo["fused_score"])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "fusion.json"
            fm.save(p)
            again = FusionModel.load(p).predict(0.07, 0.8)
        self.assertEqual(hi["fused_score"], again["fused_score"])

    def test_margin_sign_matches_label(self):
        fm = FusionModel.fit(self.s, self.c, self.y)
        for s, c in [(0.07, 0.8), (0.02, 0.3), (0.05, None), (0.03, None)]:
            r = fm.predict(s, c)
            self.assertEqual(r["margin"] >= 0, r["label"] == "glaucoma_suspected")

    def test_fusion_without_cdr_uses_anomaly_only(self):
        fm = FusionModel.fit(self.s, self.c, self.y)
        r = fm.predict(0.07, None)
        self.assertFalse(r["used_cdr"])
        self.assertIsNone(r["signals_agree"])

    def test_confidence_levels_and_range(self):
        fm = FusionModel.fit(self.s, self.c, self.y)
        for s, c in [(0.07, 0.8), (0.045, 0.55), (0.02, 0.3)]:
            r = fm.predict(s, c)
            self.assertTrue(0.0 <= r["confidence_score"] <= 1.0)
            self.assertIn(r["confidence_level"], {"Low", "Medium", "High"})
        self.assertEqual(fm.predict(0.07, 0.8)["confidence_level"], "High")

    def test_fusion_handles_missing_cdr_in_fit(self):
        c = self.c.copy()
        c[::3] = np.nan
        fm = FusionModel.fit(self.s, c, self.y)
        self.assertIn(fm.predict(0.06, 0.7)["label"], {"normal", "glaucoma_suspected"})


class AnnotateTests(unittest.TestCase):
    def test_returns_valid_png_same_size_and_labels_hot_region(self):
        img = np.full((200, 240, 3), 120, np.uint8)
        err = np.zeros((50, 60), np.float32)
        err[10:25, 30:50] = 0.2
        png, regions = annotate(img, err, 0.05, 0.1)
        out = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(out.shape[:2], (200, 240))
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0]["id"], "A1")
        self.assertEqual(regions[0]["severity"], "high")
        self.assertFalse(np.array_equal(out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR)))

    def test_blank_error_gives_no_regions(self):
        img = np.full((128, 128, 3), 90, np.uint8)
        png, regions = annotate(img, np.zeros((32, 32), np.float32), 0.05, 0.1)
        self.assertEqual(regions, [])
        self.assertTrue(len(png) > 100)

    def test_outlines_and_cdr_label_drawn(self):
        img = np.full((200, 200, 3), 60, np.uint8)
        d, c = disc_cup(80, 40)
        png, _ = annotate(img, np.zeros((50, 50), np.float32), 0.05, 0.1, d, c, 0.5)
        out = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        self.assertGreater(int((out != 60).any(axis=2).sum()), 200)

    def test_severity_ordering_and_max_regions(self):
        img = np.full((256, 256, 3), 90, np.uint8)
        err = np.zeros((64, 64), np.float32)
        for k, v in enumerate([0.06, 0.2, 0.07, 0.065]):
            err[5 + k * 14: 12 + k * 14, 10:30] = v
        _, regions = annotate(img, err, 0.05, 0.1, max_regions=3)
        self.assertLessEqual(len(regions), 3)
        self.assertEqual(regions[0]["severity"], "high")


class SplitTests(unittest.TestCase):
    def test_synthetic_splits_have_no_leakage_and_keep_classes(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = load_config(quick=True)
            cfg["data"]["synthetic_dir"] = str(Path(d) / "syn")
            cfg["paths"]["data_splits"] = str(Path(d) / "splits")
            generate(cfg["data"]["synthetic_dir"], n=60, size=64, seed=1)
            counts = make_all_splits(cfg)
            import pandas as pd
            df = pd.read_csv(Path(d) / "splits" / "classification.csv")
            assert_no_leakage(df, "path")
            for s in ("train", "val", "test"):
                self.assertEqual(set(df[df.split == s].label), {0, 1}, s)
            self.assertEqual(len(df), 60)
            self.assertIn("segmentation", counts)

    def test_leakage_check_catches_duplicates(self):
        import pandas as pd
        df = pd.DataFrame({"path": ["a", "a"], "split": ["train", "test"]})
        with self.assertRaises(AssertionError):
            assert_no_leakage(df, "path")

    def test_acrima_label_rule(self):
        from glaucoma.data.index import label_from_acrima_name
        self.assertEqual(label_from_acrima_name("Im001_ACRIMA.jpg"), 0)
        self.assertEqual(label_from_acrima_name("Im400_g_ACRIMA.jpg"), 1)


class StubAnalyzer:
    def analyze(self, rgb):
        return {"label": "normal", "anomaly_score": 0.02, "anomaly_threshold": 0.04, "cdr": 0.3, "cdr_reason": None,
                "cdr_reference": 0.65, "fused_score": 0.1, "threshold": 0.5, "confidence_score": 0.8,
                "confidence_level": "High", "signals_agree": True, "used_cdr": True, "regions": [],
                "annotated_image": "data:image/png;base64,AAAA", "image_size": {"width": 1, "height": 1},
                "model_versions": {"app": "test"}}


def png_bytes(w=128, h=128):
    ok, buf = cv2.imencode(".png", np.full((h, w, 3), 100, np.uint8))
    return buf.tobytes()


class APITests(unittest.TestCase):
    def setUp(self):
        from backend.app import create_app
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "t.sqlite3"
        self.app = create_app(analyzer=StubAnalyzer(), db_path=self.db).test_client()
        self.down = create_app(analyzer=None, db_path=self.db, load_models=False).test_client()

    def tearDown(self):
        self.tmp.cleanup()

    def post(self, client, data, name="x.png"):
        return client.post("/predict", data={"image": (io.BytesIO(data), name)}, content_type="multipart/form-data")

    def test_health_ok(self):
        r = self.app.get("/health").get_json()
        self.assertTrue(r["models_loaded"])

    def test_health_degraded_when_models_missing(self):
        r = self.down.get("/health").get_json()
        self.assertFalse(r["models_loaded"])
        self.assertEqual(r["status"], "degraded")

    def test_predict_503_when_models_missing(self):
        r = self.post(self.down, png_bytes())
        self.assertEqual(r.status_code, 503)
        self.assertIn("message", r.get_json()["error"])

    def test_predict_valid_image_and_history(self):
        r = self.post(self.app, png_bytes())
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertIn("disclaimer", body)
        self.assertEqual(body["label"], "normal")
        h = self.app.get("/history").get_json()["items"]
        self.assertEqual(len(h), 1)
        self.assertEqual(h[0]["filename"], "x.png")
        self.assertNotIn("annotated_image", h[0])

    def test_corrupt_file_is_422(self):
        r = self.post(self.app, b"not an image at all")
        self.assertEqual(r.status_code, 422)

    def test_wrong_extension_is_415(self):
        self.assertEqual(self.post(self.app, png_bytes(), "x.gif").status_code, 415)

    def test_missing_file_is_400(self):
        r = self.app.post("/predict", data={}, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)

    def test_tiny_image_is_422(self):
        self.assertEqual(self.post(self.app, png_bytes(16, 16)).status_code, 422)

    def test_oversized_upload_is_413(self):
        self.app.application.config["MAX_CONTENT_LENGTH"] = 1000
        self.assertEqual(self.post(self.app, png_bytes(512, 512)).status_code, 413)

    def test_explain_endpoint_covers_the_three_key_terms(self):
        r = self.app.get("/explain").get_json()
        for key in ("anomaly_score", "cdr", "confidence"):
            self.assertIn(key, r["glossary"])
            self.assertTrue(r["glossary"][key]["what"] and r["glossary"][key]["read"])
        self.assertIn("not give a medical diagnosis", r["disclaimer"])
        self.assertIn("NOT the probability", r["glossary"]["confidence"]["caution"])

    def test_cors_only_for_configured_origin(self):
        ok = self.app.get("/health", headers={"Origin": "http://localhost:5173"})
        bad = self.app.get("/health", headers={"Origin": "http://evil.example"})
        self.assertEqual(ok.headers.get("Access-Control-Allow-Origin"), "http://localhost:5173")
        self.assertIsNone(bad.headers.get("Access-Control-Allow-Origin"))


try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "PyTorch not installed")
class TorchTests(unittest.TestCase):
    def test_autoencoder_shapes_and_range(self):
        import torch
        from glaucoma.models.autoencoder import ConvAutoencoder
        m = ConvAutoencoder(channels=(8, 16, 32), latent_channels=8)
        y = m(torch.rand(2, 3, 64, 64))
        self.assertEqual(tuple(y.shape), (2, 3, 64, 64))
        self.assertTrue(float(y.min()) >= 0 and float(y.max()) <= 1)
        self.assertEqual(tuple(m.error_map(torch.rand(2, 3, 64, 64)).shape), (2, 64, 64))

    def test_autoencoder_with_skips_runs(self):
        import torch
        from glaucoma.models.autoencoder import ConvAutoencoder
        m = ConvAutoencoder(channels=(8, 16), latent_channels=4, skip=True)
        self.assertEqual(tuple(m(torch.rand(1, 3, 32, 32)).shape), (1, 3, 32, 32))

    def test_bad_input_size_fails_loudly(self):
        import torch
        from glaucoma.models.autoencoder import ConvAutoencoder
        with self.assertRaises(ValueError):
            ConvAutoencoder(channels=(8, 16, 32))(torch.rand(1, 3, 60, 60))

    def test_overfit_sanity_loss_goes_down(self):
        import torch
        from glaucoma.models.autoencoder import ConvAutoencoder
        torch.manual_seed(0)
        m = ConvAutoencoder(channels=(8, 16), latent_channels=8)
        x = torch.rand(8, 3, 32, 32)
        opt = torch.optim.Adam(m.parameters(), 1e-2)
        first = None
        for _ in range(40):
            loss = torch.nn.functional.l1_loss(m(x), x)
            opt.zero_grad(); loss.backward(); opt.step()
            first = first if first is not None else loss.item()
        self.assertLess(loss.item(), first * 0.8)

    def test_segmenter_output_shape_and_dice(self):
        import torch
        from glaucoma.models.segmenter import UNet, dice_score
        m = UNet(base=4)
        out = m(torch.rand(2, 3, 32, 32))
        self.assertEqual(tuple(out.shape), (2, 2, 32, 32))
        perfect = torch.full((1, 2, 8, 8), 20.0)
        self.assertAlmostEqual(float(dice_score(perfect, torch.ones(1, 2, 8, 8)).mean()), 1.0, places=3)

    def test_autoencoder_training_set_has_no_glaucoma(self):
        from glaucoma.data.datasets import ImageDataset
        import pandas as pd
        with tempfile.TemporaryDirectory() as d:
            generate(Path(d) / "syn", n=20, size=64, seed=2)
            files = sorted((Path(d) / "syn" / "images").glob("*.png"))
            df = pd.DataFrame({"path": [str(f) for f in files], "label": [int("_g_" in f.name) for f in files]})
            healthy = df[df.label == 0]
            ds = ImageDataset(healthy, 64, train=True)
            self.assertTrue(all(ds[i][1] == 0 for i in range(len(ds))))


if __name__ == "__main__":
    unittest.main()
