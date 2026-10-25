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


class SegPrepareTests(unittest.TestCase):
    def _make(self, tmp, with_cup=True, h=300, w=400):
        import json
        import cv2
        img = np.full((h, w, 3), 90, np.uint8)
        cv2.imwrite(str(tmp / "img1.jpg"), img)
        shapes = [{"label": "disc", "points": [[150, 100], [250, 100], [250, 200], [150, 200]]}]
        if with_cup:
            shapes.append({"label": "cup", "points": [[180, 130], [220, 130], [220, 170], [180, 170]]})
        (tmp / "img1.json").write_text(json.dumps({"imageHeight": h, "imageWidth": w, "shapes": shapes}))

    def test_convert_crop_and_cdr_roundtrip(self):
        import tempfile
        from pathlib import Path
        import cv2
        from glaucoma.data.seg_prepare import prepare_json_dataset
        with tempfile.TemporaryDirectory() as t:
            raw, out = Path(t) / "raw", Path(t) / "out"
            raw.mkdir(); self._make(raw)
            rep = prepare_json_dataset(raw, out, scale=2.5)
            self.assertEqual(rep["written"], 1)
            m = cv2.imread(str(out / "masks" / "g1020_img1.png"), cv2.IMREAD_GRAYSCALE)
            im = cv2.imread(str(out / "images" / "g1020_img1.png"))
            self.assertEqual(m.shape[:2], im.shape[:2])
            self.assertEqual(m.shape[0], m.shape[1])           # square crop
            self.assertEqual(m.shape[0], 252)                  # 2.5 x 101-pixel disc (polygon edges are inclusive)
            disc, cup = masks_from_refuge(m)
            r = compute_cdr(disc, cup)
            self.assertAlmostEqual(r.value, 0.41, delta=0.02)  # 41 / 101 pixel heights

    def test_missing_cup_is_reported_not_dropped_silently(self):
        import tempfile
        from pathlib import Path
        from glaucoma.data.seg_prepare import prepare_json_dataset
        with tempfile.TemporaryDirectory() as t:
            raw = Path(t) / "raw"; raw.mkdir(); self._make(raw, with_cup=False)
            rep = prepare_json_dataset(raw, Path(t) / "out", scale=2.5)
            self.assertEqual(rep["written"], 0)
            self.assertIn("no cup polygon", rep["skipped"])

    def test_bad_json_is_reported(self):
        import tempfile
        from pathlib import Path
        from glaucoma.data.seg_prepare import prepare_json_dataset
        with tempfile.TemporaryDirectory() as t:
            raw = Path(t) / "raw"; raw.mkdir()
            (raw / "x.json").write_text("{\"not\": \"labelme\"}")
            rep = prepare_json_dataset(raw, Path(t) / "out", scale=2.5)
            self.assertEqual(rep["written"], 0)
            self.assertTrue(any("unreadable" in k for k in rep["skipped"]))


class SegPrepareHardeningTests(unittest.TestCase):
    """Things that could silently ruin the segmenter's training data if the converter got them wrong."""
    H, W = 300, 400
    DISC = [[150, 100], [250, 100], [250, 200], [150, 200]]
    CUP = [[180, 130], [220, 130], [220, 170], [180, 170]]

    def _write(self, folder, stem, shapes, size=True, image_path=None, img_name=None, h=None, w=None):
        import json
        h, w = h or self.H, w or self.W
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / f"{img_name or stem}.jpg"), np.full((h, w, 3), 90, np.uint8))
        data = {"shapes": shapes}
        if size:
            data.update(imageHeight=h, imageWidth=w)
        if image_path:
            data["imagePath"] = image_path
        (folder / f"{stem}.json").write_text(json.dumps(data))

    def _prepare(self, raw, **kw):
        from glaucoma.data.seg_prepare import prepare_json_dataset
        return prepare_json_dataset(raw, raw.parent / "out", scale=2.5, **kw)

    def test_bounding_boxes_are_never_read_as_the_disc(self):
        from glaucoma.data.seg_prepare import classify_label
        self.assertEqual(classify_label("disc_bbox"), "box")
        with tempfile.TemporaryDirectory() as t:
            raw = Path(t) / "raw"
            shapes = [{"label": "disc", "points": self.DISC}, {"label": "cup", "points": self.CUP},
                      {"label": "disc", "shape_type": "rectangle", "points": [[20, 20], [380, 280]]},
                      {"label": "disc_bbox", "points": [[10, 10], [390, 10], [390, 290], [10, 290]]}]
            self._write(raw, "img1", shapes)
            rep = self._prepare(raw)
            self.assertEqual(rep["written"], 1)
            self.assertEqual(rep["ignored_boxes"], 2)
            m = cv2.imread(str(Path(t) / "out" / "masks" / "g1020_img1.png"), cv2.IMREAD_GRAYSCALE)
            r = compute_cdr(*masks_from_refuge(m))
            self.assertAlmostEqual(r.value, 0.41, delta=0.02)   # unchanged by the two boxes

    def test_several_label_spellings_are_understood(self):
        from glaucoma.data.seg_prepare import classify_label
        for lab in ("disc", "Disc", "disk", "OD", "optic_disc", "Optic Disc", "OpticDisc"):
            self.assertEqual(classify_label(lab), "disc", lab)
        for lab in ("cup", "Cup", "OC", "optic_cup", "Optic Cup", "OpticCup"):
            self.assertEqual(classify_label(lab), "cup", lab)
        self.assertIsNone(classify_label("macula"))
        with tempfile.TemporaryDirectory() as t:
            raw = Path(t) / "raw"
            self._write(raw, "a", [{"label": "OD", "points": self.DISC}, {"label": "OC", "points": self.CUP}])
            self.assertEqual(self._prepare(raw)["written"], 1)

    def test_cup_outside_disc_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            raw = Path(t) / "raw"
            far_cup = [[300, 220], [340, 220], [340, 260], [300, 260]]
            self._write(raw, "a", [{"label": "disc", "points": self.DISC}, {"label": "cup", "points": far_cup}])
            rep = self._prepare(raw)
            self.assertEqual(rep["written"], 0)
            self.assertIn("cup polygon is not inside the disc polygon", rep["skipped"])

    def test_missing_disc_is_rejected_not_replaced_by_the_cup(self):
        with tempfile.TemporaryDirectory() as t:
            raw = Path(t) / "raw"
            self._write(raw, "a", [{"label": "cup", "points": self.CUP}])
            rep = self._prepare(raw)
            self.assertEqual(rep["written"], 0)
            self.assertIn("no disc polygon", rep["skipped"])

    def test_image_found_through_image_path_when_names_differ(self):
        with tempfile.TemporaryDirectory() as t:
            raw = Path(t) / "raw"
            self._write(raw, "ann1", [{"label": "disc", "points": self.DISC}, {"label": "cup", "points": self.CUP}],
                        image_path="C:\\data\\Images\\pic1.jpg", img_name="pic1")
            self.assertEqual(self._prepare(raw)["written"], 1)

    def test_polygon_outside_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            raw = Path(t) / "raw"
            far = [[[1000, 100], [1100, 100], [1100, 200], [1000, 200]]][0]
            self._write(raw, "a", [{"label": "disc", "points": far}, {"label": "cup", "points": far}], size=False)
            rep = self._prepare(raw)
            self.assertEqual(rep["written"], 0)
            self.assertTrue(any("outside the image" in k for k in rep["skipped"]))

    def test_large_crops_are_shrunk_and_masks_keep_three_values(self):
        with tempfile.TemporaryDirectory() as t:
            raw = Path(t) / "raw"
            disc = [[500, 300], [1100, 300], [1100, 900], [500, 900]]
            cup = [[700, 500], [900, 500], [900, 700], [700, 700]]
            self._write(raw, "big", [{"label": "disc", "points": disc}, {"label": "cup", "points": cup}], h=1200, w=1600)
            rep = self._prepare(raw, max_side=512)
            self.assertEqual(rep["written"], 1)
            m = cv2.imread(str(Path(t) / "out" / "masks" / "g1020_big.png"), cv2.IMREAD_GRAYSCALE)
            im = cv2.imread(str(Path(t) / "out" / "images" / "g1020_big.png"))
            self.assertEqual(m.shape, (512, 512))
            self.assertEqual(im.shape[:2], (512, 512))
            self.assertTrue(set(np.unique(m).tolist()) <= {0, 128, 255})
            r = compute_cdr(*masks_from_refuge(m))
            self.assertAlmostEqual(r.value, 201 / 601, delta=0.02)

    def test_same_name_in_two_folders_picks_the_one_matching_the_annotation_size(self):
        with tempfile.TemporaryDirectory() as t:
            raw = Path(t) / "raw"
            shapes = [{"label": "disc", "points": self.DISC}, {"label": "cup", "points": self.CUP}]
            (raw / "a").mkdir(parents=True)
            cv2.imwrite(str(raw / "a" / "x.jpg"), np.full((200, 200, 3), 90, np.uint8))   # same name, wrong size, sorts first
            self._write(raw / "b", "x", shapes)                                           # right size + the annotation
            rep = self._prepare(raw)
            self.assertEqual(rep["written"], 1)           # would be 0 ("size differs") if the wrong file were picked
            self.assertEqual(rep["ambiguous_names"], [])

    def test_same_name_without_annotation_size_is_flagged_as_ambiguous(self):
        with tempfile.TemporaryDirectory() as t:
            raw = Path(t) / "raw"
            shapes = [{"label": "disc", "points": self.DISC}, {"label": "cup", "points": self.CUP}]
            (raw / "a").mkdir(parents=True)
            cv2.imwrite(str(raw / "a" / "x.jpg"), np.full((300, 400, 3), 90, np.uint8))
            self._write(raw / "b", "x", shapes, size=False)
            rep = self._prepare(raw)
            self.assertEqual(rep["ambiguous_names"], ["x"])


class AugmentSegTests(unittest.TestCase):
    def _data(self, size=96):
        img = np.zeros((size, size, 3), np.uint8)
        img[:] = (60, 40, 30)
        disc = np.zeros((size, size), np.uint8)
        cup = np.zeros_like(disc)
        cv2.circle(disc, (48, 48), 24, 1, -1)
        cv2.circle(cup, (48, 48), 10, 1, -1)
        img[disc > 0] = (200, 150, 90)
        img[cup > 0] = (240, 220, 180)
        return img, disc, cup

    def test_identity_when_everything_is_switched_off(self):
        from glaucoma.preprocessing import augment_seg
        img, d, c = self._data()
        i2, d2, c2 = augment_seg(img, d, c, np.random.default_rng(0), scale=(1, 1), rot=0, shift=0,
                                 colour=False, flip=False)
        np.testing.assert_array_equal(i2, img)
        np.testing.assert_array_equal(d2, d)
        np.testing.assert_array_equal(c2, c)

    def test_masks_stay_binary_cup_stays_inside_disc(self):
        from glaucoma.preprocessing import augment_seg
        img, d, c = self._data()
        for seed in range(30):
            i2, d2, c2 = augment_seg(img, d, c, np.random.default_rng(seed))
            self.assertEqual(i2.dtype, np.uint8)
            self.assertEqual(i2.shape, img.shape)
            self.assertTrue(set(np.unique(d2).tolist()) <= {0, 1})
            self.assertTrue(set(np.unique(c2).tolist()) <= {0, 1})
            self.assertEqual(int((c2 > d2).sum()), 0, f"cup outside disc for seed {seed}")

    def test_deterministic_for_a_given_seed(self):
        from glaucoma.preprocessing import augment_seg
        img, d, c = self._data()
        a = augment_seg(img, d, c, np.random.default_rng(5))
        b = augment_seg(img, d, c, np.random.default_rng(5))
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_masks_move_with_the_image(self):
        from glaucoma.preprocessing import augment_seg
        img, d, c = self._data()
        for seed in range(20):
            i2, d2, _ = augment_seg(img, d, c, np.random.default_rng(seed), colour=False)
            if d2.sum() > 50:
                self.assertGreater(float(i2[d2 > 0].mean()), float(i2[d2 == 0].mean()), f"seed {seed}")

    def test_it_really_changes_size_and_colour(self):
        from glaucoma.preprocessing import augment_seg
        img, d, c = self._data()
        areas, means = set(), set()
        for seed in range(15):
            i2, d2, _ = augment_seg(img, d, c, np.random.default_rng(seed))
            areas.add(int(d2.sum()))
            means.add(round(float(i2.mean()), 1))
        self.assertGreater(len(areas), 8)
        self.assertGreater(len(means), 8)


class CDRCheckTests(unittest.TestCase):
    @staticmethod
    def _data(rng, shift=0.3, n=60, sd=0.08):
        y = np.r_[np.zeros(n), np.ones(n)].astype(int)
        c = np.r_[rng.normal(0.4, sd, n), rng.normal(0.4 + shift, sd, n)]
        return c, y

    def test_informative_ratio_gives_no_warnings(self):
        from glaucoma.eval.cdr_check import cdr_sanity
        rng = np.random.default_rng(0)
        cv, yv = self._data(rng)
        ct, yt = self._data(rng)
        info = cdr_sanity(cv, yv, ct, yt, threshold=0.55, n_boot=50)
        self.assertGreater(info["val_auc"], 0.9)
        self.assertEqual(info["warnings"], [])
        self.assertIsNotNone(info["test_metrics"])
        self.assertGreater(info["val_mean_cdr_glaucoma"], info["val_mean_cdr_normal"])

    def test_uninformative_ratio_warns_about_chance(self):
        from glaucoma.eval.cdr_check import cdr_sanity
        rng = np.random.default_rng(1)
        cv, yv = self._data(rng, shift=0.0)
        ct, yt = self._data(rng, shift=0.0)
        info = cdr_sanity(cv, yv, ct, yt, threshold=0.4, n_boot=50)
        self.assertTrue(any("close to chance" in m for m in info["warnings"]), info["warnings"])

    def test_inverted_ratio_is_called_inverted(self):
        from glaucoma.eval.cdr_check import cdr_sanity
        rng = np.random.default_rng(2)
        cv, yv = self._data(rng, shift=-0.3)
        ct, yt = self._data(rng, shift=-0.3)
        info = cdr_sanity(cv, yv, ct, yt, threshold=0.4, n_boot=50)
        self.assertTrue(any("INVERTED" in m for m in info["warnings"]), info["warnings"])

    def test_many_missing_values_warn(self):
        from glaucoma.eval.cdr_check import cdr_sanity
        rng = np.random.default_rng(3)
        cv, yv = self._data(rng)
        ct, yt = self._data(rng)
        cv[::2] = np.nan
        info = cdr_sanity(cv, yv, ct, yt, threshold=0.55, n_boot=50)
        self.assertAlmostEqual(info["unavailable_val"], 0.5)
        self.assertTrue(any("No cup-to-disc ratio could be computed" in m for m in info["warnings"]))

    def test_all_missing_does_not_crash(self):
        from glaucoma.eval.cdr_check import cdr_sanity
        y = np.array([0, 1] * 20)
        nan = np.full(40, np.nan)
        info = cdr_sanity(nan, y, nan, y, threshold=None, n_boot=10)
        self.assertIsNone(info["val_auc"])
        self.assertIsNone(info["test_metrics"])
        self.assertTrue(info["warnings"])


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

    def test_macos_zip_junk_is_not_treated_as_images(self):
        from glaucoma.data.index import classification_index, is_junk
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "acrima"
            (root / "__MACOSX").mkdir(parents=True)
            cv2.imwrite(str(root / "Im001_ACRIMA.png"), np.full((32, 32, 3), 80, np.uint8))
            cv2.imwrite(str(root / "Im002_g_ACRIMA.png"), np.full((32, 32, 3), 90, np.uint8))
            (root / "__MACOSX" / "._Im001_ACRIMA.png").write_bytes(b"\x00\x05\x16\x07 not an image")
            (root / "._Im002_g_ACRIMA.png").write_bytes(b"\x00\x05\x16\x07 not an image")
            (root / ".DS_Store").write_bytes(b"x")
            cfg = load_config()
            cfg["data"]["source"], cfg["data"]["acrima_dir"] = "acrima", str(root)
            df = classification_index(cfg)
            self.assertEqual(sorted(Path(p).name for p in df.path), ["Im001_ACRIMA.png", "Im002_g_ACRIMA.png"])
            self.assertEqual(df.sort_values("path").label.tolist(), [0, 1])
            self.assertTrue(is_junk(root / "__MACOSX" / "x.jpg"))
            self.assertFalse(is_junk(root / "Im001_ACRIMA.png"))

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

    def test_seg_dataset_train_targets_are_binary_and_cup_inside_disc(self):
        from glaucoma.data.datasets import SegDataset
        import pandas as pd
        with tempfile.TemporaryDirectory() as d:
            generate(Path(d) / "syn", n=6, size=64, seed=3)
            imgs = sorted((Path(d) / "syn" / "images").glob("*.png"))
            df = pd.DataFrame({"image": [str(p) for p in imgs],
                               "mask": [str(Path(d) / "syn" / "masks" / p.name) for p in imgs]})
            for train in (False, True):
                ds = SegDataset(df, 64, train=train, seed=0)
                for i in range(len(ds)):
                    x, y = ds[i]
                    self.assertEqual(tuple(x.shape), (3, 64, 64))
                    self.assertEqual(tuple(y.shape), (2, 64, 64))
                    self.assertTrue(set(y.unique().tolist()) <= {0.0, 1.0})
                    self.assertEqual(int((y[1] > y[0]).sum()), 0)   # cup (channel 1) lies inside disc (channel 0)

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
