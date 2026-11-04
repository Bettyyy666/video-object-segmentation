"""Generic regressions for crop geometry, CLIP filtering, task-text counts and raw detector diagnostics.
Run with: python tests/test_diagnostics.py
"""
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seg import config as C, finalize, verify, vocab
from tools.probe_det import raw_prompt_maxima


class DiagnosticsTests(unittest.TestCase):
    def test_thin_object_endpoints_survive_clip_preprocessing(self):
        from clip.clip import _transform
        from PIL import Image
        for vertical in (False, True):
            # Deliberately touch image borders, with different marks at both tips.
            img = np.full((31, 301, 3), 90, np.uint8)
            mask = np.zeros(img.shape[:2], bool)
            mask[10:21, :] = True
            img[mask] = (0, 200, 0)
            img[10:21, :20] = (0, 0, 255)
            img[10:21, -20:] = (255, 0, 0)
            if vertical:
                img, mask = img.transpose(1, 0, 2), mask.T
            original = img.copy()
            crop = verify.masked_square_crop(img, mask)
            self.assertEqual(crop.shape[0], crop.shape[1])
            np.testing.assert_array_equal(img, original)
            transformed = _transform(224)(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))).numpy()
            red, green, blue = transformed
            self.assertGreater(int(((red > 1) & (blue < 0) & (green < 0)).sum()), 20)
            self.assertGreater(int(((blue > 1) & (red < 0) & (green < 0)).sum()), 20)
        with self.assertRaises(ValueError):
            verify.masked_square_crop(img, np.zeros(mask.shape, bool))

    def test_generic_clip_threshold(self):
        self.assertEqual(C.VERIFY_MIN_OBJECT, 0.4)
        meta = {"fps": 50, "num_frames": 100, "tracks": [
            {"id": i, "votes": {"widget": 1.0}, "first": 0, "last": 99, "n_vis": 100,
             "median_area": 900, "events": []} for i in range(1, 6)]}
        # The threshold applies to any object, only with enough supporting samples.
        scores = [0.23, 0.34, 0.77, 0.4, 0.23]
        ver = {str(i): {"object": p, "cats": {"widget": p}, "n": 2 if i == 5 else 8}
               for i, p in enumerate(scores, 1)}
        _, remap, _, _ = finalize.plan(meta, verify=ver, log=lambda *_: None)
        self.assertEqual(set(remap), {3, 4, 5})
        meta["tracks"][0]["votes"] = {"robot": 1.0}
        _, remap, _, _ = finalize.plan(meta, verify=ver, log=lambda *_: None)
        self.assertIn(1, remap)

    def test_generic_the_counts(self):
        def count(rows):
            task = {"title": "Arrange objects", "subtasks": [
                {"object": "basket", "subtask": text, "count": n} for text, n in rows]}
            return vocab.expected_counts(task, [("basket", "basket")], {})
        self.assertEqual(count([("Lift the basket", "1"), ("Move the basket", "1")]), {"basket": 1})
        for rows in [ [("Lift the basket", "2")],
                      [("Lift the basket", "1"), ("Move baskets", "2")],
                      [("Lift the basket", "1"), ("Move the second basket", "1")],
                      [("Lift the basket", "1"), ("Move the other basket", "1")],
                      [("Lift the basket", "1"), ("Move the next basket", "1")],
                      [("Lift the basket", "1-2")]]:
            self.assertEqual(count(rows), {}, rows)
        self.assertEqual(count([("Place one basket", "3")]), {"basket": 3})
        # All supplied task sheets still parse; no per-video quantity override is used.
        for i in range(1, 31):
            counts = vocab.build("task_sheets.json", f"{i:02d}_test")[-1]
            self.assertTrue(all(isinstance(n, int) and n > 0 for n in counts.values()))
        counts = vocab.build("task_sheets.json", "26_sort_cutlery_drawer_insert")[-1]
        self.assertEqual(counts, {"fork": 2, "knife": 2, "spoon": 4, "wooden utensil": 1, "insert": 1})

    def test_route_unseen_categories_to_grounding_dino(self):
        from seg.tracker import route_categories
        prompts_cats = ["cable", "cable", "strap", "organizer", "robot"]
        best = [0.012, 0.018, 0.004, 0.91, 0.001]          # robots are never routed
        self.assertEqual(route_categories(best, prompts_cats), ["cable", "strap"])
        self.assertEqual(route_categories([0.6, 0.0, 0.3, 0.9, 0.5], prompts_cats), [])   # one prompt is enough
        # generic containers added to every video are never routed (only task sheet categories are)
        self.assertEqual(route_categories([0.0, 0.0, 0.9], ["basket", "container", "fork"]), [])

    def test_grounding_dino_phrase_matching(self):
        from seg.gdino import GroundingDino
        g = GroundingDino.__new__(GroundingDino)
        g.prompts = ["cable", "coiled cable", "cable bundle", "velcro strap", "strap", "organizer box"]
        self.assertEqual(g.match("cable"), 0)
        self.assertEqual(g.match("coiled cable"), 1)
        self.assertEqual(g.match("velcro"), 3)
        self.assertEqual(g.match("strap"), 4)
        self.assertIsNone(g.match("table"))
        self.assertIsNone(g.match(""))

    def test_raw_scores_include_nonwinning_classes(self):
        import torch
        # Class 0 loses every anchor to class 1 but has a real 0.2 score.
        raw = torch.zeros((1, 6, 2))
        raw[0, 4:, :] = torch.tensor([[0.1, 0.2], [0.9, 0.8]])
        np.testing.assert_allclose(raw_prompt_maxima((raw, None), 2), [0.2, 0.9])
        with self.assertRaises(ValueError):
            raw_prompt_maxima(raw, 3)


if __name__ == "__main__":
    unittest.main()
