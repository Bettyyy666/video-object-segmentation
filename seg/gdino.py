"""Grounding DINO (HuggingFace transformers) as a second open-vocabulary detector.

Used only for categories YOLO-World cannot see in a video (seg/tracker.py probes this before tracking): e.g.
thin dark cables, where YOLO-World's best raw score over 16 sampled frames stayed below 0.02.
Returns the same (box xyxy, score, prompt index) tuples as the YOLO-World detector.
"""
import cv2
import numpy as np

from . import config as C


class GroundingDino:
    def __init__(self, prompts, device, box_thr=None, text_thr=None):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(C.GDINO_MODEL)
        self.m = AutoModelForZeroShotObjectDetection.from_pretrained(C.GDINO_MODEL).to(device).eval()
        self.prompts = [p.lower().strip() for p in prompts]
        self.text = " . ".join(self.prompts) + " ."
        self.device = device
        self.box_thr = C.GDINO_BOX_THR if box_thr is None else box_thr
        self.text_thr = C.GDINO_TEXT_THR if text_thr is None else text_thr

    def match(self, label):
        """phrase GDINO returns (may be part of a prompt, or two prompts run together) -> prompt index or None"""
        lab = str(label).lower().strip()
        if lab in self.prompts:
            return self.prompts.index(lab)
        w = set(lab.split())
        if not w:
            return None
        k = max(range(len(self.prompts)), key=lambda i: (len(w & set(self.prompts[i].split())),
                                                         -len(self.prompts[i].split())))
        return k if w & set(self.prompts[k].split()) else None

    def __call__(self, bgr):
        import torch
        from PIL import Image
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        inp = self.proc(images=img, text=self.text, return_tensors="pt").to(self.device)
        cuda = str(self.device).startswith("cuda")
        with torch.no_grad(), torch.autocast("cuda", enabled=False) if cuda else torch.no_grad():
            out = self.m(**inp)
        kw = dict(text_threshold=self.text_thr, target_sizes=[img.size[::-1]])
        try:                                                    # transformers >= 4.51
            res = self.proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=self.box_thr, **kw)
        except TypeError:                                       # older releases
            res = self.proc.post_process_grounded_object_detection(out, inp.input_ids, box_threshold=self.box_thr,
                                                                   **kw)
        res = res[0]
        labels = res["text_labels"] if "text_labels" in res else res.get("labels")
        dets = []
        for box, sc, lab in zip(res["boxes"].float().cpu().numpy(), res["scores"].float().cpu().numpy(), labels):
            k = self.match(lab)
            if k is not None:
                dets.append((np.asarray(box, float), float(sc), k))
        return dets
