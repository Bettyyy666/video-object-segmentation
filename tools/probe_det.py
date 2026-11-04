"""Probe any video's vocabulary on evenly spaced frames, without SAM 2.

Reports raw per-prompt maxima BEFORE winning-class selection, NMS and max_det,
plus maxima of the boxes actually emitted by YOLO. A missing output is not a
zero raw score. Extra prompts are diagnostic only; they never edit overrides.

    python tools/probe_det.py data/09_coil_and_stow_four_cables.mp4 \
        --extra 'wire,cord,cable tie,velcro cable tie' --json results/probe_09.json
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seg import config as C, vocab  # noqa: E402
from seg.tracker import resolve_device  # noqa: E402


def raw_prompt_maxima(preds, n_prompts):
    """YOLO-World decoded output: [batch, xywh + class scores, anchors]. Fail explicitly on API drift."""
    raw = preds[0] if isinstance(preds, (tuple, list)) else preds
    if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[1] != 4 + n_prompts:
        raise ValueError(f"unsupported YOLO-World prediction shape {tuple(raw.shape)} for {n_prompts} prompts")
    return raw[0, 4:, :].amax(dim=1).detach().float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--task-sheets", default="data/task_sheets.json")
    ap.add_argument("--overrides", default="configs/overrides.json")
    ap.add_argument("--extra", default="", help="comma-separated extra diagnostic prompts")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--json", dest="out_json", help="save settings, per-frame scores and summary")
    ap.add_argument("--gdino", action="store_true", help="also score every prompt with Grounding DINO")
    a = ap.parse_args()
    if a.n <= 0 or a.imgsz <= 0:
        ap.error("--n and --imgsz must be positive")
    if not os.path.isfile(a.video):
        ap.error(f"video not found: {a.video}")
    vid = os.path.splitext(os.path.basename(a.video))[0]
    prompts = vocab.build(a.task_sheets, vid, a.overrides)[0]
    prompts = list(dict.fromkeys(prompts + [p.strip() for p in a.extra.split(",") if p.strip()]))
    import torch
    from ultralytics import YOLOWorld
    from ultralytics.models.yolo.detect import DetectionPredictor

    class ProbePredictor(DetectionPredictor):
        def postprocess(self, preds, img, orig_imgs, **kwargs):
            self.raw_maxima = raw_prompt_maxima(preds, len(prompts))
            return super().postprocess(preds, img, orig_imgs, **kwargs)

    device = resolve_device(torch, a.device)
    cap = cv2.VideoCapture(a.video)
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or T <= 0 or fps <= 0:
        cap.release()
        ap.error("cannot read video frame count / frame rate")
    indices = np.unique(np.linspace(0, T - 1, min(T, a.n)).astype(int))
    m = YOLOWorld(C.DET_MODEL)
    m.set_classes(prompts)
    gd = None
    if a.gdino:
        from seg.gdino import GroundingDino
        gd = GroundingDino(prompts, device, box_thr=0.01, text_thr=0.01)
    records, failed = [], []
    try:
        for f in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, img = cap.read()
            if not ok:
                failed.append(int(f))
                continue
            r = m.predict(img, predictor=ProbePredictor, conf=0.001, iou=0.5, max_det=300,
                          imgsz=a.imgsz, agnostic_nms=False, verbose=False, device=device)[0]
            emitted = np.zeros(len(prompts))
            if r.boxes is not None:
                for confidence, cls in zip(r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
                    emitted[cls] = max(emitted[cls], float(confidence))
            rec = {"frame": int(f), "t": float(f / fps),
                   "raw": dict(zip(prompts, map(float, m.predictor.raw_maxima))),
                   "emitted": dict(zip(prompts, map(float, emitted)))}
            if gd is not None:
                g = np.zeros(len(prompts))
                for _, s, k in gd(img):
                    g[k] = max(g[k], s)
                rec["gdino"] = dict(zip(prompts, map(float, g)))
            records.append(rec)
            print(f"  sampled {len(records)}/{len(indices)} (t={f / fps:.2f}s)", flush=True)
    finally:
        cap.release()
    if not records:
        raise SystemExit("No sampled frames decoded; no diagnostic result.")
    print(f"\n{vid}: {len(records)}/{len(indices)} decoded frames; device={device}, imgsz={a.imgsz}")
    print("raw = before class selection/NMS; emitted = after postprocessing at conf=0.001")
    print(f"{'prompt':34s} {'raw max':>8} {'median':>8} {'>=.05':>6} {'>=.10':>6} {'>=.25':>6} {'emitted':>8}")
    summary = {}
    for p in prompts:
        raw = np.array([r["raw"][p] for r in records])
        emitted = np.array([r["emitted"][p] for r in records])
        peak = int(raw.argmax())
        summary[p] = {"raw_max": float(raw.max()), "raw_median": float(np.median(raw)),
                      "frames_ge_005": int((raw >= .05).sum()), "frames_ge_010": int((raw >= .10).sum()),
                      "frames_ge_025": int((raw >= .25).sum()), "emitted_max": float(emitted.max()),
                      "peak_frame": records[peak]["frame"], "peak_t": records[peak]["t"]}
        line = (f"{p:34s} {raw.max():8.4f} {np.median(raw):8.4f} {(raw >= .05).sum():6d} "
                f"{(raw >= .10).sum():6d} {(raw >= .25).sum():6d} {emitted.max():8.4f}")
        if gd is not None:
            g = np.array([r["gdino"][p] for r in records])
            summary[p].update({"gdino_max": float(g.max()), "gdino_frames_ge_020": int((g >= .20).sum()),
                               "gdino_frames_ge_030": int((g >= .30).sum())})
            line += f"   gdino max {g.max():.3f}  >=.20 {(g >= .20).sum():3d}  >=.30 {(g >= .30).sum():3d}"
        print(line)
    if failed:
        print(f"WARNING: failed to decode frames {failed}")
    print("Sampled-frame diagnostics only: scores do not establish localization quality or full-video recall.")
    if a.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
        with open(a.out_json, "w") as fh:
            json.dump({"video": os.path.abspath(a.video), "model": C.DET_MODEL, "device": device,
                       "imgsz": a.imgsz, "gdino": C.GDINO_MODEL if gd is not None else None,
                       "postprocess_conf": 0.001, "postprocess_iou": 0.5, "max_det": 300,
                       "requested_samples": a.n, "decoded_samples": len(records), "failed_frames": failed,
                       "prompts": prompts, "summary": summary, "frames": records}, fh, indent=2)
        print(f"Saved {a.out_json}")


if __name__ == "__main__":
    main()
