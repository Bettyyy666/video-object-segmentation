"""Self-check without ground truth.

    python tools/check_output.py results/12_fold_four_towels.json --video data/12_fold_four_towels.mp4 \
        --overlay results/12_fold_four_towels_overlay.mp4 --dump results/check/12_fold_four_towels

1. format: every frame in order, sizes, ids declared, RLE decodes, no pixel in two masks, overlay matches input
2. per-object diagnostics:
   - coverage: frames with a mask, number of disappear/reappear gaps
   - temporal IoU between consecutive frames (at 50 fps an object barely moves, so a drop means a leak,
     a lost part or an id switch)
   - area jumps (x2.5 between consecutive frames)
   - fragmentation (connected components)
3. --dump: writes the overlay frames at every flagged moment so you can look at them
   (--sample N adds N evenly spaced frames, named sample_*.jpg, for an unbiased manual spot check).
"""
import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np
from pycocotools import mask as mask_util


def vinfo(path):
    cap = cv2.VideoCapture(path)
    n = 0
    while cap.grab():
        n += 1
    out = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
           round(cap.get(cv2.CAP_PROP_FPS), 3), n)
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--video")
    ap.add_argument("--overlay")
    ap.add_argument("--dump")
    ap.add_argument("--sample", type=int, default=0, help="also dump N evenly spaced frames for manual spot checks")
    ap.add_argument("--iou-flag", type=float, default=0.5)
    a = ap.parse_args()
    d = json.load(open(a.json))
    W, H, T, fps = d["width"], d["height"], d["num_frames"], d["fps"]
    ids = {o["id"]: o for o in d["objects"]}
    errors, flags = [], []

    # ---- format
    if [f["frame"] for f in d["frames"]] != list(range(T)):
        errors.append("frames are not exactly 0..num_frames-1 in order")
    if a.video:
        vw, vh, vf, vn = vinfo(a.video)
        if (vw, vh, vn) != (W, H, T) or abs(vf - fps) > 1e-3:
            errors.append(f"JSON says {W}x{H} {fps}fps {T} frames, video is {vw}x{vh} {vf}fps {vn} frames")
    if a.overlay:
        ow, oh, of, on = vinfo(a.overlay)
        if (ow, oh, on) != (W, H, T) or abs(of - fps) > 1e-3:
            errors.append(f"overlay is {ow}x{oh} {of}fps {on} frames, expected {W}x{H} {fps}fps {T}")

    prev, area, vis, comps = {}, defaultdict(list), defaultdict(list), defaultdict(int)
    iou_hist = defaultdict(list)
    for fr in d["frames"]:
        f = fr["frame"]
        acc = np.zeros((H, W), np.uint8)
        cur = {}
        for m in fr["masks"]:
            if m["id"] not in ids:
                errors.append(f"frame {f}: undeclared id {m['id']}")
                continue
            if m["rle"]["size"] != [H, W]:
                errors.append(f"frame {f}: id {m['id']} wrong RLE size")
            bm = mask_util.decode({"size": m["rle"]["size"], "counts": m["rle"]["counts"].encode()}).astype(bool)
            acc += bm
            cur[m["id"]] = bm
            i = m["id"]
            area[i].append(int(bm.sum()))
            vis[i].append(f)
            n, _ = cv2.connectedComponents(bm.astype(np.uint8))
            if n - 1 > 3:
                comps[i] += 1
            if i in prev:
                p = prev[i]
                inter, uni = (p & bm).sum(), (p | bm).sum()
                iou = inter / uni if uni else 1.0
                iou_hist[i].append(iou)
                pa, ca = p.sum(), bm.sum()
                if iou < a.iou_flag:
                    flags.append((f, i, f"temporal IoU {iou:.2f}"))
                elif max(pa, ca) > 2.5 * max(1, min(pa, ca)):
                    flags.append((f, i, f"area {pa} -> {ca}"))
        if (acc > 1).any():
            errors.append(f"frame {f}: {(acc > 1).sum()} pixels in more than one mask")
        prev = cur

    print(f"{os.path.basename(a.json)}: {T} frames, {len(ids)} objects")
    print(f"{'id':>3} {'category':18s} {'frames':>6} {'first':>7} {'last':>7} {'gaps':>4} {'medIoU':>6} "
          f"{'IoU<.5':>6} {'frag':>5}")
    for i, o in ids.items():
        v = vis[i]
        gaps = sum(1 for x, y in zip(v, v[1:]) if y - x > 1)
        ih = iou_hist[i]
        print(f"{i:3d} {o['category'][:18]:18s} {len(v):6d} {v[0] / fps if v else -1:7.2f} "
              f"{v[-1] / fps if v else -1:7.2f} {gaps:4d} {np.median(ih) if ih else 0:6.2f} "
              f"{sum(x < a.iou_flag for x in ih):6d} {comps[i]:5d}")
        for x, y in zip(v, v[1:]):
            if y - x > 1:
                flags.append((y, i, f"reappears after {(y - x) / fps:.2f}s (check id)"))
    print("\nERRORS:" if errors else "\nformat OK")
    for e in errors[:50]:
        print("  ", e)
    flags.sort()
    print(f"\n{len(flags)} flagged moments, grouped into time ranges per id (gaps < 0.5 s merged), longest first:")
    ranges = []
    for i in ids:
        fl = sorted({f for f, j, _ in flags if j == i})
        for f in fl:
            if ranges and ranges[-1][0] == i and f - ranges[-1][2] <= fps / 2:
                ranges[-1][2], ranges[-1][3] = f, ranges[-1][3] + 1
            else:
                ranges.append([i, f, f, 1])
    for i, r0, r1, n in sorted(ranges, key=lambda r: -r[3])[:25]:
        print(f"   id {i:3d} {ids[i]['category']:15s} {r0 / fps:7.2f}-{r1 / fps:7.2f}s  {n:4d} flagged frames")

    if a.dump and a.overlay:
        os.makedirs(a.dump, exist_ok=True)
        flagged = {f for f, _, _ in flags}
        sampled = set(np.linspace(0, T - 1, a.sample).astype(int).tolist()) if a.sample else set()
        want = sorted(flagged | sampled)
        cap = cv2.VideoCapture(a.overlay)
        f, k = 0, 0
        while k < len(want) and cap.grab():
            if f == want[k]:
                _, img = cap.retrieve()
                if f in flagged:
                    cv2.imwrite(os.path.join(a.dump, f"flag_{f:06d}.jpg"), img)
                if f in sampled:
                    cv2.imwrite(os.path.join(a.dump, f"sample_{f:06d}.jpg"), img)
                k += 1
            f += 1
        print(f"dumped {len(flagged)} flagged + {len(sampled)} sampled overlay frames to {a.dump}")


if __name__ == "__main__":
    main()
