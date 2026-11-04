"""Second opinion on every tracked object with CLIP (the same text-image model YOLO-World uses internally).

For each non-robot track, up to VERIFY_SAMPLES crops of its mask (background greyed out) are scored against
"a photo of a <prompt>" for every object prompt of the video plus a few non-object texts (table, wall, shadow,
robot arm ...). Stored in work/verify.json:
    {track_id: {"object": mean probability of being one of the task objects, "cats": {category: prob}, "n": crops}}
finalize.py drops tracks CLIP calls background and uses the category probabilities next to the detector votes.
"""
import json
import os

import cv2
import numpy as np

from . import config as C

NEGATIVE = ["a wooden table top", "a white wall", "a shadow", "a robot arm", "a robot gripper",
            "an empty background"]


def masked_square_crop(img, mask):
    """Keep the entire masked object, including thin tips, before CLIP's resize/centre crop."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("cannot crop an empty mask")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    py, px = int(0.15 * (y1 - y0)) + 4, int(0.15 * (x1 - x0)) + 4
    y0, y1 = max(0, y0 - py), min(img.shape[0], y1 + py)
    x0, x1 = max(0, x0 - px), min(img.shape[1], x1 + px)
    crop = img[y0:y1, x0:x1].copy()
    crop[~mask[y0:y1, x0:x1]] = 127
    h, w = crop.shape[:2]
    s = max(h, w)
    return cv2.copyMakeBorder(crop, (s - h) // 2, s - h - (s - h) // 2,
                             (s - w) // 2, s - w - (s - w) // 2,
                             cv2.BORDER_CONSTANT, value=(127, 127, 127))


def _samples(work, meta):
    T = meta["num_frames"]
    step = max(1, T // 400)
    seen = {}                                        # id -> [(frame, area)]
    for f in range(0, T, step):
        lab = cv2.imread(os.path.join(work, "labels", f"{f:06d}.png"), cv2.IMREAD_UNCHANGED)
        if lab is None:
            continue
        ids, cnt = np.unique(lab, return_counts=True)
        for i, n in zip(ids, cnt):
            if i and n >= 400:
                seen.setdefault(int(i), []).append((f, int(n)))
    pick = {}
    for i, fa in seen.items():
        med = np.median([a for _, a in fa])
        good = [f for f, a in fa if a >= med]        # the larger half: less occluded views
        k = min(C.VERIFY_SAMPLES, len(good))
        pick[i] = [good[j] for j in np.linspace(0, len(good) - 1, k).astype(int)] if k else []
    return pick


def run(work, prompts, cats, device="cpu"):
    try:
        import clip
        import torch
        from PIL import Image
    except ImportError as e:
        print(f"  verify: CLIP not available ({e}); skipped")
        return None
    meta = json.load(open(os.path.join(work, "tracks.json")))
    robot = {t["id"] for t in meta["tracks"] if max(t["votes"], key=t["votes"].get) == "robot"}
    dev = "cuda" if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    model, prep = clip.load("ViT-B/32", device=dev, download_root="checkpoints/clip")
    obj_texts = [(p, c) for p, c in zip(prompts, cats) if c != "robot"]
    texts = [f"a photo of a {p}" for p, _ in obj_texts] + NEGATIVE
    with torch.no_grad():
        tf = model.encode_text(clip.tokenize(texts).to(dev)).float()
        tf /= tf.norm(dim=-1, keepdim=True)
    pick = {i: fs for i, fs in _samples(work, meta).items() if i not in robot}
    by_frame = {}
    for i, fs in pick.items():
        for f in fs:
            by_frame.setdefault(f, []).append(i)
    files = sorted(os.listdir(os.path.join(work, "frames")))
    probs = {i: [] for i in pick}
    for f, ids in sorted(by_frame.items()):
        img = cv2.imread(os.path.join(work, "frames", files[f]))
        lab = cv2.imread(os.path.join(work, "labels", f"{f:06d}.png"), cv2.IMREAD_UNCHANGED)
        crops = []
        for i in ids:
            c = masked_square_crop(img, lab == i)
            crops.append(prep(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))))
        with torch.no_grad():
            imf = model.encode_image(torch.stack(crops).to(dev)).float()
            imf /= imf.norm(dim=-1, keepdim=True)
            p = (100.0 * imf @ tf.T).softmax(-1).cpu().numpy()
        for i, row in zip(ids, p):
            probs[i].append(row)
    out = {}
    for i, rows in probs.items():
        if not rows:
            continue
        r = np.mean(rows, 0)
        per_cat = {}
        for (_, c), v in zip(obj_texts, r[:len(obj_texts)]):
            per_cat[c] = per_cat.get(c, 0.0) + float(v)
        out[str(i)] = {"object": round(float(r[:len(obj_texts)].sum()), 3),
                       "cats": {c: round(v, 3) for c, v in per_cat.items()}, "n": len(rows)}
    json.dump(out, open(os.path.join(work, "verify.json"), "w"), indent=1)
    print(f"  verify: CLIP checked {len(out)} tracks")
    return out
