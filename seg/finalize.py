"""Pass 2 (CPU only): label maps -> submission JSON (COCO RLE per object per frame) + overlay video.

- drops ids visible for less than MIN_TRACK_S (spurious detections)
- category = detection votes over the whole track; optional size split (dinner spoon vs teaspoon)
- unstable tracks (mask jumps around frame to frame and is in pieces, e.g. slivers left over from a duplicate id)
  are merged into the touching same-category object
- renumbers ids 1..K in order of first appearance (robot arms last)
"""
import colorsys
import json
import os
import shutil
import subprocess

import cv2
import numpy as np
from pycocotools import mask as mask_util

from . import config as C


def _size_split(tracks, split):
    for cat, names in split.items():
        group = [t for t in tracks if t["category"] == cat]
        if not group:
            continue
        areas = np.array([t["median_area"] for t in group])
        nl, ns = names.get("n_large"), names.get("n_small")
        if nl and ns:                           # the task sheet says how many of each: the nl biggest are large
            k = int(round(len(group) * nl / (nl + ns)))
            order = np.argsort(-areas)
            for r, j in enumerate(order):
                group[j]["category"] = names["large"] if r < k else names["small"]
            continue
        if len(group) == 1:
            thr = np.inf
        else:                                   # best 2-way split of sorted areas (Otsu in 1-D)
            a = np.sort(areas)
            best, thr = -1, a[-1]
            for k in range(1, len(a)):
                lo, hi = a[:k], a[k:]
                v = len(lo) * len(hi) * (lo.mean() - hi.mean()) ** 2
                if v > best:
                    best, thr = v, (a[k - 1] + a[k]) / 2
        for t, ar in zip(group, areas):
            t["category"] = names["large"] if ar >= thr else names["small"]


def _flag(cat, words):
    return any(w in cat.lower() for w in words)


def colour_name(bgr):
    h, s, v = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0].astype(int)
    if v < 60:
        return "black"
    if s < 45:
        return "white" if v > 170 else "grey"
    for top, name in [(8, "red"), (20, "orange"), (33, "yellow"), (85, "green"), (100, "cyan"), (130, "blue"),
                      (150, "purple"), (170, "pink"), (181, "red")]:
        if h < top:
            return name


def object_colours(work, T, ids, every=25):
    """mean colour of each object's pixels, median over frames sampled every `every` frames"""
    files = sorted(os.listdir(os.path.join(work, "frames")))
    acc = {i: [] for i in ids}
    for f in range(0, T, every):
        lab = cv2.imread(os.path.join(work, "labels", f"{f:06d}.png"), cv2.IMREAD_UNCHANGED)
        img = cv2.imread(os.path.join(work, "frames", files[f]))
        if lab is None or img is None:
            continue
        lab, img = lab[::2, ::2], img[::2, ::2]
        for i in ids:
            m = lab == i
            if m.sum() >= 50:
                acc[i].append(img[m].mean(0))
    return {i: colour_name(np.median(a, 0)) for i, a in acc.items() if a}


def _stability(work, T, ids, robots=(), size=None):
    """per id (2x downsampled): median IoU of its mask between consecutive frames, share of frames with >3 pieces,
    how far it travels (largest distance of its smoothed centroid from where it started, full-res px), and whether
    a robot arm ever touches it"""
    prev, ious, frag, n = {}, {i: [] for i in ids}, dict.fromkeys(ids, 0), dict.fromkeys(ids, 0)
    cents, touched = {i: [] for i in ids}, set()
    rset = list(robots)
    k3 = np.ones((2 * C.TOUCH_PX // 2 + 1,) * 2, np.uint8)
    for f in range(T):
        lab = cv2.imread(os.path.join(work, "labels", f"{f:06d}.png"), cv2.IMREAD_UNCHANGED)
        cur = {}
        if lab is not None:
            lab = lab[::2, ::2]
            for i in np.unique(lab):
                i = int(i)
                if i not in ious:
                    continue
                m = lab == i
                cur[i] = m
                n[i] += 1
                if cv2.connectedComponents(m.astype(np.uint8))[0] - 1 > 3:
                    frag[i] += 1
                if i in prev:
                    u = (prev[i] | m).sum()
                    ious[i].append((prev[i] & m).sum() / u if u else 1.0)
                if f % C.MOTION_EVERY == 0:
                    ys, xs = np.nonzero(m)
                    cents[i].append((2.0 * xs.mean(), 2.0 * ys.mean()))
            if rset and f % C.MOTION_EVERY == 0:
                rob = np.isin(lab, rset)
                if rob.any():
                    near = cv2.dilate(rob.astype(np.uint8), k3).astype(bool)
                    for i in set(cur) - touched:
                        if (cur[i] & near).any():
                            touched.add(i)
        prev = cur
    out = {}
    for i in ids:
        c = np.array(cents[i])
        if len(c) >= 3:                                     # median over 5 samples: mask flicker is not motion
            k = min(5, len(c))
            sm = np.array([np.median(c[max(0, j - k // 2):j + k // 2 + 1], 0) for j in range(len(c))])
            travel = float(np.linalg.norm(sm - sm[0], axis=1).max())
        else:
            travel = 0.0
        inside = 1.0
        if size and len(c):
            W, H = size
            mx, my = C.ROI_MARGIN * W, C.ROI_MARGIN * H
            inside = float(np.mean((c[:, 0] >= mx) & (c[:, 0] <= W - mx) & (c[:, 1] >= my) & (c[:, 1] <= H - my)))
        out[i] = (float(np.median(ious[i])) if ious[i] else 1.0, frag[i] / max(1, n[i]), travel, i in touched,
                  inside)
    return out


def unstable_tracks(work, meta):
    cand = [t["id"] for t in meta["tracks"] if max(t["votes"], key=t["votes"].get) != "robot"
            and t["n_vis"] >= C.MIN_TRACK_S * meta["fps"]]
    robots = [t["id"] for t in meta["tracks"] if max(t["votes"], key=t["votes"].get) == "robot"]
    out, stats = {}, _stability(work, meta["num_frames"], cand, robots, (meta["width"], meta["height"]))
    for i, (med, fr, *_) in stats.items():
        if med < C.UNSTABLE_IOU and fr > C.UNSTABLE_FRAG:
            out[i] = (med, fr)
    return out, stats


def _absorb(label, unstable, cat_of, remap):
    """pixels of an unstable id -> the same-category kept object touching it most (else background)"""
    for i in unstable:
        m = label == i
        if not m.any():
            continue
        ring = cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~m
        ok = [j for j in remap if cat_of.get(j) == cat_of[i]]
        nb = label[ring]
        nb = nb[np.isin(nb, ok)]
        label[m] = np.bincount(nb).argmax() if nb.size else 0
    return label


def _cat_scores(t, v):
    """category -> score in [0, 1]: detector votes and CLIP probabilities, each normalised, mixed"""
    votes = {c: x for c, x in t["votes"].items() if x > 0 and c != "robot"}
    vs = sum(votes.values()) or 1.0
    if not v or not v.get("cats") or sum(v["cats"].values()) <= 0:
        return {c: x / vs for c, x in votes.items()}
    cs = sum(v["cats"].values())
    return {c: (1 - C.VERIFY_WEIGHT) * votes.get(c, 0) / vs + C.VERIFY_WEIGHT * v["cats"].get(c, 0) / cs
            for c in set(votes) | set(v["cats"])}


def assign_categories(tracks, counts, quality):
    """Give every track a category, at most counts[c] tracks for a counted category (task sheet), greedily by
    utility = track quality x category score. A track that finds no category with score >= DUP_SCORE left is a
    duplicate. Without counts this is just the best-scoring category of each track."""
    left = {c: n for c, n in counts.items()}
    cand = sorted(((round(quality[t["id"]] * sc, 6), t.get("median_area", 0.0), -t["id"], c)
                   for t in tracks for c, sc in t["scores"].items() if sc >= C.DUP_SCORE), reverse=True)
    out = {}
    for u, _, i, c in cand:                    # ties: the bigger object (a duplicate is usually a sliver)
        i = -i
        if i in out or (c in left and left[c] <= 0):
            continue
        out[i] = c
        if c in left:
            left[c] -= 1
    return out


def plan(meta, size_split=None, deformable=(), articulated=(), skip=(), colours=None, verify=None, counts=None,
         stab=None, manipulated=(), log=print):
    """-> objects, remap {track id: output id}, duplicates {track id: category} (over the task sheet count, to be
    merged into a neighbour), category {track id: category before the size split}"""
    fps = meta["fps"]
    verify, counts = verify or {}, counts or {}
    keep = []
    for t in meta["tracks"]:
        if t["n_vis"] < C.MIN_TRACK_S * fps or t["id"] in skip:
            continue
        t = dict(t)
        big = t["median_area"] > C.MAX_OBJECT_FRAC * meta.get("width", 1) * meta.get("height", 1) \
            if "width" in meta else False
        if big and max(t["votes"], key=t["votes"].get) != "robot":
            log(f"  track {t['id']}: median mask {t['median_area'] / (meta['width'] * meta['height']):.0%} of the "
                f"frame -> table / background, dropped")
            continue
        v = verify.get(str(t["id"]))
        t["clip"] = v["object"] if v else None
        st = (stab or {}).get(t["id"])
        sc = _cat_scores(t, v)
        task_cat = bool(sc) and max(sc, key=sc.get) not in C.GENERIC_CONTAINERS   # extras: CLIP knows containers
        handled = task_cat and st is not None and len(st) >= 4 and st[2] >= C.MIN_TRAVEL_PX and st[3]
        if v and v["n"] >= 3 and v["object"] < C.VERIFY_MIN_OBJECT and max(t["votes"], key=t["votes"].get) != "robot":
            if handled:          # an arm touched it and it moved: a task object even if CLIP cannot name it (thin
                log(f"  track {t['id']}: CLIP unsure (p={v['object']:.2f}) but an arm moved it -> kept")  # dark cable)
            else:
                log(f"  track {t['id']}: CLIP says not a task object (p={v['object']:.2f}) -> dropped")
                continue
        robot = max(t["votes"], key=t["votes"].get) == "robot"
        t["category"] = "robot" if robot else None
        t["scores"] = {} if robot else _cat_scores(t, v)
        keep.append(t)
    objs = [t for t in keep if t["category"] is None]
    T = max(1, meta["num_frames"])
    quality = {t["id"]: (t["n_vis"] / T) * (stab or {}).get(t["id"], (1.0, 0))[0]
               * (t["clip"] if t["clip"] is not None else 1.0) for t in objs}
    got = assign_categories(objs, counts, quality)
    duplicates, dropped = {}, []
    for t in objs:
        top = max(t["scores"], key=t["scores"].get) if t["scores"] else None
        if t["id"] in got:
            t["category"] = got[t["id"]]
            if t["category"] != top:
                log(f"  track {t['id']}: best guess {top}, but the task sheet's {top} count is used up -> "
                    f"{t['category']}")
        elif t["clip"] is not None and t["clip"] < C.DUP_MERGE_CLIP:
            dropped.append(t["id"])
            log(f"  track {t['id']} ({top}): more ids than the task sheet's count and CLIP doubts it "
                f"(p={t['clip']:.2f}) -> dropped")
        elif not any(o["id"] in got and got[o["id"]] == top
                     and o["median_area"] < C.CONTAINER_RATIO * max(1.0, t["median_area"]) for o in objs):
            # the only objects of its category are much bigger (a strip inside the insert called 'insert'): its
            # pixels would make the container grow over whatever lies in it, so drop instead of merging
            dropped.append(t["id"])
            log(f"  track {t['id']} ({top}): more ids than the task sheet's count, much smaller than every kept "
                f"{top} -> dropped (not merged)")
        else:
            duplicates[t["id"]] = top
            log(f"  track {t['id']} ({top}): more ids than the task sheet's count -> merged into its neighbour")
    keep = [t for t in keep if t["id"] not in duplicates and t["id"] not in dropped]
    # task relevance as the task defines it: the robot touches or moves it (or puts things in / on it). A track in
    # a manipulated category (task sheet object column) that never moves AND is never touched by an arm is a
    # false detection that got a task category (a dark patch in a corner called 'cable'); containers are exempt
    still = []
    for t in keep:
        st = (stab or {}).get(t["id"])
        if st is not None and len(st) >= 5 and st[4] < 0.5:
            still.append(t["id"])
            log(f"  track {t['id']} ({t['category']}): centre in the frame border {1 - st[4]:.0%} of the time "
                f"-> outside the detection range, dropped")
            continue
        if (st is not None and len(st) >= 4 and t["category"] in manipulated
                and st[2] < C.MIN_TRAVEL_PX and not st[3]):
            still.append(t["id"])
            log(f"  track {t['id']} ({t['category']}): never moves (max {st[2]:.0f} px) and no arm ever touches it "
                f"-> not a task object, dropped")
    keep = [t for t in keep if t["id"] not in still]
    category = {t["id"]: t["category"] for t in keep}
    _size_split(keep, size_split or {})
    keep.sort(key=lambda t: (t["category"] == "robot", t["first"]))
    objects, remap = [], {}
    for new_id, t in enumerate(keep, 1):
        remap[t["id"]] = new_id
        reids = [e["frame"] / fps for e in t["events"] if e["type"] == "reid"]
        col = (colours or {}).get(t["id"])
        note = (f"{col} {t['category']}; " if col and t["category"] != "robot" else "") + \
            f"first visible t={t['first'] / fps:.2f}s"
        if reids:
            note += "; re-identified after occlusion at " + ", ".join(f"{x:.2f}s" for x in reids)
        if t.get("clip") is not None:
            note += f"; CLIP object score {t['clip']:.2f}"
        if t["category"] == "robot":
            note = "robot arm (helper class, not a task object); " + note
        cat = t["category"]
        objects.append({"id": new_id, "category": cat,
                        "deformable": cat in deformable or _flag(cat, C.DEFORMABLE),
                        "articulated": cat in articulated or (cat not in deformable and _flag(cat, C.ARTICULATED)),
                        "note": note})
    return objects, remap, duplicates, category


def color(i):
    r, g, b = colorsys.hsv_to_rgb((i * 0.61803398875) % 1.0, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


class _Writer:
    """Exact frame count and fps: raw frames piped to ffmpeg (H.264); falls back to OpenCV mp4v."""

    def __init__(self, path, W, H, fps):
        self.p = None
        from .ffmpeg import ffmpeg_bin
        exe = ffmpeg_bin()
        if exe:
            self.p = subprocess.Popen(
                [exe, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                 "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                 "-pix_fmt", "yuv420p", "-r", str(fps), path], stdin=subprocess.PIPE)
        else:
            self.w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    def write(self, img):
        if self.p:
            self.p.stdin.write(np.ascontiguousarray(img).tobytes())
        else:
            self.w.write(img)

    def close(self):
        if self.p:
            self.p.stdin.close()
            self.p.wait()
        else:
            self.w.release()


def draw(img, label, objects_by_id, remap):
    out = img.copy()
    over = img.copy()
    present = [i for i in np.unique(label) if i and int(i) in remap]
    for i in present:
        nid = remap[int(i)]
        m = label == i
        over[m] = color(nid)
    out = cv2.addWeighted(over, 0.5, out, 0.5, 0)
    for i in present:
        nid = remap[int(i)]
        m = (label == i).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, color(nid), 2)
        big = max(cnts, key=cv2.contourArea)
        mom = cv2.moments(big)
        if mom["m00"] > 0:
            cx, cy = int(mom["m10"] / mom["m00"]), int(mom["m01"] / mom["m00"])
        else:
            x, y, w, h = cv2.boundingRect(big)
            cx, cy = x + w // 2, y + h // 2
        txt = f"{nid} {objects_by_id[nid]['category']}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        x = int(np.clip(cx - tw // 2, 0, label.shape[1] - tw - 4))
        ty = int(np.clip(cy + th // 2, th + 4, label.shape[0] - 4))
        cv2.rectangle(out, (x, ty - th - 4), (x + tw + 4, ty + 2), (0, 0, 0), -1)
        cv2.putText(out, txt, (x + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color(nid), 2)
    return out


def run(work, video_name, out_json, out_overlay, size_split=None, deformable=(), articulated=(), counts=None,
        manipulated=()):
    meta = json.load(open(os.path.join(work, "tracks.json")))
    unstable, stab = unstable_tracks(work, meta)
    cat_of = {t["id"]: max(t["votes"], key=t["votes"].get) for t in meta["tracks"]}
    for i, (med, fr) in unstable.items():
        print(f"  track {i} ({cat_of[i]}): unstable (median frame-to-frame IoU {med:.2f}, in pieces in "
              f"{fr:.0%} of its frames) -> pixels merged into the touching {cat_of[i]}")
    colours = object_colours(work, meta["num_frames"], [t["id"] for t in meta["tracks"]]) if C.NOTE_COLOUR else {}
    vp = os.path.join(work, "verify.json")
    verify = json.load(open(vp)) if os.path.exists(vp) else None
    objects, remap, dups, category = plan(meta, size_split, deformable, articulated, skip=set(unstable), colours=colours,
                                verify=verify, counts=counts, stab=stab, manipulated=set(manipulated))
    unstable = {**unstable, **{i: (None, None) for i in dups}}        # both kinds merge into their neighbour
    cat_of.update(category)
    cat_of.update(dups)
    json.dump({"final_id_of_track": {str(k): v for k, v in remap.items()},
               "merged_unstable_tracks": {str(k): v for k, v in unstable.items()},
               "expected_counts": counts or {}},
              open(os.path.join(work, "final_ids.json"), "w"), indent=1)
    by_id = {o["id"]: o for o in objects}
    scores = {}
    with open(os.path.join(work, "scores.jsonl")) as fh:
        for line in fh:
            r = json.loads(line)
            scores.setdefault(r["f"], {}).update({int(k): v for k, v in r["s"].items()})
    W, H, T, fps = meta["width"], meta["height"], meta["num_frames"], meta["fps"]
    fps_out = int(round(fps)) if abs(fps - round(fps)) < 1e-3 else fps
    frames_dir = os.path.join(work, "frames")
    frame_files = sorted(os.listdir(frames_dir))
    writer = _Writer(out_overlay, W, H, fps)
    with open(out_json, "w") as fh:
        head = {"video": video_name, "width": W, "height": H, "fps": fps_out, "num_frames": T, "objects": objects}
        fh.write(json.dumps(head)[:-1] + ', "frames": [\n')
        for f in range(T):
            label = cv2.imread(os.path.join(work, "labels", f"{f:06d}.png"), cv2.IMREAD_UNCHANGED)
            if label is None:
                label = np.zeros((H, W), np.uint16)
            elif unstable:
                label = _absorb(label, unstable, cat_of, remap)
            masks = []
            for i in np.unique(label):
                if i == 0 or int(i) not in remap:
                    continue
                rle = mask_util.encode(np.asfortranarray((label == i).astype(np.uint8)))
                masks.append({"id": remap[int(i)],
                              "rle": {"size": [H, W], "counts": rle["counts"].decode("ascii")},
                              "score": scores.get(f, {}).get(int(i), 0.0)})
            masks.sort(key=lambda m: m["id"])
            fh.write(json.dumps({"frame": f, "t": round(f / fps, 3), "masks": masks}))
            fh.write(",\n" if f < T - 1 else "\n")
            img = cv2.imread(os.path.join(frames_dir, frame_files[f]))
            writer.write(draw(img, label, by_id, remap))
        fh.write("]}\n")
    writer.close()
    return objects
