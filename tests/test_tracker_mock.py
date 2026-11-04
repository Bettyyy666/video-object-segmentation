"""Control-flow test of the tracker with fake models replaying a known synthetic scene (needs torch, no GPU,
no SAM 2 / YOLO weights). Checks: ids survive occlusion (re-id), same-category instances stay apart, a late
object gets a new id, backward fill covers frames before the detector saw an object, one pixel one id.
Uses real torch if installed, otherwise tests/np_torch.py.

    python tests/test_tracker_mock.py
"""
import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

try:
    import torch
except ImportError:                     # no torch installed: numpy stand-in for the ops the tracker uses
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from np_torch import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from seg.ffmpeg import ffmpeg_bin  # noqa: E402
from seg import config as C, finalize  # noqa: E402
from seg.tracker import Tracker  # noqa: E402

T, H, W = 160, 360, 640
PROMPTS = ["towel", "fork", "knife", "robot gripper", "robot arm", "insert", "spoon"]
CATS = ["towel", "fork", "knife", "robot", "robot", "insert", "spoon"]
COLORS = {"towel": (40, 40, 200), "forkB": (200, 120, 40), "forkC": (200, 120, 40), "knife": (40, 200, 40),
          "towel2": (200, 80, 40), "towel3": (90, 40, 220), "towel4": (120, 30, 180), "arm": (230, 230, 230), "towel5": (50, 90, 200),
          "insert": (60, 60, 60), "spoon": (210, 210, 120), "forkD": (200, 120, 40), "knifeB": (40, 200, 40)}
CLS = {"towel": 0, "forkB": 1, "forkC": 1, "knife": 2, "towel2": 0, "towel3": 0, "towel4": 0, "arm": 3, "towel5": 0, "insert": 5, "spoon": 6, "forkD": 1, "knifeB": 2}
CUR = {"f": 0}


def gt(f):
    m = {}
    def box(y0, y1, x0, x1):
        a = np.zeros((H, W), bool)
        a[y0:y1, x0:x1] = True
        return a
    m["towel"] = box(200, 320, 50 + f // 4, 250 + f // 4)              # slowly moving, always visible
    if f < 60 or f >= 110:                                              # hidden 60..109 (e.g. under the cloth)
        dx = 0 if f < 60 else 30
        m["forkB"] = box(40, 60, 300 + dx, 420 + dx)
    m["forkC"] = box(100, 120, 450, 570)                                # identical-looking second fork
    m["towel5"] = box(10, 50, 100, 180)                               # half-hidden: always scored only 0.15
    m["arm"] = box(200, 360, 0, 40)                                     # robot arm entering at the bottom-left
    if f >= 60:
        m["spoon"] = box(30, 45, 500, 600)                              # put INTO the insert at 1.2 s
    m["insert"] = box(0, 90, 470, 640)                                  # cutlery tray (its mask includes the spoon)
    m["forkD"] = box(176, 186, 300, 440)                                # a fork and a knife side by side: SAM's
    m["knifeB"] = box(188, 198, 300, 440)                               # first fork mask holds both
    if f >= 100:
        m["towel2"] = box(250, 330, 400, 480)                           # 2nd towel, appears at 2 s
    if f >= 130:
        m["towel3"] = box(250, 330, 520, 620)                           # 3rd towel: beyond the first-window cap
    if f >= 100:
        m["towel4"] = box(100, 160, 450, 570)                           # partly under fork C (1/3 of its box)
    if f >= 25:
        m["knife"] = box(150, 170, 300, 440)                            # visible from 25, detected only at 100
    return m


class FakeDet:
    def __call__(self, img):
        out = [(np.array([0, 300, 40, 360], float), 0.9, 0)]           # the arm's base, scored as 'towel' (0.9)
        for name, m in gt(CUR["f"]).items():
            if name == "knife" and CUR["f"] < 100:                      # detector misses it for 3 windows
                continue
            ys, xs = np.nonzero(m)
            out.append((np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float),
                        {"arm": 0.5, "towel5": 0.15, "knifeB": 0.7}.get(name, 0.8), CLS[name]))
        return out


class FakeImg:
    def set_image(self, rgb):
        pass

    def predict(self, box, multimask_output=False):
        def iou(m):
            ys, xs = np.nonzero(m)
            b = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
            ix = max(0, min(b[2], box[2]) - max(b[0], box[0])) * max(0, min(b[3], box[3]) - max(b[1], box[1]))
            return ix / ((b[2] - b[0]) * (b[3] - b[1]) + (box[2] - box[0]) * (box[3] - box[1]) - ix)
        g = gt(CUR["f"])
        name = max(g, key=lambda k: iou(g[k]))
        best = g[name] | g["knifeB"] if name == "forkD" and CUR["f"] == 0 else g[name]     # the leak
        return best[None].astype(np.float32), np.array([0.9]), None


class FakeVP:
    """Perfect within-window tracker: follows the gt object the prompt mask overlaps; cannot find objects it
    was not prompted with (like SAM 2)."""

    def init_state(self, video_path, **kw):
        names = sorted(os.listdir(video_path))
        frames = [int(os.path.basename(os.readlink(os.path.join(video_path, n)))[:-4]) for n in names]
        return {"frames": frames, "prompts": {}}

    def add_new_mask(self, state, li, oid, mask):
        g = gt(state["frames"][li])
        name = max(g, key=lambda k: (g[k] & mask).sum())
        # SAM 2 keeps following everything the prompt mask covered (> half of another object: it follows that too)
        extra = [k for k in g if k != name and (g[k] & mask).sum() > 0.5 * g[k].sum()]
        state["prompts"][oid] = (name, extra)

    def propagate_in_video(self, state, start_frame_idx=0, reverse=False):
        order = range(start_frame_idx, -1, -1) if reverse else range(start_frame_idx, len(state["frames"]))
        ids = sorted(state["prompts"])
        for li in order:
            g = gt(state["frames"][li])
            lg = torch.full((len(ids), 1, H, W), -5.0)
            for k, oid in enumerate(ids):
                name, extra = state["prompts"][oid]
                for e in extra:
                    if e in g and e != "spoon":
                        lg[k, 0][torch.from_numpy(g[e])] = 4.0
                if name == "towel3" and state["frames"][li] >= 158 and not reverse:   # ... then loses everything
                    pass
                elif name == "towel3" and state["frames"][li] >= 155 and not reverse:   # SAM 2 jumps onto towel2
                    lg[k, 0][torch.from_numpy(g["towel2"])] = 8.0
                elif name in g:
                    lg[k, 0][torch.from_numpy(g[name])] = 3.0 if name == "insert" else 5.0   # things in it win
            yield li, ids, lg

    def reset_state(self, state):
        pass


def main():
    C.ROI_MARGIN = 0.0      # this small synthetic scene puts objects against the frame edges on purpose
    tmp = tempfile.mkdtemp()
    src = os.path.join(tmp, "src")
    os.makedirs(src)
    for f in range(T):
        img = np.full((H, W, 3), 110, np.uint8)
        for name, m in gt(f).items():
            img[m] = COLORS[name]
        cv2.imwrite(os.path.join(src, f"{f:06d}.png"), img)
    video = os.path.join(tmp, "26_sort_cutlery_drawer_insert.mp4")
    subprocess.run([ffmpeg_bin(), "-loglevel", "error", "-y", "-framerate", "50", "-i", os.path.join(src, "%06d.png"),
                    "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", video], check=True)

    tr = Tracker.__new__(Tracker)
    tr.torch, tr.dev, tr.cats = torch, "cpu", CATS
    tr.det, tr.vp, tr.ip = FakeDet(), FakeVP(), FakeImg()
    tr.tracks, tr.next_id, tr.caps = {}, 1, None
    tr.det_log, tr.weak, tr.glue_seen = [], [], []
    tr.fixed_caps = {"fork": 3, "knife": 2, "towel": 5}          # as parsed from a task sheet (spoon, insert: none)
    orig = tr._seed
    tr._seed = lambda s, img: (CUR.__setitem__("f", s), orig(s, img))[1]

    work = os.path.join(tmp, "work")
    meta = tr.run(video, work)
    print(json.dumps(meta["tracks"], indent=1)[:1500])
    out_json = os.path.join(tmp, "out.json")
    objs = finalize.run(work, os.path.basename(video), out_json, os.path.join(tmp, "out_overlay.mp4"))
    d = json.load(open(out_json))
    cats = sorted(o["category"] for o in objs)
    assert cats == ["fork"] * 3 + ["insert", "knife", "knife", "robot", "spoon"] + ["towel"] * 5, cats

    from pycocotools import mask as mu
    seen = {}
    for fr in d["frames"]:
        for m in fr["masks"]:
            bm = mu.decode({"size": m["rle"]["size"], "counts": m["rle"]["counts"].encode()}).astype(bool)
            g = gt(fr["frame"])
            name = max(g, key=lambda k: (g[k] & bm).sum())
            seen.setdefault(name, {}).setdefault(m["id"], []).append(fr["frame"])
    for name, by_id in seen.items():
        assert len(by_id) == 1, f"{name} got several ids: { {k: (v[0], v[-1]) for k, v in by_id.items()} }"
    t5 = list(seen["towel5"].values())[0]
    assert t5 == list(range(T)), (t5[:3], len(t5))                   # weak towel confirmed at 2nd sighting, backfilled
    fb = list(seen["forkB"].values())[0]
    kn = list(seen["knife"].values())[0]
    assert fb == list(range(60)) + list(range(110, T)), (fb[:3], fb[55:70])
    assert kn == list(range(25, T)), kn[:5]
    t3 = list(seen["towel3"].values())[0]
    assert t3 == list(range(130, 155)), (t3[:3], t3[-3:])            # jump at 155 blocked, towel2 keeps its id                              # deep backfill reached frame 25
    assert list(seen["forkB"])[0] != list(seen["forkC"])[0]
    kb = list(seen["knifeB"].values())[0]
    assert kb == list(range(T)), (kb[:3], len(kb))                   # split off the fork's mask, backfilled to 0
    fd = list(seen["forkD"].values())[0]
    assert fd == list(range(T)), (fd[:3], len(fd))
    sp = list(seen["spoon"].values())[0]
    assert sp == list(range(60, T)), (sp[:3], len(sp))               # spoon in the insert is its own object
    assert len(seen["insert"]) == 1 and len(list(seen["insert"].values())[0]) == T
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "check_output.py"), out_json, "--video", video,
                        "--overlay", os.path.join(tmp, "out_overlay.mp4")], capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-500:])
    assert "format OK" in r.stdout
    print("OK: one id per object through occlusion; object detected 1.5 s late still has masks from its first "
          "visible frame (deep backfill)")


if __name__ == "__main__":
    main()
