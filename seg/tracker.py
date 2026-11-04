"""Pass 1: detect -> SAM 2 video propagation in short windows -> per-frame label maps.

Per window [s, e] (WINDOW frames):
  1. objects still visible at the end of the previous window are re-prompted at frame s with their own mask
     (mask prompts keep the id through deformation: a towel being folded, a cable being coiled);
  2. YOLO-World runs on frame s; a detection not covered by any current mask is an object that just appeared
     or reappeared -> SAM 2 image mask -> re-identified with a lost object (appearance + position + category)
     or given a new id;
  3. (re)appearing objects are also propagated BACKWARDS through the previous window, so they get masks on the
     frames where they were already visible but not yet detected;
  4. all objects are propagated forward s..e; overlapping masks are resolved per pixel by the highest logit
     (one pixel -> at most one object).
Writes work/labels/<frame>.png (uint16 id map), work/scores.jsonl, work/tracks.json.
"""
import json
import os
import shutil
import subprocess
from collections import Counter

import cv2
import numpy as np

from . import config as C


# ---------------------------------------------------------------- helpers
def extract_frames(video, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.isfile(video):
        raise FileNotFoundError(f"video not found: {video}")
    if not os.listdir(out_dir):
        from .ffmpeg import ffmpeg_bin
        exe = ffmpeg_bin()
        if exe is None:
            raise RuntimeError("ffmpeg not found: `pip install imageio-ffmpeg` (or load an ffmpeg module)")
        base = [exe, "-loglevel", "error", "-i", video]
        tail = ["-q:v", "2", "-start_number", "0", os.path.join(out_dir, "%06d.jpg")]
        # keep every decoded frame exactly once (ffmpeg >= 5.1: -fps_mode; older builds only know -vsync)
        r = subprocess.run(base + ["-fps_mode", "passthrough"] + tail, capture_output=True, text=True)
        if r.returncode != 0 and "fps_mode" in r.stderr:
            r = subprocess.run(base + ["-vsync", "0"] + tail, capture_output=True, text=True)
        if r.returncode != 0:
            shutil.rmtree(out_dir, ignore_errors=True)          # no half-extracted frames left behind
            raise RuntimeError(f"ffmpeg failed on {video}:\n{r.stderr}")
    names = sorted(os.listdir(out_dir))
    return [os.path.join(out_dir, n) for n in names]


def hsv_hist(bgr, mask):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], mask.astype(np.uint8), [16, 4], [0, 180, 0, 256]).ravel()
    return h / (h.sum() + 1e-6)


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _dilate(mask, px):
    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), k) > 0


def centroid(mask):
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()]) if len(xs) else np.array([np.nan, np.nan])


class Track:
    def __init__(self, tid, cat, mask, frame, hist):
        self.id, self.votes = tid, Counter({cat: 1.0})
        self.mask, self.hist = mask, hist
        self.active, self.first, self.last = True, frame, frame
        self.n_vis, self.areas, self.events = 0, [], []

    @property
    def cat(self):
        return self.votes.most_common(1)[0][0]

    def meta(self):
        return {"id": self.id, "votes": dict(self.votes), "first": self.first, "last": self.last,
                "n_vis": self.n_vis, "median_area": float(np.median(self.areas)) if self.areas else 0.0,
                "events": self.events}


def route_categories(best, cats):
    """best: highest YOLO-World score per prompt over the probe frames -> object categories none of whose prompts
    reaches ROUTE_MIN (YOLO-World cannot see them in this video; they go to Grounding DINO)"""
    top = {}
    for s, c in zip(best, cats):
        if c != "robot" and c not in C.GENERIC_CONTAINERS:   # the optional extras never get a second detector:
            top[c] = max(top.get(c, 0.0), float(s))             # in 26 that made tracking ~30x slower
    return sorted(c for c, s in top.items() if s < C.ROUTE_MIN)


class Detector:
    def __init__(self, prompts, device, cats=None):
        from ultralytics import YOLOWorld
        self.m = YOLOWorld(C.DET_MODEL)
        self.m.set_classes(prompts)
        self.device = device
        self.prompts, self.cats = list(prompts), list(cats or prompts)
        self.routed, self.gd, self.gd_idx = [], None, []
        self.probe_best = None

    def probe(self, paths):
        """YOLO-World on PROBE_FRAMES evenly spaced frames; categories it never scores >= ROUTE_MIN are detected
        with Grounding DINO instead for the whole video."""
        idx = np.unique(np.linspace(0, len(paths) - 1, min(len(paths), C.PROBE_FRAMES)).astype(int))
        best = np.zeros(len(self.prompts))
        for f in idx:
            r = self.m.predict(cv2.imread(paths[f]), conf=0.001, agnostic_nms=False, verbose=False,
                               device=self.device)[0]
            if r.boxes is not None:
                for s, k in zip(r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
                    best[k] = max(best[k], float(s))
        self.probe_best = {p: round(float(s), 4) for p, s in zip(self.prompts, best)}
        print("  detector probe (YOLO-World best score per prompt): " +
              ", ".join(f"'{p}' {s:.3f}" for p, s in self.probe_best.items()), flush=True)
        routed = route_categories(best, self.cats)
        if not routed:
            return
        try:
            from .gdino import GroundingDino
            self.gd_idx = [k for k, c in enumerate(self.cats) if c in routed]
            self.gd = GroundingDino([self.prompts[k] for k in self.gd_idx], self.device)
            self.routed = routed
            print(f"  YOLO-World cannot see {routed} here (best < {C.ROUTE_MIN}): Grounding DINO detects them",
                  flush=True)
        except Exception as e:
            print(f"  Grounding DINO unavailable ({type(e).__name__}: {e}); {routed} stay with YOLO-World", flush=True)
            self.gd, self.gd_idx = None, []

    def __call__(self, bgr):
        # per-class NMS: the arm base must not lose its 'robot arm' box to a higher-scoring 'towel' box on the
        # same pixels; duplicates across object prompts are resolved by the tracker's coverage test
        r = self.m.predict(bgr, conf=C.DET_CONF, agnostic_nms=False, verbose=False, device=self.device)[0]
        out = []
        if r.boxes is not None and len(r.boxes):
            b = r.boxes
            out = [d for d in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int))
                   if d[2] not in self.gd_idx]
        if self.gd is not None:
            out += [(box, s, self.gd_idx[k]) for box, s, k in self.gd(bgr)]
        return out


def resolve_device(torch, device):
    if device in (None, "auto"):
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"                       # Apple-silicon GPU
        return "cpu"
    return f"cuda:{device}" if str(device).isdigit() else device


# ---------------------------------------------------------------- tracker
class Tracker:
    def __init__(self, prompts, cats, device="auto", sam2="base_plus", counts=None):
        import torch
        from sam2.build_sam import build_sam2, build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.torch = torch
        self.dev = resolve_device(torch, device)
        self.cats = cats
        ckpt, cfg = C.SAM2_MODELS[sam2]
        print(f"  device={self.dev}  sam2={sam2}", flush=True)
        self.det = Detector(prompts, self.dev, cats)
        self.vp = build_sam2_video_predictor(cfg, ckpt, device=self.dev)
        self.ip = SAM2ImagePredictor(build_sam2(cfg, ckpt, device=self.dev))
        self.tracks, self.next_id, self.caps = {}, 1, None
        self.det_log, self.weak = [], []
        self.fixed_caps = dict(counts or {})     # instances per category parsed from the task sheet (vocab.py)
        self.glue_seen = []

    # ---- id bookkeeping
    def _count_exact(self, cat):
        return sum(1 for t in self.tracks.values() if t.cat == cat)

    def _glue_candidate(self, cat, conf, host, box, s):
        """A confident detection of a counted category that is still short of its task sheet number, lying on an
        object of another category: that object's mask may hold two touching things (fork + knife side by side).
        Only if the same situation repeats at the same place (GLUE_PERSIST windows) -- one-off partial detections
        of the host itself do not split it."""
        if (cat == "robot" or host.cat == "robot" or host.cat == cat or cat not in self.fixed_caps
                or conf < C.NEW_CONF or self._count_exact(cat) >= self.fixed_caps[cat]):
            return False
        bx = np.array(box, float)
        hits = sum(1 for g in self.glue_seen if g["cat"] == cat and s - g["frame"] <= 10 * C.WINDOW
                   and _iou(g["box"], bx) >= 0.5)
        if not any(g["frame"] == s and _iou(g["box"], bx) >= 0.9 for g in self.glue_seen):
            self.glue_seen.append({"frame": s, "cat": cat, "box": bx})
        return hits + 1 >= C.GLUE_PERSIST

    @staticmethod
    def _typical_area(t):
        """usual size of an object (median over its life so far): a towel stays 'big' while it is half hidden"""
        return float(np.median(t.areas)) if t.areas else float(t.mask.sum())

    def _count(self, cat):
        if cat in self.fixed_caps:                # counted categories share one budget: the detector often calls a
            return sum(1 for t in self.tracks.values() if t.cat in self.fixed_caps)   # fork 'spoon', a strap 'cable'
        return sum(1 for t in self.tracks.values() if t.cat == cat)

    def _cap(self, cat):
        if cat in self.fixed_caps:                # task sheet total + slack: a spurious id must not lock a real one
            n = sum(self.fixed_caps.values())     # out; finalize.py enforces the exact counts
            return n + max(1, int(round(C.COUNT_SLACK * n)))
        if self.caps is None:
            return 10 ** 6
        return self.caps.get(cat, C.CAP_UNSEEN)

    def _reid(self, mask, cat, img):
        h, c = hsv_hist(img, mask), centroid(mask)
        best, bs = None, -1.0
        for t in self.tracks.values():
            if t.active or ((t.cat == "robot") != (cat == "robot")):
                continue
            sim = float(np.minimum(h, t.hist).sum())
            d = float(np.linalg.norm(c - centroid(t.mask)))
            confusable = t.cat in self.fixed_caps and cat in self.fixed_caps    # fork / spoon / knife ...
            sc = 0.6 * sim + 0.4 * np.exp(-d / C.REID_DIST_PX) - (0.3 if t.cat != cat and not confusable else 0.0)
            if sc > bs:
                best, bs = t, sc
        if best is not None and bs >= C.REID_ACCEPT:
            return best, bs
        if best is not None and self._count(cat) >= self._cap(cat) and (bs >= C.REID_FORCE or cat in self.fixed_caps):
            return best, bs                    # counted categories at their budget: every new one is a lost one
        return None, bs

    # ---- per-window seeding
    def _seed(self, s, img):
        """Returns objects that (re)appear at frame s. Every detection's fate is logged to detections.jsonl."""
        H, W = img.shape[:2]
        robot = lambda c: c == "robot"
        live = [t for t in self.tracks.values() if t.active]          # grows as objects appear below
        dets = sorted(self.det(img), key=lambda d: (self.cats[d[2]] != "robot", -d[1]))   # arms first
        appeared, image_set = [], False
        wbest = {}                                   # track id -> (conf, cat): best detection on it in this window
        vote = lambda t, c, p: wbest.__setitem__(t.id, max(wbest.get(t.id, (0.0, c)), (float(p), c)))
        area_of = {t.id: int(t.mask.sum()) for t in live}
        for box, conf, k in dets:
            cat = self.cats[k]
            x1, y1, x2, y2 = np.clip(box, 0, [W - 1, H - 1, W - 1, H - 1]).astype(int)
            rec = {"frame": s, "cat": cat, "conf": round(float(conf), 3), "box": [int(x1), int(y1), int(x2), int(y2)]}
            self.det_log.append(rec)
            if x2 <= x1 or y2 <= y1:
                rec["why"] = "empty box"
                continue
            area = (x2 - x1) * (y2 - y1)
            same_kind = [t for t in live if robot(t.cat) == robot(cat)]   # arms never "cover" objects and vice versa
            # an object lying in / on a much bigger object of another category (cutlery in the insert, a bundle in
            # the organizer) is not covered by it: it is a separate object
            container = lambda t, a: (not robot(cat) and t.cat != cat
                                      and area_of.get(t.id, int(t.mask.sum())) >= C.CONTAINER_RATIO * a)
            best, cov = None, 0.0
            for t in same_kind:
                if container(t, area):
                    continue
                c = t.mask[y1:y2, x1:x2].sum() / area
                if c > cov:
                    best, cov = t, float(c)
            if best is not None and cov >= C.COVERED_FRAC:
                if not self._glue_candidate(cat, conf, best, [x1, y1, x2, y2], s):
                    vote(best, cat, conf)                              # existing object: detection only votes
                    rec["why"] = f"covered by id {best.id} ({cov:.2f})"
                    continue
                rec["glue_host"] = best.id                             # SAM decides below
            if conf < C.NEW_CONF:
                # weak detection: accept it only if the same category was weakly seen at the same place recently
                # (a towel half-hidden in the pile scores 0.1-0.2 every window; a one-off false positive does not)
                bx = np.array([x1, y1, x2, y2], float)
                hit = any(p["cat"] == cat and s - p["frame"] <= C.WEAK_MEMORY and _iou(p["box"], bx) >= 0.5
                          for p in self.weak)
                self.weak.append({"frame": s, "cat": cat, "box": bx})
                short = cat in self.fixed_caps and self._count(cat) < sum(self.fixed_caps.values())
                if not hit and short:                  # the task sheet says some of this category are still missing
                    rec["weak_short"] = True
                elif not hit:
                    rec["why"] = f"conf < {C.NEW_CONF}" + (f" (best cover id {best.id} {cov:.2f})" if best else "")
                    continue
                else:
                    rec["weak_confirmed"] = True
            if not image_set:
                self.ip.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                image_set = True
            m, _, _ = self.ip.predict(box=np.array([x1, y1, x2, y2]), multimask_output=False)
            m = m[0] > 0.5
            union_same = np.zeros((H, W), bool)
            for t in same_kind:
                if not container(t, max(1, int(m.sum()))):
                    union_same |= t.mask
            if m.sum() < C.MIN_AREA:
                rec["why"] = "SAM mask too small"
                continue
            if not robot(cat):
                my, mx = int(C.ROI_MARGIN * H), int(C.ROI_MARGIN * W)
                inside = float(m[my:H - my, mx:W - mx].sum() / m.sum())
                if inside < 0.5:
                    rec["why"] = f"outside the detection range ({1 - inside:.0%} of its mask in the frame border)"
                    continue
            if not robot(cat) and m.sum() > C.MAX_OBJECT_FRAC * H * W:
                rec["why"] = f"SAM mask covers {m.sum() / (H * W):.0%} of the frame: table / background"
                continue
            ov = float((m & union_same).sum() / m.sum())
            if ov > 0.5:
                host = max(same_kind, key=lambda t: (m & t.mask).sum())
                glued = (host.mask.sum() >= C.GLUE_RATIO * m.sum()
                         and (rec.get("glue_host") == host.id
                              or self._glue_candidate(cat, conf, host, [x1, y1, x2, y2], s)))
                if not glued:
                    if rec.get("glue_host") == host.id:
                        vote(host, cat, conf)
                    rec["why"] = f"SAM mask {ov:.2f} inside existing same-kind masks (mostly id {host.id} {host.cat})"
                    continue
                rec["glue_host"] = host.id                             # host holds this object too: split it off
            robots = [t for t in live if robot(t.cat)]
            if robot(cat):
                touch = [t for t in robots if (m & _dilate(t.mask, C.ROBOT_JOIN_PX)).any()]
                if touch:                                       # another part of an arm we already track
                    t = max(touch, key=lambda t: (m & _dilate(t.mask, C.ROBOT_JOIN_PX)).sum())
                    objs = np.zeros((H, W), bool)
                    for o in live:
                        if not robot(o.cat):
                            objs |= o.mask
                    t.mask = t.mask | (m & ~objs)                 # not over objects it holds (cable in a gripper)
                    vote(t, cat, conf)
                    rec["why"] = f"merged into robot id {t.id}"
                    continue
            else:
                runion = np.zeros((H, W), bool)
                for t in robots:
                    runion |= t.mask
                rov = float((m & runion).sum() / m.sum())
                if rov > C.ON_ROBOT_FRAC:
                    rec["why"] = f"part of a robot arm ({rov:.2f} of its mask)"
                    continue
            t, sc = self._reid(m, cat, img)
            if t is not None:
                t.events.append({"type": "reid", "frame": s, "score": round(float(sc), 3)})
                vote(t, cat, conf)
                rec["why"] = f"re-identified as id {t.id} ({sc:.2f})"
            else:
                # the cap only forces re-use of a LOST id; if every object of this category is still on screen,
                # an uncovered detection cannot be one of them -> it is a genuinely new object (e.g. a 4th towel
                # that sat in the pile during the first window)
                lost_same = any(not x.active and x.cat == cat for x in self.tracks.values())
                if cat in self.fixed_caps and self._count(cat) >= self._cap(cat):
                    # the task sheet says how many there are: all ids exist -> duplicate or false positive
                    rec["why"] = f"counted categories full ({self._cap(cat)} per task sheet), no lost id matched"
                    continue
                if self.caps is not None and self._count(cat) >= self._cap(cat) and lost_same:
                    rec["why"] = "category at cap, lost ids exist but none matched"
                    continue
                t = Track(self.next_id, cat, m, s, hsv_hist(img, m))
                self.tracks[t.id] = t
                self.next_id += 1
                t.events.append({"type": "new", "frame": s, "conf": round(float(conf), 3)})
                if rec.get("glue_host"):
                    t.split_from = {rec["glue_host"]}
                rec["why"] = f"NEW id {t.id}" + (" (weak, seen here before)" if rec.get("weak_confirmed") else "") \
                    + (" (weak, category short of task sheet count)" if rec.get("weak_short") else "") \
                    + (f" (split off id {rec['glue_host']}, which held it too)" if rec.get("glue_host") else "")
            t.active, t.mask = True, m
            # a towel pulled out of a pile was part of the pile's mask: take its pixels out of every other
            # same-category prompt, so the next window does not start with two ids claiming the same cloth
            for o in live:
                if o is not t and not robot(o.cat) and (o.cat == t.cat or container(o, int(m.sum()))
                                                        or o.id == rec.get("glue_host")) \
                        and (o.mask & m).any():
                    rest = o.mask & ~m
                    if rest.sum() >= C.MIN_AREA:
                        o.mask = rest
                        rec["why"] += f"; carved out of id {o.id}"
            if t not in live:
                live.append(t)
            appeared.append(t)
        for tid, (p, c) in wbest.items():
            self.tracks[tid].votes[c] += p
        if self.caps is None:                                  # first window defines how many of each exist
            cnt = Counter(t.cat for t in self.tracks.values())
            self.caps = {c: n + C.CAP_SLACK for c, n in cnt.items()}
        return appeared

    # ---- SAM 2 propagation over a list of global frame indices
    def _propagate(self, frames, paths, prompts, reverse=False):
        """prompts: [(local_idx, track)] -> yields (local_idx, ids, logits[N,H,W] on GPU)"""
        wdir = os.path.join(self.work, "win")
        shutil.rmtree(wdir, ignore_errors=True)
        os.makedirs(wdir)
        for i, f in enumerate(frames):
            os.symlink(os.path.abspath(paths[f]), os.path.join(wdir, f"{i:05d}.jpg"))
        state = self.vp.init_state(video_path=wdir, offload_video_to_cpu=True)
        for li, t in prompts:
            self.vp.add_new_mask(state, li, t.id, t.mask)
        start = prompts[0][0]
        for li, ids, logits in self.vp.propagate_in_video(state, start_frame_idx=start, reverse=reverse):
            lg = logits[:, 0].float()
            if "cuda" not in str(self.dev):    # mps/cpu: do the per-pixel bookkeeping on CPU (unified memory)
                lg = lg.cpu()
            yield li, list(ids), lg
        self.vp.reset_state(state)

    def _jump_guard(self, ids, lg, prev, dead, f, drop=None):
        """SAM 2 failure mode: when its target is hidden, an object's mask jumps onto a look-alike neighbour
        (one towel taking over another). Detect it frame by frame on a 4x-downsampled grid: the new mask barely
        overlaps the object's own previous mask but mostly lies on another same-category object's previous mask.
        Such an object is switched off for the rest of the window (-> lost -> re-identified when re-detected)."""
        small = (lg[:, ::4, ::4] > 0).cpu().numpy()
        for k, i in enumerate(ids):
            if i in dead:
                lg[k] = -10.0
                continue
            a = small[k]
            n = int(a.sum())
            if i in prev and n >= 8:
                p = prev[i]
                own = (a & p).sum() / max(1, (a | p).sum())
                if drop is not None and (i not in drop or own < drop[i][0]):
                    drop[i] = (float(own), int(f))
                cat = self.tracks[int(i)].cat
                other = max([(a & q).sum() / n for j, q in prev.items()
                             if j != i and j not in dead and self.tracks[int(j)].cat == cat] or [0.0])
                if own < C.JUMP_OWN_IOU and other > C.JUMP_ONTO:
                    dead.add(i)
                    lg[k] = -10.0
                    self.tracks[int(i)].events.append({"type": "jump_blocked", "frame": int(f),
                                                       "own_iou": round(float(own), 2), "onto": round(float(other), 2)})
                    continue
            prev[i] = a

    def _combine(self, ids, lg):
        torch = self.torch
        mx, am = lg.max(0)
        idt = torch.tensor(ids, device=lg.device)
        label = torch.where(mx > 0, idt[am], torch.zeros_like(am))
        prob = torch.sigmoid(mx)
        n = int(max(ids)) + 1
        sums = torch.bincount(label.flatten(), weights=prob.flatten(), minlength=n)
        cnts = torch.bincount(label.flatten(), minlength=n)
        sc = {int(i): round(float(sums[i] / cnts[i]), 4) for i in ids if cnts[i] > 0}
        return label.to(torch.int32).cpu().numpy().astype(np.uint16), sc

    def _backfill(self, buffer, appeared, s, paths):
        """Run (re)appearing objects backwards through the buffered previous window, free pixels only."""
        torch = self.torch
        frames = [b["frame"] for b in buffer] + [s]
        li_s = len(frames) - 1
        for li, ids, lg in self._propagate(frames, paths, [(li_s, t) for t in appeared], reverse=True):
            if li == li_s:
                continue
            b = buffer[li]
            mx, am = lg.max(0)
            idt = torch.tensor(ids, device=lg.device)
            put = (mx > 0) & torch.from_numpy(self._takeable(b["label"], appeared)).to(lg.device)
            if not put.any():
                continue
            lab = torch.from_numpy(b["label"].astype(np.int32)).to(lg.device)
            lab = torch.where(put, idt[am].to(torch.int32), lab)
            b["label"] = lab.cpu().numpy().astype(np.uint16)
            prob = torch.sigmoid(mx)
            for i in ids:
                sel = put & (idt[am] == i)
                if sel.any():
                    b["scores"][int(i)] = round(float(prob[sel].mean()), 4)

    def _takeable(self, label, tracks):
        """Pixels a backfilled object may take: unassigned, held by an object of the same category (a towel
        lifted off a pile was part of the pile's mask before it was detected on its own), or held by a much
        bigger object of another category that contains it (a spoon lying in the insert)."""
        cats = {t.cat for t in tracks}
        small = min(max(1, int(t.mask.sum())) for t in tracks)
        same = [i for i, t in self.tracks.items() if t.cat in cats]          # includes the objects themselves
        holders = [i for i, o in self.tracks.items() if o.cat not in cats and o.cat != "robot"
                   and int(o.mask.sum()) >= C.CONTAINER_RATIO * small]
        holders += [i for t in tracks for i in getattr(t, "split_from", ())]   # the mask that used to hold it
        return (label == 0) | np.isin(label, same + holders)

    def _deep_backfill(self, t, f_hi, mask, f_stop, paths):
        """Keep propagating t backwards window by window over frames already on disk, until frame f_stop
        or until it is not visible for a whole window."""
        keep = t.mask
        while f_hi > f_stop:
            f_lo = max(f_stop, f_hi - C.WINDOW)
            frames = list(range(f_lo, f_hi + 1))
            t.mask, first = mask, None
            for li, ids, lg in self._propagate(frames, paths, [(len(frames) - 1, t)], reverse=True):
                if li == len(frames) - 1:
                    continue
                m = (lg[ids.index(t.id)] > 0).cpu().numpy()
                if m.sum() < C.MIN_AREA:
                    continue
                f = frames[li]
                path = os.path.join(self.work, "labels", f"{f:06d}.png")
                lab = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                put = m & self._takeable(lab, [t])
                if put.sum() < C.MIN_AREA:
                    continue
                if not (lab == t.id).any():
                    t.n_vis += 1
                    t.areas.append(int(put.sum()))
                lab[put] = t.id
                cv2.imwrite(path, lab)
                prob = 1.0 / (1.0 + np.exp(-lg[ids.index(t.id)].cpu().numpy()[put].astype(np.float64)))
                self.score_fh.write(json.dumps({"f": f, "s": {str(t.id): round(float(prob.mean()), 4)}}) + "\n")
                t.first = min(t.first, f)
                if li == 0:
                    first = m
            if first is None:                                   # not visible at the window's first frame: stop
                break
            mask, f_hi = first, f_lo
        t.events.append({"type": "deep_backfill", "until": int(t.first)})
        t.mask = keep

    def _flush(self, buffer):
        for b in buffer:
            cv2.imwrite(os.path.join(self.work, "labels", f"{b['frame']:06d}.png"), b["label"])
            ids, cnt = np.unique(b["label"], return_counts=True)
            for i, n in zip(ids, cnt):
                if i == 0 or int(i) not in self.tracks:
                    continue
                t = self.tracks[int(i)]
                t.n_vis += 1
                t.areas.append(int(n))
                t.first, t.last = min(t.first, b["frame"]), max(t.last, b["frame"])
            self.score_fh.write(json.dumps({"f": b["frame"], "s": {str(k): v for k, v in b["scores"].items()
                                                                   if k in set(ids.tolist())}}) + "\n")

    # ---- main loop
    def run(self, video, work):
        self.work = work
        os.makedirs(os.path.join(work, "labels"), exist_ok=True)
        paths = extract_frames(video, os.path.join(work, "frames"))
        T = len(paths)
        if hasattr(self.det, "probe"):
            self.det.probe(paths)
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
        cap.release()
        H, W = cv2.imread(paths[0]).shape[:2]
        self.score_fh = open(os.path.join(work, "scores.jsonl"), "w")
        torch = self.torch
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if "cuda" in str(self.dev) else torch.no_grad()
        buffer, s = [], 0
        with torch.inference_mode(), ctx:
            while True:
                e = min(s + C.WINDOW, T - 1)
                img = cv2.imread(paths[s])
                appeared = self._seed(s, img)
                if C.BACKFILL and buffer and appeared:
                    stop = {t.id: (t.last if t.events[-1]["type"] == "reid" else 0) for t in appeared}
                    self._backfill(buffer, appeared, s, paths)
                    f0 = buffer[0]["frame"]
                    for t in appeared:                          # still visible at the window's first frame?
                        m0 = buffer[0]["label"] == t.id
                        if f0 > stop[t.id] and m0.sum() >= C.MIN_AREA:
                            self._deep_backfill(t, f0, m0, stop[t.id], paths)
                self._flush(buffer)
                buffer = []
                last_out = e if e == T - 1 else e - 1          # frame e is output by the next window
                active = [t for t in self.tracks.values() if t.active]
                if not active:
                    buffer = [{"frame": f, "label": np.zeros((H, W), np.uint16), "scores": {}}
                              for f in range(s, last_out + 1)]
                else:
                    seen_at_e, prev, dead, drop = {}, {}, set(), {}
                    for li, ids, lg in self._propagate(list(range(s, e + 1)), paths, [(0, t) for t in active]):
                        f = s + li
                        self._jump_guard(ids, lg, prev, dead, f, drop)
                        if f <= last_out:
                            label, sc = self._combine(ids, lg)
                            buffer.append({"frame": f, "label": label, "scores": sc})
                        vis = (lg > 0).sum((1, 2)).cpu().numpy()
                        for i, v in zip(ids, vis):
                            t = self.tracks[int(i)]
                            if v >= C.MIN_AREA and (f == e or li % 5 == 0):
                                t.mask = (lg[ids.index(i)] > 0).cpu().numpy()
                            if f == e:
                                seen_at_e[int(i)] = v >= C.MIN_AREA
                    img_e = cv2.imread(paths[e])
                    for t in active:
                        # swapped ids: SAM 2 moved this id onto a different-looking object. Check the colour right
                        # after the frame where its mask changed most (whether or not it is still visible at e).
                        own_min, df = drop.get(t.id, (1.0, None))
                        if t.cat != "robot" and df is not None and own_min < C.SWAP_SUSPECT_IOU:
                            fc = min(df + 2, buffer[-1]["frame"]) if buffer else None
                            bc = next((x for x in buffer if x["frame"] == fc), None)
                            mc = (bc["label"] == t.id) if bc is not None else None
                            if mc is not None and mc.sum() >= C.MIN_AREA and self._typical_area(t) >= C.APPEAR_MIN_AREA:
                                sim = float(np.minimum(hsv_hist(cv2.imread(paths[fc]), mc), t.hist).sum())
                                if sim < C.APPEAR_MIN:
                                    for x in buffer:
                                        if x["frame"] >= df:
                                            x["label"] = np.where(x["label"] == t.id, 0, x["label"]).astype(np.uint16)
                                            x["scores"].pop(t.id, None)
                                    t.active = False
                                    t.events.append({"type": "appearance_swap", "frame": int(df), "sim": round(sim, 2)})
                                    continue
                        if seen_at_e.get(t.id, False):                  # visible at e -> carried, else lost
                            h = hsv_hist(img_e, t.mask)
                            sim = float(np.minimum(h, t.hist).sum())
                            if t.cat != "robot" and sim < C.APPEAR_MIN and self._typical_area(t) >= C.APPEAR_MIN_AREA:
                                df = drop.get(t.id, (1.0, s))[1]
                                for x in buffer:
                                    if x["frame"] >= df:
                                        x["label"] = np.where(x["label"] == t.id, 0, x["label"]).astype(np.uint16)
                                        x["scores"].pop(t.id, None)
                                t.active = False
                                t.events.append({"type": "appearance_swap", "frame": int(df), "sim": round(sim, 2)})
                                continue
                            t.hist = 0.7 * t.hist + 0.3 * h
                        else:
                            t.active = False
                            t.events.append({"type": "lost", "frame": e})
                print(f"  frames {s}-{last_out}/{T - 1}: {len(active)} active, {len(self.tracks)} ids", flush=True)
                if e == T - 1:
                    break
                s = e
            self._flush(buffer)
        self.score_fh.close()
        meta = {"video": os.path.basename(video), "width": W, "height": H, "fps": fps, "num_frames": T,
                "detector": {"probe_best": getattr(self.det, "probe_best", None),
                             "grounding_dino_for": getattr(self.det, "routed", [])},
                "tracks": [t.meta() for t in self.tracks.values()]}
        json.dump(meta, open(os.path.join(work, "tracks.json"), "w"), indent=1)
        with open(os.path.join(work, "detections.jsonl"), "w") as fh:
            for r in self.det_log:
                fh.write(json.dumps(r) + "\n")
        return meta
