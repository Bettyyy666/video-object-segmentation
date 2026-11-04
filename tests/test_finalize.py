"""CPU-only check of pass 2 + the checker on synthetic label maps (no models needed).

    python tests/test_finalize.py
"""
import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from seg.ffmpeg import ffmpeg_bin  # noqa: E402
from seg import finalize  # noqa: E402

T, W, H, FPS = 60, 1280, 720, 50.0


def main():
    tmp = tempfile.mkdtemp()
    work = os.path.join(tmp, "work")
    os.makedirs(os.path.join(work, "frames"))
    os.makedirs(os.path.join(work, "labels"))
    scores = open(os.path.join(work, "scores.jsonl"), "w")
    for f in range(T):
        img = np.full((H, W, 3), 90, np.uint8)
        lab = np.zeros((H, W), np.uint16)
        lab[100:200, 100 + f:400 + f] = 3                       # big spoon, moving right
        lab[400:440, 300:400] = 5                               # small spoon
        if not 20 <= f < 35:                                    # towel hidden for 15 frames, same id after
            lab[300:500, 700:1000] = 7
        if f < 5:
            lab[600:620, 50:70] = 9                             # spurious, 5 frames -> dropped
        if not 20 <= f < 35:                                    # duplicate towel id: slivers on the towel's edge,
            y = 300 + (f * 37) % 180                            # jumping every frame, in pieces
            for k in range(4):
                lab[y:y + 6, 1000 + 8 * k:1004 + 8 * k] = 11
        lab[50:90, 1100:1200] = 13                             # stable, but CLIP says table -> dropped
        if f < 30:
            lab[600:650, 900:1000] = 15                         # 2nd 'towel' while the task sheet says 1 towel
        cv2.imwrite(os.path.join(work, "frames", f"{f:06d}.jpg"), img)
        cv2.imwrite(os.path.join(work, "labels", f"{f:06d}.png"), lab)
        scores.write(json.dumps({"f": f, "s": {"3": 0.9, "5": 0.8, "7": 0.95, "9": 0.3}}) + "\n")
    scores.close()
    tracks = [
        {"id": 3, "votes": {"spoon": 2.0}, "first": 0, "last": 59, "n_vis": 60, "median_area": 30000, "events": []},
        {"id": 5, "votes": {"spoon": 1.5, "fork": 0.2}, "first": 0, "last": 59, "n_vis": 60, "median_area": 4000,
         "events": []},
        {"id": 7, "votes": {"towel": 3.0}, "first": 0, "last": 59, "n_vis": 45, "median_area": 60000,
         "events": [{"type": "lost", "frame": 20}, {"type": "reid", "frame": 35, "score": 0.7}]},
        {"id": 11, "votes": {"towel": 1.0}, "first": 0, "last": 59, "n_vis": 45, "median_area": 100, "events": []},
        {"id": 13, "votes": {"towel": 2.0}, "first": 0, "last": 59, "n_vis": 60, "median_area": 4000, "events": []},
        {"id": 15, "votes": {"towel": 0.5}, "first": 0, "last": 29, "n_vis": 30, "median_area": 5000, "events": []},
        {"id": 9, "votes": {"fork": 0.3}, "first": 0, "last": 4, "n_vis": 5, "median_area": 400, "events": []},
    ]
    json.dump({"video": "x.mp4", "width": W, "height": H, "fps": FPS, "num_frames": T, "tracks": tracks},
              open(os.path.join(work, "tracks.json"), "w"))
    json.dump({"13": {"object": 0.08, "cats": {"towel": 0.08}, "n": 8},
               "15": {"object": 0.9, "cats": {"towel": 0.9}, "n": 8},
               "7": {"object": 0.95, "cats": {"towel": 0.95}, "n": 8}}, open(os.path.join(work, "verify.json"), "w"))
    video = os.path.join(tmp, "x.mp4")
    subprocess.run([ffmpeg_bin(), "-loglevel", "error", "-y", "-framerate", "50", "-i",
                    os.path.join(work, "frames", "%06d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p", video],
                   check=True)
    out_json, out_ov = os.path.join(tmp, "x.json"), os.path.join(tmp, "x_overlay.mp4")
    objs = finalize.run(work, "x.mp4", out_json, out_ov,
                        size_split={"spoon": {"small": "teaspoon", "large": "dinner spoon"}}, counts={"towel": 1})
    print(json.dumps(objs, indent=1))
    d = json.load(open(out_json))
    assert [o["category"] for o in d["objects"]] == ["dinner spoon", "teaspoon", "towel"]
    assert d["objects"][2]["deformable"] and "re-identified" in d["objects"][2]["note"]
    assert len(d["frames"]) == T and d["fps"] == 50
    towel = [f["frame"] for f in d["frames"] if any(m["id"] == 3 for m in f["masks"])]
    assert towel == [f for f in range(T) if not 20 <= f < 35]
    from pycocotools import mask as mu
    m = next(m for m in d["frames"][0]["masks"] if m["id"] == 3)
    bm = mu.decode({"size": m["rle"]["size"], "counts": m["rle"]["counts"].encode()})
    assert bm.sum() == 200 * 300 + 4 * 24, bm.sum()            # slivers of the duplicate id joined the towel
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "check_output.py"), out_json,
                        "--video", video, "--overlay", out_ov, "--dump", os.path.join(tmp, "check")],
                       capture_output=True, text=True)
    print(r.stdout, r.stderr)
    assert "format OK" in r.stdout
    assert "id   3 towel              0.70-   0.70s" in r.stdout      # the 0.3 s occlusion of the towel
    # a 'towel' as big as most of the frame is the table, whatever the detector and CLIP say
    meta = {"fps": 50, "num_frames": 100, "width": 100, "height": 100, "tracks": [
        {"id": 1, "votes": {"towel": 5.0}, "first": 0, "last": 99, "n_vis": 100, "median_area": 6000, "events": []},
        {"id": 2, "votes": {"towel": 5.0}, "first": 0, "last": 99, "n_vis": 100, "median_area": 1500, "events": []},
        {"id": 3, "votes": {"robot": 5.0}, "first": 0, "last": 99, "n_vis": 100, "median_area": 6000, "events": []}]}
    _, remap, _, _ = finalize.plan(meta, log=lambda *_: None)
    assert set(remap) == {2, 3}, remap
    # task sheet: 1 insert. A small strip inside it also called 'insert' must vanish, not grow the insert
    meta = {"fps": 50, "num_frames": 100, "width": 100, "height": 100, "tracks": [
        {"id": 1, "votes": {"insert": 5.0}, "first": 0, "last": 99, "n_vis": 100, "median_area": 3000, "events": []},
        {"id": 2, "votes": {"insert": 5.0}, "first": 0, "last": 99, "n_vis": 100, "median_area": 200, "events": []},
        {"id": 3, "votes": {"towel": 5.0}, "first": 0, "last": 99, "n_vis": 100, "median_area": 900, "events": []},
        {"id": 4, "votes": {"towel": 5.0}, "first": 0, "last": 99, "n_vis": 50, "median_area": 800, "events": []}]}
    ver = {k: {"object": 0.9, "cats": {"insert": 0.9} if k in "12" else {"towel": 0.9}, "n": 8} for k in "1234"}
    _, remap, dup, _ = finalize.plan(meta, verify=ver, counts={"insert": 1, "towel": 1}, log=lambda *_: None)
    assert set(remap) == {1, 3} and dup == {4: "towel"}, (remap, dup)     # towel duplicate (similar size): merged
    # relevance: a handled-category track that never moves and is never touched goes; touched or moved stays;
    # containers (not in the task sheet's object column) stay regardless
    meta = {"fps": 50, "num_frames": 100, "tracks": [
        {"id": i, "votes": {c: 5.0}, "first": 0, "last": 99, "n_vis": 100, "median_area": 500, "events": []}
        for i, c in [(1, "cable"), (2, "cable"), (3, "cable"), (4, "organizer")]]}
    stab = {1: (1.0, 0.0, 3.0, False, 1.0), 2: (1.0, 0.0, 3.0, True, 1.0), 3: (1.0, 0.0, 300.0, False, 1.0),
            4: (1.0, 0.0, 0.0, False, 1.0)}
    _, remap, _, _ = finalize.plan(meta, stab=stab, manipulated={"cable"}, log=lambda *_: None)
    assert set(remap) == {2, 3, 4}, remap
    ver = {"1": {"object": 0.1, "cats": {"cable": 0.1}, "n": 8}, "3": {"object": 0.1, "cats": {"cable": 0.1}, "n": 8}}
    stab[3] = (1.0, 0.0, 300.0, True, 1.0)                                 # CLIP unsure, but an arm moved it: kept
    _, remap, _, _ = finalize.plan(meta, stab=stab, verify=ver, manipulated={"cable"}, log=lambda *_: None)
    assert 3 in remap and 1 not in remap, remap
    meta["tracks"][2]["votes"] = {"container": 5.0}                       # a generic extra gets no such pass
    ver["3"] = {"object": 0.1, "cats": {"container": 0.1}, "n": 8}
    _, remap, _, _ = finalize.plan(meta, stab=stab, verify=ver, manipulated={"cable", "container"},
                                   log=lambda *_: None)
    assert 3 not in remap, remap
    meta["tracks"][2]["votes"] = {"cable": 5.0}
    stab[3] = (1.0, 0.0, 300.0, True, 0.2)                                 # lives in the frame border
    _, remap, _, _ = finalize.plan(meta, stab=stab, manipulated={"cable"}, log=lambda *_: None)
    assert set(remap) == {2, 4}, remap
    # the detector calls all cutlery 'spoon'; CLIP and the task sheet's counts (1 each) sort it out
    meta = {"fps": 50, "num_frames": 100, "tracks": [
        {"id": i, "votes": {"spoon": 5.0, "fork": f}, "first": 0, "last": 99, "n_vis": 100, "median_area": 900,
         "events": []} for i, f in [(1, 0.0), (2, 1.0), (3, 0.0)]]}
    ver = {"1": {"object": 0.9, "cats": {"spoon": 0.8, "fork": 0.05, "knife": 0.05}, "n": 8},
           "2": {"object": 0.9, "cats": {"spoon": 0.3, "fork": 0.5, "knife": 0.1}, "n": 8},
           "3": {"object": 0.9, "cats": {"spoon": 0.2, "fork": 0.1, "knife": 0.6}, "n": 8}}
    o, _, dup, _ = finalize.plan(meta, verify=ver, counts={"spoon": 1, "fork": 1, "knife": 1}, log=lambda *_: None)
    assert [x["category"] for x in o] == ["spoon", "fork", "knife"] and not dup, o
    o, _, dup, _ = finalize.plan(meta, verify=ver, log=lambda *_: None)                 # no counts: CLIP leads
    assert [x["category"] for x in o] == ["spoon", "fork", "knife"], o
    # size split with task sheet numbers: 2 + 2, although an occluded dinner spoon looks small
    tr = [{"category": "spoon", "median_area": a} for a in (900, 880, 400, 380)]
    finalize._size_split(tr, {"spoon": {"small": "teaspoon", "large": "dinner spoon", "n_small": 2, "n_large": 2}})
    assert [t["category"] for t in tr] == ["dinner spoon", "dinner spoon", "teaspoon", "teaspoon"], tr
    tr = [{"category": "spoon", "median_area": a} for a in (900, 600, 560, 380)]   # Otsu alone: 1 large, 3 small
    finalize._size_split(tr, {"spoon": {"small": "teaspoon", "large": "dinner spoon", "n_small": 2, "n_large": 2}})
    assert [t["category"] for t in tr] == ["dinner spoon", "dinner spoon", "teaspoon", "teaspoon"], tr
    print("OK:", tmp)


if __name__ == "__main__":
    main()
