"""One command per video:

    python segment.py data/12_fold_four_towels.mp4 --task-sheets data/task_sheets.json --out results

writes results/12_fold_four_towels.json, results/12_fold_four_towels_overlay.mp4,
working files in results/work/12_fold_four_towels/ (frames, label maps, track log).
"""
import argparse
import json
import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # Apple GPU: ops MPS lacks fall back to CPU

from seg import finalize, vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--task-sheets", default="data/task_sheets.json")
    ap.add_argument("--overrides", default="configs/overrides.json")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("--sam2", default="base_plus", choices=["tiny", "small", "base_plus", "large"])
    ap.add_argument("--finalize-only", action="store_true", help="reuse the label maps of a previous run")
    ap.add_argument("--no-verify", action="store_true", help="skip the CLIP second opinion on each object")
    ap.add_argument("--reverify", action="store_true", help="recompute CLIP scores, including with --finalize-only")
    a = ap.parse_args()
    if a.no_verify and a.reverify:
        ap.error("--no-verify and --reverify cannot be used together")

    if not os.path.isfile(a.video):
        raise SystemExit(f"video not found: {a.video}  (run `ls data` to see the exact file names)")
    vid = os.path.splitext(os.path.basename(a.video))[0]
    work = os.path.join(a.out, "work", vid)
    os.makedirs(work, exist_ok=True)
    prompts, cats, split, deform, artic, counts = vocab.build(a.task_sheets, vid, a.overrides)
    print(f"{vid}: prompts -> categories: " + ", ".join(f"'{p}'->{c}" for p, c in zip(prompts, cats)))
    print(f"{vid}: expected instances parsed from the task sheet: {counts or 'none'}")
    manip = vocab.manipulated_categories(a.task_sheets, vid, a.overrides)
    print(f"{vid}: handled objects (must move or be touched by an arm): {manip}")

    t0 = time.time()
    if not a.finalize_only:
        from seg.tracker import Tracker
        Tracker(prompts, cats, a.device, a.sam2, counts).run(a.video, work)
    t1 = time.time()
    if a.reverify and os.path.exists(os.path.join(work, "verify.json")):
        os.remove(os.path.join(work, "verify.json"))  # failed refresh must not silently reuse old crops' scores
    if not a.no_verify and (a.reverify or not a.finalize_only or not os.path.exists(os.path.join(work, "verify.json"))):
        try:                                     # a second opinion only: never lose a finished tracking run to it
            from seg import verify
            verify.run(work, prompts, cats, "cuda" if a.device in ("auto", "cuda") else a.device)
        except Exception as e:
            print(f"  verify failed ({type(e).__name__}: {e}); continuing without CLIP check")
    elif a.no_verify and os.path.exists(os.path.join(work, "verify.json")):
        os.remove(os.path.join(work, "verify.json"))
    t15 = time.time()
    objs = finalize.run(work, os.path.basename(a.video), os.path.join(a.out, vid + ".json"),
                        os.path.join(a.out, vid + "_overlay.mp4"), split, deform, artic, counts, manip)
    t2 = time.time()
    meta = json.load(open(os.path.join(work, "tracks.json")))
    print(f"{vid}: {len(objs)} objects ({', '.join(o['category'] for o in objs)}) | "
          f"{meta['num_frames']} frames | tracking {t1 - t0:.0f}s, CLIP check {t15 - t1:.0f}s, writing {t2 - t15:.0f}s")


if __name__ == "__main__":
    main()
