"""One-screen diagnosis of a finished run, in FINAL ids (as in the JSON / overlay):
expected vs found instances, per object: life span, occlusion / re-id / swap events; detector decisions per category.

    python tools/report.py results 12_fold_four_towels
"""
import json
import os
import re
import sys
from collections import Counter

out, vid = sys.argv[1], sys.argv[2]
work = os.path.join(out, "work", vid)
meta = json.load(open(os.path.join(work, "tracks.json")))
fin = json.load(open(os.path.join(work, "final_ids.json")))
res = json.load(open(os.path.join(out, vid + ".json")))
fps = meta["fps"]
fid = {int(k): v for k, v in fin["final_id_of_track"].items()}
objs = {o["id"]: o for o in res["objects"]}

found = Counter(o["category"] for o in res["objects"])
print(f"== {vid}: {meta['num_frames']} frames")
print("expected (parsed from task sheet):", fin.get("expected_counts") or "-")
print("found:", dict(found))
print("merged into a neighbour (unstable / over the count):", list(fin["merged_unstable_tracks"]) or "none")
vp = os.path.join(work, "verify.json")
ver = json.load(open(vp)) if os.path.exists(vp) else {}
print("\nfinal id  category        note")
for i, o in objs.items():
    print(f"  {i:3d}    {o['category']:15s} {o['note']}")
print("\nevents (track -> final id):")
for t in meta["tracks"]:
    ev = []
    for e in t["events"]:
        if e["type"] in ("new", "reid", "lost", "appearance_swap", "jump_blocked"):
            ev.append(f"{e['type']}@{e['frame'] / fps:.1f}s")
        elif e["type"] == "deep_backfill":
            ev.append(f"backfill->{e['until'] / fps:.1f}s")
    v = ver.get(str(t["id"]))
    cl = f"clip {v['object']:.2f} {max(v['cats'], key=v['cats'].get) if v['cats'] else ''}" if v else ""
    print(f"  track {t['id']:3d} -> {str(fid.get(t['id'], 'dropped')):>7}  {max(t['votes'], key=t['votes'].get):12s}"
          f" vis {t['n_vis'] / fps:6.1f}s {cl:22s} " + " ".join(ev[:12]) + (" ..." if len(ev) > 12 else ""))
dl = os.path.join(work, "detections.jsonl")
if not os.path.exists(dl):
    sys.exit(0)
print("\ndetector decisions:")
c = Counter((r["cat"], re.sub(r"[\d.]+", "#", r.get("why", "?")))
            for r in map(json.loads, open(dl)))
for (cat, why), n in sorted(c.items(), key=lambda kv: (kv[0][0], -kv[1])):
    print(f"  {n:5d}  {cat:15s} {why}")
