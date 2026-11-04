"""What happened to every detection at the window starts (why an object did or did not get an id).

    python tools/why.py results/work/12_fold_four_towels/detections.jsonl            # summary
    python tools/why.py results/work/12_fold_four_towels/detections.jsonl --cat towel --frames 0-300
"""
import argparse
import json
import re
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--cat")
    ap.add_argument("--frames", help="a-b")
    a = ap.parse_args()
    lo, hi = (map(int, a.frames.split("-")) if a.frames else (0, 10 ** 9))
    rows = [json.loads(l) for l in open(a.log)]
    rows = [r for r in rows if lo <= r["frame"] <= hi and (not a.cat or r["cat"] == a.cat)]
    summary = Counter((r["cat"], re.sub(r"[\d.]+", "#", r.get("why", "?"))) for r in rows)
    print("decision counts (numbers replaced by #):")
    for (cat, why), n in summary.most_common():
        print(f"  {n:5d}  {cat:15s} {why}")
    if a.cat or a.frames:
        print("\nframe  conf   box                   decision")
        for r in rows:
            print(f"{r['frame']:5d}  {r['conf']:.2f}  {str(r['box']):20s}  {r.get('why', '?')}")


if __name__ == "__main__":
    main()
