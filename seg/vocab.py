"""Which objects to look for, from the task sheet text only (never from the video).

task-related = what the robot touches / moves (the task sheet's `object` column) + what things are put into or
onto ("in the next empty compartment", "onto the strap" ...). configs/overrides.json can replace the prompt list
for a video (hand-written from the task sheet; listed in the README as a manual step).
"""
import json
import os
import re

from . import config as C

FUNCTION = {"off", "over", "it", "its", "rest", "park", "third", "diagonal", "mouth", "neck", "opening", "stage",
            "beside", "part", "releases", "release", "first", "second", "last", "same", "way", "upright", "flat",
            "level", "square", "place", "turn", "pass", "hold", "near", "far", "free", "lifts", "output", "below",
            "floor", "base"}
CUT = {"and", "the", "to", "of", "with", "while", "so", "until", "before", "after", "by", "for", "from", "at"}


def find_task(task_sheets, episode_id):
    tasks = json.load(open(task_sheets))["tasks"]
    key = re.sub(r"^\d+_", "", episode_id)
    for t in tasks:
        if t["task"] == key:
            return t
    m = re.match(r"(\d+)_", episode_id)
    if m and 1 <= int(m.group(1)) <= len(tasks):
        return tasks[int(m.group(1)) - 1]
    raise KeyError(episode_id)


def receptacles(task):
    text = " . ".join(s["subtask"] + " . " + s["ends_when"] for s in task["subtasks"]).lower()
    out = []
    for m in re.finditer(r"\b(?:in|into|onto|on|inside)\s+((?:the|a|an|its|one|each|next|empty|other)\s+)+"
                         r"([a-z-]+(?:\s+[a-z-]+)?)", text):
        words = []
        for w in m.group(2).split():
            if w in CUT:
                break
            words.append(w)
        if not words:
            continue
        noun = words[-1]
        if len(noun) < 3 or noun in C.NOT_OBJECTS or noun in C.GENERIC or noun in FUNCTION or noun in out:
            continue
        if noun.endswith("s") and noun[:-1] in out:
            continue
        out.append(noun)
    return out


NUM = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}


def _head(name):
    w = name.lower().split()[-1]
    return w[:-1] if w.endswith("s") and not w.endswith("ss") and len(w) > 3 else w


def expected_counts(task, pairs, size_split, per_object=None):
    """How many instances of each output category the task sheet implies, from its text only:
    - a row "<verb> one/a <object> ..." with count k: k instances (rows like that for the same object add up:
      "one grocery from Crate 1" x6 + "one grocery from Crate 2" x6 = 12)
    - the title: "Fold four towels", "Pair and fold six pairs of socks" (pairs count twice)
    - objects only ever named "the <object>" in rows whose count is 1: 1 (the insert, the wooden utensil)
    Never counted: "the day label" with count 4 (four labels), objects also called "the second / other / another /
    next <object>" when the count found is 1 ("the second bag").
    The estimate errs high on purpose: a cap that is too high only loses protection against duplicate ids, a cap
    that is too low would throw away real objects."""
    one, more, title, the_ok, the_bad = {}, set(), {}, set(), set()
    for s in task["subtasks"]:
        o = s["object"].strip().lower()
        if not o or o in ("–", "-") or o in C.GENERIC:
            continue
        h, text = _head(o), s["subtask"].lower()
        m = re.match(r"\s*(\d+)", s["count"])
        k = int(m.group(1)) if m else 0
        if s["count"].strip() != "1":
            the_bad.add(o)                           # EVERY row must say count=1, even if it omits "the"
        if re.search(rf"\b(one|a|an)\s+([a-z-]+\s+){{0,2}}{h}s?\b", text):
            if k > 0:
                one[o] = one.get(o, 0) + k
        if re.search(rf"\b(second|other|another|next|third|remaining)\s+([a-z-]+\s+){{0,2}}{h}s?\b", text):
            more.add(o)
        if re.search(rf"\bthe\s+([a-z-]+\s+){{0,2}}{h}s?\b", text):
            (the_ok if k == 1 and s["count"].strip() == "1" else the_bad).add(o)
    objs = {s["object"].strip().lower() for s in task["subtasks"]} - {"–", "-"}
    for m in re.finditer(r"\b(" + "|".join(NUM) + r"|\d+)\s+(pairs? of\s+)?([a-z-]+(?:\s+[a-z-]+)?)",
                         task["title"].lower()):
        n = int(m.group(1)) if m.group(1).isdigit() else NUM[m.group(1)]
        n *= 2 if m.group(2) else 1
        words = m.group(3).split()
        for o in objs | {c for _, c in pairs}:
            if _head(o) in [_head(w) for w in words]:
                title[o] = max(title.get(o, 0), n)
    per_obj = {}
    for o in objs | set(title):
        n = max(one.get(o, 0), title.get(o, 0))
        if n == 0 and o in the_ok and o not in the_bad and o not in more:
            n = 1
        if n == 1 and o in more:
            n = 0
        if n:
            per_obj[o] = n
    # object names -> output categories (an override may rename / merge them)
    cats = {c for _, c in pairs if c != "robot"}
    if per_object is not None:
        per_object.update(per_obj)
    split_of = {names[k].lower(): c for c, names in size_split.items() for k in ("small", "large")}
    out, summed = {}, {}
    for o, n in per_obj.items():
        if o in split_of:                                    # dinner spoon + teaspoon -> spoon (both exist)
            summed[split_of[o]] = summed.get(split_of[o], 0) + n
            continue
        c = o if o in cats else next((c for p, c in pairs if c != "robot" and _head(o) in p.lower().split()), None)
        if c is None:
            c = next((c for c in cats if _head(c) == _head(o)), None)
        if c is not None:
            out[c] = max(out.get(c, 0), n)                   # bundle = the same cable, so max, not sum
    for c, n in summed.items():
        out[c] = max(out.get(c, 0), n)
    return out


def manipulated_categories(task_sheets, episode_id, overrides_path="configs/overrides.json"):
    """categories that must be touched by an arm or move to count as task objects: the task sheet's object column
    (what the robot handles) and the generic containers added to every video. Exempt: containers / places the task
    text names as destinations (organizer, compartments ...), which matter because things are put in them."""
    task = find_task(task_sheets, episode_id)
    prompts, cats, split = build(task_sheets, episode_id, overrides_path)[:3]
    pairs = [(p, c) for p, c in zip(prompts, cats) if c != "robot"]
    catset = {c for _, c in pairs}
    split_of = {names[k].lower(): c for c, names in split.items() for k in ("small", "large")}
    out = set()
    for s in task["subtasks"]:
        o = s["object"].strip().lower()
        if not o or o in ("–", "-") or o in C.GENERIC:
            continue
        if o in split_of:
            out.add(split_of[o])
            continue
        c = o if o in catset else next((c for p, c in pairs if _head(o) in p.lower().split()), None)
        c = c or next((c for c in catset if _head(c) == _head(o)), None)
        if c:
            out.add(c)
    out |= {g for g in C.GENERIC_CONTAINERS if g in catset}    # unnamed containers: only if touched or moved
    return sorted(out)


def build(task_sheets, episode_id, overrides_path="configs/overrides.json", robot=C.TRACK_ROBOT):
    """-> prompts [str], category of each prompt [str], size_split {category: {small, large, n_small, n_large}},
    deformable, articulated, counts {category: expected instances, parsed from the task sheet text}"""
    task = find_task(task_sheets, episode_id)
    ov = {}
    if overrides_path and os.path.exists(overrides_path):
        allov = json.load(open(overrides_path))
        ov = allov.get(episode_id) or allov.get(task["task"]) or {}     # by task name: clips like *_15s match too
    if "prompts" in ov:
        pairs = [tuple(p) for p in ov["prompts"]]
    else:
        objs = []
        for s in task["subtasks"]:
            o = s["object"].strip()
            if o and o not in ("–", "-") and o not in C.GENERIC and o not in objs:
                objs.append(o)
        pairs = [(o, o) for o in objs] + [(r, r) for r in receptacles(task) if r not in objs]
    words = {w for p, c in pairs for w in (p.lower().split() + c.lower().split())}
    pairs += [(g, g) for g in C.GENERIC_CONTAINERS if g not in words and _head(g) not in words]
    if robot:
        pairs += [(p, "robot") for p in C.ROBOT_PROMPTS]
    prompts = [p for p, _ in pairs]
    cats = [c for _, c in pairs]
    split, per = ov.get("size_split", {}), {}
    counts = expected_counts(task, pairs, split, per)
    split = {c: dict(names, n_small=per.get(names["small"].lower()), n_large=per.get(names["large"].lower()))
             for c, names in split.items()}                  # "Place one dinner spoon" x2, "... one teaspoon" x2
    return prompts, cats, split, ov.get("deformable", []), ov.get("articulated", []), counts
