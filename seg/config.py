"""Tunable constants."""

# ---- models ----
DET_MODEL = "yolov8x-worldv2.pt"                  # YOLO-World, open vocabulary
SAM2_MODELS = {                                   # --sam2 <name>: (checkpoint, config name inside the sam2 package)
    "tiny": ("checkpoints/sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"),
    "small": ("checkpoints/sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
    "base_plus": ("checkpoints/sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    "large": ("checkpoints/sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
}

# ---- detection ----
PROBE_FRAMES = 16            # before tracking, YOLO-World runs on this many evenly spaced frames ...
ROUTE_MIN = 0.05             # ... a category none of whose prompts ever scores this much goes to Grounding DINO
GDINO_MODEL = "IDEA-Research/grounding-dino-base"
GDINO_BOX_THR = 0.20         # Grounding DINO box / text thresholds (its scores are not on YOLO-World's scale;
GDINO_TEXT_THR = 0.20        # 0.20-0.25 passes the tracker's weak-detection check, >= 0.25 starts ids directly)
DET_CONF = 0.10              # keep everything above this as a candidate
NEW_CONF = 0.25              # a detection must be at least this confident to start a NEW object id ...
WEAK_MEMORY = 200            # ... or (above DET_CONF) seen at the same place (IoU>=0.5) within this many frames
COVERED_FRAC = 0.5           # ONE existing same-kind object covers this much of the detection box -> same object

# ---- tracking ----
WINDOW = 50                  # frames per SAM 2 window (1 s at 50 fps); objects are re-seeded at each window start
ROI_MARGIN = 0.10            # detection range: the frame minus a band of this width on every side. New objects
                             # (not arms, which enter from the edges) must lie mostly inside it
MAX_OBJECT_FRAC = 0.35       # a non-robot mask covering more of the frame than this is the table / background
MIN_AREA = 30                # px; smaller masks count as "not visible"
BACKFILL = True              # propagate newly (re)appearing objects backwards through the previous window
TRACK_ROBOT = True           # segment the two arms as category 'robot' (they compete for pixels -> less leakage)
ROBOT_PROMPTS = ["robot gripper", "robot arm"]
# things the task sheet does not name but the robot may touch (where the straps are kept, ...): task-related by the
# task's own definition (touched / moved / things put in or on it). Added to every video unless a prompt already
# uses the word; kept in the output only if an arm touches it or it moves.
GENERIC_CONTAINERS = ["basket", "box", "tray", "bin", "container"]
ROBOT_JOIN_PX = 15           # a robot detection touching an existing arm mask (within this) extends that arm
ON_ROBOT_FRAC = 0.3          # an object candidate with more than this share of its mask on an arm is part of the arm

JUMP_OWN_IOU = 0.3           # a mask overlapping its own previous mask less than this ...
JUMP_ONTO = 0.6              # ... while lying this much on another same-category object = SAM 2 jumped targets
APPEAR_MIN = 0.4             # colour-histogram similarity to the object's own record below this = swapped ids
APPEAR_MIN_AREA = 4000       # px, median size over the track; smaller objects (cutlery) skip the colour checks:
                             # their histograms are noise. (Per frame it would also skip a towel mid-swap.)
GLUE_RATIO = 1.6             # host mask this much bigger than a short category's detection inside it ...
GLUE_PERSIST = 3             # ... in this many windows at the same place: the host holds two objects, split
MIN_TRAVEL_PX = 40           # a manipulated object (task sheet object column) that never moves this far ...
TOUCH_PX = 10                # ... and that no arm mask ever comes within this many px of is not a task object
MOTION_EVERY = 10            # frames between motion / contact samples
CONTAINER_RATIO = 4.0        # a mask this many times bigger than a detection, of another category, holds it
SWAP_SUSPECT_IOU = 0.5       # a frame where an object's mask changed this much (IoU with previous) gets a colour check

# ---- re-identification after occlusion ----
REID_ACCEPT = 0.35           # score above which a reappearing detection takes a lost object's id
REID_FORCE = 0.10            # when the category is already at its cap, accept a weaker match instead of a new id
REID_DIST_PX = 200.0         # spatial scale of the distance term
COUNT_SLACK = 0.25           # tracker id budget for counted categories = task sheet total * (1 + this), at least +1
CAP_SLACK = 1                # max ids per category = count seen in the first window + CAP_SLACK
CAP_UNSEEN = 4               # categories not seen in the first window (e.g. hidden in a box at the start)

# ---- finalize ----
NOTE_COLOUR = False          # colour words in the note: off, the mean colour of a mask is not reliable (metal, light)
MIN_TRACK_S = 0.5            # drop ids visible for less than this (spurious detections)
VERIFY_SAMPLES = 8           # CLIP second opinion: crops per object
VERIFY_MIN_OBJECT = 0.4      # mean probability of "one of the task objects" below this -> not an object, dropped
VERIFY_WEIGHT = 0.8          # category = (1 - w) * detector votes + w * CLIP (both normalised). CLIP looks at 8 whole-
                             # object crops; YOLO-World's per-box class on small cutlery was 'spoon' for every
                             # utensil in 26, while CLIP's labels matched the task sheet counts (4 / 2 / 1 / 1)
DUP_MERGE_CLIP = 0.5         # an id over the count with a CLIP object score below this is dropped, not merged
DUP_SCORE = 0.05             # a category scoring below this is not an option for a track (finalize assignment)
UNSTABLE_IOU = 0.6           # a track whose median frame-to-frame mask IoU is below this ...
UNSTABLE_FRAG = 0.5          # ... and that is in >3 pieces in this share of its frames is not a coherent object
                             # (typically the leftover slivers of a duplicate id): its pixels join the touching
                             # same-category object, or are dropped

DEFORMABLE = ["towel", "cloth", "cable", "strap", "bundle", "bag", "bubble", "wrap", "garment", "t-shirt", "shirt",
              "pants", "sock", "film", "lace", "liner", "tape", "label", "backing", "flower", "foam", "pouch",
              "pad", "band", "bandage", "wipe", "sheet", "ham", "slice"]
ARTICULATED = ["drawer", "scissors", "lid", "suitcase", "kit box", "lunchbox", "pencil case", "tongs", "backpack",
               "box", "caddy", "shoe", "pump"]

# generic task-sheet names that make useless open-vocabulary prompts
GENERIC = {"half", "setting", "pair", "item", "supply", "nest", "piece", "stack", "stacks", "bins", "pile"}
# nouns after "in/into/onto the ..." that are places, not objects
NOT_OBJECTS = {"area", "zone", "location", "spot", "side", "table", "middle", "center", "edge", "corner", "place",
               "position", "front", "back", "left", "right", "half", "top", "bottom", "end", "way", "air", "turn",
               "gripper", "grippers", "hand", "hold", "row", "line", "order", "space", "view", "lamp", "camera",
               "towel", "fold", "pile", "work", "working", "finished", "home"}
