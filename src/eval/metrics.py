"""
Evaluation metrics for E0-R
============================
Computes: BLEU-1..4, METEOR, ROUGE-L, CIDEr, CHAIR_s, CHAIR_i

NO SPICE.

Dependencies
------------
    pip install pycocoevalcap pycocotools nltk

CHAIR implementation follows the original paper:
  Rohrbach et al., "Object Hallucination in Image Captioning", EMNLP 2018.

COCO 80-class object vocabulary is bundled here and augmented with
synonyms from the COCO panoptic annotations (the same list used in the
official CHAIR code).
"""

from __future__ import annotations

import json
import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# COCO 80 object vocabulary  +  synonyms used by CHAIR
# (canonical name → set of surface forms that count as the same object)
# ---------------------------------------------------------------------------

COCO_OBJECTS: dict[str, list[str]] = {
    "person":       ["person", "people", "man", "woman", "child", "boy", "girl",
                     "kid", "human", "rider", "pedestrian"],
    "bicycle":      ["bicycle", "bike", "cycle"],
    "car":          ["car", "automobile", "vehicle", "sedan", "cab", "taxi"],
    "motorcycle":   ["motorcycle", "motorbike", "moped", "scooter"],
    "airplane":     ["airplane", "aeroplane", "plane", "aircraft", "jet"],
    "bus":          ["bus", "coach", "minibus"],
    "train":        ["train", "locomotive", "railway"],
    "truck":        ["truck", "lorry", "van"],
    "boat":         ["boat", "ship", "vessel", "canoe", "kayak", "yacht", "sailboat"],
    "traffic light":["traffic light", "traffic signal", "stoplight"],
    "fire hydrant": ["fire hydrant", "hydrant"],
    "stop sign":    ["stop sign"],
    "parking meter":["parking meter", "meter"],
    "bench":        ["bench", "seat"],
    "bird":         ["bird", "crow", "sparrow", "seagull", "duck", "pigeon"],
    "cat":          ["cat", "kitten", "kitty"],
    "dog":          ["dog", "puppy", "pup", "hound"],
    "horse":        ["horse", "pony", "stallion", "mare"],
    "sheep":        ["sheep", "lamb", "ram", "ewe"],
    "cow":          ["cow", "cattle", "bull", "calf", "ox"],
    "elephant":     ["elephant"],
    "bear":         ["bear", "grizzly", "polar bear", "panda"],
    "zebra":        ["zebra"],
    "giraffe":      ["giraffe"],
    "backpack":     ["backpack", "rucksack", "knapsack", "bag"],
    "umbrella":     ["umbrella", "parasol"],
    "handbag":      ["handbag", "purse", "clutch"],
    "tie":          ["tie", "necktie", "bowtie", "bow tie"],
    "suitcase":     ["suitcase", "luggage", "baggage", "case"],
    "frisbee":      ["frisbee", "disc"],
    "skis":         ["skis", "ski"],
    "snowboard":    ["snowboard"],
    "sports ball":  ["ball", "football", "soccer ball", "baseball", "basketball",
                     "tennis ball", "volleyball", "rugby ball"],
    "kite":         ["kite"],
    "baseball bat": ["baseball bat", "bat"],
    "baseball glove":["baseball glove", "glove", "mitt"],
    "skateboard":   ["skateboard", "board"],
    "surfboard":    ["surfboard", "board"],
    "tennis racket":["tennis racket", "racket", "racquet"],
    "bottle":       ["bottle", "flask", "jar"],
    "wine glass":   ["wine glass", "glass", "goblet"],
    "cup":          ["cup", "mug", "tumbler"],
    "fork":         ["fork"],
    "knife":        ["knife", "blade", "cleaver"],
    "spoon":        ["spoon", "ladle"],
    "bowl":         ["bowl", "dish"],
    "banana":       ["banana"],
    "apple":        ["apple"],
    "sandwich":     ["sandwich", "burger", "sub", "hoagie"],
    "orange":       ["orange", "tangerine", "clementine"],
    "broccoli":     ["broccoli"],
    "carrot":       ["carrot"],
    "hot dog":      ["hot dog", "hotdog", "sausage", "frankfurter"],
    "pizza":        ["pizza", "pie"],
    "donut":        ["donut", "doughnut"],
    "cake":         ["cake", "cupcake", "torte"],
    "chair":        ["chair", "seat", "stool", "armchair"],
    "couch":        ["couch", "sofa", "settee", "loveseat"],
    "potted plant": ["plant", "potted plant", "houseplant", "flower pot"],
    "bed":          ["bed", "mattress", "cot", "bunk"],
    "dining table": ["table", "dining table", "desk"],
    "toilet":       ["toilet", "lavatory", "commode", "wc"],
    "tv":           ["tv", "television", "monitor", "screen", "display"],
    "laptop":       ["laptop", "notebook", "computer"],
    "mouse":        ["mouse", "computer mouse"],
    "remote":       ["remote", "remote control"],
    "keyboard":     ["keyboard"],
    "cell phone":   ["cell phone", "phone", "mobile", "smartphone", "iphone"],
    "microwave":    ["microwave", "microwave oven"],
    "oven":         ["oven", "stove", "range"],
    "toaster":      ["toaster"],
    "sink":         ["sink", "basin"],
    "refrigerator": ["refrigerator", "fridge", "freezer"],
    "book":         ["book", "novel", "textbook", "magazine"],
    "clock":        ["clock", "watch", "timer"],
    "vase":         ["vase", "pot", "urn"],
    "scissors":     ["scissors", "shears"],
    "teddy bear":   ["teddy bear", "teddy", "stuffed animal", "plush"],
    "hair drier":   ["hair drier", "hair dryer", "dryer", "blow dryer"],
    "toothbrush":   ["toothbrush"],
}

# Flat token set for fast lookup  (lowercased surface forms → canonical)
_SURFACE_TO_CANONICAL: dict[str, str] = {}
for _canon, _surfaces in COCO_OBJECTS.items():
    for _s in _surfaces:
        _SURFACE_TO_CANONICAL[_s.lower()] = _canon


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PUNCT = str.maketrans("", "", string.punctuation)

def _tokenise(text: str) -> list[str]:
    return text.lower().translate(_PUNCT).split()


def _objects_in_text(text: str) -> set[str]:
    """Return set of canonical COCO object names mentioned in *text*."""
    tokens = _tokenise(text)
    found: set[str] = set()
    # Check 1-gram and 2-gram surface forms
    for i, tok in enumerate(tokens):
        if tok in _SURFACE_TO_CANONICAL:
            found.add(_SURFACE_TO_CANONICAL[tok])
        if i < len(tokens) - 1:
            bigram = tok + " " + tokens[i + 1]
            if bigram in _SURFACE_TO_CANONICAL:
                found.add(_SURFACE_TO_CANONICAL[bigram])
    return found


# ---------------------------------------------------------------------------
# CHAIR
# ---------------------------------------------------------------------------

def compute_chair(
    predictions: list[dict],
    annotations_file: str,
) -> dict[str, float]:
    """
    Compute CHAIR_s and CHAIR_i.

    Parameters
    ----------
    predictions      : list of {"image_id": int, "caption": str}
    annotations_file : path to captions_val2017.json

    Returns
    -------
    {"chair_s": float, "chair_i": float}
    """
    # ── Load ground-truth captions ─────────────────────────────────
    with open(annotations_file) as f:
        gt_data = json.load(f)

    gt_objects: dict[int, set[str]] = defaultdict(set)
    for ann in gt_data["annotations"]:
        for obj in _objects_in_text(ann["caption"]):
            gt_objects[ann["image_id"]].add(obj)

    # ── Compute CHAIR ──────────────────────────────────────────────
    hallucinated_sentences = 0
    hallucinated_objects   = 0
    total_sentences        = 0
    total_objects          = 0

    for pred in predictions:
        iid     = pred["image_id"]
        caption = pred["caption"]

        pred_objs = _objects_in_text(caption)
        gt_objs   = gt_objects.get(iid, set())

        hallucinated = pred_objs - gt_objs

        total_sentences += 1
        if hallucinated:
            hallucinated_sentences += 1

        total_objects          += len(pred_objs)
        hallucinated_objects   += len(hallucinated)

    chair_s = hallucinated_sentences / max(total_sentences, 1)
    chair_i = hallucinated_objects   / max(total_objects,   1)

    return {
        "chair_s": round(chair_s * 100, 4),   # percentage
        "chair_i": round(chair_i * 100, 4),
    }


# ---------------------------------------------------------------------------
# Standard metrics via pycocoevalcap
# ---------------------------------------------------------------------------

def compute_metrics(
    predictions: list[dict],
    annotations_file: str,
) -> dict[str, float]:
    """
    Compute BLEU-1..4, METEOR, ROUGE-L, CIDEr using pycocoevalcap.

    Parameters
    ----------
    predictions      : list of {"image_id": int, "caption": str}
    annotations_file : path to captions_val2017.json

    Returns
    -------
    dict with keys: bleu_1, bleu_2, bleu_3, bleu_4, meteor, rouge_l, cider
    """
    try:
        from pycocotools.coco import COCO
        from pycocoevalcap.eval import COCOEvalCap
    except ImportError as e:
        raise ImportError(
            "pycocoevalcap and pycocotools are required for metric computation.\n"
            "Install with: pip install pycocoevalcap pycocotools"
        ) from e

    coco_gt   = COCO(annotations_file)
    coco_dt   = coco_gt.loadRes(predictions)
    evaluator = COCOEvalCap(coco_gt, coco_dt)
    evaluator.params["image_id"] = coco_dt.getImgIds()
    evaluator.evaluate()

    raw = {k.lower(): v for k, v in evaluator.eval.items()}

    # pycocoevalcap keys: Bleu_1..4, METEOR, ROUGE_L, CIDEr
    return {
        "bleu_1"  : round(raw.get("bleu_1", 0.0) * 100, 4),
        "bleu_2"  : round(raw.get("bleu_2", 0.0) * 100, 4),
        "bleu_3"  : round(raw.get("bleu_3", 0.0) * 100, 4),
        "bleu_4"  : round(raw.get("bleu_4", 0.0) * 100, 4),
        "meteor"  : round(raw.get("meteor", 0.0)  * 100, 4),
        "rouge_l" : round(raw.get("rouge_l", 0.0) * 100, 4),
        "cider"   : round(raw.get("cider",   0.0),       4),
    }


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------

def evaluate_all(
    predictions: list[dict],
    annotations_file: str,
) -> dict[str, float]:
    """Run compute_metrics + compute_chair and return merged dict."""
    metrics = compute_metrics(predictions, annotations_file)
    chair   = compute_chair(predictions, annotations_file)
    return {**metrics, **chair}
