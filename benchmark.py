"""
Benchmark dog_detector's accuracy on two public datasets, each used for what it
measures best:

  COCO val2017      -> DETECTION. Real-world scenes, often several dogs, carefully
                       labeled boxes. Held out from the detector's training data.
                       Reports dog AP50 (the standard detection score), recall,
                       precision and false dogs per image.
  Oxford-IIIT Pet   -> BREED. 3,669 test photos of 25 dog and 12 cat breeds, not
  (test split)         taken from ImageNet, so the classifier hasn't seen them.
                       Reports end-to-end breed accuracy (detector box -> breed, what
                       the app does), full-image accuracy (isolates the classifier),
                       confidence calibration, and cats mistaken for dogs.

Accuracy percentages come with 95% confidence intervals, so you can tell whether
a change is a real improvement or noise.

Downloads on first run, cached in ./data:
  Oxford-IIIT Pet test split from Hugging Face (timm/oxford-iiit-pet), ~430 MB
  COCO val2017 annotations (~250 MB, only the val file is kept) + images (~780 MB)

Usage:
    python benchmark.py                          # both datasets, full (most accurate)
    python benchmark.py --limit 300              # quick check: subset of each dataset
    python benchmark.py --datasets oxford        # just one dataset
    python benchmark.py --accurate               # benchmark the larger models
    python benchmark.py --classifier accurate --det-size 640 --proposals 300   # mix + speed settings
    python benchmark.py --crop-pad 0.15          # try a different crop padding
    python benchmark.py --checkpoint finetuned.pt   # evaluate fine-tuned classifier
    python benchmark.py --baseline benchmark_results/<run>/summary.json   # show deltas
    python benchmark.py --no-calibration         # raw confidences (see calibrate.py)

Each run writes summary.json plus per-image CSVs to benchmark_results/<timestamp>/.
"""

import argparse
import csv
import io
import json
import math
import random
import shutil
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from dog_detector import (COCO_DOG_LABEL, DogBreedDetector, add_model_args, model_kwargs,
                          pick_device)

# --- Dataset sources ----------------------------------------------------------
# Pinned to a specific revision so results stay comparable over time.
OXFORD_URL = ("https://huggingface.co/datasets/timm/oxford-iiit-pet/resolve/"
              "089695c834a7deb60505b7cc506672db1c31a6aa/data/{split}-00000-of-00001.parquet")
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_ANN_MEMBER = "annotations/instances_val2017.json"
COCO_IMG_URL = "http://images.cocodataset.org/val2017/{}"

# Oxford-IIIT Pet dog breed -> ImageNet class name(s) counted as correct.
# Pit bull / Staffordshire labels are genuinely ambiguous, so either is accepted.
# Breeds with no ImageNet equivalent (NEW_OXFORD_BREEDS) still count for dog-vs-cat
# detection but stay out of the headline breed accuracy, so it remains comparable
# between models. If train_breeds.py added them, they're reported separately.
OXFORD_TO_IMAGENET = {
    "American Pit Bull Terrier": ["American Staffordshire terrier", "Staffordshire bullterrier"],
    "Basset Hound": ["basset"],
    "Beagle": ["beagle"],
    "Boxer": ["boxer"],
    "Chihuahua": ["Chihuahua"],
    "English Cocker Spaniel": ["cocker spaniel"],
    "English Setter": ["English setter"],
    "German Shorthaired": ["German short-haired pointer"],
    "Great Pyrenees": ["Great Pyrenees"],
    "Japanese Chin": ["Japanese spaniel"],
    "Keeshond": ["keeshond"],
    "Leonberger": ["Leonberg"],
    "Miniature Pinscher": ["miniature pinscher"],
    "Newfoundland": ["Newfoundland"],
    "Pomeranian": ["Pomeranian"],
    "Pug": ["pug"],
    "Saint Bernard": ["Saint Bernard"],
    "Samoyed": ["Samoyed"],
    "Scottish Terrier": ["Scotch terrier"],
    "Staffordshire Bull Terrier": ["Staffordshire bullterrier", "American Staffordshire terrier"],
    "Wheaten Terrier": ["soft-coated wheaten terrier"],
    "Yorkshire Terrier": ["Yorkshire terrier"],
}
NEW_OXFORD_BREEDS = ["American Bulldog", "Havanese", "Shiba Inu"]
OXFORD_CAT_BREEDS = {"Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair",
                     "Egyptian Mau", "Maine Coon", "Persian", "Ragdoll", "Russian Blue",
                     "Siamese", "Sphynx"}

IOU_MATCH = 0.5
SWEEP_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CALIBRATION_BINS = 10


# --- Small helpers ------------------------------------------------------------

def download(url: str, dest: Path) -> Path:
    """Download url to dest (atomically, with progress) unless it already exists."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        done, last = 0, 0.0
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if time.time() - last > 2:
                last = time.time()
                print(f"  {done / 1e6:.0f} / {total / 1e6:.0f} MB" if total else f"  {done / 1e6:.0f} MB")
    tmp.rename(dest)
    return dest


def box_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def inside_fraction(det, region) -> float:
    """Fraction of det's area that lies inside region (used for COCO crowd boxes)."""
    ix1, iy1 = max(det[0], region[0]), max(det[1], region[1])
    ix2, iy2 = min(det[2], region[2]), min(det[3], region[3])
    area = (det[2] - det[0]) * (det[3] - det[1])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1) / area if area > 0 else 0.0


def wilson(k: int, n: int) -> dict | None:
    """Percentage with a 95% Wilson confidence interval."""
    if n == 0:
        return None
    z, p = 1.96, k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {"pct": round(100 * p, 2), "ci95": [round(100 * (centre - half), 2),
                                               round(100 * (centre + half), 2)], "n": n}


def calibration(pairs) -> dict:
    """pairs: [(confidence, correct)]. Accuracy per confidence bin + expected calibration error."""
    if not pairs:
        return {"ece_pct": None, "bins": []}
    bins = defaultdict(list)
    for conf, ok in pairs:
        bins[min(int(conf * CALIBRATION_BINS), CALIBRATION_BINS - 1)].append((conf, ok))
    rows, ece = [], 0.0
    for b in sorted(bins):
        items = bins[b]
        mean_conf = sum(c for c, _ in items) / len(items)
        acc = sum(ok for _, ok in items) / len(items)
        ece += len(items) / len(pairs) * abs(acc - mean_conf)
        rows.append({"bin": f"{b / CALIBRATION_BINS:.1f}-{(b + 1) / CALIBRATION_BINS:.1f}",
                     "count": len(items), "mean_conf_pct": round(100 * mean_conf, 1),
                     "accuracy_pct": round(100 * acc, 1)})
    return {"ece_pct": round(100 * ece, 2), "bins": rows}


def build_label_map(model: DogBreedDetector) -> dict[str, set[int]]:
    """Oxford breed -> set of indices into model.breed_names. Fails loudly on typos."""
    lookup = {name.lower(): i for i, name in enumerate(model.breed_names)}
    label_map = {}
    for oxford, imagenet_names in OXFORD_TO_IMAGENET.items():
        missing = [n for n in imagenet_names if n.lower() not in lookup]
        if missing:
            raise SystemExit(f"Label map error: {missing} not among ImageNet dog classes")
        label_map[oxford] = {lookup[n.lower()] for n in imagenet_names}
    return label_map


def new_breed_map(model: DogBreedDetector) -> dict[str, set[int]]:
    """Oxford breeds outside the headline number that this model can name."""
    return {b: {model.breed_names.index(b)} for b in NEW_OXFORD_BREEDS if b in model.breed_names}


class Timer:
    def __init__(self):
        self.total, self.calls = 0.0, 0

    def __enter__(self):
        self.t0 = time.perf_counter()

    def __exit__(self, *exc):
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        self.total += time.perf_counter() - self.t0
        self.calls += 1

    def ms(self):
        return round(1000 * self.total / self.calls, 1) if self.calls else None


# --- Datasets -----------------------------------------------------------------

def load_oxford(data_dir: Path, limit: int | None, seed: int, split: str = "test"):
    """Yields (image_id, PIL image, breed name, is_dog).

    split="test" is for benchmarking; split="train" (Oxford's trainval) is for fitting
    anything - calibration, fine-tuning - so the test numbers stay honest.
    """
    path = download(OXFORD_URL.format(split=split), data_dir / "oxford-iiit-pet-hf" / f"{split}.parquet")
    table = pq.read_table(path)
    meta = json.loads(table.schema.metadata[b"huggingface"])
    raw_names = meta["info"]["features"]["label"]["names"]
    names = [" ".join(p.title() for p in n.split("_")) for n in raw_names]
    unknown = set(names) - set(OXFORD_TO_IMAGENET) - OXFORD_CAT_BREEDS - set(NEW_OXFORD_BREEDS)
    if unknown:
        raise SystemExit(f"Unexpected Oxford breed names: {unknown}")

    indices = list(range(table.num_rows))
    if limit:
        indices = random.Random(seed).sample(indices, min(limit, len(indices)))
    images, labels, ids = table.column("image"), table.column("label"), table.column("image_id")
    for i in indices:
        breed = names[labels[i].as_py()]
        img = Image.open(io.BytesIO(images[i].as_py()["bytes"])).convert("RGB")
        yield ids[i].as_py(), img, breed, breed not in OXFORD_CAT_BREEDS


def load_coco(data_dir: Path, limit: int | None, seed: int):
    """Yields (file name, PIL image, dog boxes, crowd dog boxes), boxes as [x1,y1,x2,y2]."""
    root = data_dir / "coco"
    ann_path = root / "instances_val2017.json"
    if not ann_path.exists():
        zip_path = download(COCO_ANN_URL, root / "annotations_trainval2017.zip")
        with zipfile.ZipFile(zip_path) as z, z.open(COCO_ANN_MEMBER) as src, open(ann_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        zip_path.unlink()              # the other annotation files aren't needed
    coco = json.loads(ann_path.read_text())

    dogs, crowds = defaultdict(list), defaultdict(list)
    for a in coco["annotations"]:
        if a["category_id"] == COCO_DOG_LABEL:   # torchvision uses COCO's category ids
            x, y, w, h = a["bbox"]
            (crowds if a["iscrowd"] else dogs)[a["image_id"]].append([x, y, x + w, y + h])
    files = {im["id"]: im["file_name"] for im in coco["images"]}
    ids = sorted(files)
    if limit:
        # Keep the dog / no-dog mix representative: sample from each group in proportion.
        rng = random.Random(seed)
        with_dogs = [i for i in ids if i in dogs]
        without = [i for i in ids if i not in dogs]
        k = max(1, round(limit * len(with_dogs) / len(ids)))
        ids = sorted(rng.sample(with_dogs, min(k, len(with_dogs))) +
                     rng.sample(without, min(limit - k, len(without))))

    img_dir = root / "val2017"
    img_dir.mkdir(parents=True, exist_ok=True)
    missing = [files[i] for i in ids if not (img_dir / files[i]).exists()]
    if missing:
        print(f"Downloading {len(missing)} COCO images...")
        with ThreadPoolExecutor(16) as pool:
            for n, _ in enumerate(pool.map(
                    lambda f: download_quiet(COCO_IMG_URL.format(f), img_dir / f), missing), 1):
                if n % 500 == 0 or n == len(missing):
                    print(f"  {n}/{len(missing)}")
    for i in ids:
        img = Image.open(img_dir / files[i]).convert("RGB")
        yield files[i], img, dogs.get(i, []), crowds.get(i, [])


def download_quiet(url: str, dest: Path):
    tmp = dest.with_suffix(".part")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as f:
                shutil.copyfileobj(resp, f)
            tmp.rename(dest)
            return
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2)


# --- Evaluation -----------------------------------------------------------------

def coco_ap50(det_records, n_gt: int) -> float | None:
    """COCO-style 101-point interpolated average precision at IoU 0.5."""
    if n_gt == 0:
        return None
    recs = sorted((r for r in det_records if r["match"] != "ignore"), key=lambda r: -r["score"])
    tp = fp = 0
    precision, recall = [], []
    for r in recs:
        tp += r["match"] == "tp"
        fp += r["match"] == "fp"
        precision.append(tp / (tp + fp))
        recall.append(tp / n_gt)
    # Make precision monotonically decreasing, then sample at 101 recall points.
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    ap, j = 0.0, 0
    for t in [k / 100 for k in range(101)]:
        while j < len(recall) and recall[j] < t:
            j += 1
        ap += precision[j] if j < len(recall) else 0.0
    return round(100 * ap / 101, 2)


def eval_coco(model, args, det_timer):
    per_image, det_records, n_gt = [], [], 0
    start = time.time()
    for n, (name, img, gts, crowds) in enumerate(load_coco(args.data_dir, args.limit, args.seed), 1):
        with det_timer:
            dets = model.detect_dogs(TF.to_tensor(img))        # scores >= 0.05
        dets.sort(key=lambda d: -d[1])                          # highest score first
        n_gt += len(gts)
        matched = [False] * len(gts)
        for box, score in dets:                                 # greedy, highest score first
            best, best_j = 0.0, -1
            for j, gt in enumerate(gts):
                if not matched[j] and (iou := box_iou(box, gt)) > best:
                    best, best_j = iou, j
            if best >= IOU_MATCH:
                matched[best_j] = True
                det_records.append({"image": name, "score": score, "match": "tp", "iou": best})
            elif any(inside_fraction(box, c) >= IOU_MATCH for c in crowds):
                det_records.append({"image": name, "score": score, "match": "ignore", "iou": 0.0})
            else:
                det_records.append({"image": name, "score": score, "match": "fp", "iou": best})
        per_image.append({"image": name, "dogs": len(gts), "has_dog": bool(gts),
                          "max_dog_score": round(max((s for _, s in dets), default=0.0), 4)})
        if n % 500 == 0:
            print(f"  COCO {n} images ({time.time() - start:.0f}s)")

    def at(t):
        kept = [r for r in det_records if r["score"] >= t]
        tp = sum(r["match"] == "tp" for r in kept)
        fp = sum(r["match"] == "fp" for r in kept)
        no_dog = [p for p in per_image if not p["has_dog"]]
        return {
            "recall": wilson(tp, n_gt),
            "precision": wilson(tp, tp + fp),
            "false_dogs_per_100_images": round(100 * fp / len(per_image), 2),
            "no_dog_images_with_false_dog": wilson(sum(p["max_dog_score"] >= t for p in no_dog),
                                                   len(no_dog)),
            "mean_iou_of_hits": round(sum(r["iou"] for r in kept if r["match"] == "tp") / tp, 3) if tp else None,
        }

    summary = {
        "images": len(per_image), "images_with_dogs": sum(p["has_dog"] for p in per_image),
        "dogs": n_gt,
        "ap50": coco_ap50(det_records, n_gt),
        "at_threshold": at(args.threshold),
        "threshold_sweep": [{"threshold": t, **{k: v for k, v in at(t).items()
                                                 if k in ("recall", "precision", "false_dogs_per_100_images")}}
                            for t in SWEEP_THRESHOLDS],
    }
    return summary, per_image


def eval_oxford(model, args, label_map, det_timer, cls_timer):
    rows = []
    start = time.time()
    for n, (image_id, img, breed, is_dog) in enumerate(load_oxford(args.data_dir, args.limit, args.seed), 1):
        rgb = TF.to_tensor(img)
        with det_timer:
            dets = model.detect_dogs(rgb)
        dets.sort(key=lambda d: -d[1])
        shown = [(b, s) for b, s in dets if s >= args.threshold]   # what the app would draw
        row = {"image": image_id, "breed": breed, "is_dog": is_dog,
               "max_dog_score": round(max((s for _, s in dets), default=0.0), 4),
               "dog_detected": bool(shown)}

        if is_dog and breed in label_map:
            valid = label_map[breed]
            _, H, W = rgb.shape
            boxes = [[0.0, 0.0, float(W), float(H)]] + ([shown[0][0]] if shown else [])
            with cls_timer:
                probs = model.breed_probs(rgb, boxes).cpu()
            top5 = probs[0].topk(5).indices.tolist()
            row.update({"full_pred": model.breed_names[top5[0]], "full_top1": top5[0] in valid,
                        "full_top5": any(i in valid for i in top5),
                        "full_conf": round(probs[0, top5[0]].item(), 4)})
            if shown:
                pred = probs[1].argmax().item()
                row.update({"e2e_pred": model.breed_names[pred], "e2e_top1": pred in valid,
                            "e2e_conf": round(probs[1, pred].item(), 4)})
            else:
                row["e2e_top1"] = False      # no box drawn -> the app shows nothing
        rows.append(row)
        if n % 500 == 0:
            print(f"  Oxford {n} images ({time.time() - start:.0f}s)")

    dogs = [r for r in rows if r["is_dog"]]
    cats = [r for r in rows if not r["is_dog"]]
    scored = [r for r in dogs if "full_top1" in r and r["breed"] in OXFORD_TO_IMAGENET]
    new = [r for r in dogs if "full_top1" in r and r["breed"] not in OXFORD_TO_IMAGENET]

    per_breed, confusions = {}, Counter()
    for breed in sorted({r["breed"] for r in scored}):
        br = [r for r in scored if r["breed"] == breed]
        per_breed[breed] = wilson(sum(r["full_top1"] for r in br), len(br))
        confusions.update((breed, r["full_pred"]) for r in br if not r["full_top1"])

    summary = {
        "images": len(rows), "dogs": len(dogs), "cats": len(cats),
        "dogs_scored_for_breed": len(scored),
        "breed_top1_end_to_end": wilson(sum(r["e2e_top1"] for r in scored), len(scored)),
        "breed_top1_full_image": wilson(sum(r["full_top1"] for r in scored), len(scored)),
        "breed_top5_full_image": wilson(sum(r["full_top5"] for r in scored), len(scored)),
        "new_breeds_end_to_end": {b: wilson(sum(r["e2e_top1"] for r in new if r["breed"] == b),
                                            sum(r["breed"] == b for r in new))
                                  for b in sorted({r["breed"] for r in new})},
        "dogs_detected": wilson(sum(r["dog_detected"] for r in dogs), len(dogs)),
        "cats_called_dogs": wilson(sum(r["dog_detected"] for r in cats), len(cats)),
        # Calibration of the % the app shows next to the breed.
        "confidence": calibration([(r["e2e_conf"], r["e2e_top1"]) for r in scored if "e2e_conf" in r]),
        "threshold_sweep": [{"threshold": t,
                             "dogs_detected": wilson(sum(r["max_dog_score"] >= t for r in dogs), len(dogs)),
                             "cats_called_dogs": wilson(sum(r["max_dog_score"] >= t for r in cats), len(cats))}
                            for t in SWEEP_THRESHOLDS],
        "per_breed_top1_full_image": per_breed,
        "top_confusions": [{"true": t, "predicted": p, "count": c}
                           for (t, p), c in confusions.most_common(10)],
    }
    return summary, rows


# --- Reporting --------------------------------------------------------------------

# (label, path into summary, higher is better)
HEADLINE = [
    ("Breed accuracy, end-to-end", ("oxford", "breed_top1_end_to_end", "pct"), True),
    ("Breed accuracy, full image", ("oxford", "breed_top1_full_image", "pct"), True),
    ("Breed top-5, full image", ("oxford", "breed_top5_full_image", "pct"), True),
    ("Dogs detected (Oxford)", ("oxford", "dogs_detected", "pct"), True),
    ("Cats called dogs (Oxford)", ("oxford", "cats_called_dogs", "pct"), False),
    ("Confidence error (ECE)", ("oxford", "confidence", "ece_pct"), False),
    ("Dog AP50 (COCO)", ("coco", "ap50"), True),
    ("Dog recall (COCO)", ("coco", "at_threshold", "recall", "pct"), True),
    ("Dog precision (COCO)", ("coco", "at_threshold", "precision", "pct"), True),
    ("False dogs / 100 imgs (COCO)", ("coco", "at_threshold", "false_dogs_per_100_images"), False),
    ("Detector ms/image", ("speed_ms", "detector"), False),
    ("Classifier ms/batch", ("speed_ms", "classifier"), False),
]


def get(d, path):
    for k in path:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def fmt(w):
    return "n/a" if w is None else f"{w['pct']:.1f}"


def print_report(s: dict, baseline: dict | None):
    r = s["run"]
    print(f"\n=== Benchmark: {r['models']}, threshold {r['threshold']}, crop pad {r['crop_pad']}"
          + (f", checkpoint {r['checkpoint']}" if r["checkpoint"] else "")
          + f", temperature {r.get('temperature', 1.0):.3f}"
          + (f", limit {r['limit']}" if r["limit"] else "") + " ===")
    if "oxford" in s:
        o = s["oxford"]
        print(f"Oxford-IIIT Pet: {o['images']} images ({o['dogs']} dogs, "
              f"{o['dogs_scored_for_breed']} with an ImageNet breed, {o['cats']} cats)")
    if "coco" in s:
        c = s["coco"]
        print(f"COCO val2017:    {c['images']} images ({c['images_with_dogs']} with dogs, {c['dogs']} dogs)")
    print()

    for label, path, higher_better in HEADLINE:
        val = get(s, path)
        if val is None and get(s, path[:1]) is None:
            continue                                   # dataset not run
        line = f"  {label:<30}{'n/a' if val is None else val:>7}"
        ci = get(s, path[:-1] + ("ci95",)) if path[-1] == "pct" else None
        line += f"  [{ci[0]:.1f}-{ci[1]:.1f}]" if ci else " " * 14
        old = get(baseline, path) if baseline else None
        if val is not None and old is not None:
            delta = val - old
            mark = "" if abs(delta) < 1e-9 else ("better" if (delta > 0) == higher_better else "worse")
            line += f"  was {old:<7} {delta:+.2f} {mark}"
        print(line)
    print("  (brackets = 95% confidence interval)")

    if "coco" in s:
        print("\n  Detection threshold sweep (COCO) - pick the trade-off you want for --threshold:")
        print(f"    {'thresh':>6}  {'recall %':>8}  {'precision %':>11}  {'false dogs/100 imgs':>19}")
        for row in s["coco"]["threshold_sweep"]:
            print(f"    {row['threshold']:>6}  {fmt(row['recall']):>8}  {fmt(row['precision']):>11}  "
                  f"{row['false_dogs_per_100_images']:>19}")

    if "oxford" in s:
        o = s["oxford"]
        print("\n  Confidence calibration (Oxford) - shown % vs. how often it's actually right:")
        for b in o["confidence"]["bins"]:
            print(f"    {b['bin']}: n={b['count']:<5} says {b['mean_conf_pct']:>5}%  right {b['accuracy_pct']:>5}%")
        weakest = sorted(o["per_breed_top1_full_image"].items(), key=lambda kv: kv[1]["pct"])[:5]
        print("\n  Weakest breeds (full-image top-1):")
        for breed, w in weakest:
            print(f"    {breed:<28}{w['pct']:>6.1f}%  (n={w['n']})")
        if o["top_confusions"]:
            print("\n  Most common mistakes:")
            for m in o["top_confusions"][:5]:
                print(f"    {m['true']} -> {m['predicted']}  x{m['count']}")


def write_csv(path: Path, rows):
    if not rows:
        return
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("image", "breed"), k))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description="Benchmark dog_detector on COCO + Oxford-IIIT Pet")
    p.add_argument("--datasets", nargs="+", default=["oxford", "coco"], choices=["oxford", "coco"])
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="dataset cache directory")
    p.add_argument("--limit", type=int, help="evaluate a random subset of N images per dataset")
    p.add_argument("--seed", type=int, default=0, help="seed for --limit sampling")
    add_model_args(p)
    p.add_argument("--threshold", type=float, default=0.6, help="detection threshold (app default 0.6)")
    p.add_argument("--crop-pad", type=float, default=0.05, help="padding around each crop")
    p.add_argument("--checkpoint", help="classifier state_dict to evaluate (e.g. fine-tuned)")
    p.add_argument("--baseline", help="earlier summary.json to compare against")
    p.add_argument("--out-dir", type=Path, default=Path("benchmark_results"))
    p.add_argument("--cpu", action="store_true", help="force CPU")
    p.add_argument("--no-calibration", action="store_true",
                   help="evaluate raw confidences (ignore calibration.json)")
    args = p.parse_args()

    baseline = json.loads(Path(args.baseline).read_text()) if args.baseline else None
    device = torch.device("cpu") if args.cpu else pick_device()
    print(f"Loading models on {device}...")
    # Keep every dog detection down to 0.05 so AP and threshold sweeps can be computed.
    kwargs = model_kwargs(args)
    if args.checkpoint:
        kwargs["extra_breeds"] = False     # a checkpoint replaces the 118-breed classifier
    model = DogBreedDetector(device, det_threshold=0.05, **kwargs,
                             crop_pad=args.crop_pad, calibrated=not args.no_calibration)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_classifier(state.get("state_dict", state))
        print(f"Loaded classifier weights from {args.checkpoint}")
        if model.temperature != 1.0:
            print("Warning: calibration.json was fitted for the pretrained classifier, not this "
                  "checkpoint. Use --no-calibration or re-fit calibration for these weights.")
    label_map = build_label_map(model) | new_breed_map(model)

    # Warm up (GPU kernels compile on first use) so timings reflect steady state.
    dummy = torch.rand(3, 480, 640)
    for _ in range(2):
        model.detect_dogs(dummy)
        model.breed_probs(dummy, [[0.0, 0.0, 640.0, 480.0], [100.0, 100.0, 400.0, 400.0]])

    det_timer, cls_timer = Timer(), Timer()
    summary = {"run": {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "models": f"{model.detector_kind} detector @ {model.det_size}px / {model.proposals} "
                  f"proposals + {model.classifier_kind} classifier",
        "detector": model.detector_kind, "classifier": model.classifier_kind,
        "det_size": model.det_size, "proposals": model.proposals, "threshold": args.threshold,
        "crop_pad": args.crop_pad, "checkpoint": args.checkpoint, "limit": args.limit,
        "seed": args.seed, "device": str(device), "temperature": model.temperature,
    }}
    out = args.out_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    if "oxford" in args.datasets:
        print("\n--- Oxford-IIIT Pet (breed accuracy) ---")
        summary["oxford"], rows = eval_oxford(model, args, label_map, det_timer, cls_timer)
        write_csv(out / "oxford_predictions.csv", rows)
    if "coco" in args.datasets:
        print("\n--- COCO val2017 (detection accuracy) ---")
        summary["coco"], rows = eval_coco(model, args, det_timer)
        write_csv(out / "coco_images.csv", rows)
    summary["speed_ms"] = {"detector": det_timer.ms(), "classifier": cls_timer.ms()}

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print_report(summary, baseline)
    print(f"\nSaved results to {out}/")


if __name__ == "__main__":
    main()
