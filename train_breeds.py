"""
Teach the classifier dog breeds that ImageNet doesn't have.

The ImageNet classifier stays frozen: its features and its 118 dog-breed rows are
kept, and this trains new rows for the breeds in EXTRA_BREEDS. The result is one
small layer saved to breed_head_<classifier>.pth, which dog_detector.py picks up
automatically, so inference costs the same as before.

Training data, all expected under ./data (see the README for where to get it):
  Tsinghua Dogs     new breeds, and images of the 118 existing breeds, which teach
                    the new rows what they are *not*.
  benni-ben breeds  new breeds Tsinghua doesn't have.
  Oxford-IIIT Pet   trainval only, and only for breeds no other source covers.
  Udacity dogImages more Australian Shepherds (download_breed_photos.py).
  Wikimedia Commons more Australian Shepherds (download_breed_photos.py), used only
                    once reviewed (see REVIEWED_PHOTOS).
  Open Images       photos from download_open_images.py, used only once reviewed.
                    Its train split adds Australian Shepherds and Dachshunds; its
                    test split is an independent check that is reported at the end
                    and never trained on or used to pick a head.
  Your own dog      webcam photos saved with `dog_detector.py --collect BREED`, in
                    data/my_dogs/. Re-run this script after collecting more.

These rules keep the numbers honest:
  - Any training image that matches an Oxford image (test or trainval) or an Open
    Images test photo is dropped. benni-ben is compiled from other datasets, so it
    does contain Oxford photos.
  - Repeats of a photo within a breed are dropped before the train/validation split,
    otherwise validation scores those breeds on images the head trained on.
  - Oxford trainval is otherwise left out of training, because calibrate.py fits on it.
    It's also the guardrail for choosing a head: Tsinghua validation rewards fitting
    Tsinghua's photos, even when that makes the head worse on other people's dogs.

Usage:
    python train_breeds.py                 # fast classifier
    python train_breeds.py --accurate      # accurate classifier
    python calibrate.py [--accurate]       # then refit the confidence %
"""

import argparse
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from benchmark import build_label_map, load_oxford
from dog_detector import (BREED_HEAD_FILE, CALIBRATION_FILE, MY_DOGS_DIR, DogBreedDetector,
                          add_model_args, crop_box, model_kwargs, pick_device)

# New breed -> its folder name in each source (Oxford names as benchmark.py prints them).
# Dalmatian isn't new, but Tsinghua lacks it, so without these images it would have no
# examples in training or validation.
EXTRA_BREEDS = {
    "Dalmatian": {"benni": "Dalmatian"},
    "American Bulldog": {"oxford": "American Bulldog"},
    "American Hairless Terrier": {"benni": "American Hairless"},
    "Australian Shepherd": {"tsinghua": "Australian_Shepherd", "benni": "Australian Shepherd",
                            "udacity": "Australian Shepherd"},
    "Basset Fauve De Bretagne": {"benni": "Basset Fauve De Bretagne"},
    "Bichon Frise": {"tsinghua": "Bichon_Frise"},
    "Cane Corso": {"tsinghua": "Cane_Carso", "benni": "Italian Cane Corso"},
    "Chinese Crested": {"tsinghua": "Chinese_Crested_Dog"},
    "Cockapoo": {"benni": "Cockapoo"},
    "Fila Brasileiro": {"tsinghua": "Fila Braziliero"},
    "Greyhound": {"benni": "Greyhound"},
    "Havanese": {"benni": "Havanese", "oxford": "Havanese"},
    "Japanese Spitz": {"tsinghua": "Japanese_Spitzes", "benni": "Japanese Spitz"},
    "Labradoodle": {"benni": "Labradoodle"},
    "Shar Pei": {"benni": "Shar Pei"},
    "Shiba Inu": {"tsinghua": "Shiba_Dog", "benni": "Shiba Inu"},
    "Smooth Fox Terrier": {"benni": "Smooth Fox Terrier"},
    # Only in Open Images (see download_open_images.py for its label names).
    "Dachshund": {"openimages": "Dachshund"},
}
MIN_TRAIN_IMAGES = 50      # fewer than this after de-duplication: leave the breed out
# Tsinghua folders that are not usable breed labels: a grooming style, a mixed-breed
# group, an unclear label, and wild canids.
TSINGHUA_SKIP = {"teddy", "chinese_rural_dog", "Black_sable", "dhole", "dingo",
                 "African_hunting_dog"}
TSINGHUA_ALIASES = {"Brabancon_griffo": "Brabancon Griffon"}
HASH_DISTANCE = 4          # dHash bits apart that still count as the same photo
VAL_EVERY = 5              # sources without a validation split: hold out every 5th image
# Your webcam photos are held out in blocks of this many consecutive frames, because
# neighbouring frames are near-copies and would leak between training and validation.
OWN_BLOCK = 10
# Open Images and Commons labels are often wrong - Open Images' "Bulldog" photos are half
# French Bulldogs, its "Australian Shepherds" mostly other dogs - so a photo from either
# is only used if a person has checked the dog crop and marked it keep in this file
# (columns: photo, its path under data/; breed; verdict). Open Images breeds in
# OPEN_IMAGES_TRAIN are trained on (their test photos still aren't); its other
# reviewed breeds are only in the test report.
REVIEWED_PHOTOS = Path(__file__).with_name("reviewed_photos.csv")
OPEN_IMAGES_TRAIN = {"Australian Shepherd", "Dachshund"}
OXFORD_TOLERANCE = 0.01    # a head may lose at most this much accuracy on Oxford trainval
# Head variants tried: (decay on the new rows, pull of the old rows towards ImageNet,
# class balance). A pull of None freezes the old rows. Balance is the power of
# 1 / (images of that breed) each training image is weighted by: 0.5 softens the
# imbalance, 1.0 makes every breed count equally - e.g. 222 Australian Shepherds
# against 2,538 Border Collies, the breed they're most often mistaken for.
VARIANTS = [(new, pull, balance) for new in (1e-4, 1e-3) for pull in (None, 1.0, 0.1, 0.01)
            for balance in (0.5, 1.0)]
# Taken off every new breed's score. New breeds are mostly look-alikes of old ones
# (American Bulldog / Staffordshire), and this stops them claiming borderline dogs.
NEW_BREED_HANDICAPS = (0.0, 0.5, 1.0, 1.5, 2.0)


def simple_name(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalpha())


# --- Matching photos against Oxford ---------------------------------------------

def dhash(img: Image.Image) -> int:
    """64-bit perceptual hash: survives resizing and re-compression."""
    px = np.asarray(img.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    return int.from_bytes(np.packbits(px[:, 1:] > px[:, :-1]).tobytes(), "big")


def held_out_fingerprints(data_dir: Path, test_items):
    """Hashes of every photo training must not see: Oxford dogs and Open Images test."""
    images = (img for split in ("test", "train")
              for _, img, _, is_dog in load_oxford(data_dir, None, 0, split=split) if is_dog)
    hashes, digests = [], set()
    for img in images:
        hashes.append(dhash(img))
        digests.add(hashlib.sha256(img.tobytes()).digest())
    for item in test_items:
        with Image.open(item["path"]) as img:
            img = img.convert("RGB")
            hashes.append(dhash(img))
            digests.add(hashlib.sha256(img.tobytes()).digest())
    return np.array(hashes, dtype=np.uint64), digests


def drop_held_out_matches(items, held_out_hashes, held_out_digests):
    kept, dropped = [], Counter()
    for item in items:
        try:
            with Image.open(item["path"]) as img:
                img = img.convert("RGB")
                item["hash"] = dhash(img)
                near = np.bitwise_count(held_out_hashes ^ np.uint64(item["hash"])).min()
                same = hashlib.sha256(img.tobytes()).digest() in held_out_digests
        except OSError:
            dropped["unreadable"] += 1
            continue
        if same or near <= HASH_DISTANCE:
            dropped[item["source"]] += 1
        else:
            kept.append(item)
    print(f"  dropped: {dict(dropped)}, kept {len(kept)}")
    return kept


def drop_repeats(items):
    """Keep one copy of each photo within a breed, preferring the training copy."""
    seen, kept = {}, []
    for item in sorted(items, key=lambda i: i["split"] == "val"):
        hashes = seen.setdefault(item["breed"], np.zeros(0, dtype=np.uint64))
        h = np.uint64(item["hash"])
        if len(hashes) and np.bitwise_count(hashes ^ h).min() <= HASH_DISTANCE:
            continue
        seen[item["breed"]] = np.append(hashes, h)
        kept.append(item)
    print(f"  {len(items) - len(kept)} repeated photos dropped")
    return kept


def assign_splits(items):
    """Hold out every VAL_EVERY-th image of each breed from sources without a split."""
    seen = Counter()
    for item in items:
        if item["split"] is None:
            seen[item["breed"]] += 1
            item["split"] = "val" if seen[item["breed"]] % VAL_EVERY == 0 else "train"


# --- Collecting training images -------------------------------------------------

def read_list(path: Path):
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    return [tuple(line.strip().replace("\\", "/").split("/")[-2:]) for line in lines if line.strip()]


def tsinghua_items(root: Path, breed_names):
    """Tsinghua images with their hand-drawn whole-dog boxes."""
    known = {simple_name(n): n for n in breed_names}
    new = {src["tsinghua"]: breed for breed, src in EXTRA_BREEDS.items() if "tsinghua" in src}
    items = []
    for split, list_name in (("train", "train.lst"), ("val", "validation.lst")):
        for folder, file in read_list(root / "TrainAndValList" / list_name):
            raw = folder.split("-", 2)[2]
            if raw in TSINGHUA_SKIP:
                continue
            breed = new.get(raw) or known.get(simple_name(TSINGHUA_ALIASES.get(raw, raw)))
            if breed is None:
                raise SystemExit(f"Tsinghua breed not recognised: {raw}")
            note = root / "Low-Annotations" / folder / f"{file}.xml"
            body = ET.fromstring(note.read_bytes().strip()).find(".//bodybndbox")
            box = [int(body.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]
            items.append({"path": str(root / "low-resolution" / folder / file), "breed": breed,
                          "split": split, "box": box, "source": "tsinghua"})
    return items


def benni_items(root: Path):
    items = []
    for breed, src in EXTRA_BREEDS.items():
        if "benni" not in src:
            continue
        files = sorted(p for p in (root / "Dog Breeds" / src["benni"]).iterdir()
                       if not p.name.startswith("."))
        items += [{"path": str(p), "breed": breed, "box": None, "source": "benni", "split": None}
                  for p in files]
    return items


def oxford_items(data_dir: Path, out_dir: Path):
    """Save Oxford trainval photos of the breeds that need them; Oxford has no box file."""
    wanted = {src["oxford"]: breed for breed, src in EXTRA_BREEDS.items() if "oxford" in src}
    out_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for image_id, img, breed, _ in load_oxford(data_dir, None, 0, split="train"):
        if breed in wanted:
            path = out_dir / f"{image_id}.png"
            if not path.exists():
                img.save(path)
            items.append({"path": str(path), "breed": wanted[breed], "box": None, "source": "oxford",
                          "split": None, "hash": dhash(img)})
    return items


def udacity_items(root: Path):
    items = []
    for breed, src in EXTRA_BREEDS.items():
        if "udacity" in src and (root / src["udacity"]).exists():
            items += [{"path": str(p), "breed": breed, "box": None, "source": "udacity",
                       "split": None} for p in sorted((root / src["udacity"]).glob("*.jpg"))]
    return items


def reviewed_items(data_dir: Path, folder: str, source: str):
    """Photos under data/<folder>/ that REVIEWED_PHOTOS marks keep."""
    if not REVIEWED_PHOTOS.exists():
        return []
    with open(REVIEWED_PHOTOS, newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r["verdict"] == "keep" and r["photo"].startswith(folder + "/")]
    return [{"path": str(data_dir / r["photo"]), "breed": r["breed"], "box": None,
             "source": source, "split": "test" if "/test/" in r["photo"] else None}
            for r in rows]


def open_images_items(data_dir: Path):
    """Reviewed photos from download_open_images.py; test photos are kept for the report."""
    items = reviewed_items(data_dir, "open-images", "openimages")
    train = [i for i in items if i["split"] is None and i["breed"] in OPEN_IMAGES_TRAIN]
    return train, [i for i in items if i["split"] == "test"]


def own_items(root: Path, breed_names):
    """Webcam photos from dog_detector.py --collect: <root>/<breed>/<session>/NNNN.jpg."""
    items = []
    if not root.exists():
        return items
    for breed_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        if breed_dir.name not in breed_names:
            raise SystemExit(f"{breed_dir}: not a breed this head knows")
        photos = sorted(breed_dir.glob("*/*.jpg"))
        for n, photo in enumerate(photos):
            with Image.open(photo) as img:
                h = dhash(img.convert("RGB"))
            box = json.loads(photo.with_suffix(".json").read_text())["box"]
            split = "val" if (n // OWN_BLOCK) % VAL_EVERY == VAL_EVERY - 1 else "train"
            items.append({"path": str(photo), "breed": breed_dir.name, "box": box,
                          "split": split, "source": "own", "hash": h})
    return items


def add_detector_boxes(items, model):
    """Box the dog the way the app will. Images with no detected dog are dropped, and so
    are Open Images and Commons photos with more than one: their label is for the whole
    photo, so it can't be pinned to one dog."""
    kept = []
    for item in items:
        if item["box"] is None:
            with Image.open(item["path"]) as img:
                dets = model.detect_dogs(TF.to_tensor(img.convert("RGB")))
            if not dets or (item["source"] in ("openimages", "commons") and len(dets) > 1):
                continue
            item["box"] = max(dets, key=lambda d: d[1])[0]
        kept.append(item)
    print(f"  {len(items) - len(kept)} images without exactly one usable dog dropped")
    return kept


def build_manifest(args, model):
    cache = args.data_dir / "breed_training" / "manifest.json"
    if cache.exists():
        return json.loads(cache.read_text())
    print("Collecting images...")
    items = tsinghua_items(args.data_dir / "tsinghua-dogs" / "extracted", model.breed_names)
    items += benni_items(args.data_dir / "benni-dog-breeds" / "extracted")
    items += udacity_items(args.data_dir / "udacity-dogs")
    items += reviewed_items(args.data_dir, "commons-dogs", "commons")
    oi_train, oi_test = open_images_items(args.data_dir)
    items += oi_train
    print(f"  {len(items)} candidates. Comparing with every Oxford dog photo and "
          f"{len(oi_test)} Open Images test photos...")
    items = drop_held_out_matches(items, *held_out_fingerprints(args.data_dir, oi_test))
    items += oxford_items(args.data_dir, args.data_dir / "breed_training" / "oxford")
    items = drop_repeats(items)
    items = add_detector_boxes(items + oi_test, model)
    assign_splits(items)

    counts = Counter(i["breed"] for i in items if i["split"] == "train")
    for source in ("udacity", "commons", "openimages"):
        by_breed = Counter(i["breed"] for i in items if i["source"] == source and i["split"] != "test")
        print(f"  {source} photos kept per breed: {dict(sorted(by_breed.items()))}")
    too_few = [b for b in EXTRA_BREEDS if counts[b] < MIN_TRAIN_IMAGES]
    print(f"  left out, under {MIN_TRAIN_IMAGES} training images: "
          f"{ {b: counts[b] for b in too_few} }")
    items = [i for i in items if i["breed"] not in too_few]
    cache.write_text(json.dumps(items))
    return items


# --- Features -------------------------------------------------------------------

class Crops(Dataset):
    """Each dog crop exactly as the app prepares it, plus a mirrored copy."""

    def __init__(self, items, transform, crop_pad):
        self.items, self.transform, self.crop_pad = items, transform, crop_pad

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        with Image.open(self.items[i]["path"]) as img:
            rgb = TF.to_tensor(img.convert("RGB"))
        crop = crop_box(rgb, self.items[i]["box"], self.crop_pad)
        return torch.stack([self.transform(crop), self.transform(crop.flip(-1))])


@torch.inference_mode()
def extract_features(args, model, items, cache: Path | None):
    """float16 array [N, 2, D]: classifier features of each crop and its mirror image."""
    if cache and cache.exists():
        return np.load(cache)
    loader = DataLoader(Crops(items, model.cls_transform, args.crop_pad), batch_size=args.batch,
                        num_workers=args.workers, persistent_workers=args.workers > 0)
    out = []
    for n, batch in enumerate(loader, 1):
        feats = model.classifier(batch.flatten(0, 1).to(model.device))
        out.append(feats.view(len(batch), 2, -1).half().cpu().numpy())
        if n % 100 == 0:
            print(f"  {n * args.batch}/{len(items)} crops")
    feats = np.concatenate(out)
    if cache:
        np.save(cache, feats)
    return feats


# --- Fitting the head -----------------------------------------------------------

@torch.inference_mode()
def oxford_guardrail(args, model):
    """Features and accepted-breed masks of the Oxford trainval dogs calibrate.py uses."""
    cache = args.data_dir / "breed_training" / f"guardrail_{model.classifier_kind}.pt"
    if cache.exists():
        return torch.load(cache, weights_only=True)
    label_map = build_label_map(model)
    feats, masks = [], []
    for _, img, breed, is_dog in load_oxford(args.data_dir, None, 0, split="train"):
        if not is_dog or breed not in label_map:
            continue
        rgb = TF.to_tensor(img)
        dets = sorted(model.detect_dogs(rgb), key=lambda d: -d[1])
        if dets:
            crop = model.cls_transform(crop_box(rgb, dets[0][0], args.crop_pad))
            feats.append(model.classifier(crop[None].to(model.device))[0].cpu())
            mask = torch.zeros(len(model.breed_names), dtype=torch.bool)
            mask[list(label_map[breed])] = True
            masks.append(mask)
    result = (torch.stack(feats).clone(), torch.stack(masks))
    torch.save(result, cache)
    return result


@torch.no_grad()
def guardrail_accuracy(feats, masks, w, b, device):
    pred = (feats.to(device).float() @ w.T + b).argmax(1).cpu()
    in_old = pred < masks.shape[1]
    right = masks[torch.arange(len(pred)), pred.clamp(max=masks.shape[1] - 1)]
    return (right & in_old).float().mean().item()


def fit_head(x, y, old_w, old_b, n_classes, new_decay, old_pull, balance, epochs, device):
    """Train the breed layer with class-balanced cross-entropy.

    old_pull=None keeps the ImageNet rows exactly as they are and trains only the new
    ones. Otherwise the old rows may move, at this cost for leaving their ImageNet values.
    """
    n_old = len(old_w)
    w = torch.zeros(n_classes, old_w.shape[1], device=device)
    b = torch.zeros(n_classes, device=device)
    w[:n_old], b[:n_old] = old_w, old_b
    # New breeds start from the bottom of the old rows' range rather than at zero.
    b[n_old:] = old_b.min()
    w.requires_grad_(), b.requires_grad_()
    x, y = x.to(device), y.to(device)
    counts = torch.bincount(y, minlength=n_classes).float().clamp(min=1)
    class_weight = (counts.mean() / counts) ** balance
    opt = torch.optim.AdamW([w, b], lr=3e-3, weight_decay=0.0)
    steps = epochs * (len(x) // 2048 + 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(epochs):
        for idx in torch.randperm(len(x), device=device).split(2048):
            loss = F.cross_entropy(x[idx].float() @ w.T + b, y[idx], weight=class_weight)
            loss = loss + new_decay * (w[n_old:] ** 2).sum(dim=1).mean()
            if old_pull is not None:
                loss = loss + old_pull * ((w[:n_old] - old_w) ** 2).sum(dim=1).mean()
            opt.zero_grad()
            loss.backward()
            if old_pull is None:
                w.grad[:n_old] = 0
                b.grad[:n_old] = 0
            opt.step()
            sched.step()
    return w.detach(), b.detach()


@torch.no_grad()
def scores(x, y, w, b, n_old, device):
    """Mean per-breed accuracy on the old and on the new breeds, plus the % of old-breed
    dogs wrongly given a new breed."""
    pred = torch.cat([(xb.to(device).float() @ w.T + b).argmax(1).cpu() for xb in x.split(4096)])
    per_breed = {int(c): (pred[y == c] == c).float().mean().item() for c in y.unique()}
    old = [v for c, v in per_breed.items() if c < n_old]
    new = [v for c, v in per_breed.items() if c >= n_old]
    return {"old": float(np.mean(old)), "new": float(np.mean(new)) if new else 0.0,
            "old_called_new": (pred[y < n_old] >= n_old).float().mean().item()}, per_breed


@torch.no_grad()
def test_report(x, y, old_head, new_head, names, n_old, device, previous=None):
    """Per-breed accuracy on Open Images test, next to the ImageNet-only head and, if
    there is one, the head this run replaces (`previous`: its saved file contents).

    The ImageNet head can't name new breeds, so it scores 0 on them by definition.
    """
    x = x.to(device).float()

    def predict(head, head_names):
        pred = (x @ head[0].to(device).T + head[1].to(device)).argmax(1).cpu()
        index = {n: i for i, n in enumerate(names)}  # compare breeds by name
        return torch.tensor([index.get(head_names[p], -1) for p in pred.tolist()])

    columns = {"ImageNet": predict(old_head, names[:n_old])}
    if previous is not None:
        columns["previous"] = predict((previous["head"]["weight"], previous["head"]["bias"]),
                                      previous["names"])
    columns["new head"] = predict(new_head, names)
    print(f"\nOpen Images test (never trained on or used for choosing):")
    print(f"  {'breed':<28}{'dogs':>6}" + "".join(f"{c:>10}" for c in columns))
    for c in y.unique().tolist():
        m = y == c
        print(f"  {names[c]:<28}{int(m.sum()):>6}" + "".join(
            f"{(pred[m] == c).float().mean():>10.0%}" for pred in columns.values()))
    print(f"  {'all':<28}{len(y):>6}" + "".join(
        f"{(pred == y).float().mean():>10.0%}" for pred in columns.values()))


def main():
    p = argparse.ArgumentParser(description="Train extra breeds for dog_detector")
    add_model_args(p)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--crop-pad", type=float, default=0.05, help="crop padding (app default)")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true", help="force CPU")
    args = p.parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cpu") if args.cpu else pick_device()
    kwargs = model_kwargs(args) | {"extra_breeds": False}
    model = DogBreedDetector(device, crop_pad=args.crop_pad, **kwargs)
    (args.data_dir / "breed_training").mkdir(exist_ok=True)

    items = build_manifest(args, model)
    print(f"Extracting {model.classifier_kind} features for {len(items)} crops...")
    cache = args.data_dir / "breed_training" / f"features_{model.classifier_kind}.npy"
    feats = torch.from_numpy(extract_features(args, model, items, cache))
    # Your own photos change between runs, so they're never cached.
    own = drop_repeats(own_items(MY_DOGS_DIR, set(model.breed_names) | set(EXTRA_BREEDS)))
    if own:
        print(f"Your photos: {dict(Counter(i['breed'] for i in own))}")
        own_feats = torch.from_numpy(extract_features(args, model, own, None))
        items, feats = items + own, torch.cat([feats, own_feats])

    old_names = model.breed_names
    names = old_names + sorted({i["breed"] for i in items} - set(old_names))
    labels = torch.tensor([names.index(i["breed"]) for i in items])
    split = [i["split"] for i in items]
    is_train = torch.tensor([s == "train" for s in split])
    is_val = torch.tensor([s == "val" for s in split])
    is_test = torch.tensor([s == "test" for s in split])
    x_train = feats[is_train].flatten(0, 1)
    y_train = labels[is_train].repeat_interleave(2)
    x_val, y_val = feats[is_val][:, 0], labels[is_val]
    print(f"{len(names)} breeds ({len(names) - len(old_names)} new), {int(is_train.sum())} "
          f"train / {int(is_val.sum())} validation / {int(is_test.sum())} test dogs")

    old_w = model.breed_head.weight.data.to(device)
    old_b = model.breed_head.bias.data.to(device)
    base, _ = scores(x_val, y_val, old_w, old_b, len(old_names), device)
    guard_x, guard_masks = oxford_guardrail(args, model)
    guard_base = guardrail_accuracy(guard_x, guard_masks, old_w, old_b, device)
    print(f"ImageNet rows alone: old breeds {base['old']:.1%}, Oxford trainval {guard_base:.1%}")

    # Oxford test is never touched here: heads are compared on validation images, with
    # Oxford trainval as a limit on how much the existing breeds may get worse.
    results = []
    print(f"{'new decay':>10}{'old pull':>10}{'balance':>9}{'handicap':>10}{'old':>8}{'new':>8}"
          f"{'old->new':>10}{'Oxford':>8}")
    for new_decay, old_pull, balance in VARIANTS:
        w, raw_b = fit_head(x_train, y_train, old_w, old_b, len(names), new_decay, old_pull,
                            balance, args.epochs, device)
        for handicap in NEW_BREED_HANDICAPS:
            b = raw_b.clone()
            b[len(old_names):] -= handicap
            s, per_breed = scores(x_val, y_val, w, b, len(old_names), device)
            guard = guardrail_accuracy(guard_x, guard_masks, w, b, device)
            print(f"{new_decay:>10}{str(old_pull):>10}{balance:>9}{handicap:>10}{s['old']:>8.1%}"
                  f"{s['new']:>8.1%}{s['old_called_new']:>10.1%}{guard:>8.1%}")
            results.append({"new": s["new"], "guard": guard, "w": w, "b": b,
                            "per_breed": per_breed, "new_decay": new_decay,
                            "old_pull": old_pull, "balance": balance, "handicap": handicap})

    allowed = [r for r in results if r["guard"] >= guard_base - OXFORD_TOLERANCE]
    best = (max(allowed, key=lambda r: r["new"]) if allowed
            else max(results, key=lambda r: r["guard"]))
    if not allowed:
        print(f"WARNING: every head lost over {OXFORD_TOLERANCE:.0%} on Oxford trainval.")
    w, b, per_breed = best["w"], best["b"], best["per_breed"]
    print(f"\nKept: new decay {best['new_decay']}, old pull {best['old_pull']}, "
          f"balance {best['balance']}, handicap {best['handicap']}: new {best['new']:.1%}, Oxford {best['guard']:.1%}")
    val_counts = Counter(y_val.tolist())
    is_own_val = torch.tensor([i["source"] == "own" for i in items])[is_val]
    if is_own_val.any():
        pred = (x_val[is_own_val].to(device).float() @ w.T + b).argmax(1).cpu()
        right = int((pred == y_val[is_own_val]).sum())
        print(f"Your held-back photos: {right}/{int(is_own_val.sum())} labelled correctly")
    for c in range(len(old_names), len(names)):
        n = int((y_train == c).sum()) // 2
        print(f"  {names[c]:<28}{per_breed.get(c, float('nan')):>7.1%}  "
              f"({n} train, {val_counts[c]} validation)")

    path = CALIBRATION_FILE.with_name(BREED_HEAD_FILE.format(model.classifier_kind))
    if is_test.any():
        previous = torch.load(path, weights_only=True) if path.exists() else None
        test_report(feats[is_test][:, 0], labels[is_test], (old_w, old_b), (w, b), names,
                    len(old_names), device, previous)

    torch.save({"names": names, "head": {"weight": w.cpu(), "bias": b.cpu()}}, path)
    print(f"\nSaved {path.name}. Now run calibrate.py, then benchmark.py.")


if __name__ == "__main__":
    main()
