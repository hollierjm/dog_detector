"""
Download Open Images photos of the dog breeds train_breeds.py uses.

Open Images tags whole photos with labels that people have checked; there are no
boxes for breeds, so train_breeds.py finds the dog with the detector. This keeps a
photo only if exactly one of BREEDS is confirmed on it (a photo tagged both
Australian Shepherd and Border Collie is ambiguous) and it needs no rotation.

  train split -> training and validation photos for train_breeds.py
  test split  -> an independent check that is never trained on

Writes data/open-images/images/<split>/<id>.jpg and data/open-images/labels.csv, which
also records each photo's license, author and Flickr page for attribution. The photos
are listed by Open Images as CC BY 2.0.

Usage:
    python download_open_images.py
"""

import csv
import io
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GCS = "https://storage.googleapis.com/openimages/"
CLASSES_URL = GCS + "v7/oidv7-class-descriptions.csv"
LABELS_URL = GCS + "v7/oidv7-{split}-annotations-human-imagelabels.csv"
INFO_URLS = {"train": GCS + "v6/oidv6-train-images-with-labels-with-rotation.csv",
             "test": GCS + "2018_04/test/test-images-with-rotation.csv"}
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/{split}/{id}.jpg"

# Open Images label -> breed name in dog_detector.
BREEDS = {
    "Australian shepherd": "Australian Shepherd",
    "Border collie": "Border Collie",
    "Australian kelpie": "Kelpie",
    "Golden retriever": "Golden Retriever",
    "Bernese mountain dog": "Bernese Mountain Dog",
    "Dachshund": "Dachshund",
    "Bulldog": "Bulldog",
    "Akita": "Akita",
    "Akita inu": "Akita",
    "Labradoodle": "Labradoodle",
    "Cockapoo": "Cockapoo",
    "Greyhound": "Greyhound",
    "Japanese spitz": "Japanese Spitz",
    "Shiba inu": "Shiba Inu",
    "Shar pei": "Shar Pei",
    "Havanese": "Havanese",
    "Bichon Frise": "Bichon Frise",
    "Basset fauve de bretagne": "Basset Fauve De Bretagne",
    "Cane corso": "Cane Corso",
}


def stream_csv(url):
    """Rows of a (possibly multi-GB) CSV, read as it downloads."""
    with urllib.request.urlopen(url, timeout=120) as response:
        yield from csv.DictReader(io.TextIOWrapper(response, encoding="utf-8"))


def fetch(url: str, path: Path) -> bool:
    if path.exists():
        return True
    for attempt in range(6):
        try:
            data = urllib.request.urlopen(url, timeout=60).read()
            path.write_bytes(data)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:  # removed from the mirror; retrying won't help
                return False
            error = e
            time.sleep(2 ** attempt)
        except OSError as e:
            error = e
            time.sleep(2 ** attempt)
    print(f"  {path.stem}: {error}")
    return False


def select_photos(label_ids):
    """Photos with exactly one confirmed breed and no rotation, from both splits."""
    rows = []
    for split in ("train", "test"):
        print(f"Reading {split} labels (large file, streamed)...")
        breeds = {}
        for r in stream_csv(LABELS_URL.format(split=split)):
            if r["LabelName"] in label_ids and float(r["Confidence"]) == 1:
                breeds.setdefault(r["ImageID"], set()).add(label_ids[r["LabelName"]])
        single = {i: b.pop() for i, b in breeds.items() if len(b) == 1}
        print(f"  {len(breeds)} photos, {len(single)} with exactly one breed")

        print(f"Reading {split} photo details (license, author, rotation)...")
        kept = 0
        for r in stream_csv(INFO_URLS[split]):
            if r["ImageID"] in single:
                if r["Rotation"] not in ("0", "0.0"):
                    continue
                rows.append({"image_id": r["ImageID"], "split": split,
                             "breed": single[r["ImageID"]], "license": r["License"],
                             "author": r["Author"], "page": r["OriginalLandingURL"]})
                kept += 1
        print(f"  {kept} photos need no rotation")
    return rows


def write_csv(path: Path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    root = Path(__file__).with_name("data") / "open-images"
    root.mkdir(parents=True, exist_ok=True)
    selected = root / "selected.csv"  # cached, so a rerun only retries downloads
    if selected.exists():
        rows = list(csv.DictReader(open(selected)))
    else:
        label_ids = {}
        for row in csv.reader(io.TextIOWrapper(urllib.request.urlopen(CLASSES_URL), encoding="utf-8")):
            if row[1] in BREEDS:
                label_ids[row[0]] = BREEDS[row[1]]
        missing = set(BREEDS.values()) - set(label_ids.values())
        if missing:
            raise SystemExit(f"Labels not found in Open Images: {missing}")
        rows = select_photos(label_ids)
        write_csv(selected, rows)

    print(f"Downloading {len(rows)} photos...")
    for split in ("train", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
    jobs = [(IMAGE_URL.format(split=r["split"], id=r["image_id"]),
             root / "images" / r["split"] / f"{r['image_id']}.jpg") for r in rows]
    with ThreadPoolExecutor(6) as pool:
        ok = list(pool.map(lambda job: fetch(*job), jobs))
    rows = [r for r, good in zip(rows, ok) if good]
    print(f"  {ok.count(False)} failed")

    write_csv(root / "labels.csv", rows)
    counts = Counter((r["split"], r["breed"]) for r in rows)
    for breed in sorted(set(BREEDS.values())):
        print(f"  {breed:<28}{counts['train', breed]:>6} train {counts['test', breed]:>5} test")
    print(f"Saved {root / 'labels.csv'}")


if __name__ == "__main__":
    main()
