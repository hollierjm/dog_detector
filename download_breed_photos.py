"""
Download extra photos of breeds the model confuses, from two more sources.

  Udacity dogImages   the dataset from Udacity's dog-breed project, sorted by breed.
                      Only the folders in UDACITY are read, straight out of the
                      1 GB zip, so the rest is never downloaded. Its license is
                      unclear, so it is for local training only, like Tsinghua.
  Wikimedia Commons   photos from the breed's category (and the subcategories in
                      COMMONS), at 960 px wide. Each photo's license, author and page
                      is saved for attribution. Commons categories include drawings,
                      several dogs at once and mislabelled dogs, so these photos are
                      only trained on once reviewed (see train_breeds.py).

Writes data/udacity-dogs/<breed>/*.jpg, data/commons-dogs/<breed>/*.jpg and
data/commons-dogs/labels.csv. Photos already on disk are skipped.

Usage:
    python download_breed_photos.py
"""

import csv
import hashlib
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

UDACITY_ZIP = "https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip"
UDACITY = {"012.Australian_shepherd": "Australian Shepherd"}
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
COMMONS = {"Australian Shepherd": ["Category:Australian Shepherd",
                                   "Category:Australian Shepherd heads",
                                   "Category:Australian Shepherd puppies",
                                   "Category:Australian Shepherd in non-standard coat colors"]}
# Commons asks automated clients to identify themselves and not to hammer the API.
HEADERS = {"User-Agent": "dog-detector-training/1.0 (breed classifier; python-urllib)"}
PAUSE = 1.0  # seconds between Commons requests
# Commons only serves thumbnails at standard widths to bots without throttling them
# (https://www.mediawiki.org/wiki/Common_thumbnail_sizes).
COMMONS_WIDTH = 960


class HttpRangeFile(io.RawIOBase):
    """A remote file read with HTTP range requests, so zipfile only fetches what it needs."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(request, timeout=60) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, buffer):
        if self.pos >= self.size or not len(buffer):
            return 0
        end = min(self.pos + len(buffer), self.size) - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(request, timeout=120) as r:
            data = r.read()
        buffer[:len(data)] = data
        self.pos += len(data)
        return len(data)


def download_udacity(root: Path):
    archive = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(UDACITY_ZIP), buffer_size=1 << 22))
    for info in archive.infolist():
        parts = info.filename.split("/")  # dogImages/<split>/<folder>/<file>
        if info.is_dir() or len(parts) != 4 or parts[2] not in UDACITY:
            continue
        if not parts[3].lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        # The Udacity train/valid/test split is ignored; train_breeds.py makes its own.
        path = root / UDACITY[parts[2]] / f"{parts[1]}_{parts[3]}"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(info))
    for breed in sorted(set(UDACITY.values())):
        print(f"  Udacity {breed}: {len(list((root / breed).glob('*')))} photos")


def commons_query(**params):
    params.update(action="query", format="json", maxlag=5)
    url = COMMONS_API + "?" + urllib.parse.urlencode(params)
    for attempt in range(5):
        time.sleep(PAUSE * 2 ** attempt)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r:
                result = json.load(r)
            if "error" not in result:  # a maxlag error means "busy, try again"
                return result
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503):
                raise
    raise SystemExit(f"Commons kept refusing: {url}")


def commons_files(category: str):
    """Photo files in one category, with an 800 px thumbnail URL and license details."""
    files, cont = [], {}
    while True:
        result = commons_query(generator="categorymembers", gcmtitle=category, gcmtype="file",
                               gcmlimit=50, prop="imageinfo", iiprop="url|extmetadata|mime",
                               iiurlwidth=COMMONS_WIDTH, **cont)
        for page in result.get("query", {}).get("pages", {}).values():
            info = page["imageinfo"][0]
            if info["mime"] not in ("image/jpeg", "image/png"):
                continue  # drawings and diagrams are usually SVG; videos are skipped too
            meta = info.get("extmetadata", {})
            files.append({"title": page["title"], "url": info.get("thumburl", info["url"]),
                          "license": meta.get("LicenseShortName", {}).get("value", ""),
                          "author": meta.get("Artist", {}).get("value", ""),
                          "page": info["descriptionurl"]})
        if "continue" not in result:
            return files
        cont = result["continue"]


def commons_fetch(url: str, path: Path) -> bool:
    for attempt in range(5):
        time.sleep(PAUSE * 4 ** attempt)  # back off hard when throttled
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS),
                                        timeout=60) as r:
                path.write_bytes(r.read())
            return True
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503):
                return False
    return False


def download_commons(root: Path):
    rows = []
    for breed, categories in COMMONS.items():
        seen = set()
        for category in categories:
            for f in commons_files(category):
                if f["title"] in seen:
                    continue
                seen.add(f["title"])
                # Named from the title, so a reviewed photo keeps its name across runs.
                name = hashlib.sha1(f["title"].encode()).hexdigest()[:16] + ".jpg"
                path = root / breed / name
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if not commons_fetch(f["url"], path):
                        print(f"  skipped {f['title']}")
                        continue
                rows.append({"file": f"{breed}/{name}", "breed": breed, **f})
        print(f"  Commons {breed}: {sum(r['breed'] == breed for r in rows)} photos")
    with open(root / "labels.csv", "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    data = Path(__file__).with_name("data")
    print("Udacity dogImages...")
    download_udacity(data / "udacity-dogs")
    print("Wikimedia Commons...")
    download_commons(data / "commons-dogs")


if __name__ == "__main__":
    main()
