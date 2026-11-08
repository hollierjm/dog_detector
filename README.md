# Dog Detector

Real-time dog detection and breed classification from a webcam or image, using PyTorch.

1. **Detection:** a COCO-pretrained Faster R-CNN finds every dog in the frame.
2. **Breed classification:** each dog crop goes to an ImageNet classifier (ResNet50, or EfficientNetV2 with `--accurate`), restricted to its 118 dog-breed classes, plus any extra breeds you train (see [More breeds](#more-breeds)).
3. **Display:** a box and breed label are drawn around each dog.

No training required for the 118 breeds. Pretrained weights download automatically on first run.

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```sh
python dog_detector.py                          # default webcam, fast models
python dog_detector.py --accurate               # slower, more accurate models
python dog_detector.py --camera 1               # use a different camera
python dog_detector.py --image dog.jpg          # run on a single image
python dog_detector.py --image dog.jpg --out annotated.jpg
python dog_detector.py --threshold 0.5          # detection confidence (default 0.6)
python dog_detector.py --cpu                    # force CPU

# Mix and tune models (each is benchmarked below)
python dog_detector.py --detector accurate --classifier fast
python dog_detector.py --det-size 640 --proposals 300   # smaller detector input = faster
```

Press `q` or `Esc` to quit the video window. It runs on CUDA or Apple Silicon (MPS) automatically when available.

On macOS, give your terminal app camera access in **System Settings > Privacy & Security > Camera**.

## Accuracy

Measured with `benchmark.py` at threshold 0.6. Breed accuracy is the full pipeline
(detect, then classify) on the Oxford-IIIT Pet test split (2,186 dogs); detection is on
COCO val2017 (218 dogs). Speed is model time per image on Apple Silicon (MPS).

| Mode | Correct breed (95% CI) | COCO dog AP50 | ms / image |
|---|---|---|---|
| Default (fast) | **84.4%** (82.8–85.8) | 76.4 | 43 |
| `--accurate` | **90.2%** (88.8–91.3) | 82.6 | 84 |

The `--accurate` detector runs at a 640px input with 300 region proposals. That is 1.8x
faster than torchvision's 800px / 1000 proposals (147 ms/image) with the same breed
accuracy (89.8% before). Calibrated confidence error (ECE) is 2–3% in both modes,
down from 21.3% uncalibrated for the fast models.

```sh
python benchmark.py                 # full benchmark, downloads datasets to ./data
python benchmark.py --limit 300     # quick check
```

## More breeds

`train_breeds.py` adds 14 breeds ImageNet doesn't have: American Bulldog, Australian
Shepherd, Basset Fauve de Bretagne, Bichon Frise, Cane Corso, Chinese Crested, Cockapoo,
Fila Brasileiro, Greyhound, Havanese, Japanese Spitz, Labradoodle, Shar Pei and Shiba Inu.
The classifier's features stay frozen; only its last layer changes, so speed is unchanged.
The result is `breed_head_<classifier>.pth`, which the app loads automatically when
present. `--imagenet-breeds` ignores it. The accuracy table above is without a head.

```sh
python train_breeds.py --classifier fast --detector accurate
python calibrate.py
python benchmark.py --datasets oxford
```

It expects these under `./data`, unzipped into `extracted/`:
[Tsinghua Dogs](https://cg.cs.tsinghua.edu.cn/ThuDogs/) (low-resolution images, annotations
and split, in `tsinghua-dogs/`) and
[benni-ben/dog-breeds](https://huggingface.co/datasets/benni-ben/dog-breeds)
(in `benni-dog-breeds/`). Tsinghua Dogs is published for academic use; check both
datasets' terms before redistributing images or trained weights.

More Australian Shepherds come from `python download_breed_photos.py` (Udacity's
dogImages and Wikimedia Commons) and `python download_open_images.py` (Open Images V7,
whose test split also gives an independent per-breed check). Commons and Open Images
labels are often wrong (Open Images' "Australian Shepherds" are mostly other dogs, its
"Bulldogs" half French Bulldogs), so their photos are only used once a person has
checked the dog crop: `reviewed_photos.csv` records each keep/reject decision.

Oxford test, same 22 breeds and code as the table above, plus the three Oxford breeds
that only the trained heads can name:

| Classifier | 118 breeds | 132 breeds | American Bulldog | Havanese | Shiba Inu |
|---|---|---|---|---|---|
| Fast | 84.4% (82.8–85.8) | 82.9% (81.3–84.5) | 89% | 78% | 90% |
| Accurate | 90.2% (88.8–91.3) | 87.7% (86.3–89.1) | 95% | 87% | 99% |

Adding breeds costs accuracy on the existing ones, mostly from look-alikes such as
Staffordshire and American Bulldog. Training weights each breed by how rare it is, so breeds with
few photos (429 Australian Shepherds against 2,538 Border Collies) aren't drowned out.
With the extra Australian Shepherds, 82% of held-back ones are named correctly (74%
before, on a smaller held-back set), and Oxford accuracy is unchanged. The accurate head falls outside the 118-breed range,
so it isn't enabled by default. Training images matching any Oxford photo are removed
first: `benni-ben` contains Oxford images, which would otherwise inflate these numbers.

### Training on your own dog

If the app struggles with your dog, collect webcam photos of it and retrain. Keep only
your dog in view and move it around: different angles, distances and lighting.

```sh
python dog_detector.py --collect "Australian Shepherd"   # q to stop; repeat any time
python train_breeds.py --classifier fast --detector accurate
python calibrate.py
```

Photos go to `data/my_dogs/<breed>/` (gitignored). Training reports how many of your
held-back photos it labels correctly.

## Confidence calibration

The pretrained classifier is under-confident: before calibration, predictions it
showed as 45% were right 79% of the time. `calibrate.py` fits a *temperature*
on Oxford-IIIT Pet's trainval split and saves it to `calibration.json`. The app then
divides the breed scores by it before the softmax, so the shown % matches how often
it's right. This changes only the percentages, never which breed is predicted.

```sh
python calibrate.py              # fast models
python calibrate.py --accurate   # accurate models
python dog_detector.py --no-calibration   # show raw percentages
```
