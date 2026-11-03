# Dog Detector

Real-time dog detection and breed classification from a webcam or image, using PyTorch.

1. **Detection:** a COCO-pretrained Faster R-CNN finds every dog in the frame.
2. **Breed classification:** each dog crop goes to an ImageNet classifier (ResNet50, or EfficientNetV2 with `--accurate`), restricted to the 120 dog-breed classes.
3. **Display:** a box and breed label are drawn around each dog.

No training required. Pretrained weights download automatically on first run.

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
