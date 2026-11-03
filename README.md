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
```

Press `q` or `Esc` to quit the video window. It runs on CUDA or Apple Silicon (MPS) automatically when available.

On macOS, give your terminal app camera access in **System Settings > Privacy & Security > Camera**.

## Accuracy

Measured with `benchmark.py` (fast models, threshold 0.6):

| Metric | Result |
|---|---|
| Correct breed, full pipeline (Oxford-IIIT Pet test, 2,186 dogs) | **84.4%** (95% CI 82.8–85.8) |
| Correct breed in top 5 | 99.0% |
| Correct breed, full pipeline, `--accurate` models | **89.8%** (95% CI 88.4–91.0) |
| Dog detection AP50 (COCO val2017, 218 dogs) | **76.4** |
| Confidence calibration error (ECE) | 3.0% (21.3% before calibration) |

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
