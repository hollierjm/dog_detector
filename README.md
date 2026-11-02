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
