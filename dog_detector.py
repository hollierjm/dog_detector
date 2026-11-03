"""
Real-time dog detection + breed classification from a webcam, using PyTorch.

Pipeline (two pretrained models, no training required):
  1. Object detector (Faster R-CNN, COCO-pretrained) finds every "dog" in the frame.
  2. Each dog crop is passed to an ImageNet classifier (EfficientNetV2 / ResNet50);
     we keep only the 118 ImageNet dog-breed classes and pick the most likely breed.
  3. A red box is drawn around each dog with the breed label above it.

Usage:
    python dog_detector.py                    # default webcam, fast models
    python dog_detector.py --accurate         # slower, more accurate models
    python dog_detector.py --camera 1         # use a different camera
    python dog_detector.py --image dog.jpg    # run on a single image instead

Press 'q' or ESC to quit.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import torch
import torchvision.transforms.functional as TF
from torchvision.models import (
    EfficientNet_V2_S_Weights,
    ResNet50_Weights,
    efficientnet_v2_s,
    resnet50,
)
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn_v2,
)

COCO_DOG_LABEL = 18                    # "dog" in torchvision's COCO label map
IMAGENET_DOG_RANGE = range(151, 269)   # ImageNet classes 151..268 are dog breeds
RED = (0, 0, 255)                      # OpenCV uses BGR
WHITE = (255, 255, 255)
CALIBRATION_FILE = Path(__file__).with_name("calibration.json")


def load_temperature(key: str) -> float:
    """Temperature for this model setup from calibration.json, or 1.0 if not calibrated."""
    try:
        entry = json.loads(CALIBRATION_FILE.read_text())[key]
        return float(entry["temperature"])
    except (FileNotFoundError, KeyError, ValueError, TypeError):
        return 1.0


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DogBreedDetector:
    def __init__(self, device: torch.device, accurate: bool = False,
                 det_threshold: float = 0.6, crop_pad: float = 0.05,
                 calibrated: bool = True):
        self.device = device
        self.det_threshold = det_threshold
        self.crop_pad = crop_pad

        # --- 1. Dog detector -------------------------------------------------
        if accurate:
            det_weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self.detector = fasterrcnn_resnet50_fpn_v2(weights=det_weights)
        else:
            det_weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            self.detector = fasterrcnn_mobilenet_v3_large_fpn(weights=det_weights)
        self.detector.eval().to(device)

        # --- 2. Breed classifier ---------------------------------------------
        if accurate:
            cls_weights = EfficientNet_V2_S_Weights.DEFAULT
            self.classifier = efficientnet_v2_s(weights=cls_weights)
        else:
            cls_weights = ResNet50_Weights.DEFAULT
            self.classifier = resnet50(weights=cls_weights)
        self.classifier.eval().to(device)
        self.cls_transform = cls_weights.transforms()

        categories = cls_weights.meta["categories"]
        self.dog_idx = torch.tensor(list(IMAGENET_DOG_RANGE), device=device)
        self.breed_names = [categories[i].replace("_", " ").title()
                            for i in IMAGENET_DOG_RANGE]

        # --- 3. Confidence calibration ---------------------------------------
        # A temperature fitted by calibrate.py makes the shown % match how often
        # the prediction is actually right. 1.0 means "no calibration".
        self.calibration_key = "accurate" if accurate else "fast"
        self.temperature = load_temperature(self.calibration_key) if calibrated else 1.0

    @torch.inference_mode()
    def detect_dogs(self, rgb: torch.Tensor):
        """rgb: float tensor [3,H,W] in 0..1. Returns list of (box, score)."""
        out = self.detector([rgb.to(self.device)])[0]
        keep = (out["labels"] == COCO_DOG_LABEL) & (out["scores"] >= self.det_threshold)
        boxes = out["boxes"][keep].cpu()
        scores = out["scores"][keep].cpu()
        return list(zip(boxes.tolist(), scores.tolist()))

    @torch.inference_mode()
    def breed_logits(self, rgb: torch.Tensor, boxes) -> torch.Tensor:
        """Raw (uncalibrated) dog-breed scores for each box crop: tensor [N, n_breeds]."""
        _, H, W = rgb.shape
        crops = []
        for x1, y1, x2, y2 in boxes:
            # Pad the crop slightly so the classifier sees the whole dog.
            pw, ph = self.crop_pad * (x2 - x1), self.crop_pad * (y2 - y1)
            x1, y1 = max(0, int(x1 - pw)), max(0, int(y1 - ph))
            x2, y2 = min(W, int(x2 + pw)), min(H, int(y2 + ph))
            x2, y2 = max(x2, x1 + 1), max(y2, y1 + 1)   # never an empty crop
            crop = rgb[:, y1:y2, x1:x2]
            crops.append(self.cls_transform(crop))
        batch = torch.stack(crops).to(self.device)

        # Keep only the 118 dog-breed classes.
        return self.classifier(batch).index_select(1, self.dog_idx)

    def breed_probs(self, rgb: torch.Tensor, boxes) -> torch.Tensor:
        """Calibrated probabilities over the dog breeds for each box crop: [N, n_breeds].

        Dividing by the temperature before the softmax corrects over- or
        under-confidence without changing which breed ranks first.
        """
        return (self.breed_logits(rgb, boxes) / self.temperature).softmax(dim=1)

    def classify_breeds(self, rgb: torch.Tensor, boxes):
        """Classify each box crop. Returns list of (breed_name, probability)."""
        if not boxes:
            return []
        probs, idx = self.breed_probs(rgb, boxes).max(dim=1)
        return [(self.breed_names[i], p) for i, p in zip(idx.tolist(), probs.tolist())]

    def process_frame(self, frame_bgr):
        """Detect + classify + draw on an OpenCV BGR frame (in place)."""
        rgb = TF.to_tensor(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        detections = self.detect_dogs(rgb)
        boxes = [b for b, _ in detections]
        breeds = self.classify_breeds(rgb, boxes)

        for (x1, y1, x2, y2), (breed, prob) in zip(boxes, breeds):
            draw_labeled_box(frame_bgr, int(x1), int(y1), int(x2), int(y2),
                             f"{breed} ({prob:.0%})")
        return frame_bgr, len(boxes)


def draw_labeled_box(img, x1, y1, x2, y2, text):
    # Scale line/text size with the image so labels stay readable at any resolution.
    s = max(img.shape[1] / 1280, 0.5)
    cv2.rectangle(img, (x1, y1), (x2, y2), RED, max(2, int(3 * s)))

    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, max(1, int(2 * s))
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    # Place label above the box; if there's no room, put it just inside the top.
    top = y1 - th - baseline - 6
    if top < 0:
        top = y1
    cv2.rectangle(img, (x1, top), (x1 + tw + 8, top + th + baseline + 6), RED, -1)
    cv2.putText(img, text, (x1 + 4, top + th + 3), font, scale, WHITE, thick, cv2.LINE_AA)


def run_webcam(model: DogBreedDetector, camera: int):
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera {camera}. On macOS, grant camera access to your "
            "terminal app in System Settings > Privacy & Security > Camera."
        )

    print("Running — press 'q' or ESC in the video window to quit.")
    prev = time.time()
    fps = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera.")
                break

            frame, n_dogs = model.process_frame(frame)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
            prev = now
            cv2.putText(frame, f"FPS: {fps:.1f}  Dogs: {n_dogs}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Dog Breed Detector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_image(model: DogBreedDetector, path: str, out_path: str | None):
    frame = cv2.imread(path)
    if frame is None:
        raise SystemExit(f"Could not read image: {path}")
    frame, n = model.process_frame(frame)
    print(f"Found {n} dog(s).")
    if out_path:
        cv2.imwrite(out_path, frame)
        print(f"Saved annotated image to {out_path}")
    else:
        cv2.imshow("Dog Breed Detector", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description="Webcam dog detector + breed labeler (PyTorch)")
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--accurate", action="store_true",
                   help="use larger, more accurate (slower) models")
    p.add_argument("--threshold", type=float, default=0.6,
                   help="dog detection confidence threshold (default 0.6)")
    p.add_argument("--image", help="run on an image file instead of the webcam")
    p.add_argument("--out", help="with --image: save annotated result here instead of showing it")
    p.add_argument("--cpu", action="store_true", help="force CPU")
    p.add_argument("--no-calibration", action="store_true",
                   help="show raw (uncalibrated) confidence percentages")
    args = p.parse_args()

    device = torch.device("cpu") if args.cpu else pick_device()
    print(f"Loading models on {device} (first run downloads weights)...")
    model = DogBreedDetector(device, accurate=args.accurate, det_threshold=args.threshold,
                             calibrated=not args.no_calibration)
    if model.temperature != 1.0:
        print(f"Confidence calibrated (temperature {model.temperature:.3f})")

    if args.image:
        run_image(model, args.image, args.out)
    else:
        run_webcam(model, args.camera)


if __name__ == "__main__":
    main()
