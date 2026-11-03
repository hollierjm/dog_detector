"""
Real-time dog detection + breed classification from a webcam, using PyTorch.

Pipeline (two pretrained models, no training required):
  1. Object detector (Faster R-CNN, COCO-pretrained) finds every "dog" in the frame.
  2. Each dog crop is passed to an ImageNet classifier (EfficientNetV2 / ResNet50);
     we keep only the 118 ImageNet dog-breed classes and pick the most likely breed.
  3. A red box is drawn around each dog with the breed label above it.

Usage:
    python dog_detector.py                    # default webcam, fast models
    python dog_detector.py --accurate         # slower, more accurate models (~2x slower)
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
# Detector input size (short side, px) and region proposals per frame, per detector.
# Chosen with benchmark.py: for the accurate detector, 640px / 300 proposals is 1.8x
# faster than torchvision's 800 / 1000 with the same breed accuracy (90.2% vs 89.8%).
# The fast detector keeps 800 / 1000: smaller sizes cost it recall on distant dogs.
DETECTOR_DEFAULTS = {"fast": (800, 1000), "accurate": (640, 300)}
# Webcam mode runs the models at most once per this many seconds (10x/s);
# frames in between reuse the most recent boxes and labels.
ANALYZE_INTERVAL = 0.1


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
    """
    detector / classifier: "fast" or "accurate", chosen independently
    (accurate=True is shorthand for both "accurate").
    det_size:  short side (pixels) frames are resized to before detection.
               Smaller is faster. None = the detector's default (DETECTOR_DEFAULTS).
    proposals: candidate regions the detector scores per frame. None = default.
    """

    def __init__(self, device: torch.device, accurate: bool = False,
                 det_threshold: float = 0.6, crop_pad: float = 0.05,
                 calibrated: bool = True, detector: str | None = None,
                 classifier: str | None = None, det_size: int | None = None,
                 proposals: int | None = None):
        self.device = device
        self.det_threshold = det_threshold
        self.crop_pad = crop_pad
        self.detector_kind = detector or ("accurate" if accurate else "fast")
        self.classifier_kind = classifier or ("accurate" if accurate else "fast")

        # --- 1. Dog detector -------------------------------------------------
        if self.detector_kind == "accurate":
            det_weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self.detector = fasterrcnn_resnet50_fpn_v2(weights=det_weights)
        else:
            det_weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            self.detector = fasterrcnn_mobilenet_v3_large_fpn(weights=det_weights)
        default_size, default_proposals = DETECTOR_DEFAULTS[self.detector_kind]
        det_size = det_size or default_size
        proposals = proposals or default_proposals
        # Input resolution: keep torchvision's 800:1333 short:long side ratio.
        self.detector.transform.min_size = (det_size,)
        self.detector.transform.max_size = round(det_size * 1333 / 800)
        # Candidate regions kept before / after the region proposal step.
        self.detector.rpn._pre_nms_top_n["testing"] = proposals
        self.detector.rpn._post_nms_top_n["testing"] = proposals
        self.det_size, self.proposals = det_size, proposals
        self.detector.eval().to(device)

        # --- 2. Breed classifier ---------------------------------------------
        if self.classifier_kind == "accurate":
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
        # Confidence is a property of the classifier, so it's keyed by classifier.
        self.calibration_key = self.classifier_kind
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

    def score_frame(self, frame_bgr):
        """Detect dogs in an OpenCV BGR frame and score every breed for each.

        Returns (boxes, probs): boxes are (x1, y1, x2, y2) in pixels and probs is
        a CPU tensor [N, n_breeds] of calibrated breed probabilities.
        """
        rgb = TF.to_tensor(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        boxes = [b for b, _ in self.detect_dogs(rgb)]
        if not boxes:
            return [], torch.empty(0, len(self.breed_names))
        return boxes, self.breed_probs(rgb, boxes).cpu()

    def label(self, probs: torch.Tensor) -> str:
        """Display label for one dog's breed probabilities: top breed and its %."""
        p, i = probs.max(dim=0)
        return f"{self.breed_names[int(i)]} ({float(p):.0%})"

    def analyze(self, frame_bgr):
        """Detect + classify dogs in an OpenCV BGR frame.

        Returns list of (box, label) where box is (x1, y1, x2, y2) in pixels.
        """
        boxes, probs = self.score_frame(frame_bgr)
        return [(box, self.label(p)) for box, p in zip(boxes, probs)]

    def process_frame(self, frame_bgr):
        """Detect + classify + draw on an OpenCV BGR frame (in place)."""
        results = self.analyze(frame_bgr)
        draw_results(frame_bgr, results)
        return frame_bgr, len(results)


def box_iou(a, b) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter)


class BreedTracker:
    """Keeps each dog's breed label steady across webcam frames.

    A single frame's top breed often flips between look-alike breeds (e.g.
    Border Collie / Collie) as the dog moves. Instead, each dog is followed from
    frame to frame by box overlap, its breed probabilities are averaged over
    time, and the label shows the breed with the highest average.

    min_iou:    box overlap needed to count as the same dog as last time.
    max_misses: frames a dog can go undetected before its history is dropped,
                so a brief missed detection doesn't reset the label.
    min_weight: weight of the newest frame once a dog has been seen 1/min_weight
                times; before that it's a plain average. Keeping a floor lets
                the label still change if it was wrong at first.
    """

    def __init__(self, min_iou: float = 0.3, max_misses: int = 5, min_weight: float = 0.1):
        self.min_iou, self.max_misses, self.min_weight = min_iou, max_misses, min_weight
        self.tracks = []   # dicts: box, probs (running average), seen, misses

    def update(self, boxes, probs: torch.Tensor):
        """Add one frame's detections; returns the smoothed probs for each box."""
        # Greedily pair each box with the unclaimed track it overlaps most.
        pairs = sorted(((box_iou(b, t["box"]), i, j)
                        for i, b in enumerate(boxes) for j, t in enumerate(self.tracks)),
                       reverse=True)
        match, used = {}, set()
        for iou, i, j in pairs:
            if iou < self.min_iou:
                break
            if i not in match and j not in used:
                match[i] = j
                used.add(j)

        smoothed = []
        for i, (box, p) in enumerate(zip(boxes, probs)):
            if i in match:
                t = self.tracks[match[i]]
                t["seen"] += 1
                w = max(1.0 / t["seen"], self.min_weight)
                t["probs"] = (1 - w) * t["probs"] + w * p
                t["box"], t["misses"] = box, 0
            else:
                t = {"box": box, "probs": p.clone(), "seen": 1, "misses": 0}
                self.tracks.append(t)
                used.add(len(self.tracks) - 1)
            smoothed.append(t["probs"])

        # Age out dogs that weren't seen this frame.
        for j, t in enumerate(self.tracks):
            if j not in used:
                t["misses"] += 1
        self.tracks = [t for t in self.tracks if t["misses"] <= self.max_misses]
        return smoothed


def draw_results(img, results):
    """Draw the (box, label) pairs returned by DogBreedDetector.analyze."""
    for (x1, y1, x2, y2), label in results:
        draw_labeled_box(img, int(x1), int(y1), int(x2), int(y2), label)


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


def run_webcam(model: DogBreedDetector, camera: int, interval: float = ANALYZE_INTERVAL):
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera {camera}. On macOS, grant camera access to your "
            "terminal app in System Settings > Privacy & Security > Camera."
        )

    print("Running — press 'q' or ESC in the video window to quit.")
    prev = time.time()
    fps = 0.0
    last_analyzed = float("-inf")
    results = []
    tracker = BreedTracker()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera.")
                break

            # Only run the models every `interval` seconds; in between, draw the
            # last results on the fresh frame so the video stays smooth.
            if time.time() - last_analyzed >= interval:
                last_analyzed = time.time()
                boxes, probs = model.score_frame(frame)
                smoothed = tracker.update(boxes, probs)
                results = [(b, model.label(p)) for b, p in zip(boxes, smoothed)]
            draw_results(frame, results)
            n_dogs = len(results)

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


def add_model_args(p: argparse.ArgumentParser):
    """Model-choice flags shared by dog_detector.py, benchmark.py and calibrate.py."""
    p.add_argument("--accurate", action="store_true",
                   help="use the larger, more accurate (slower) detector AND classifier")
    p.add_argument("--detector", choices=["fast", "accurate"],
                   help="detector to use (overrides --accurate)")
    p.add_argument("--classifier", choices=["fast", "accurate"],
                   help="breed classifier to use (overrides --accurate)")
    p.add_argument("--det-size", type=int,
                   help="detector input size, short side in px; smaller is faster "
                        "(default: 800 fast detector, 640 accurate)")
    p.add_argument("--proposals", type=int,
                   help="candidate regions per frame; fewer is faster "
                        "(default: 1000 fast detector, 300 accurate)")


def model_kwargs(args) -> dict:
    return {"accurate": args.accurate, "detector": args.detector,
            "classifier": args.classifier, "det_size": args.det_size,
            "proposals": args.proposals}


def main():
    p = argparse.ArgumentParser(description="Webcam dog detector + breed labeler (PyTorch)")
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    add_model_args(p)
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
    model = DogBreedDetector(device, det_threshold=args.threshold,
                             calibrated=not args.no_calibration, **model_kwargs(args))
    print(f"Detector: {model.detector_kind} ({model.det_size}px, {model.proposals} proposals), "
          f"classifier: {model.classifier_kind}")
    if model.temperature != 1.0:
        print(f"Confidence calibrated (temperature {model.temperature:.3f})")

    if args.image:
        run_image(model, args.image, args.out)
    else:
        run_webcam(model, args.camera)


if __name__ == "__main__":
    main()
