"""
Calibrate the confidence % that dog_detector shows next to each breed.

Fits a single "temperature" T so that, e.g., predictions shown as 80% are right
about 80% of the time. The breed probabilities become softmax(logits / T):
  T < 1 sharpens them (fixes UNDER-confidence),
  T > 1 softens them (fixes OVER-confidence).
Dividing every score by the same number never changes which breed ranks first,
so accuracy is untouched - only the shown % changes.

T is fitted on Oxford-IIIT Pet's trainval split (never the test split, which
benchmark.py uses), using exactly what the app shows: the detector's box, then the
classifier. The result is saved to calibration.json, which dog_detector.py and
benchmark.py load automatically.

Usage:
    python calibrate.py              # calibrate the fast models
    python calibrate.py --accurate   # calibrate the --accurate models
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torchvision.transforms.functional as TF

from benchmark import build_label_map, calibration, load_oxford
from dog_detector import CALIBRATION_FILE, DogBreedDetector, add_model_args, model_kwargs, pick_device


def collect(model: DogBreedDetector, args, label_map):
    """Raw breed scores for every detected dog + a mask of the accepted labels."""
    logits, masks = [], []
    for n, (_, img, breed, is_dog) in enumerate(
            load_oxford(args.data_dir, args.limit, args.seed, split="train"), 1):
        if n % 500 == 0:
            print(f"  {n} images, {len(logits)} dogs collected")
        if not is_dog or breed not in label_map:
            continue
        rgb = TF.to_tensor(img)
        dets = sorted(model.detect_dogs(rgb), key=lambda d: -d[1])
        if not dets:
            continue                       # the app shows nothing, so nothing to calibrate
        logits.append(model.breed_logits(rgb, [dets[0][0]])[0].cpu())
        mask = torch.zeros(len(model.breed_names), dtype=torch.bool)
        mask[list(label_map[breed])] = True
        masks.append(mask)
    # clone(): tensors made under inference_mode can't take part in autograd.
    return torch.stack(logits).float().clone(), torch.stack(masks)


def nll(logits, masks, temperature):
    """Negative log-likelihood of the correct breed(s) - the standard calibration loss."""
    log_probs = (logits / temperature).log_softmax(dim=1)
    return -log_probs.masked_fill(~masks, float("-inf")).logsumexp(dim=1).mean()


def fit_temperature(logits, masks) -> float:
    # Optimise log(T) so T stays positive.
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = nll(logits, masks, log_t.exp())
        loss.backward()
        return loss

    opt.step(closure)
    return log_t.exp().item()


def describe(logits, masks, temperature):
    probs = (logits / temperature).softmax(dim=1)
    conf, pred = probs.max(dim=1)
    correct = masks[torch.arange(len(pred)), pred]
    cal = calibration(list(zip(conf.tolist(), correct.tolist())))
    return {"nll": round(nll(logits, masks, temperature).item(), 4),
            "ece_pct": cal["ece_pct"],
            "mean_conf_pct": round(100 * conf.mean().item(), 1),
            "accuracy_pct": round(100 * correct.float().mean().item(), 1)}


def main():
    p = argparse.ArgumentParser(description="Fit confidence calibration for dog_detector")
    add_model_args(p)
    p.add_argument("--threshold", type=float, default=0.6, help="detection threshold (app default)")
    p.add_argument("--crop-pad", type=float, default=0.05, help="crop padding (app default)")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--limit", type=int, help="use a random subset of N trainval images")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true", help="force CPU")
    args = p.parse_args()

    device = torch.device("cpu") if args.cpu else pick_device()
    print(f"Loading models on {device}...")
    model = DogBreedDetector(device, det_threshold=args.threshold, **model_kwargs(args),
                             crop_pad=args.crop_pad, calibrated=False)
    label_map = build_label_map(model)

    print("Collecting predictions on Oxford-IIIT Pet trainval (not the test split)...")
    logits, masks = collect(model, args, label_map)
    temperature = fit_temperature(logits, masks)
    before, after = describe(logits, masks, 1.0), describe(logits, masks, temperature)

    print(f"\nFitted temperature: {temperature:.3f}  "
          f"({'sharpens: model was under-confident' if temperature < 1 else 'softens: model was over-confident'})")
    print(f"  {'':<14}{'before':>10}{'after':>10}")
    for key, label in [("mean_conf_pct", "avg shown %"), ("accuracy_pct", "accuracy %"),
                       ("ece_pct", "ECE %"), ("nll", "NLL")]:
        print(f"  {label:<14}{before[key]:>10}{after[key]:>10}")
    print("  (accuracy can't change - only the shown % does; ECE should drop)")

    data = json.loads(CALIBRATION_FILE.read_text()) if CALIBRATION_FILE.exists() else {}
    data[model.calibration_key] = {
        "temperature": round(temperature, 4),
        "fitted_on": "Oxford-IIIT Pet trainval (timm/oxford-iiit-pet 'train' split)",
        "dogs": len(logits), "threshold": args.threshold, "crop_pad": args.crop_pad,
        "before": before, "after": after,
        "date": datetime.now().isoformat(timespec="seconds"),
    }
    CALIBRATION_FILE.write_text(json.dumps(data, indent=2) + "\n")
    print(f"\nSaved to {CALIBRATION_FILE.name} under '{model.calibration_key}'. "
          "dog_detector.py and benchmark.py will use it automatically.")


if __name__ == "__main__":
    main()
