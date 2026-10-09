"""Experiment 5: transformation-aware robust fine-tuning (EfficientNet-B0).

Fine-tunes the frozen clean baseline checkpoint on TRAIN data only, with a
fixed per-sample augmentation policy (clean / JPEG / resize /
recompression / combined C1-C2), selecting on CLEAN validation accuracy.
Test data is never constructed, loaded, or evaluated here. CUDA required.

Launch later with (training NOT run by any check in this task):
    conda run -n ml_clean python scripts/train_robust.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset import DEFAULT_MANIFEST, DEFAULT_ROOT, GenImageDataset, load_manifest
from src.model import build_model, load_checkpoint
from src.preprocessing import IMAGE_SIZE, get_transform
from src.train import (
    SEED,
    build_optimizer,
    set_seed,
    train_one_epoch,
    validate,
)
from src.transformations import (
    apply_jpeg_compression,
    apply_recompression,
    apply_resize,
)

# ----------------------------------------------------------------------------
# Approved policy (FIXED — not CLI-overridable; any change needs team approval)
# ----------------------------------------------------------------------------
P_CLEAN, P_JPEG, P_RESIZE, P_RECOMP, P_COMBINED = 0.40, 0.20, 0.15, 0.15, 0.10
assert abs((P_CLEAN + P_JPEG + P_RESIZE + P_RECOMP + P_COMBINED) - 1.0) < 1e-12
JPEG_QUALITIES = (90, 70, 50, 30)
RESIZE_SCALES = (0.75, 0.50)
RECOMP_QUALITIES = (90, 70)
RECOMP_PASSES = 2
# C1: resize 0.50 -> JPEG 50 ; C2: JPEG 50 -> resize 0.50
COMBINED_CHOICES = ("C1", "C2")
C_SCALE, C_QUALITY = 0.50, 50

# Approved hyperparameters (CLI defaults mirror these).
MODEL_NAME = "efficientnet_b0"
INIT_CHECKPOINT = Path("models/baseline/efficientnet_b0_best.pth")
OPTIMIZER, LR, WEIGHT_DECAY, EPOCHS, BATCH_SIZE = "adam", 1e-4, 1e-4, 10, 16
NUM_WORKERS = 0

EXPECTED_TRAIN = (5600, 2800, 2800)  # n, real, ai
EXPECTED_VAL = (1200, 600, 600)

BEST_NAME = "efficientnet_b0_robust_best.pth"
HISTORY_NAME = "robust_training_history.json"
CONFIG_NAME = "robust_training_config.json"


def sample_augmentation(rng: random.Random) -> tuple[str, object]:
    """Draw one policy outcome; returns ``(name, pil_step)``.

    ``pil_step`` maps a PIL image to a PIL image, or is ``None`` for the
    clean portion. Never inspects any label.
    """
    roll = rng.random()
    if roll < P_CLEAN:
        return "clean", None
    if roll < P_CLEAN + P_JPEG:
        quality = rng.choice(JPEG_QUALITIES)
        return f"jpeg_q{quality}", lambda im, q=quality: apply_jpeg_compression(im, q)
    if roll < P_CLEAN + P_JPEG + P_RESIZE:
        scale = rng.choice(RESIZE_SCALES)
        return f"resize_{scale}", lambda im, s=scale: apply_resize(im, s)
    if roll < P_CLEAN + P_JPEG + P_RESIZE + P_RECOMP:
        quality = rng.choice(RECOMP_QUALITIES)
        return (f"recomp_q{quality}_p{RECOMP_PASSES}",
                lambda im, q=quality: apply_recompression(im, q, passes=RECOMP_PASSES))
    combo = rng.choice(COMBINED_CHOICES)
    if combo == "C1":
        return "combined_C1", lambda im: apply_jpeg_compression(apply_resize(im, C_SCALE), C_QUALITY)
    return "combined_C2", lambda im: apply_resize(apply_jpeg_compression(im, C_QUALITY), C_SCALE)


class RobustTrainTransform:
    """Train-split transform: stochastic policy, then deterministic preprocessing.

    Holds one ``random.Random`` stream per DataLoader worker id (main
    process = -1), each seeded from ``(seed, worker_id)``, so the policy is
    reproducible for a fixed worker count. Operates on a fresh PIL copy;
    dataset files are only read.
    """

    def __init__(self, seed: int = SEED, image_size: int = IMAGE_SIZE) -> None:
        self.seed = seed
        self.base = get_transform("train", image_size, augment=False)
        self._rngs: dict[int, random.Random] = {}
        self.last_category: str = "clean"  # introspection for tests only

    def _rng(self) -> random.Random:
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else -1
        rng = self._rngs.get(wid)
        if rng is None:
            rng = random.Random(self.seed * 100003 + (wid + 1))
            self._rngs[wid] = rng
        return rng

    def sample_category(self) -> tuple[str, object]:
        return sample_augmentation(self._rng())

    def __call__(self, image):
        image = image.copy()
        name, step = self.sample_category()
        self.last_category = name
        if step is not None:
            image = step(image)
        return self.base(image)


def _worker_init(seed: int):
    def _init(worker_id: int) -> None:
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    return _init


def _counts(rows) -> tuple[int, int, int]:
    n0 = sum(1 for r in rows if r.label == 0)
    return len(rows), n0, len(rows) - n0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robust fine-tuning (train/val only).")
    parser.add_argument("--optimizer", default=OPTIMIZER)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--init-checkpoint", default=str(INIT_CHECKPOINT))
    parser.add_argument("--checkpoint-dir", default="models/robust")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "GPU/CUDA is required for robust training "
            f"(cuda.is_available()={torch.cuda.is_available()}). "
            "Refusing to fall back to CPU."
        )
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    set_seed(args.seed)

    # Train/val only — the string "test" must never select data here.
    train_rows = load_manifest(DEFAULT_MANIFEST, DEFAULT_ROOT, split="train")
    val_rows = load_manifest(DEFAULT_MANIFEST, DEFAULT_ROOT, split="val")
    assert _counts(train_rows) == EXPECTED_TRAIN, _counts(train_rows)
    assert _counts(val_rows) == EXPECTED_VAL, _counts(val_rows)
    print(f"Train: {len(train_rows)} (2800/2800)  Val: {len(val_rows)} (600/600)")

    ckpt_path = Path(args.init_checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Init checkpoint missing: {ckpt_path}")
    model, bundle = load_checkpoint(ckpt_path, model_name=MODEL_NAME)
    assert bundle.get("model_name") == MODEL_NAME, bundle.get("model_name")
    head = model.classifier[1]
    assert isinstance(head, nn.Linear) and head.out_features == 2, (
        f"Unexpected classifier head: {head}")
    assert all(p.requires_grad for p in model.parameters()), "params must all train"
    model.to(device)
    print(f"Initialized from {ckpt_path} (head: Linear(*, 2)); all params trainable.")

    out_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [out_dir / BEST_NAME, out_dir / HISTORY_NAME, out_dir / CONFIG_NAME]
    existing = [str(p) for p in targets if p.exists()]
    if existing and not args.overwrite:
        raise RuntimeError(
            "Refusing to overwrite existing robust outputs: "
            + ", ".join(existing) + " (pass --overwrite to replace).")

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        GenImageDataset(split="train", transform=RobustTrainTransform(args.seed),
                        manifest_path=DEFAULT_MANIFEST, root=DEFAULT_ROOT),
        batch_size=args.batch_size, shuffle=True, num_workers=NUM_WORKERS,
        generator=generator, worker_init_fn=_worker_init(args.seed),
    )
    val_loader = DataLoader(
        GenImageDataset(split="val", transform=get_transform("val"),
                        manifest_path=DEFAULT_MANIFEST, root=DEFAULT_ROOT),
        batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, args.optimizer, args.lr, args.weight_decay)

    policy = {
        "clean": P_CLEAN, "jpeg": {"p": P_JPEG, "qualities": list(JPEG_QUALITIES)},
        "resize": {"p": P_RESIZE, "scales": list(RESIZE_SCALES)},
        "recompression": {"p": P_RECOMP, "qualities": list(RECOMP_QUALITIES),
                          "passes": RECOMP_PASSES},
        "combined": {"p": P_COMBINED, "choices": list(COMBINED_CHOICES),
                     "C1": f"resize({C_SCALE})->jpeg({C_QUALITY})",
                     "C2": f"jpeg({C_QUALITY})->resize({C_SCALE})"},
        "unseen_reserved": ["resize_0.25", "recomp_q50", "recomp_q30", "C3"],
    }
    history: dict = {"train_loss": [], "train_acc": [], "val_loss": [],
                     "val_acc": [], "best_epoch": -1, "best_val_acc": -1.0,
                     "best_val_loss": None,
                     "selection_rule": "highest clean val accuracy; "
                                       "tie -> lower clean val loss, then earlier",
                     "config": {
                         "model": MODEL_NAME, "init_checkpoint": str(ckpt_path),
                         "optimizer": args.optimizer, "lr": args.lr,
                         "weight_decay": args.weight_decay, "epochs": args.epochs,
                         "batch_size": args.batch_size, "seed": args.seed,
                         "device": "cuda", "policy": policy,
                         "train_n": len(train_rows), "val_n": len(val_rows),
                     }}

    best_acc, best_loss = -1.0, float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"[robust] epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
            best_acc, best_loss = val_acc, val_loss
            history["best_epoch"], history["best_val_acc"] = epoch, val_acc
            history["best_val_loss"] = val_loss
            torch.save({"model_name": MODEL_NAME, "num_classes": 2,
                        "label_map": {"real": 0, "ai": 1}, "epoch": epoch,
                        "config": history["config"], "state_dict": model.state_dict()},
                       out_dir / BEST_NAME)
    with open(out_dir / HISTORY_NAME, "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / CONFIG_NAME, "w") as f:
        json.dump({"config": history["config"],
                   "selection_rule": history["selection_rule"]}, f, indent=2)
    print(f"[robust] best epoch={history['best_epoch']} "
          f"best_val_acc={history['best_val_acc']:.4f}")


if __name__ == "__main__":
    main()
