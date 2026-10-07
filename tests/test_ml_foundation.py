"""Lightweight verification for the Person 1 ML foundation.

Run from the repository root::

    python tests/test_ml_foundation.py

Uses only stdlib + torch/torchvision/PIL/numpy (no pytest needed).
Exits nonzero on the first failure. The smoke-training section uses a
tiny image subset and writes checkpoints to the system temp dir; the
real dataset is never modified.

GPU REQUIREMENT (standing rule): any model forward test, smoke
training, or training-related test MUST run on CUDA. The suite prints
the detected GPU and selected device, and STOPS with an explicit
message instead of silently falling back to CPU when CUDA is
unavailable. Lightweight non-model checks (dataset, preprocessing)
may run on CPU.
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import (  # noqa: E402
    LABEL_AI,
    LABEL_MAP,
    LABEL_REAL,
    GenImageDataset,
    load_manifest,
)
from src.model import (  # noqa: E402
    build_model,
    count_parameters,
    load_checkpoint,
    save_checkpoint,
)
from src.preprocessing import (  # noqa: E402
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_eval_transform,
    build_train_transform,
    get_transform,
    load_image,
)
from src.train import (  # noqa: E402
    TrainConfig,
    fit,
    make_dataloaders,
    set_seed,
)

PASS = []


def require_cuda() -> torch.device:
    """Enforce the GPU rule for all model/training tests.

    Prints the torch build, CUDA availability, detected GPU name, and
    selected device. If CUDA is unavailable, STOPS here with an explicit
    error instead of silently falling back to CPU.

    Returns:
        ``torch.device("cuda")`` when a GPU is present.

    Raises:
        RuntimeError: When CUDA is unavailable.
    """
    available = torch.cuda.is_available()
    print(f"torch={torch.__version__} cuda_available={available}")
    if not available:
        raise RuntimeError(
            "STOP: CUDA is unavailable on this machine, so model forward, "
            "smoke-training, and training-related tests will NOT run on "
            "CPU. Re-run on a CUDA machine. "
            "(Dataset/preprocessing checks above already passed on CPU.)"
        )
    name = torch.cuda.get_device_name(0)
    device = torch.device("cuda")
    print(f"detected GPU: {name} | selected device: {device}")
    return device


def check(name: str, condition: bool) -> None:
    """Record a named check; raise immediately on failure."""
    if not condition:
        raise AssertionError(f"FAILED: {name}")
    PASS.append(name)
    print(f"  ok: {name}")


def test_dataset() -> None:
    print("[dataset]")
    check("label mapping real=0 ai=1",
          LABEL_REAL == 0 and LABEL_AI == 1
          and LABEL_MAP == {"real": 0, "ai": 1})

    train_rows = load_manifest(split="train")
    val_rows = load_manifest(split="val")
    test_rows = load_manifest(split="test")
    check("train rows = 5600", len(train_rows) == 5600)
    check("val rows = 1200", len(val_rows) == 1200)
    check("test rows = 1186", len(test_rows) == 1186)

    def counts(rows):
        real = sum(1 for r in rows if r.label == LABEL_REAL)
        ai = sum(1 for r in rows if r.label == LABEL_AI)
        return real, ai

    check("train 2800/2800", counts(train_rows) == (2800, 2800))
    check("val 600/600", counts(val_rows) == (600, 600))
    check("test 600 real / 586 ai", counts(test_rows) == (600, 586))
    check("every referenced file exists (load would have raised)",
          all(r.path.is_file() for r in train_rows + val_rows + test_rows))

    # Labels must come from the manifest label column, not filenames:
    # a file NAMED like a Real image but LABELED ai must yield label 1.
    from PIL import Image as PILImage
    with tempfile.TemporaryDirectory() as tmp:
        tmproot = Path(tmp)
        (tmproot / "train" / "ai").mkdir(parents=True)
        PILImage.new("RGB", (16, 16), (1, 2, 3)).save(
            tmproot / "train" / "ai" / "real_00001.jpeg")
        fake = tmproot / "manifest.csv"
        with open(fake, "w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["filename", "split", "label", "generator", "path"]
            )
            w.writeheader()
            w.writerow({
                "filename": "real_00001.jpeg", "split": "train",
                "label": "ai", "generator": "real", "path": "ignored",
            })
        tricky = load_manifest(
            manifest_path=fake, root=tmproot, split="train")
        check("no filename-based label inference (label column wins)",
              len(tricky) == 1 and tricky[0].label == LABEL_AI)
        bad = tmproot / "bad.csv"
        with open(bad, "w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["filename", "split", "label", "generator", "path"]
            )
            w.writeheader()
            w.writerow({
                "filename": "x.png", "split": "train", "label": "maybe",
                "generator": "g", "path": "ignored",
            })
        try:
            load_manifest(manifest_path=bad, root=tmproot, split="train")
            check("unknown label string raises", False)
        except ValueError:
            check("unknown label string raises", True)

    ds = GenImageDataset(split="val", transform=build_eval_transform())
    check("val dataset len = 1200", len(ds) == 1200)
    img, label = ds[0]
    check("dataset item is (tensor, long label)",
          isinstance(img, torch.Tensor) and label.dtype == torch.long
          and label.item() in (0, 1))
    check("dataset labels match manifest rows",
          all(GenImageDataset(split="val")[i][1].item() == r.label
              for i, r in enumerate(load_manifest(split="val")[:8])))
    check("class_counts sane", ds.class_counts() == {0: 600, 1: 600})


def test_preprocessing() -> None:
    print("[preprocessing]")
    check("imagenet constants explicit",
          IMAGENET_MEAN == [0.485, 0.456, 0.406]
          and IMAGENET_STD == [0.229, 0.224, 0.225]
          and IMAGE_SIZE == 224)

    first = load_manifest(split="train")[0].path
    img = load_image(first)
    check("real image loads as RGB", img.mode == "RGB")

    from PIL import Image as PILImage
    gray = PILImage.new("L", (32, 32), 128)
    rgba = PILImage.new("RGBA", (32, 32), (10, 20, 30, 0))
    with tempfile.TemporaryDirectory() as tmp:
        gp, ap = Path(tmp) / "g.png", Path(tmp) / "a.png"
        gray.save(gp)
        rgba.save(ap)
        check("grayscale -> RGB", load_image(gp).mode == "RGB")
        check("RGBA -> RGB", load_image(ap).mode == "RGB")

    t_train = build_train_transform()
    t_eval = build_eval_transform()
    out = t_eval(img)
    check("eval output shape [3,224,224]",
          tuple(out.shape) == (3, IMAGE_SIZE, IMAGE_SIZE))
    check("train and eval transforms callable",
          callable(t_train) and callable(t_eval))
    check("eval deterministic (two passes identical)",
          bool((t_eval(img) == t_eval(img)).all()))
    try:
        get_transform("val", augment=True)
        check("val+augment raises", False)
    except ValueError:
        check("val+augment raises", True)
    try:
        get_transform("nope")
        check("unknown split raises", False)
    except ValueError:
        check("unknown split raises", True)


def test_models(device: torch.device) -> None:
    print(f"[models] device={device}")
    for name in ("resnet50", "efficientnet_b0"):
        model = build_model(name, pretrained=False).to(device)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=device))
        check(f"{name} forward -> [2,2]", tuple(out.shape) == (2, 2))
        check(f"{name} has trainable params", count_parameters(model) > 0)
    try:
        build_model("vit_huge")
        check("unknown model raises", False)
    except ValueError:
        check("unknown model raises", True)

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "m.pth"
        model = build_model("resnet50", pretrained=False).to(device)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        save_checkpoint(ckpt, model, "resnet50", extra={"epoch": 3})
        check("checkpoint file created", ckpt.is_file())
        loaded, bundle = load_checkpoint(ckpt, map_location=device)
        loaded = loaded.to(device)
        check("reloaded model params live on CUDA",
              all(p.is_cuda for p in loaded.parameters()))
        check("checkpoint reload keeps architecture+weights",
              bundle["model_name"] == "resnet50"
              and bundle["extra"] == {"epoch": 3}
              and all(bool((loaded.state_dict()[k] == v).all())
                      for k, v in before.items()))


def test_training_smoke(device: torch.device) -> None:
    print(f"[training smoke] device={device}")
    check("device is CUDA (no silent CPU fallback)", str(device) == "cuda")
    set_seed(42)
    check("set_seed runs", True)

    # Tiny stratified subset: 4 real + 4 ai for train, 2+2 for val.
    train_rows = load_manifest(split="train")
    val_rows = load_manifest(split="val")
    real_idx = [i for i, r in enumerate(train_rows) if r.label == 0][:4]
    ai_idx = [i for i, r in enumerate(train_rows) if r.label == 1][:4]
    v_real = [i for i, r in enumerate(val_rows) if r.label == 0][:2]
    v_ai = [i for i, r in enumerate(val_rows) if r.label == 1][:2]
    train_loader, val_loader = make_dataloaders(
        batch_size=4, train_indices=real_idx + ai_idx,
        val_indices=v_real + v_ai,
    )
    check("smoke loaders built (8 train / 4 val)",
          len(train_loader.dataset) == 8 and len(val_loader.dataset) == 4)
    import src.train as train_module
    check("no test-split loader in training module",
          not hasattr(train_module, "make_test_loader"))

    with tempfile.TemporaryDirectory() as tmp:
        cfg = TrainConfig(
            model_name="resnet50", pretrained=False, epochs=1,
            batch_size=4, checkpoint_dir=Path(tmp), device="cuda",
            train_indices=real_idx + ai_idx, val_indices=v_real + v_ai,
        )
        history = fit(cfg)
        check("smoke run used the 8/4 subset (acc fractions match)",
              (history["train_acc"][0] * 8).is_integer()
              and (history["val_acc"][0] * 4).is_integer())
        check("history has one epoch of train/val metrics",
              len(history["train_loss"]) == 1
              and len(history["val_loss"]) == 1
              and len(history["train_acc"]) == 1
              and len(history["val_acc"]) == 1)
        check("history values finite",
              all(isinstance(v[0], float) and v[0] == v[0]
                  for v in (history["train_loss"], history["val_loss"],
                            history["train_acc"], history["val_acc"])))
        check("best checkpoint selected on val",
              history["best_epoch"] == 1
              and (Path(tmp) / "resnet50_best.pth").is_file()
              and (Path(tmp) / "resnet50_last.pth").is_file()
              and (Path(tmp) / "resnet50_history.json").is_file())
        loaded, bundle = load_checkpoint(
            Path(tmp) / "resnet50_best.pth", map_location=device)
        check("best checkpoint reloads", bundle["model_name"] == "resnet50")


def main() -> None:
    # CPU-allowed: dataset and preprocessing involve no models.
    test_dataset()
    test_preprocessing()
    # GPU-gated: stops with an explicit message when CUDA is absent.
    device = require_cuda()
    test_models(device)
    test_training_smoke(device)
    print(f"\nALL {len(PASS)} CHECKS PASSED (full training NOT run)")


if __name__ == "__main__":
    main()
