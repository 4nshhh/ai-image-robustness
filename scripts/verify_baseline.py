"""Verify the clean EfficientNet-B0 baseline on GPU (no CPU fallback).

Loads the finalized test split and the clean baseline checkpoint, evaluates
with the existing ``src.evaluate.evaluate_model`` (threshold 0.5), prints
metrics plus confusion matrix, and compares against Krish's reference
results within tolerance. No retraining, no test-set modification, no
robustness transformations.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset import DEFAULT_MANIFEST, DEFAULT_ROOT, GenImageDataset
from src.evaluate import evaluate_model
from src.model import load_checkpoint
from src.preprocessing import get_transform

DEVICE = "cuda"
BATCH_SIZE = 32
NUM_WORKERS = 0
THRESHOLD = 0.5
TOLERANCE = 1e-4

CHECKPOINT = Path("models/baseline/efficientnet_b0_best.pth")

REFERENCE = {
    "accuracy": 0.9073,
    "precision": 0.9034,
    "recall": 0.9096,  # AI recall
    "f1": 0.9065,
    "roc_auc": 0.9649,
    "ai_false_negative_rate": 0.0904,
}

EXPECTED_TEST_N = 1186
EXPECTED_REAL = 600
EXPECTED_AI = 586


def require_cuda() -> torch.device:
    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU/CUDA is required for baseline verification "
            f"(torch {torch.__version__} reports no CUDA device). "
            "Refusing to fall back to CPU — fix the CUDA setup and re-run."
        )
    index = torch.cuda.current_device()
    print(f"Device: {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(index)}")
    print(f"CUDA device index: {index}")
    return torch.device(f"{DEVICE}:{index}")


def check_git_safety() -> None:
    status = subprocess.run(
        ["git", "status", "--short"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=True,
    ).stdout
    print("--- git status --short ---")
    print(status if status.strip() else "(clean)")
    bad = [
        line
        for line in status.splitlines()
        if line.strip().endswith((".pth", ".pt", ".ckpt", ".png", ".jpeg", ".jpg"))
        or "data/splits/" in line
        or "data/raw/" in line
        or "data/processed/" in line
    ]
    if bad:
        raise RuntimeError(
            "Dataset images or checkpoints appear tracked/staged:\n"
            + "\n".join(bad)
        )
    print("Git safety OK: no dataset images or checkpoints tracked/staged.")


def main() -> None:
    device = require_cuda()

    test_set = GenImageDataset(
        split="test",
        transform=get_transform("test"),
        manifest_path=DEFAULT_MANIFEST,
        root=DEFAULT_ROOT,
    )
    assert all(
        Path(row.path).parent.parent.name == "test" for row in test_set.rows
    ), "Non-test rows leaked into the test split."
    counts = test_set.class_counts()
    n_real, n_ai = counts.get(0, 0), counts.get(1, 0)
    print(f"Test samples: {len(test_set)} (Real={n_real}, AI={n_ai})")
    if (len(test_set), n_real, n_ai) != (EXPECTED_TEST_N, EXPECTED_REAL, EXPECTED_AI):
        raise RuntimeError(
            f"Unexpected test composition: got {(len(test_set), n_real, n_ai)}, "
            f"expected {(EXPECTED_TEST_N, EXPECTED_REAL, EXPECTED_AI)}."
        )

    model, bundle = load_checkpoint(CHECKPOINT, model_name="efficientnet_b0", map_location="cuda")
    model.to(device)
    off_device = [n for n, p in model.named_parameters() if p.device.type != "cuda"]
    if off_device:
        raise RuntimeError(f"Parameters not on CUDA: {off_device[:5]}")
    print(f"Model on CUDA: all {sum(1 for _ in model.parameters())} parameter tensors verified.")
    print(f"Checkpoint: model={bundle.get('model_name')}, "
          f"num_classes={bundle.get('num_classes')}, label_map={bundle.get('label_map')}")

    loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    out = evaluate_model(model, loader, device, THRESHOLD)
    m = out["metrics"]
    print(f"Evaluated samples: {out['n_samples']}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]:\n{m['confusion_matrix']}")
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc",
                "ai_recall", "ai_false_negative_rate"):
        print(f"{key}: {m[key]:.4f}")

    mismatches = [
        f"{k}: got {m[k]:.6f}, reference {v:.4f}"
        for k, v in REFERENCE.items()
        if abs(float(m[k]) - v) > TOLERANCE
    ]
    if mismatches:
        print("BASELINE MISMATCH — stopping for investigation:")
        print("\n".join(mismatches))
        sys.exit(1)
    print(f"Baseline VERIFIED: all reference metrics match within {TOLERANCE}.")

    check_git_safety()


if __name__ == "__main__":
    main()
