"""Experiment 1: JPEG compression robustness (EfficientNet-B0, clean checkpoint).

Evaluates the EXACT SAME 1,186-image held-out test set and the EXACT SAME
checkpoint as the verified clean baseline, under: clean, JPEG Q90 / Q70 /
Q50 / Q30. JPEG is applied to the PIL image of EVERY test image (both
classes) before the unchanged clean preprocessing. No retraining, no
threshold change (0.5), no test-set modification. CUDA required.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset import DEFAULT_MANIFEST, DEFAULT_ROOT, load_image, load_manifest
from src.evaluate import evaluate_model
from src.experiment_config import ExperimentConfig
from src.model import load_checkpoint
from src.preprocessing import get_transform
from src.results import ExperimentResult
from src.transformations import apply_jpeg_compression

DEVICE = "cuda"
BATCH_SIZE = 32
NUM_WORKERS = 0
THRESHOLD = 0.5
TOLERANCE = 1e-4
SEED = 42

MODEL_NAME = "efficientnet_b0"
CHECKPOINT = Path("models/baseline/efficientnet_b0_best.pth")
EXPERIMENT = "jpeg_robustness"

# (condition label, jpeg quality or None for clean)
CONDITIONS: list[tuple[str, int | None]] = [
    ("clean", None),
    ("jpeg_q90", 90),
    ("jpeg_q70", 70),
    ("jpeg_q50", 50),
    ("jpeg_q30", 30),
]

EXPECTED_N, EXPECTED_REAL, EXPECTED_AI = 1186, 600, 586

# Verified clean baseline (Krish + independent GPU reproduction).
BASELINE_REFERENCE = {
    "accuracy": 0.9073,
    "precision": 0.9034,
    "recall": 0.9096,
    "f1": 0.9065,
    "roc_auc": 0.9649,
    "ai_false_negative_rate": 0.0904,
}


class JpegTestDataset(Dataset):
    """Test rows with optional PIL-level JPEG applied before clean preprocessing.

    The transform is applied unconditionally to every row (both classes);
    original files are only read, never written.
    """

    def __init__(self, rows, quality: int | None) -> None:
        self.rows = rows
        self.quality = quality
        self.transform = get_transform("test")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = load_image(row.path)  # fresh PIL read every time
        if self.quality is not None:
            image = apply_jpeg_compression(image, self.quality)
        return self.transform(image), torch.tensor(row.label, dtype=torch.long)


def require_cuda() -> None:
    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU/CUDA required; refusing CPU fallback.")
    print(f"Device: {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")


def snapshot_files(rows) -> dict[str, tuple[int, int]]:
    return {str(r.path): (r.path.stat().st_size, r.path.stat().st_mtime_ns) for r in rows}


def main() -> None:
    require_cuda()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    rows = load_manifest(
        manifest_path=DEFAULT_MANIFEST, root=DEFAULT_ROOT, split="test"
    )
    counts = {0: 0, 1: 0}
    for r in rows:
        counts[r.label] += 1
    print(f"Test rows: {len(rows)} (Real={counts[0]}, AI={counts[1]})")
    assert (len(rows), counts[0], counts[1]) == (EXPECTED_N, EXPECTED_REAL, EXPECTED_AI)
    before_files = snapshot_files(rows)

    model, bundle = load_checkpoint(CHECKPOINT, model_name=MODEL_NAME, map_location="cuda")
    model.to(DEVICE)
    assert all(p.device.type == "cuda" for p in model.parameters())
    params_before = [p.detach().clone() for p in model.parameters()]

    metric_rows: list[dict] = []
    pred_rows: list[dict] = []
    cm_rows: list[dict] = []
    for condition, quality in CONDITIONS:
        config = ExperimentConfig(
            model_name=MODEL_NAME,
            experiment_name=f"{EXPERIMENT}_{condition}",
            split="test",
            transformation="none" if quality is None else "jpeg",
            transformation_params={} if quality is None else {"quality": quality},
            condition="clean" if quality is None else "transformed",
            training_condition="clean",
            seed=SEED,
        )
        dataset = JpegTestDataset(rows, quality)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS)
        out = evaluate_model(model, loader, DEVICE, THRESHOLD)
        assert out["n_samples"] == EXPECTED_N, (condition, out["n_samples"])
        assert list(out["y_true"]) == [r.label for r in rows], "order/label drift"
        n0 = int((out["y_true"] == 0).sum())
        assert (n0, out["n_samples"] - n0) == (EXPECTED_REAL, EXPECTED_AI)

        result = ExperimentResult.from_metrics(config, out["metrics"])
        m = out["metrics"]
        metric_rows.append({
            "condition": condition, "jpeg_quality": quality if quality is not None else "",
            "n_samples": out["n_samples"], "accuracy": m["accuracy"],
            "precision": m["precision"], "ai_recall": m["ai_recall"],
            "ai_f1": m["f1"], "roc_auc": m["roc_auc"],
            "ai_fnr": m["ai_false_negative_rate"],
            "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"],
        })
        cm_rows.append({"condition": condition, "jpeg_quality": metric_rows[-1]["jpeg_quality"],
                        "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"]})
        for row, t, p, prob in zip(rows, out["y_true"], out["y_pred"], out["y_prob"]):
            pred_rows.append({
                "filename": row.filename, "split": row.split,
                "generator": row.generator, "true_label": int(t),
                "pred_label": int(p), "ai_prob": float(prob),
                "condition": condition,
                "jpeg_quality": quality if quality is not None else "",
            })
        (REPO_ROOT / "results" / "experiment_logs").mkdir(parents=True, exist_ok=True)
        with open(REPO_ROOT / "results" / "experiment_logs"
                  / f"{EXPERIMENT}_{condition}_{MODEL_NAME}.json", "w") as f:
            json.dump({
                "run_id": run_id, "config": config.to_dict(),
                "result": result.to_dict(), "checkpoint": str(CHECKPOINT),
                "threshold": THRESHOLD, "device": DEVICE,
                "batch_size": BATCH_SIZE, "seed": SEED,
            }, f, indent=2)
        print(f"[{condition}] n={out['n_samples']} acc={m['accuracy']:.4f} "
              f"prec={m['precision']:.4f} ai_rec={m['ai_recall']:.4f} "
              f"f1={m['f1']:.4f} auc={m['roc_auc']:.4f} fnr={m['ai_false_negative_rate']:.4f}")

    # Validation: clean reproduces verified baseline; weights/files untouched.
    clean = metric_rows[0]
    assert clean["condition"] == "clean"
    ref = {"accuracy": BASELINE_REFERENCE["accuracy"], "precision": BASELINE_REFERENCE["precision"],
           "ai_recall": BASELINE_REFERENCE["recall"], "ai_f1": BASELINE_REFERENCE["f1"],
           "roc_auc": BASELINE_REFERENCE["roc_auc"], "ai_fnr": BASELINE_REFERENCE["ai_false_negative_rate"]}
    bad = [f"{k}: got {clean[k]:.6f}, ref {v:.4f}" for k, v in ref.items()
           if abs(clean[k] - v) > TOLERANCE]
    if bad:
        print("CLEAN REPRODUCTION MISMATCH:"); print("\n".join(bad)); sys.exit(1)
    print("Clean condition reproduces the verified baseline within 1e-4.")
    for b, p in zip(params_before, model.parameters()):
        assert torch.equal(b, p), "model parameters changed during evaluation"
    print("Model parameters unchanged.")
    assert snapshot_files(rows) == before_files, "test files were modified"
    print("Original test images untouched (size+mtime identical).")

    (REPO_ROOT / "results" / "metrics").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "results" / "predictions").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "results" / "confusion_matrices").mkdir(parents=True, exist_ok=True)
    with open(REPO_ROOT / "results" / "metrics" / f"{EXPERIMENT}_{MODEL_NAME}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys())); w.writeheader(); w.writerows(metric_rows)
    with open(REPO_ROOT / "results" / "predictions" / f"{EXPERIMENT}_{MODEL_NAME}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pred_rows[0].keys())); w.writeheader(); w.writerows(pred_rows)
    with open(REPO_ROOT / "results" / "confusion_matrices" / f"{EXPERIMENT}_{MODEL_NAME}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cm_rows[0].keys())); w.writeheader(); w.writerows(cm_rows)
    print("Saved metrics / predictions / confusion matrices / experiment logs.")

    print("\nCondition | Accuracy | Precision | AI Recall | AI F1 | ROC-AUC | AI FNR")
    for r in metric_rows:
        print(f"{r['condition']:9s} | {r['accuracy']:.4f} | {r['precision']:.4f} | "
              f"{r['ai_recall']:.4f} | {r['ai_f1']:.4f} | {r['roc_auc']:.4f} | {r['ai_fnr']:.4f}")


if __name__ == "__main__":
    main()
