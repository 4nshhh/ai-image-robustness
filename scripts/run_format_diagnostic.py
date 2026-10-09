"""Experiment 9: evaluation-only format-normalization diagnostic.

Re-encodes held-out test images into common formats (lossless PNG; JPEG
Q95 with 4:4:4 chroma subsampling) to measure how the frozen baseline and
robust EfficientNet-B0 checkpoints respond when the perfect Real-JPEG /
AI-PNG metadata separation is neutralized at input time. Eval only: no
training, no tuning, threshold 0.5, CUDA required. Originals untouched.

Fixed settings (documented): PNG = lossless round-trip; JPEG = quality 95,
subsampling=0 (4:4:4, minimal chroma damage to isolate container effects
from heavy compression). Re-encoding happens on decoded RGB copies BEFORE
the unchanged 224x224 model preprocessing.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
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

DEVICE = "cuda"
BATCH_SIZE = 32
NUM_WORKERS = 0
THRESHOLD = 0.5
TOLERANCE = 1e-4
SEED = 42

MODEL_NAME = "efficientnet_b0"
MODELS = {
    "baseline": Path("models/baseline/efficientnet_b0_best.pth"),
    "robust": Path("models/robust/efficientnet_b0_robust_best.pth"),
}
EXPERIMENT = "format_diagnostic"

JPEG_QUALITY = 95
JPEG_SUBSAMPLING = 0  # 4:4:4


def reencode_png(image: Image.Image) -> Image.Image:
    """Lossless PNG round-trip of a decoded RGB image (new object)."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    with Image.open(buf) as decoded:
        decoded.load()
        return decoded.convert("RGB")


def reencode_jpeg(image: Image.Image, quality: int = JPEG_QUALITY,
                  subsampling: int = JPEG_SUBSAMPLING) -> Image.Image:
    """JPEG round-trip at fixed quality/subsampling (new object)."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality,
                              subsampling=subsampling)
    buf.seek(0)
    with Image.open(buf) as decoded:
        decoded.load()
        return decoded.convert("RGB")


# (condition, pipeline label, pil step or None, transformation, params)
CONDITIONS = [
    ("clean", "none", None, "none", {}),
    ("png", "png-lossless", reencode_png, "format",
     {"target_format": "PNG", "lossless": True}),
    ("jpeg_q95", "jpeg(Q95,s0)", reencode_jpeg, "format",
     {"target_format": "JPEG", "quality": JPEG_QUALITY,
      "subsampling": JPEG_SUBSAMPLING}),
]

EXPECTED_N, EXPECTED_REAL, EXPECTED_AI = 1186, 600, 586
RATE_KEYS = ["accuracy", "precision", "ai_recall", "ai_f1", "roc_auc",
             "ai_fnr", "real_recall", "fpr"]


class FormatDataset(Dataset):
    """Test rows with optional format re-encode before clean preprocessing."""

    def __init__(self, rows, step) -> None:
        self.rows = rows
        self.step = step
        self.transform = get_transform("test")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = load_image(row.path)
        if self.step is not None:
            image = self.step(image)
        return self.transform(image), torch.tensor(row.label, dtype=torch.long)


def require_cuda() -> None:
    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU/CUDA required; refusing CPU fallback.")
    print(f"Device: {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")


def main() -> None:
    require_cuda()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    rows = load_manifest(DEFAULT_MANIFEST, DEFAULT_ROOT, split="test")
    counts = {0: 0, 1: 0}
    for r in rows:
        counts[r.label] += 1
    assert (len(rows), counts[0], counts[1]) == (EXPECTED_N, EXPECTED_REAL, EXPECTED_AI)
    print(f"Test rows: {len(rows)} (Real={counts[0]}, AI={counts[1]})")
    before_files = {str(r.path): (r.path.stat().st_size, r.path.stat().st_mtime_ns) for r in rows}

    metric_rows: list[dict] = []
    pred_rows: list[dict] = []
    cm_rows: list[dict] = []
    reference_y_true = None
    per_mc: dict[tuple[str, str], dict] = {}

    for model_id, ckpt in MODELS.items():
        model, bundle = load_checkpoint(ckpt, model_name=MODEL_NAME, map_location="cuda")
        assert bundle.get("model_name") == MODEL_NAME
        assert model.classifier[1].out_features == 2
        model.to(DEVICE)
        assert all(p.device.type == "cuda" for p in model.parameters())
        params_before = [p.detach().clone() for p in model.parameters()]

        for condition, pipeline, step, transformation, params in CONDITIONS:
            config = ExperimentConfig(
                model_name=f"{MODEL_NAME}_{model_id}",
                experiment_name=f"{EXPERIMENT}_{condition}_{model_id}",
                split="test", transformation=transformation,
                transformation_params=dict(params),
                condition="clean" if step is None else "transformed",
                training_condition="clean" if model_id == "baseline" else "robust",
                seed=SEED,
            )
            loader = DataLoader(FormatDataset(rows, step), batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=NUM_WORKERS)
            out = evaluate_model(model, loader, DEVICE, THRESHOLD)
            assert out["n_samples"] == EXPECTED_N, (model_id, condition)
            assert list(out["y_true"]) == [r.label for r in rows], "label drift"
            if reference_y_true is None:
                reference_y_true = list(out["y_true"])
            assert list(out["y_true"]) == reference_y_true, "y_true mismatch"
            n0 = int((out["y_true"] == 0).sum())
            assert (n0, out["n_samples"] - n0) == (EXPECTED_REAL, EXPECTED_AI)
            probs = np.asarray(out["y_prob"], dtype=float)
            assert np.all(np.isfinite(probs)) and probs.min() >= 0.0 and probs.max() <= 1.0

            result = ExperimentResult.from_metrics(config, out["metrics"])
            m = out["metrics"]
            metric_rows.append({
                "model": model_id, "condition": condition, "pipeline": pipeline,
                "n_samples": out["n_samples"], "accuracy": m["accuracy"],
                "precision": m["precision"], "ai_recall": m["ai_recall"],
                "ai_f1": m["f1"], "roc_auc": m["roc_auc"],
                "ai_fnr": m["ai_false_negative_rate"],
                "real_recall": m["real_recall"], "fpr": m["false_positive_rate"],
                "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"],
            })
            cm_rows.append({"model": model_id, "condition": condition,
                            "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"]})
            per_mc[(model_id, condition)] = metric_rows[-1]
            for row, t, p, prob in zip(rows, out["y_true"], out["y_pred"], out["y_prob"]):
                pred_rows.append({
                    "example_id": str(row.path), "filename": row.filename,
                    "split": row.split, "generator": row.generator,
                    "true_label": int(t), "model": model_id,
                    "condition": condition, "pipeline": pipeline,
                    "ai_prob": float(prob), "pred_label": int(p),
                })
            (REPO_ROOT / "results" / "experiment_logs").mkdir(parents=True, exist_ok=True)
            with open(REPO_ROOT / "results" / "experiment_logs"
                      / f"{EXPERIMENT}_{condition}_{model_id}.json", "w") as f:
                json.dump({"run_id": run_id, "config": config.to_dict(),
                           "result": result.to_dict(), "checkpoint": str(ckpt),
                           "threshold": THRESHOLD, "device": DEVICE,
                           "batch_size": BATCH_SIZE, "seed": SEED,
                           "jpeg_quality": JPEG_QUALITY,
                           "jpeg_subsampling": JPEG_SUBSAMPLING}, f, indent=2)
            print(f"[{model_id:8s} {condition:8s}] n={out['n_samples']} "
                  f"acc={m['accuracy']:.4f} ai_rec={m['ai_recall']:.4f} "
                  f"fnr={m['ai_false_negative_rate']:.4f} auc={m['roc_auc']:.4f} "
                  f"CM=[[ {m['tn']},{m['fp']}],[ {m['fn']},{m['tp']}]]")

        for b, p in zip(params_before, model.parameters()):
            assert torch.equal(b, p), f"{model_id} parameters changed"
        print(f"{model_id}: parameters unchanged.")

    # Baseline clean must reproduce the verified reference.
    bc = per_mc[("baseline", "clean")]
    ref = {"accuracy": 0.9073, "precision": 0.9034, "ai_recall": 0.9096,
           "ai_f1": 0.9065, "roc_auc": 0.9649, "ai_fnr": 0.0904}
    bad = [f"{k}: got {bc[k]:.6f}, ref {v:.4f}" for k, v in ref.items()
           if abs(bc[k] - v) > TOLERANCE]
    if bad:
        print("BASELINE CLEAN MISMATCH:"); print("\n".join(bad)); sys.exit(1)
    print("Baseline clean reproduces verified reference within 1e-4.")
    after_files = {str(r.path): (r.path.stat().st_size, r.path.stat().st_mtime_ns) for r in rows}
    assert after_files == before_files, "test files were modified"
    print("Test images and checkpoints untouched.")
    assert len(metric_rows) == 6, len(metric_rows)

    comp_rows = []
    for condition, pipeline, _, _, _ in CONDITIONS:
        b, r = per_mc[("baseline", condition)], per_mc[("robust", condition)]
        row = {"condition": condition, "pipeline": pipeline, "n_samples": EXPECTED_N}
        for k in RATE_KEYS:
            row[f"baseline_{k}"] = b[k]
            row[f"robust_{k}"] = r[k]
            row[f"delta_pp_{k}"] = (r[k] - b[k]) * 100.0
        for prefix, src in (("baseline", b), ("robust", r)):
            for c in ("tn", "fp", "fn", "tp"):
                row[f"{prefix}_{c}"] = src[c]
        comp_rows.append(row)

    for sub in ("metrics", "predictions", "confusion_matrices"):
        (REPO_ROOT / "results" / sub).mkdir(parents=True, exist_ok=True)
    with open(REPO_ROOT / "results" / "metrics" / f"{EXPERIMENT}_{MODEL_NAME}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys())); w.writeheader(); w.writerows(metric_rows)
    with open(REPO_ROOT / "results" / "predictions" / f"{EXPERIMENT}_{MODEL_NAME}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pred_rows[0].keys())); w.writeheader(); w.writerows(pred_rows)
    with open(REPO_ROOT / "results" / "confusion_matrices" / f"{EXPERIMENT}_{MODEL_NAME}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cm_rows[0].keys())); w.writeheader(); w.writerows(cm_rows)
    with open(REPO_ROOT / "results" / "metrics" / f"baseline_vs_robust_format_{MODEL_NAME}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comp_rows[0].keys())); w.writeheader(); w.writerows(comp_rows)
    print("Saved metrics (6 rows) / predictions / confusion matrices / comparison / logs.")

    print("\nModel | Condition | Accuracy | AI Recall | AI FNR | ROC-AUC | CM")
    for r in metric_rows:
        print(f"{r['model']:8s} | {r['condition']:8s} | {r['accuracy']:.4f} | "
              f"{r['ai_recall']:.4f} | {r['ai_fnr']:.4f} | {r['roc_auc']:.4f} | "
              f"[[{r['tn']},{r['fp']}],[{r['fn']},{r['tp']}]]")


if __name__ == "__main__":
    main()
