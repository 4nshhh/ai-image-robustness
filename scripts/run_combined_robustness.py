"""Experiment 4: Combined transformation robustness (EfficientNet-B0).

Same 1,186-image test set + same checkpoint as the verified baseline, under:
clean, C1 = Resize 0.50 -> JPEG Q50, C2 = JPEG Q50 -> Resize 0.50,
C3 = Resize 0.50 -> JPEG Q50 -> JPEG Q50. Pipelines are expressed with the
existing ``apply_pipeline`` (order preserved; C3 holds two sequential JPEG
passes) and applied to the PIL image of EVERY test image (both classes)
BEFORE the unchanged clean preprocessing. No retraining, threshold 0.5,
CUDA required.
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
from src.transformations import (
    apply_jpeg_compression,
    apply_pipeline,
    apply_resize,
)

DEVICE = "cuda"
BATCH_SIZE = 32
NUM_WORKERS = 0
THRESHOLD = 0.5
TOLERANCE = 1e-4
SEED = 42

MODEL_NAME = "efficientnet_b0"
CHECKPOINT = Path("models/baseline/efficientnet_b0_best.pth")
EXPERIMENT = "combined_robustness"

SCALE, QUALITY = 0.50, 50


def _resize(im):
    return apply_resize(im, SCALE)


def _jpeg(im):
    return apply_jpeg_compression(im, QUALITY)


# (condition, pipeline string, ordered steps or None, resize_scale, jpeg_quality, jpeg_passes)
CONDITIONS: list[tuple[str, str, list | None, float | None, int | None, int]] = [
    ("clean", "none", None, None, None, 0),
    ("c1_resize_jpeg", "resize(0.50)->jpeg(Q50)", [_resize, _jpeg], SCALE, QUALITY, 1),
    ("c2_jpeg_resize", "jpeg(Q50)->resize(0.50)", [_jpeg, _resize], SCALE, QUALITY, 1),
    ("c3_resize_jpeg_jpeg", "resize(0.50)->jpeg(Q50)->jpeg(Q50)",
     [_resize, _jpeg, _jpeg], SCALE, QUALITY, 2),
]

EXPECTED_N, EXPECTED_REAL, EXPECTED_AI = 1186, 600, 586

BASELINE_REFERENCE = {
    "accuracy": 0.9073,
    "precision": 0.9034,
    "recall": 0.9096,
    "f1": 0.9065,
    "roc_auc": 0.9649,
    "ai_false_negative_rate": 0.0904,
}


class CombinedTestDataset(Dataset):
    """Test rows with an optional ordered PIL pipeline before clean preprocessing."""

    def __init__(self, rows, steps: list | None) -> None:
        self.rows = rows
        self.steps = steps
        self.transform = get_transform("test")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = load_image(row.path)
        if self.steps is not None:
            image = apply_pipeline(image, self.steps)
        return self.transform(image), torch.tensor(row.label, dtype=torch.long)


def require_cuda() -> None:
    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU/CUDA required; refusing CPU fallback.")
    print(f"Device: {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"CUDA device index: {torch.cuda.current_device()}")


def snapshot_files(rows) -> dict[str, tuple[int, int]]:
    return {str(r.path): (r.path.stat().st_size, r.path.stat().st_mtime_ns) for r in rows}


def main() -> None:
    require_cuda()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    rows = load_manifest(manifest_path=DEFAULT_MANIFEST, root=DEFAULT_ROOT, split="test")
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
    reference_y_true = None
    for condition, pipeline, steps, scale, quality, passes in CONDITIONS:
        config = ExperimentConfig(
            model_name=MODEL_NAME,
            experiment_name=f"{EXPERIMENT}_{condition}",
            split="test",
            transformation="none" if steps is None else "combined",
            transformation_params={} if steps is None else {
                "pipeline": pipeline,
                "resize_scale": scale,
                "jpeg_quality": quality,
                "jpeg_passes": passes,
            },
            condition="clean" if steps is None else "transformed",
            training_condition="clean",
            seed=SEED,
        )
        loader = DataLoader(CombinedTestDataset(rows, steps), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS)
        out = evaluate_model(model, loader, DEVICE, THRESHOLD)
        assert out["n_samples"] == EXPECTED_N, (condition, out["n_samples"])
        assert list(out["y_true"]) == [r.label for r in rows], "order/label drift"
        if reference_y_true is None:
            reference_y_true = list(out["y_true"])
        assert list(out["y_true"]) == reference_y_true, "y_true changed across conditions"
        n0 = int((out["y_true"] == 0).sum())
        assert (n0, out["n_samples"] - n0) == (EXPECTED_REAL, EXPECTED_AI)

        result = ExperimentResult.from_metrics(config, out["metrics"])
        m = out["metrics"]
        metric_rows.append({
            "condition": condition, "pipeline": pipeline,
            "resize_scale": "" if scale is None else scale,
            "jpeg_quality": "" if quality is None else quality,
            "jpeg_passes": passes, "n_samples": out["n_samples"],
            "accuracy": m["accuracy"], "precision": m["precision"],
            "ai_recall": m["ai_recall"], "ai_f1": m["f1"],
            "roc_auc": m["roc_auc"], "ai_fnr": m["ai_false_negative_rate"],
            "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"],
        })
        cm_rows.append({"condition": condition, "pipeline": pipeline,
                        "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"]})
        for row, t, p, prob in zip(rows, out["y_true"], out["y_pred"], out["y_prob"]):
            pred_rows.append({
                "filename": row.filename, "split": row.split,
                "generator": row.generator, "true_label": int(t),
                "pred_label": int(p), "ai_prob": float(prob),
                "condition": condition, "pipeline": pipeline,
                "resize_scale": "" if scale is None else scale,
                "jpeg_quality": "" if quality is None else quality,
                "jpeg_passes": passes,
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
              f"f1={m['f1']:.4f} auc={m['roc_auc']:.4f} fnr={m['ai_false_negative_rate']:.4f} "
              f"CM=[[ {m['tn']},{m['fp']}],[ {m['fn']},{m['tp']}]]")

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

    for sub in ("metrics", "predictions", "confusion_matrices"):
        (REPO_ROOT / "results" / sub).mkdir(parents=True, exist_ok=True)
    with open(REPO_ROOT / "results" / "metrics" / f"{EXPERIMENT}_{MODEL_NAME}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys())); w.writeheader(); w.writerows(metric_rows)
    with open(REPO_ROOT / "results" / "predictions" / f"{EXPERIMENT}_{MODEL_NAME}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pred_rows[0].keys())); w.writeheader(); w.writerows(pred_rows)
    with open(REPO_ROOT / "results" / "confusion_matrices" / f"{EXPERIMENT}_{MODEL_NAME}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cm_rows[0].keys())); w.writeheader(); w.writerows(cm_rows)
    print("Saved metrics / predictions / confusion matrices / experiment logs.")

    print("\nCondition | Pipeline | Accuracy | Precision | AI Recall | AI F1 | ROC-AUC | AI FNR")
    for r in metric_rows:
        print(f"{r['condition']:18s} | {r['pipeline']:30s} | {r['accuracy']:.4f} | "
              f"{r['precision']:.4f} | {r['ai_recall']:.4f} | {r['ai_f1']:.4f} | "
              f"{r['roc_auc']:.4f} | {r['ai_fnr']:.4f}")


if __name__ == "__main__":
    main()
