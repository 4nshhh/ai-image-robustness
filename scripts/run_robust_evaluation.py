"""Experiment 6: baseline vs robust evaluation (EfficientNet-B0, eval only).

Evaluates the frozen clean baseline checkpoint and the epoch-8 robust
checkpoint on the same 1,186-image test set under all 15 protocol
conditions (clean + JPEG x4 + resize x3 + recompression x4 + combined x3).
No retraining, no tuning, threshold 0.5, CUDA required.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
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
    apply_recompression,
    apply_resize,
)

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
EXPERIMENT = "robust_evaluation"


def _jpeg(q):
    return lambda im: apply_jpeg_compression(im, q)


def _resize(s):
    return lambda im: apply_resize(im, s)


def _recomp(q):
    return lambda im: apply_recompression(im, q, passes=2)


def _c1(im):
    return apply_pipeline(im, [lambda x: apply_resize(x, 0.50),
                               lambda x: apply_jpeg_compression(x, 50)])


def _c2(im):
    return apply_pipeline(im, [lambda x: apply_jpeg_compression(x, 50),
                               lambda x: apply_resize(x, 0.50)])


def _c3(im):
    return apply_pipeline(im, [lambda x: apply_resize(x, 0.50),
                               lambda x: apply_jpeg_compression(x, 50),
                               lambda x: apply_jpeg_compression(x, 50)])


# (condition, pipeline label, pil step or None, transformation, params)
CONDITIONS = [
    ("clean", "none", None, "none", {}),
    ("jpeg_q90", "jpeg(Q90)", _jpeg(90), "jpeg", {"quality": 90}),
    ("jpeg_q70", "jpeg(Q70)", _jpeg(70), "jpeg", {"quality": 70}),
    ("jpeg_q50", "jpeg(Q50)", _jpeg(50), "jpeg", {"quality": 50}),
    ("jpeg_q30", "jpeg(Q30)", _jpeg(30), "jpeg", {"quality": 30}),
    ("resize_075", "resize(0.75)", _resize(0.75), "resize", {"scale": 0.75}),
    ("resize_050", "resize(0.50)", _resize(0.50), "resize", {"scale": 0.50}),
    ("resize_025", "resize(0.25)", _resize(0.25), "resize", {"scale": 0.25}),
    ("recomp_q90_p2", "recomp(Q90)x2", _recomp(90), "recompression", {"quality": 90, "passes": 2}),
    ("recomp_q70_p2", "recomp(Q70)x2", _recomp(70), "recompression", {"quality": 70, "passes": 2}),
    ("recomp_q50_p2", "recomp(Q50)x2", _recomp(50), "recompression", {"quality": 50, "passes": 2}),
    ("recomp_q30_p2", "recomp(Q30)x2", _recomp(30), "recompression", {"quality": 30, "passes": 2}),
    ("c1_resize_jpeg", "resize(0.50)->jpeg(Q50)", _c1, "combined",
     {"pipeline": "resize(0.50)->jpeg(Q50)", "resize_scale": 0.50, "jpeg_quality": 50, "jpeg_passes": 1}),
    ("c2_jpeg_resize", "jpeg(Q50)->resize(0.50)", _c2, "combined",
     {"pipeline": "jpeg(Q50)->resize(0.50)", "resize_scale": 0.50, "jpeg_quality": 50, "jpeg_passes": 1}),
    ("c3_resize_jpeg_jpeg", "resize(0.50)->jpeg(Q50)->jpeg(Q50)", _c3, "combined",
     {"pipeline": "resize(0.50)->jpeg(Q50)->jpeg(Q50)", "resize_scale": 0.50, "jpeg_quality": 50, "jpeg_passes": 2}),
]

EXPECTED_N, EXPECTED_REAL, EXPECTED_AI = 1186, 600, 586

# Reference values live inline in the clean-reproduction check below.

RATE_KEYS = ["accuracy", "precision", "ai_recall", "ai_f1", "roc_auc",
             "ai_fnr", "real_recall", "fpr"]


class ConditionDataset(Dataset):
    """Test rows with an optional PIL step before clean preprocessing."""

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
    assert len(CONDITIONS) == 15, f"expected 15 conditions, got {len(CONDITIONS)}"

    rows = load_manifest(DEFAULT_MANIFEST, DEFAULT_ROOT, split="test")
    counts = {0: 0, 1: 0}
    for r in rows:
        counts[r.label] += 1
    assert (len(rows), counts[0], counts[1]) == (EXPECTED_N, EXPECTED_REAL, EXPECTED_AI)
    print(f"Test rows: {len(rows)} (Real={counts[0]}, AI={counts[1]})")
    before_files = {str(r.path): (r.path.stat().st_size, r.path.stat().st_mtime_ns) for r in rows}
    expected_ids = [str(r.path) for r in rows]

    metric_rows: list[dict] = []
    pred_rows: list[dict] = []
    cm_rows: list[dict] = []
    reference_y_true = None
    per_model_condition: dict[tuple[str, str], dict] = {}

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
            loader = DataLoader(ConditionDataset(rows, step), batch_size=BATCH_SIZE,
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
            per_model_condition[(model_id, condition)] = metric_rows[-1]
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
                           "batch_size": BATCH_SIZE, "seed": SEED}, f, indent=2)
            print(f"[{model_id:8s} {condition:18s}] acc={m['accuracy']:.4f} "
                  f"ai_rec={m['ai_recall']:.4f} fnr={m['ai_false_negative_rate']:.4f} "
                  f"auc={m['roc_auc']:.4f}")

        for b, p in zip(params_before, model.parameters()):
            assert torch.equal(b, p), f"{model_id} parameters changed"
        print(f"{model_id}: parameters unchanged.")

    # Clean-baseline reproduction for the BASELINE checkpoint.
    base_clean = per_model_condition[("baseline", "clean")]
    ref_map = {"accuracy": ("accuracy", 0.9073), "precision": ("precision", 0.9034),
               "recall": ("ai_recall", 0.9096), "f1": ("ai_f1", 0.9065),
               "roc_auc": ("roc_auc", 0.9649),
               "ai_false_negative_rate": ("ai_fnr", 0.0904)}
    bad = [f"{k}: got {base_clean[col]:.6f}, ref {v:.4f}"
           for k, (col, v) in ref_map.items()
           if abs(base_clean[col] - v) > TOLERANCE]
    if bad:
        print("BASELINE CLEAN MISMATCH:"); print("\n".join(bad)); sys.exit(1)
    print("Baseline clean reproduces verified reference within 1e-4.")

    after_files = {str(r.path): (r.path.stat().st_size, r.path.stat().st_mtime_ns) for r in rows}
    assert after_files == before_files, "test files were modified"
    print("Test images and checkpoints untouched.")

    assert len(metric_rows) == 30, len(metric_rows)
    assert len({(r["model"], r["condition"]) for r in metric_rows}) == 30

    # Comparison rows: baseline vs robust per condition.
    comp_rows = []
    for condition, pipeline, _, _, _ in CONDITIONS:
        b = per_model_condition[("baseline", condition)]
        r = per_model_condition[("robust", condition)]
        row = {"condition": condition, "pipeline": pipeline, "n_samples": EXPECTED_N}
        for k in RATE_KEYS:
            row[f"baseline_{k}"] = b[k]
            row[f"robust_{k}"] = r[k]
            row[f"delta_pp_{k}"] = (r[k] - b[k]) * 100.0
        row["baseline_tn"], row["baseline_fp"] = b["tn"], b["fp"]
        row["baseline_fn"], row["baseline_tp"] = b["fn"], b["tp"]
        row["robust_tn"], row["robust_fp"] = r["tn"], r["fp"]
        row["robust_fn"], row["robust_tp"] = r["fn"], r["tp"]
        comp_rows.append(row)

    for sub in ("metrics", "predictions", "confusion_matrices"):
        (REPO_ROOT / "results" / sub).mkdir(parents=True, exist_ok=True)
    with open(REPO_ROOT / "results" / "metrics" / f"{EXPERIMENT}_{MODEL_NAME}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys())); w.writeheader(); w.writerows(metric_rows)
    with open(REPO_ROOT / "results" / "predictions" / f"{EXPERIMENT}_{MODEL_NAME}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pred_rows[0].keys())); w.writeheader(); w.writerows(pred_rows)
    with open(REPO_ROOT / "results" / "confusion_matrices" / f"{EXPERIMENT}_{MODEL_NAME}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cm_rows[0].keys())); w.writeheader(); w.writerows(cm_rows)
    with open(REPO_ROOT / "results" / "metrics" / f"baseline_vs_robust_{MODEL_NAME}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comp_rows[0].keys())); w.writeheader(); w.writerows(comp_rows)
    print("Saved metrics (30 rows) / predictions / confusion matrices / comparison / logs.")

    print("\nCondition | Baseline acc/AIrec/FNR || Robust acc/AIrec/FNR || dAIrec_pp")
    for c in comp_rows:
        print(f"{c['condition']:18s} | {c['baseline_accuracy']:.4f}/{c['baseline_ai_recall']:.4f}/{c['baseline_ai_fnr']:.4f} "
              f"|| {c['robust_accuracy']:.4f}/{c['robust_ai_recall']:.4f}/{c['robust_ai_fnr']:.4f} "
              f"|| {c['delta_pp_ai_recall']:+.2f}")


if __name__ == "__main__":
    main()
