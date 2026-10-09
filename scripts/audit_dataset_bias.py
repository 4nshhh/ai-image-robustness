"""Experiment 8: dataset format/metadata bias audit (read-only diagnostic).

Inspects the frozen manifest + images for class-correlated properties
(extension, decoded format, geometry, color mode, JPEG quantization
presence) that could act as label shortcuts. Descriptive + one labeled
DIAGNOSTIC metadata classifier (fit on train, selected on val, reported
on val and held-out test without tuning). NOT a causal test; does not
resolve any confound; never modifies data, splits, models, or results.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset import DEFAULT_MANIFEST, DEFAULT_ROOT, load_manifest

SEED = 42
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class Counter2(dict):
    """Tiny counter (avoids importing collections for one use)."""

    def __init__(self, items):
        super().__init__()
        for k in items:
            self[k] = self.get(k, 0) + 1


def probe(row) -> dict:
    """Read-only metadata probe of one manifest row (never modifies files)."""
    rec = {"extension": Path(row.filename).suffix.lower(), "readable": True,
           "format": None, "width": None, "height": None, "mode": None,
           "has_quantization": None, "n_q_tables": None, "error": None}
    try:
        with Image.open(row.path) as img:
            img.load()
            rec["format"] = img.format
            rec["width"], rec["height"] = img.size
            rec["mode"] = img.mode
            qtables = getattr(img, "quantization", None)
            if qtables:
                rec["has_quantization"] = True
                rec["n_q_tables"] = len(qtables)
            else:
                rec["has_quantization"] = False
                rec["n_q_tables"] = 0
    except Exception as exc:  # record, never substitute
        rec.update(readable=False, error=f"{type(exc).__name__}: {exc}")
    return rec


def desc(vals: list) -> dict:
    a = np.asarray(vals, dtype=float)
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "p50": float(np.median(a)), "max": float(a.max())}


def main() -> None:
    rows = load_manifest(DEFAULT_MANIFEST, DEFAULT_ROOT)
    print(f"Manifest rows: {len(rows)}")
    assert len(rows) == 7986, len(rows)

    records = []
    for i, row in enumerate(rows):
        rec = {"split": row.split, "label": row.label, "generator": row.generator,
               "filename": row.filename, **probe(row)}
        records.append(rec)
        if (i + 1) % 2000 == 0:
            print(f"  probed {i + 1}/{len(rows)}")
    bad = [r for r in records if not r["readable"]]
    print(f"Unreadable: {len(bad)}")
    for r in bad[:10]:
        print("  MISSING:", r["filename"], r["error"])

    ok = [r for r in records if r["readable"]]
    for r in ok:
        r["aspect"] = r["width"] / r["height"]
        r["mpix"] = r["width"] * r["height"] / 1e6

    summary_rows: list[dict] = []
    audit: dict = {"run_id": RUN_ID, "seed": SEED, "n_manifest": len(rows),
                   "n_unreadable": len(bad), "splits": {}}
    for split in ("train", "val", "test"):
        audit["splits"][split] = {}
        for label in (0, 1):
            grp = [r for r in ok if r["split"] == split and r["label"] == label]
            ext = Counter2([r["extension"] for r in grp])
            fmt = Counter2([str(r["format"]) for r in grp])
            modes = Counter2([r["mode"] for r in grp])
            q = sum(1 for r in grp if r["has_quantization"])
            stats = {k: desc([r[k] for r in grp])
                     for k in ("width", "height", "aspect", "mpix")}
            audit["splits"][split][str(label)] = {
                "n": len(grp), "extensions": ext, "decoded_formats": fmt,
                "modes": modes, "n_with_quantization": q, "geometry": stats}
            summary_rows.append({
                "split": split, "label": label, "n": len(grp),
                "extensions": json.dumps(ext), "formats": json.dumps(fmt),
                "modes": json.dumps(modes), "n_quantized": q,
                "width_mean": round(stats["width"]["mean"], 1),
                "height_mean": round(stats["height"]["mean"], 1),
                "aspect_mean": round(stats["aspect"]["mean"], 4),
                "mpix_mean": round(stats["mpix"]["mean"], 3)})
            print(f"[{split} label={label}] n={len(grp)} ext={ext} fmt={fmt} "
                  f"mode={modes} quant={q} WxH={stats['width']['mean']:.0f}x"
                  f"{stats['height']['mean']:.0f}")

    # ---- labeled DIAGNOSTIC: can trivial metadata predict the label? ----
    # Fit on train, model/threshold selection on val, test reported once.
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    def feats(rs):
        return np.array([[r["width"], r["height"], r["aspect"], r["mpix"],
                          1.0 if r["extension"] in (".jpg", ".jpeg") else 0.0,
                          1.0 if str(r["format"]).upper() == "JPEG" else 0.0,
                          1.0 if r["mode"] == "RGB" else 0.0] for r in rs])

    diag = {}
    for C in (0.1, 1.0, 10.0):
        clf = LogisticRegression(C=C, max_iter=2000)
        tr = [r for r in ok if r["split"] == "train"]
        va = [r for r in ok if r["split"] == "val"]
        clf.fit(feats(tr), [r["label"] for r in tr])
        acc = accuracy_score([r["label"] for r in va], clf.predict(feats(va)))
        diag[str(C)] = round(float(acc), 4)
    best_C = max(diag, key=diag.get)
    te = [r for r in ok if r["split"] == "test"]
    clf = LogisticRegression(C=float(best_C), max_iter=2000)
    tr = [r for r in ok if r["split"] == "train"]
    clf.fit(feats(tr), [r["label"] for r in tr])
    va_acc = accuracy_score([r["label"] for r in va], clf.predict(feats(va)))
    te_acc = accuracy_score([r["label"] for r in te], clf.predict(feats(te)))
    audit["diagnostic"] = {
        "note": "DIAGNOSTIC ONLY — separability probe, not a causal test",
        "features": ["width", "height", "aspect", "mpix", "is_jpeg_ext",
                     "is_jpeg_format", "is_rgb"],
        "val_acc_by_C": diag, "selected_C": best_C,
        "val_accuracy": round(float(va_acc), 4),
        "test_accuracy_report_only": round(float(te_acc), 4)}
    print(f"Diagnostic val acc by C: {diag} -> C={best_C} "
          f"val={va_acc:.4f} test(report-only)={te_acc:.4f}")

    (REPO_ROOT / "results" / "metrics").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "results" / "experiment_logs").mkdir(parents=True, exist_ok=True)
    with open(REPO_ROOT / "results" / "metrics" / "dataset_bias_audit.csv",
              "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    with open(REPO_ROOT / "results" / "experiment_logs" / "dataset_bias_audit.json",
              "w") as f:
        json.dump(audit, f, indent=2)
    print("Saved metrics/dataset_bias_audit.csv + experiment_logs/dataset_bias_audit.json")


if __name__ == "__main__":
    main()
