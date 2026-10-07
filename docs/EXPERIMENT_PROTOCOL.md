# Experiment Protocol — Robustness Study

Status preamble (audited against the repository): Sections 3.1–3.5 are
COMPLETED — clean baseline plus JPEG, resize, recompression, and combined
evaluations have been executed with results saved under `results/` (see
the Experiment Matrix, Section 11). Sections 6–7 (robust training,
unseen levels) are PLANNED and have NOT been run. No results are claimed
in this document; measured numbers live in `results/` and
`docs/DEVELOPMENT_LOG.md`.

## 1. Research Question

How do JPEG compression, resizing, recompression, and combinations of
these social-media-style transformations affect AI-generated image
detector performance, and can transformation-aware training improve
robustness?

Binary task, fixed label mapping: **0 = Real, 1 = AI-generated**.

## 2. Prerequisites (Person 1 Dependencies) — Delivery Status

The following were required before Section 3; all except the full bias
analysis are delivered and verified in the repository:

1. Finalized dataset with manifest — DELIVERED (`manifest.csv`, 7,986 rows).
2. Frozen train/validation/test splits — DELIVERED (seed 42; test isolated).
3. Shared preprocessing, identical for both classes — DELIVERED
   (`src/preprocessing.py`; test: resize 224x224 → ToTensor → ImageNet norm).
4. Clean-trained baseline checkpoints + training config — DELIVERED
   (ResNet-50 and EfficientNet-B0 histories committed; weights gitignored).
5. Primary-model selection — DELIVERED: **EfficientNet-B0**
   (`models/baseline/efficientnet_b0_best.pth`), threshold fixed at 0.5.
6. Full bias analysis — PARTIAL (integrity/duplicates audited; the
   confirmed Real-JPEG / AI-PNG format confound is documented in
   Section 9 and must temper conclusions, not block measurement).

### 2.1 Frozen Setup (Locked; Do Not Change Without Team Decision)

- Dataset: 7,986 images (`data/splits/genimage_8000_split_v4/`, seed 42).
- Train: 5,600 (2,800 Real / 2,800 AI). Validation: 1,200 (600 / 600).
- Test: 1,186 (600 Real / 586 AI) — the ONLY evaluation split.
- Primary checkpoint: `models/baseline/efficientnet_b0_best.pth`.
- Clean baseline reference (threshold 0.5): accuracy 0.9073, precision
  0.9034, AI recall 0.9096, F1 0.9065, ROC-AUC 0.9649, AI FNR 0.0904,
  CM [[543, 57], [53, 533]] — reproduced within 1e-4 by every
  experiment's clean control (see result CSVs).

## 3. Experiment Groups

All robustness evaluations use the **same frozen baseline checkpoint**
on the **same untouched test split**. Only the test-time transformation
changes between conditions.

### 3.1 Clean baseline evaluation

Evaluate the clean-trained model on the clean test set. This is the
reference point every degradation is measured against. Record the full
metric set (Section 5).

### 3.2 JPEG compression — EXECUTED

Single JPEG compression applied at test time to both classes. Executed
levels: **Q90, Q70, Q50, Q30** (results saved; see Matrix). Purpose: measure how the detector degrades as
compression strengthens, since compression can remove the
high-frequency artifacts detectors rely on — a hypothesis to test,
not a fact assumed upfront.

### 3.3 Image resizing / downscaling — EXECUTED

Controlled downscaling preserving aspect ratio, applied to both classes.
Executed levels: **0.75x, 0.50x, 0.25x** (results saved; see Matrix). Purpose: social platforms
routinely downscale uploads; measure whether lost resolution destroys
detection signal.

### 3.4 JPEG recompression (double compression) — EXECUTED

Recompression means sequential JPEG encode→decode→encode cycles
(e.g. quality Q applied twice), which is **different from a single
compression at the same quality** and is implemented as a separate
operation (`apply_recompression`, not repeated `apply_jpeg_compression`
calls conflated with single compression). Executed: two passes at each
of the Section 3.2 levels. Purpose: uploads are often compressed more
than once across platforms.

### 3.5 Combined transformations — EXECUTED

A small, justified set — not the full cross-product. Executed
(CORRECTION: the runs used JPEG **Q50**, not the Q70 originally proposed
here; the table below records what was actually executed):

| ID | Combination | Rationale |
|---|---|---|
| C1 | Resize 0.50 → JPEG Q50 | Typical platform pipeline (downscale then compress) |
| C2 | JPEG Q50 → Resize 0.50 | Reverse order; order may matter and is recorded |
| C3 | Resize 0.50 → JPEG Q50 → JPEG Q50 | Downscale plus double compression (heavy pipeline) |

Purpose: test whether combined damage is additive and whether order
matters. Extend only with explicit justification.

## 4. Experiment Fairness Controls

- Same test set across all conditions in Sections 3.1–3.5.
- Same frozen baseline checkpoint throughout the robustness evaluation;
  never retrain or swap the model between transformation conditions.
- Same preprocessing unless preprocessing itself is the study variable.
- Same classification threshold (default 0.5; any change is a new,
  documented experiment).
- Same metric set, same evaluation code (`evaluate_model`,
  `compute_classification_metrics`, `run_experiment`, `ExperimentResult`).
- Transformations applied equally to Real and AI images — never compress
  only one class.
- Test set never used for training, tuning, or transformation-aware
  data creation (no leakage).
- Every run records: exact transformation name + parameters and order,
  model/checkpoint identity, dataset/split identity, seed, threshold.

Explicit separation of the two studies:

- **A. Robustness evaluation** — clean-trained model evaluated on
  transformed test data (Sections 3.1–3.5). Measures degradation.
- **B. Transformation-aware training** — model trained with controlled
  transformations, evaluated on clean and transformed data (Section 6).
  Measures whether robustness improves.

Never mix A and B in one comparison; each answers a different question.

## 5. Metrics

Every evaluation records: accuracy, precision, recall, F1, AI-class
recall, AI false-negative rate, Real-class recall, false-positive rate,
ROC-AUC (from probabilities; `None`/unavailable when probabilities are
absent — never invented), confusion matrix, and number of samples.

**AI-class recall and AI false-negative rate are the most important
secondary metrics after the headline set**: a false negative is an
AI-generated image that evades detection, which is the key failure mode
of this system. Do not judge conditions by accuracy alone.

## 6. Robust Training (PLANNED — Rules Fixed Before Implementation)

Compare the **standard model** (the frozen EfficientNet-B0 clean
baseline, Section 2.1) against a **robust model** trained with controlled
transformations, on the clean test set AND the exact robustness
conditions of Sections 3.2–3.5, using the identical protocol of
Sections 4–5. The classification task does not change (Real vs AI).
Architecture stays EfficientNet-B0 unless the team explicitly decides
otherwise with justification.

Binding rules (no implementation may start until each is written down):

1. TRAIN data only for training. The 5,600-image train split is the only
   source of training pixels.
2. Validation (1,200) may be used for model selection/tuning. Test data
   (1,186) must NEVER influence training, hyperparameters, augmentation
   policy, or checkpoint selection.
3. The training transformation policy must be explicitly specified first:
   (a) clean + transformed mix vs transformed-only; (b) exact
   transformation types and parameter ranges/levels; (c) per-sample
   transformation probabilities; (d) the random seed governing the policy.
   All four must be fixed, seeded, and reproducible.
4. No training-time transformation mechanism currently exists in
   `src/preprocessing.py` / `src/train.py` (only `RandomHorizontalFlip`
   for train); the mechanism must be designed and documented — not
   improvised inside the training loop.
5. Robust checkpoint saved under `models/robust/`, never committed to Git
   (same ignore rules as baseline weights).
6. Evaluation reuses the finished experiment scripts/procedure on the same
   16 robustness conditions, so baseline-vs-robust numbers are directly
   comparable; also verify clean-image performance did not regress.
7. Training transformations (what the model saw) and evaluation
   transformations (what it is tested on) are recorded separately; the
   optional unseen-levels test (Section 7) stays OPTIONAL and never
   replaces the core same-condition comparison.

Level/probability values remain TBD pending team approval — they are NOT
set by this document.

## 7. Optional Advanced Experiment: Unseen Transformations

OPTIONAL / ADVANCED — perform only if time and data permit, and do not
assume it will happen. Train the robust model on a subset of
transformation levels (e.g. JPEG Q90 + Q70) and evaluate on held-out
levels (e.g. Q50 + Q30) or held-out combinations. This tests
generalization rather than memorization of specific artifacts. Claim
generalization only if this experiment is actually run.

## 8. Result Storage

Each experiment preserves, at minimum: the `ExperimentConfig`, the
`ExperimentResult` (via `to_dict()` / `to_record()`), model/checkpoint
identity, dataset/split identity, full transformation parameters and
order, predictions/probabilities when useful, and a timestamp or run
identifier once that mechanism exists. Per-condition JSON logs plus
metrics/predictions/confusion-matrix CSVs (as produced by the
`scripts/run_*_robustness.py` runs) are the implemented form of this.

## 9. Bias / Dataset Control

### 9.1 Confirmed format confound (must temper JPEG conclusions)

Verified directly from `manifest.csv`: **every Real image is `.jpeg`
(4,000) and every AI image is `.png` (3,986), in all three splits.**
This is a CONFIRMED dataset property, not a hypothesis.

Consequences, stated carefully:

- JPEG-compression robustness results (Sections 3.2, 3.4, and the JPEG
  legs of 3.5) measure the detector on images whose classes already
  differ in compression history. Do NOT claim compression alone causes
  the observed degradation without acknowledging this confound.
- What the JPEG results DO validly show: performance of this fixed
  detector on compressed inputs as encountered — the deployment-relevant
  measurement — plus the cross-condition pattern (e.g. second-pass
  effects, error-mode asymmetries), which a pure format shortcut cannot
  fully explain.
- The resize results are less entangled with this confound (downscaling
  applies symmetrically to already-decoded pixels), but the same caution
  applies to any claim about learned artifacts vs dataset shortcuts.

### 9.2 Remaining audit checklist

No robustness result is fully interpretable until the audit also covers
systematic Real-vs-AI differences in: image resolution and aspect ratio,
file size, generator distribution, and preprocessing. High clean accuracy
may reflect such shortcuts rather than genuine generation artifacts — do
not claim otherwise until the audit rules each out. Do not assert any
unchecked bias exists; check it.

## 10. Result Interpretation

- Compare each transformed condition against the clean baseline using
  absolute metric values; report degradation relative to clean.
- Examine confusion matrices per condition, with attention to AI false
  negatives (evaded AI images) and their association with compression
  level, resolution, or generator — descriptively, not causally.
- Distinguish observation (what the numbers show), interpretation
  (supported explanation), and hypothesis (needs further evidence).
- No causal claims from a single experiment; no invented thresholds for
  "significant degradation"; identical protocol for every model
  comparison; conclusions stay within the tested conditions.

## 11. Experiment Matrix

| Experiment ID | Group | Transformation | Parameter(s) | Model condition | Eval split | Status |
|---|---|---|---|---|---|---|
| base_clean | Clean baseline | none | — | clean-trained | test | COMPLETED |
| jpeg_q90 | JPEG | jpeg | quality=90 | clean-trained | test | COMPLETED |
| jpeg_q70 | JPEG | jpeg | quality=70 | clean-trained | test | COMPLETED |
| jpeg_q50 | JPEG | jpeg | quality=50 | clean-trained | test | COMPLETED |
| jpeg_q30 | JPEG | jpeg | quality=30 | clean-trained | test | COMPLETED |
| resize_075 | Resize | resize | scale=0.75 | clean-trained | test | COMPLETED |
| resize_050 | Resize | resize | scale=0.50 | clean-trained | test | COMPLETED |
| resize_025 | Resize | resize | scale=0.25 | clean-trained | test | COMPLETED |
| recomp_q90x2 | Recompression | recompression | quality=90, passes=2 | clean-trained | test | COMPLETED |
| recomp_q70x2 | Recompression | recompression | quality=70, passes=2 | clean-trained | test | COMPLETED |
| recomp_q50x2 | Recompression | recompression | quality=50, passes=2 | clean-trained | test | COMPLETED |
| recomp_q30x2 | Recompression | recompression | quality=30, passes=2 | clean-trained | test | COMPLETED |
| combined_c1 | Combined | resize→jpeg | scale=0.50, quality=50 | clean-trained | test | COMPLETED |
| combined_c2 | Combined | jpeg→resize | quality=50, scale=0.50 | clean-trained | test | COMPLETED |
| combined_c3 | Combined | resize→jpeg→jpeg | scale=0.50, quality=50, passes=2 | clean-trained | test | COMPLETED |
| robust_train | Robust training | TBD (Section 6 rules) | TBD — needs approval | transformation-aware | test (clean + transformed) | PLANNED / PENDING APPROVAL |
| unseen_levels | Unseen levels (OPTIONAL) | TBD | TBD | transformation-aware | test (held-out levels) | PLANNED / OPTIONAL |

Sections 3.1–3.5 are COMPLETED with results in `results/`; Sections 6–7
are PLANNED and have NOT been run.
