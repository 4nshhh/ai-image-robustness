# Experiment Protocol — Robustness Study

Planning/specification document. No experiment has been executed under
this protocol; every entry below is PLANNED or PENDING. No results,
thresholds, or transformation choices beyond the proposed levels are
claimed here.

## 1. Research Question

How do JPEG compression, resizing, recompression, and combinations of
these social-media-style transformations affect AI-generated image
detector performance, and can transformation-aware training improve
robustness?

Binary task, fixed label mapping: **0 = Real, 1 = AI-generated**.

## 2. Prerequisites (Person 1 Dependencies)

Nothing in Sections 3–7 may start until Person 1 delivers:

1. Finalized, audited dataset with metadata and documented bias findings.
2. Frozen train/validation/test splits (test set isolated thereafter).
3. Shared preprocessing pipeline, identical for both classes.
4. Candidate architectures with a clean-trained baseline checkpoint and
   its exact training configuration (seed, image size, batch size,
   optimizer, epochs, threshold).
5. Primary-model selection decision with justification.

## 3. Experiment Groups

All robustness evaluations use the **same frozen baseline checkpoint**
on the **same untouched test split**. Only the test-time transformation
changes between conditions.

### 3.1 Clean baseline evaluation

Evaluate the clean-trained model on the clean test set. This is the
reference point every degradation is measured against. Record the full
metric set (Section 5).

### 3.2 JPEG compression

Single JPEG compression applied at test time. Proposed levels: **Q90,
Q70, Q50, Q30** (configurable; adjust only with justification after
initial runs). Purpose: measure how the detector degrades as
compression strengthens, since compression can remove the
high-frequency artifacts detectors rely on — a hypothesis to test,
not a fact assumed upfront.

### 3.3 Image resizing / downscaling

Controlled downscaling preserving aspect ratio. Proposed levels:
**0.75x, 0.50x, 0.25x** (configurable). Purpose: social platforms
routinely downscale uploads; measure whether lost resolution destroys
detection signal.

### 3.4 JPEG recompression (double compression)

Recompression means sequential JPEG encode→decode→encode cycles
(e.g. quality Q applied twice), which is **different from a single
compression at the same quality** and is implemented as a separate
operation (`apply_recompression`, not repeated `apply_jpeg_compression`
calls conflated with single compression). Proposed: two passes at each
of the Section 3.2 levels. Purpose: uploads are often compressed more
than once across platforms.

### 3.5 Combined transformations

A small, justified set — not the full cross-product. Proposed:

| ID | Combination | Rationale |
|---|---|---|
| C1 | Resize 0.50 → JPEG Q70 | Typical platform pipeline (downscale then compress) |
| C2 | JPEG Q70 → Resize 0.50 | Reverse order; order may matter and is recorded |
| C3 | Resize 0.50 → JPEG Q70 → JPEG Q70 | Downscale plus double compression (heavy pipeline) |

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

## 6. Robust Training (Later Experiment)

Compare a **standard model** (trained on clean data) against a **robust
model** (trained with controlled transformations applied during
training) on clean and transformed test data, using the identical
protocol of Sections 4–5. The classification task does not change
(Real vs AI).

The exact transformation levels/probabilities for training are
**deliberately undecided here** — finalize them only after the baseline
robustness results (Sections 3.1–3.5) and Person 1's training pipeline
are available. Also check whether robust training hurts clean-image
performance. Implementation details depending on Person 1's pipeline
are out of scope for this protocol.

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
identifier once that mechanism exists. File-writing machinery is a
later implementation task, not part of this protocol.

## 9. Bias / Dataset Control

No robustness result is interpretable until the dataset audit is done.
Before drawing conclusions, confirm the audit checked for systematic
Real-vs-AI differences in: JPEG/compression characteristics and file
format, image resolution and aspect ratio, file size, generator
distribution, and preprocessing. High clean accuracy may reflect such
shortcuts rather than genuine generation artifacts — do not claim
otherwise until the audit rules it out. Do not assert any of these
biases exist in the final dataset before the audit confirms them.

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
| base_clean | Clean baseline | none | — | clean-trained | test | PLANNED / PENDING BASELINE |
| jpeg_q90 | JPEG | jpeg | quality=90 | clean-trained | test | PLANNED / PENDING BASELINE |
| jpeg_q70 | JPEG | jpeg | quality=70 | clean-trained | test | PLANNED / PENDING BASELINE |
| jpeg_q50 | JPEG | jpeg | quality=50 | clean-trained | test | PLANNED / PENDING BASELINE |
| jpeg_q30 | JPEG | jpeg | quality=30 | clean-trained | test | PLANNED / PENDING BASELINE |
| resize_075 | Resize | resize | scale=0.75 | clean-trained | test | PLANNED / PENDING BASELINE |
| resize_050 | Resize | resize | scale=0.50 | clean-trained | test | PLANNED / PENDING BASELINE |
| resize_025 | Resize | resize | scale=0.25 | clean-trained | test | PLANNED / PENDING BASELINE |
| recomp_q90x2 | Recompression | recompression | quality=90, passes=2 | clean-trained | test | PLANNED / PENDING BASELINE |
| recomp_q70x2 | Recompression | recompression | quality=70, passes=2 | clean-trained | test | PLANNED / PENDING BASELINE |
| recomp_q50x2 | Recompression | recompression | quality=50, passes=2 | clean-trained | test | PLANNED / PENDING BASELINE |
| recomp_q30x2 | Recompression | recompression | quality=30, passes=2 | clean-trained | test | PLANNED / PENDING BASELINE |
| combined_c1 | Combined | resize→jpeg | scale=0.50, quality=70 | clean-trained | test | PLANNED / PENDING BASELINE |
| combined_c2 | Combined | jpeg→resize | quality=70, scale=0.50 | clean-trained | test | PLANNED / PENDING BASELINE |
| combined_c3 | Combined | resize→jpeg→jpeg | scale=0.50, quality=70, passes=2 | clean-trained | test | PLANNED / PENDING BASELINE |
| robust_train | Robust training | TBD after baseline | TBD | transformation-aware | test (clean + transformed) | PLANNED / PENDING DATASET |
| unseen_levels | Unseen levels (OPTIONAL) | TBD | TBD | transformation-aware | test (held-out levels) | PLANNED / OPTIONAL |

All statuses are PLANNED or PENDING. Nothing has been run; nothing is
COMPLETED.
