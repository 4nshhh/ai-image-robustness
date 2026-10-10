# Robust Detection of AI-Generated Images Under Social Media-Style Compression and Transformations — Report Source

> Source document for the final academic report. Every number below is taken from a saved artifact cited in place. Statements are labeled **Observation** (directly measured), **Interpretation** (supported explanation), or **Hypothesis** (requires further evidence).

## Title

Robust Detection of AI-Generated Images Under Social Media-Style Compression and Transformations: A Controlled Robustness Study with Transformation-Aware Training

## Abstract

AI-image detectors are typically evaluated on clean images, yet deployed images undergo compression and downscaling. We study an EfficientNet-B0 Real-vs-AI detector on a frozen 1,186-image test set under 15 conditions (clean, JPEG Q90–Q30, resize 0.75–0.25, two-pass recompression Q90–Q30, three ordered combined pipelines). **Observations:** the clean baseline (accuracy 90.73%, AI recall 90.96%) degrades monotonically under JPEG/recompression through AI-recall collapse (Q30: 55.46%), while resizing instead collapses Real precision; transformation order reverses the failure mode (C1 vs C2). A transformation-aware fine-tune (fixed 40/20/15/15/10 policy, best clean-val epoch 8) preserves AI recall across JPEG-family and combined conditions (e.g. C3: 54.78% → 91.47%, +36.69 pp) at a clean-accuracy cost of 1.77 pp, and improves held-out unseen conditions. **Limitation:** Real images are 100% JPEG while AI images are 100% PNG in file metadata, so compression findings are entangled with format bias; a metadata-only classifier reaches 1.0000 accuracy. No general-purpose detection is claimed.

## 1. Introduction and Motivation

Synthetic imagery is cheap to produce and easy to share; platforms recompress and downscale uploads as a matter of course. A detector that excels on pristine files but collapses on everyday transformations is of limited practical value. This work therefore evaluates robustness first and treats clean accuracy as one metric among several.

## 2. Problem Statement

Binary classification of images as Real (label 0) or AI-generated (label 1), evaluated not only on clean inputs but under a fixed, pre-registered set of social-media-style degradations applied identically to both classes.

## 3. Research Question

How do JPEG compression, resizing, recompression, and combinations of these transformations affect AI-generated image detection performance, and can transformation-aware training improve robustness?

## 4. Objectives and Contributions

1. Frozen benchmark: 7,986-image Tiny-GenImage split (train 5,600 / val 1,200 / test 1,186), seed 42.
2. Clean baselines for ResNet-50 and EfficientNet-B0 with full metric records.
3. Controlled robustness evaluation: 15 conditions, identical code/threshold/preprocessing, per-image prediction records.
4. Transformation-aware EfficientNet-B0 and a 30-row head-to-head comparison with percentage-point deltas.
5. Quantified dataset-format confound plus a format-normalization diagnostic.
6. Read-only results dashboard and reproducible scripts for every measurement.

## 5. Related Work (verify before submission)

- Zhu et al., *GenImage: A Million-Scale Benchmark for Detecting AI-Generated Images* (arXiv:2306.08571) — source domain of our subset; motivates large-scale generator-diverse evaluation. Verify citation details against the arXiv record.
- Grommelt et al., *Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets* (arXiv:2403.17608) — motivates our compression/format-bias audit. Verify details.
- Li et al., *Detecting Compressed AI-Generated Images via Phase Spectrum Robustness* (CVPR 2026, as cited in project docs) — motivates robustness under compression. **Author list and venue details must be verified against the publisher record before submission.**
- Broader robustness/augmentation literature: **not yet surveyed — open item for the team**; do not cite additional works until read.

## 6. Dataset and Split Methodology

Tiny-GenImage subset (`data/splits/genimage_8000_split_v4/manifest.csv`, seed 42): 4,000 Real / 3,986 AI across 7 generators (adm 569, biggan 570, glide 569, midjourney 569, sdv5 570, vqdm 572, wukong 567). Target was 8,000; 14 test/AI files were unrecoverable from the source export and excluded without substitution (decision record: `RESULTS_REFERENCE.md` dataset table and the manifest `integrity_note`). SHA-256 audit: zero cross-split duplicates. Final: train 5,600 (2,800/2,800), val 1,200 (600/600), test 1,186 (600/586). Images are gitignored; manifest + `split_metadata.json` are committed.

## 7. Models and Preprocessing

EfficientNet-B0 (primary) and ResNet-50, 2-class heads, RGB → 224×224 → tensor → ImageNet normalization, threshold 0.5. Baseline: Adam, lr/weight-decay 1e-4, 10 epochs, batch 16, best-validation-accuracy selection, train pixels only. Robust: fine-tuned from the baseline checkpoint under the policy below; best clean-val epoch 8 (0.9008). Test data never influenced training, tuning, or selection.

## 8. Transformation Definitions

JPEG Q{90,70,50,30} (single encode–decode); resize {0.75,0.50,0.25}, aspect-preserving; recompression = two sequential same-quality passes (an operation distinct from single compression); C1 = Resize 50%→JPEG Q50, C2 = reverse, C3 = Resize→JPEG→JPEG. Research transforms precede the single final preprocessing; both classes always transformed; originals never modified (size+mtime asserted).

## 9. Transformation-Aware Training Policy

Per-image draw, both classes identically, seed 42: 40% clean; 20% JPEG Q∼{90,70,50,30}; 15% resize ∼{0.75,0.50}; 15% recompression Q∼{90,70}×2; 10% combined C1/C2. Held out of training (unseen): resize 0.25, recompression Q50/Q30, C3. Config: `models/robust/robust_training_config.json`; entry point `scripts/train_robust.py`.

## 10. Evaluation Metrics and Protocol

Accuracy, precision, AI recall, F1, ROC-AUC (from AI probabilities), AI FNR, Real recall, FPR, TN/FP/FN/TP (rows actual, columns predicted), n and support. Every condition: n=1,186 (600/586), y_true identical, clean control reproducing baseline within 1e-4 (asserted at runtime). Rate differences reported in percentage points (pp), never conflated with relative percent change.

## 11. Results

Clean baselines (threshold 0.5, n=1,186): EfficientNet-B0 — acc 90.73%, prec 90.34%, AI recall 90.96%, F1 90.65%, AUC 0.9649, FNR 9.04%, CM [[543,57],[53,533]] (`results/metrics/baseline_efficientnet_b0_clean.json`); ResNet-50 — acc 90.81%, prec 94.09%, AI recall 86.86%, F1 90.33%, AUC 0.9682, CM [[568,32],[77,509]]. EfficientNet-B0 selected as primary for its balanced recall/F1.

Robustness (baseline AI recall → robust; pp deltas from `results/metrics/baseline_vs_robust_efficientnet_b0.csv`): JPEG Q90 84.98%→91.13% (+6.14), Q70 74.74%→89.76% (+15.02), Q50 67.75%→89.08% (+21.33), Q30 55.46%→87.54% (+32.08). Recompression tracks single JPEG within ~0.2pp (second pass adds ~nothing). Resize 0.75/0.50/0.25 baseline AI recall 95.39%/95.90%/90.61% with accuracy 86.85%/77.07%/53.71% (Real precision collapse, not AI misses); robust keeps recall 94.71%/95.05%/94.88% with accuracy 87.44%/85.33%/64.92%. Combined: C1 55.80%→91.47% (+35.67), C2 88.23%→93.17% (+4.95), C3 54.78%→91.47% (+36.69). Clean cost of robustness: accuracy 90.73%→88.95% (−1.77pp), precision −3.82pp, AUC −0.58pp; clean AI recall +1.02pp. Full 30-row table: `results/metrics/robust_evaluation_efficientnet_b0.csv`. Plots: `results/graphs/` (recall/accuracy/FNR comparisons, JPEG and resize sweeps, key confusion matrices).

## 12. Discussion and Interpretation

**Observation:** JPEG-family damage concentrates in AI false negatives; resize damage concentrates in Real false positives; pipeline order selects the failure mode (C1≈JPEG pattern with Real side identical to clean; C2≈resize pattern). **Interpretation:** lossy compression appears to erase the high-frequency cues this detector uses for the AI class, while downscaling blurs the texture cues distinguishing Real images — consistent with, but not proven by, the asymmetric confusion patterns. **Hypothesis (untested):** frequency-aware or multi-scale training would close the remaining resize-0.25 gap. The robust model converts most JPEG-family collapse into preserved recall at modest clean cost; unseen conditions improve, suggesting policy-level rather than level-memorized robustness — still within one dataset.

## 13. Dataset Bias and Format-Normalization Findings

**Observation:** format separation is perfect (100% Real `.jpeg`/JPEG with quantization tables; 100% AI `.png`/PNG, none; `results/metrics/dataset_bias_audit.csv`); AI images are always square (128–1024px), Real are variable rectangles; RGBA occurs only in AI (~14%), grayscale-L only in Real (~1.4%); a 7-feature metadata classifier scores 1.0000 on validation and held-out test. Format diagnostic (n=1,186/condition): lossless PNG re-encode changes zero predictions for either model; JPEG Q95 costs baseline −2.39pp / robust −0.90pp AI recall (`results/metrics/format_diagnostic_efficientnet_b0.csv`). **Interpretation:** container identity carries no signal to these classifiers (they see decoded RGB); heavy-compression damage is lossy-pixel damage. **Unproven:** that the networks use the format shortcut, and that any normalization removes the bias. JPEG conclusions must always be reported with the confound attached.

## 14. Threats to Validity and Limitations

Single dataset and 7 generators; one frozen test set reused across 19 model-conditions (no tuning on it, but shared-sample comparisons); perfect format/geometry/mode shortcuts available to any model; no statistical-significance testing performed — report CIs as future work, do not claim significance; exact GPU bit-reproducibility not claimed; base-environment NumPy/SciPy incompatibility documented (tests ran under `ml_clean`); no externalPLATFORM, generator, or editor generalization tested.

## 15. Conclusion and Future Work

Under controlled conditions, transformation-aware training substantially improves JPEG/recompression/combined robustness and unseen-level generalization within this dataset at a −1.77pp clean-accuracy cost, while resize-0.25 remains hard. Future work: significance testing, frequency-aware architectures, broader generators and real platform pipelines, bias-mitigated data collection, and full browser-tested public deployment.

## References

See [`REFERENCES.md`](REFERENCES.md). Verify Li et al. author/venue details before submission; related-work survey is otherwise open (Section 5).
