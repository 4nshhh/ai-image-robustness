# Presentation Source — AI Image Robustness Lab (≈11 slides, ~15 minutes)

> Slide plan only. All numbers are validated (see `RESULTS_REFERENCE.md`). Recommended visuals are existing files under `results/graphs/` — no visual is invented; none is missing except a dashboard screenshot (capture the Overview tab before presenting).

## Slide 1 — Title (1 min)
- **Message:** What we studied and what we found, in one line.
- **Bullets:** Title; team + roles; one-line result: transformation-aware training preserves AI recall under compression at a 1.77 pp clean-accuracy cost.
- **Visual:** Dashboard Overview screenshot (TO CREATE — screenshot, not data).
- **Notes:** State the research question verbatim; promise the bias caveat up front.

## Slide 2 — Problem & Research Question (1.5 min)
- **Message:** Clean-benchmark accuracy overstates deployed reliability.
- **Bullets:** uploads are recompressed/downscaled; detectors evaluated on pristine files; our question (verbatim RQ).
- **Visual:** none (spoken framing).
- **Notes:** Define Real=0 / AI=1 immediately.

## Slide 3 — Dataset (1.5 min)
- **Message:** Frozen, audited, imbalanced-test benchmark.
- **Bullets:** Tiny-GenImage 7,986; train 5,600 / val 1,200 / test 1,186 (600/586); 7 generators; 14 files unrecoverable, zero SHA-256 duplicates; seed 42.
- **Visual:** split-counts table (from `split_metadata.json`).
- **Notes:** Emphasize the test set never touched training/tuning/selection.

## Slide 4 — Method: Models & Protocol (1.5 min)
- **Message:** One fair pipeline for every number shown.
- **Bullets:** EfficientNet-B0 primary (+ResNet-50 baseline); 224px/ImageNet/threshold 0.5; 15 conditions; identical code; pp deltas; clean control reproduces baseline within 1e-4 every run.
- **Visual:** protocol pipeline diagram (TO CREATE — diagram of existing design, no new data).
- **Notes:** ResNet-50: acc 90.81%, AI recall 86.86% — precise but misses more AI.

## Slide 5 — Transformation Conditions (1 min)
- **Message:** Exactly what "robustness" means here.
- **Bullets:** JPEG Q90–Q30; resize 0.75/0.50/0.25; recompression = 2 sequential passes; C1/C2/C3 with order preserved; both classes always transformed.
- **Visual:** none (table on slide).
- **Notes:** Stress order preservation; C1 vs C2 is the order experiment.

## Slide 6 — Baseline Results (2 min)
- **Message:** Compression destroys AI recall; resizing destroys Real precision.
- **Bullets:** clean 90.73% acc / 90.96% recall; JPEG Q30 recall 55.46%; resize 0.25 accuracy 53.71% with recall 90.61%; recompression ≈ single JPEG; C1/C2 fail oppositely.
- **Visual:** `results/graphs/ai_recall_all_conditions_baseline_vs_robust.png` (baseline series); `ai_recall_vs_jpeg_quality.png`.
- **Notes:** Walk the two failure modes via confusion counts; no causal claims.

## Slide 7 — Transformation-Aware Training (1.5 min)
- **Message:** Fixed, logged policy — no test-guided tuning.
- **Bullets:** from baseline checkpoint; 40/20/15/15/10 mix; Q/scales as specified; resize-0.25, recomp Q50/Q30, C3 held unseen; Adam 1e-4, 10 epochs, batch 16, seed 42; best clean-val epoch 8 (0.9008).
- **Visual:** policy table (from `robust_training_config.json`).
- **Notes:** Unseen reservations were declared before training.

## Slide 8 — Comparative Results (2 min)
- **Message:** Recall preserved nearly everywhere for −1.77 pp clean accuracy.
- **Bullets:** JPEG Q30 +32.08, C3 +36.69, C1 +35.67 pp AI recall; clean 90.73%→88.95%; resize dips negligible (−0.68/−0.85); unseen all improve; resize-0.25 still weakest (64.92%).
- **Visual:** `accuracy_all_conditions_baseline_vs_robust.png`; `ai_fnr_all_conditions_baseline_vs_robust.png`; confusion-matrix figure.
- **Notes:** Read the trade-off slide verbatim; never claim universal superiority.

## Slide 9 — Dataset Bias (1.5 min)
- **Message:** Perfect format separation bounds every JPEG conclusion.
- **Bullets:** 100% Real JPEG / 100% AI PNG; RGBA-only-AI; square-AI; metadata classifier 1.0000 val/test; PNG re-encode changes zero predictions; Q95 costs little.
- **Visual:** bias summary table (from `dataset_bias_audit.csv`).
- **Notes:** Descriptive only; networks' shortcut use unproven; bias unresolved.

## Slide 10 — Limitations (1 min)
- **Message:** Scope discipline.
- **Bullets:** one dataset/generators; shared test set; no significance testing; no bit-reproducibility claim; no external generalization; format confound.
- **Visual:** none.
- **Notes:** Invite scrutiny; list each limit plainly.

## Slide 11 — Conclusion & Future Work (1 min)
- **Message:** Answer the RQ within scope, then next steps.
- **Bullets:** robustness improves substantially at small clean cost; order matters; bias work remains; future: significance tests, frequency-aware models, real platform pipelines, bias-mitigated data.
- **Visual:** single takeaway numbers (clean 90.73→88.95; C3 recall 54.78→91.47).
- **Notes:** Close by restating the question and the bounded answer.
