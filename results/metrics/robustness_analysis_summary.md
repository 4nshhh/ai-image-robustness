# Robustness Analysis Summary (generated from saved CSVs)

Source: `results/metrics/robust_evaluation_efficientnet_b0.csv` and
`results/metrics/baseline_vs_robust_efficientnet_b0.csv` (30 model-condition rows, n=1,186 each). Deltas are percentage points.

## Clean-performance trade-off (EfficientNet-B0, threshold 0.5)

- Baseline clean accuracy 0.9073 -> robust 0.8895 (-1.77 pp).
- Baseline clean AI recall 0.9096 -> robust 0.9198 (+1.02 pp); AI FNR 0.0904 -> 0.0802.
- Clean ROC-AUC 0.9649 -> 0.9591 (-0.58 pp).

## Largest AI-recall gains (robust minus baseline, pp)

- c3_resize_jpeg_jpeg: +36.69 pp.
- c1_resize_jpeg: +35.67 pp.
- recomp_q30_p2: +32.25 pp.
- jpeg_q30: +32.08 pp.
- recomp_q50_p2: +21.67 pp.

## Regressions / negligible change (AI recall delta < 0 pp)

- resize_050: -0.85 pp.
- resize_075: -0.68 pp.

## Unseen-by-training conditions (excluded from the training policy)

- resize_025: baseline AI recall 0.9061 -> robust 0.9488 (+4.27 pp).
- recomp_q50_p2: baseline AI recall 0.6724 -> robust 0.8891 (+21.67 pp).
- recomp_q30_p2: baseline AI recall 0.5529 -> robust 0.8754 (+32.25 pp).
- c3_resize_jpeg_jpeg: baseline AI recall 0.5478 -> robust 0.9147 (+36.69 pp).

## Interpretation limits

- Observed patterns only; no causal mechanism is claimed.
- Known confound: Real images are file-format/metadata-wise JPEG while AI images are predominantly PNG, so JPEG-related changes may reflect both compression effects and format/distribution bias.
- Results cover 15 fixed conditions on one frozen test set; they do not demonstrate universal real-world robustness.
