# Results Reference — Authoritative Numbers and Their Sources

> Every metric below names condition, model, value, and source artifact. Rate differences are **percentage points (pp)**. No relative-percent changes are reported. `n = 1,186` (600 Real / 586 AI) for every row unless noted.

## Dataset and Splits

| Item | Value | Source |
|---|---|---|
| Total / Real / AI | 7,986 / 4,000 / 3,986 | `data/splits/genimage_8000_split_v4/split_metadata.json` |
| Train / Val / Test | 5,600 (2,800/2,800) / 1,200 (600/600) / 1,186 (600/586) | same + `manifest.csv` (7,986 rows) |
| Seed / generators | 42 / adm 569, biggan 570, glide 569, midjourney 569, sdv5 570, vqdm 572, wukong 567 | same |
| Duplicates / missing | 0 cross-split SHA-256 groups; 14 test/AI files unrecoverable | `data/splits/genimage_8000_split_v4/split_metadata.json` (`integrity_note`) |

## Clean Baselines (threshold 0.5)

| Model | Acc | Prec | AI Rec | F1 | AUC | FNR | CM [[TN,FP],[FN,TP]] | Source |
|---|---|---|---|---|---|---|---|---|
| EfficientNet-B0 | 0.9073 | 0.9034 | 0.9096 | 0.9065 | 0.9649 | 0.0904 | [[543,57],[53,533]] | `results/metrics/baseline_efficientnet_b0_clean.json` |
| ResNet-50 | 0.9081 | 0.9409 | 0.8686 | 0.9033 | 0.9682 | 0.1314 | [[568,32],[77,509]] | `results/metrics/baseline_resnet50_clean.json` |
| Robust EffNet-B0 | 0.8895 | 0.8652 | 0.9198 | 0.8916 | 0.9591 | 0.0802 | [[516,84],[47,539]] | `results/metrics/robust_evaluation_efficientnet_b0.csv` (clean rows) |

Robust training: from baseline checkpoint; Adam lr/wd 1e-4, 10 epochs, batch 16, seed 42, CUDA; best clean-val epoch 8 (0.9008). Source: `models/robust/robust_training_history.json`, `models/robust/robust_training_config.json`.

## Condition Results (baseline → robust AI recall, pp change)

Source for all rows: `results/metrics/baseline_vs_robust_efficientnet_b0.csv` (full metric + CM rows in `results/metrics/robust_evaluation_efficientnet_b0.csv`).

| Condition | Baseline AI rec | Robust AI rec | Δ (pp) | Baseline acc | Robust acc |
|---|---|---|---|---|---|
| clean | 0.9096 | 0.9198 | +1.02 | 0.9073 | 0.8895 |
| jpeg_q90 | 0.8498 | 0.9113 | +6.14 | 0.8727 | 0.8820 |
| jpeg_q70 | 0.7474 | 0.8976 | +15.02 | 0.8272 | 0.8727 |
| jpeg_q50 | 0.6775 | 0.8908 | +21.33 | 0.7934 | 0.8718 |
| jpeg_q30 | 0.5546 | 0.8754 | +32.08 | 0.7428 | 0.8659 |
| resize_075 | 0.9539 | 0.9471 | −0.68 | 0.8685 | 0.8744 |
| resize_050 | 0.9590 | 0.9505 | −0.85 | 0.7707 | 0.8533 |
| resize_025 | 0.9061 | 0.9488 | +4.27 | 0.5371 | 0.6492 |
| recomp_q90_p2 | 0.8481 | 0.9147 | +6.66 | 0.8727 | 0.8853 |
| recomp_q70_p2 | 0.7440 | 0.8942 | +15.02 | 0.8272 | 0.8735 |
| recomp_q50_p2 | 0.6724 | 0.8891 | +21.67 | 0.7917 | 0.8718 |
| recomp_q30_p2 | 0.5529 | 0.8754 | +32.25 | 0.7428 | 0.8676 |
| c1_resize_jpeg | 0.5580 | 0.9147 | +35.67 | 0.7336 | 0.8288 |
| c2_jpeg_resize | 0.8823 | 0.9317 | +4.95 | 0.7715 | 0.8541 |
| c3_resize_jpeg_jpeg | 0.5478 | 0.9147 | +36.69 | 0.7310 | 0.8255 |

## Unseen-by-Training Conditions

resize_025, recomp_q50_p2, recomp_q30_p2, c3 (excluded from the augmentation policy; see training config). All improve (see Δ column above); resize_025 remains weakest absolute for both models.

## Bias Audit

100% Real `.jpeg`/JPEG with quantization tables vs 100% AI `.png`/PNG without; RGBA only in AI (~14%); grayscale-L only in Real (~1.4%); AI always square 128–1024px, Real variable rectangles; 7-feature metadata classifier: val 1.0000, held-out test 1.0000; 0 unreadable files. Sources: `results/metrics/dataset_bias_audit.csv`, `results/experiment_logs/dataset_bias_audit.json`.

## Format-Normalization Diagnostic (n=1,186/row)

Lossless PNG: zero prediction changes for either model. JPEG Q95 (4:4:4): baseline AI recall 0.9096→0.8857 (−2.39pp); robust 0.9198→0.9096 (−1.02pp). Source: `results/metrics/format_diagnostic_efficientnet_b0.csv`, `results/metrics/baseline_vs_robust_format_efficientnet_b0.csv`.

## Artifact Paths

- Metrics: `results/metrics/{robust_evaluation,baseline_vs_robust,baseline_efficientnet_b0_clean,baseline_resnet50_clean,format_diagnostic,baseline_vs_robust_format,dataset_bias_audit}.csv|json`, `robustness_analysis_summary.md`
- Plots: `results/graphs/` (7 PNGs: recall/accuracy/FNR comparisons, JPEG + resize sweeps, key CMs per model)
- Per-image predictions: `results/predictions/` (large; not needed for figures)
- Logs: `results/experiment_logs/` (per-condition JSONs incl. `dataset_bias_audit.json`)
- Training record: `models/robust/robust_training_{history,config}.json` (weights gitignored)

## Reconciliation Note

No discrepancies found: clean controls in every experiment CSV reproduce the baseline JSONs exactly (full precision); comparison-CSV deltas recompute from model rows; CM cells sum to 1,186 in all 30+ rows checked. (Historical note: retired scaffold docs `methodology.md` / `results_summary.md` / `experiment_log.md` once held TBD/NOT RUN placeholders; the decision record they contained survives in `split_metadata.json` and `docs/DEVELOPMENT_LOG.md`.)
