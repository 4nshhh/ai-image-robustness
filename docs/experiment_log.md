# Experiment Log

> Scaffold only. Record each real experiment here. Do NOT fabricate results.
> Use TBD / NOT RUN for pending experiments.

## Experiment: baseline_resnet50_clean

- Status: NOT RUN
- Model: TBD
- Training data: TBD
- Test data: TBD
- Transformations: None
- Results:
  - Accuracy: TBD
  - Precision: TBD
  - Recall: TBD
  - F1: TBD
  - ROC-AUC: TBD
  - AI Recall: TBD
- Observations: TBD

## Dataset Decision: finalize 7,986-image dataset (integrity audit)

- Date: 2026-10-07 (krish-branch, uncommitted decision, no experiment run)
- Original target: 8,000 images (train 5,600 / val 1,200 / test 1,200).
- Finding: 14 `test/ai` files absent from the exported ZIP; exact recovery
  impossible from local sources; train/val same-named files proven different
  images by SHA-256 and explicitly NOT substituted.
- Cross-split SHA-256 audit of the 7,986 present images: zero exact
  duplicate groups (no train/test, train/val, or val/test leakage by content).
- Decision (methodological, not an oversight): accept a final 7,986-image
  dataset — train 5,600 (2,800 Real / 2,800 AI), val 1,200 (600 / 600),
  test 1,186 (600 Real / 586 AI). `manifest.csv` pruned to present files
  only; `split_metadata.json` recomputed accordingly.
- Consequence for future experiments: test AI class is 586, not 600; all
  test metrics must use n = 1,186 until stated otherwise.
