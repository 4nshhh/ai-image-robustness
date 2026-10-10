# Robust Detection of AI-Generated Images Under Social Media-Style Compression and Transformations

Binary image classification (Real vs AI-generated) with a focus on **robustness**: how social-media-style transformations degrade detection, and whether transformation-aware training restores it.

## Research Question

How do JPEG compression, resizing, recompression, and combinations of these transformations affect AI-generated image detection performance, and can transformation-aware training improve robustness?

## Motivation and Objectives

Detectors are usually evaluated on clean images, but deployed images are routinely recompressed and downscaled. This project (1) measures degradation under controlled transformations on a frozen test set, (2) tests whether training with the same transformation families improves robustness, and (3) audits dataset biases that could confound the conclusions. Label mapping throughout: **0 = Real, 1 = AI-generated**.

## Contributions and Scope

- Controlled robustness study: clean + 14 transformed conditions on one frozen 1,186-image test set, same checkpoint, preprocessing, threshold (0.5), and code.
- Transformation-aware EfficientNet-B0 (fine-tuned from the baseline under a logged stochastic policy), compared head-to-head across all 15 conditions (30 model-condition evaluations).
- Format/metadata bias audit with a perfect-separation finding, plus a format-normalization diagnostic.
- Read-only Streamlit dashboard (`app/app.py`) presenting only validated saved results.
- In scope: JPEG, resize, recompression, 3 combined pipelines, clean/transformed comparison, bias audit. Out of scope: new generators, video, real-world deployment claims.

## Key Findings (validated; deltas in percentage points)

| Condition | Baseline AI recall | Robust AI recall | Change |
|---|---|---:|---:|
| Clean | 90.96% | 91.98% | +1.02 pp |
| JPEG Q30 | 55.46% | 87.54% | +32.08 pp |
| Resize 50% | 95.90% | 95.05% | −0.85 pp |
| Recompression Q30 ×2 | 55.29% | 87.54% | +32.25 pp |
| Resize → JPEG → JPEG (C3) | 54.78% | 91.47% | +36.69 pp |

- Baseline clean accuracy 90.73%; robust clean accuracy 88.95% (−1.77 pp) — robustness costs some clean accuracy.
- JPEG/recompression collapse baseline AI recall (Q30: 90.96% → ~55%); robust training largely preserves it (~88%).
- Resizing fails oppositely: Real precision collapses (false alarms), AI recall stays high for both models.
- Transformation order matters: C1 (Resize→JPEG) behaves like JPEG; C2 (JPEG→Resize) behaves like resize.
- Unseen-by-training conditions (Resize 25%, Recompression Q50/Q30, C3) all improve substantially.
- Full tables: [`docs/RESULTS_REFERENCE.md`](docs/RESULTS_REFERENCE.md); raw data: [`results/metrics/baseline_vs_robust_efficientnet_b0.csv`](results/metrics/baseline_vs_robust_efficientnet_b0.csv).

## Dashboard

Five sections (Overview, Transformation Robustness, Model Comparison, Dataset Bias & Format, Methodology & Limitations). Read-only: no uploads, inference, GPU, checkpoints, or raw data needed — it reads small saved artifacts under `results/metrics/` and `results/experiment_logs/`.

```powershell
streamlit run app/app.py
```

No dashboard screenshot is committed yet; capture the Overview tab before submission. Deployment: Streamlit Community Cloud, entry point `app/app.py`, no secrets.

## Repository Structure

```text
├── AGENTS.md                  # methodology + contributor rules (binding)
├── README.md
├── requirements.txt
├── .streamlit/config.toml     # dashboard theme
├── app/app.py                 # read-only Streamlit dashboard
├── data/splits/genimage_8000_split_v4/  # manifest.csv + split_metadata.json (images gitignored)
├── docs/                      # EXPERIMENT_PROTOCOL, DEVELOPMENT_LOG, FINAL_REPORT_SOURCE,
│                              # PRESENTATION_SOURCE, RESULTS_REFERENCE, REFERENCES, …
├── models/baseline|robust/    # histories + configs committed; *.pth weights gitignored
├── notebooks/01–04            # analysis/training/experiment/robust-training notebooks
├── results/metrics|graphs|confusion_matrices|predictions|experiment_logs/
├── scripts/                   # verify_baseline, run_*_robustness, train_robust, plotting, audit, app
├── src/                       # dataset, preprocessing, model, train, transformations,
│                              # evaluate, metrics, experiment_config, results, experiment_runner
└── tests/test_ml_foundation.py
```

## Environment Setup

```powershell
conda create -n ml_clean python=3.11
conda activate ml_clean
pip install -r requirements.txt
```

Core dependencies: `torch`, `torchvision`, `numpy`, `pandas`, `Pillow`, `opencv-python`, `scikit-learn`, `matplotlib`, `tqdm`, `streamlit`, `jupyter`. GPU runs used CUDA builds (e.g. torch 2.13+cu126); the dashboard itself needs no GPU.

## Dataset

Tiny-GenImage subset, 7,986 images: train 5,600 (2,800/2,800), val 1,200 (600/600), test 1,186 (600 Real / 586 AI), seed 42. Seven AI generators (adm, biggan, glide, midjourney, sdv5, vqdm, wukong; ~569–572 images each). 14 test/AI files were unrecoverable from the source export (see `docs/RESULTS_REFERENCE.md`); SHA-256 audit found zero cross-split duplicates. Manifest: `data/splits/genimage_8000_split_v4/manifest.csv` (images themselves are gitignored — obtain separately; see `docs/EXPERIMENT_PROTOCOL.md`).

## Models and Workflow

EfficientNet-B0 (primary; also a ResNet-50 baseline: clean acc 90.81%, AI recall 86.86%), 2-class head, ImageNet normalization, 224×224 input, threshold 0.5. Baseline: Adam, lr/weight-decay 1e-4, 10 epochs, batch 16, best-validation-accuracy selection. Robust: fine-tuned from the baseline checkpoint with a fixed policy (40% clean / 20% JPEG Q{90,70,50,30} / 15% resize {0.75,0.50} / 15% recompression Q{90,70}×2 / 10% combined C1–C2; seed 42), best clean-val epoch 8 (0.9008). Train-only pixels for training; test never used for tuning or selection. Reproduce: `conda run -n ml_clean python scripts/verify_baseline.py`, `scripts/run_jpeg_robustness.py`, `scripts/run_resize_robustness.py`, `scripts/run_recompression_robustness.py`, `scripts/run_combined_robustness.py`, `scripts/run_robust_evaluation.py` (train later via `scripts/train_robust.py`).

## Transformations and Evaluation

JPEG Q{90,70,50,30}; resize {0.75,0.50,0.25} aspect-preserving; recompression = two sequential same-quality passes; C1 = Resize 50%→JPEG Q50, C2 = reverse, C3 = Resize→JPEG→JPEG. Research transforms precede the single final preprocessing; both classes always transformed. Metrics: accuracy, precision, AI recall, F1, ROC-AUC (from AI probabilities), AI FNR, Real recall, FPR, TN/FP/FN/TP. Rate differences in percentage points.

## Results and Visualizations

- [`results/metrics/baseline_vs_robust_efficientnet_b0.csv`](results/metrics/baseline_vs_robust_efficientnet_b0.csv) — 15 conditions with pp deltas.
- [`results/metrics/robust_evaluation_efficientnet_b0.csv`](results/metrics/robust_evaluation_efficientnet_b0.csv) — 30 model-condition rows.
- [`results/graphs/`](results/graphs/) — AI recall/accuracy/FNR comparisons, JPEG/resize sweeps, key confusion matrices.
- [`results/metrics/robustness_analysis_summary.md`](results/metrics/robustness_analysis_summary.md) — generated summary.
- [`results/predictions/`](results/predictions/) — per-image records (large; not needed for the dashboard).

## Reproducibility

Fixed seed (42), frozen manifest/splits, pinned transform parameters and order, logged configs per run (`results/experiment_logs/`), deterministic preprocessing. Exact GPU bit-reproducibility is not claimed. Clean controls reproduce the baseline within 1e-4 in every experiment script (asserted at runtime).

## Limitations and Dataset Bias

**Confirmed confound:** 100% of Real images are `.jpeg`/JPEG (with quantization tables) vs 100% of AI images `.png`/PNG (none); AI images are always square (128–1024px), Real are variable rectangles; a 7-feature metadata classifier reaches 1.0000 validation/test accuracy (`results/experiment_logs/dataset_bias_audit.json`). JPEG-family results therefore entangle compression damage with format bias — reported as observation, not causal proof. Lossless PNG re-encoding changes no prediction; JPEG Q95 costs little, so heavy-compression damage is lossy-pixel damage, not container identity. No general-purpose detection claim; unseen generators/platforms untested.

## References

- Zhu et al., GenImage: A Million-Scale Benchmark for Detecting AI-Generated Images. <https://arxiv.org/abs/2306.08571>
- Grommelt et al., Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets. <https://arxiv.org/abs/2403.17608>
- Li et al., Detecting Compressed AI-Generated Images via Phase Spectrum Robustness, CVPR 2026. (As cited in project docs; verify author list before submission — see [`docs/REFERENCES.md`](docs/REFERENCES.md).)

## Team Contributions

Per `AGENTS.md` (roles; names as identified by the team):

- Person 1 (Krish) — Data + Model Lead: dataset acquisition/audit/splits, preprocessing, ResNet-50/EfficientNet-B0, training pipeline, baseline + robust training.
- Person 2 (Ansh) — Experiments + Evaluation + Application: transformations, metrics/evaluation/config/results/runner, robustness experiments 1–9, visualization, Streamlit dashboard.
- Person 3 — Documentation + Presentation: report, slides, final presentation.
