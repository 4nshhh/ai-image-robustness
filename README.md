# Robust Detection of AI-Generated Images Under Social Media-Style Compression and Transformations

## Project Overview

Binary image-classification system distinguishing **Real (0)** vs **AI-generated (1)** images,
with a focus on robustness under realistic transformations (JPEG compression, resizing,
recompression, combined transforms) and transformation-aware training.

Scaffold only — dataset processing, training, transformations, evaluation, and the
Streamlit app are NOT yet implemented.

## Research Question

How do social-media-style transformations such as JPEG compression, resizing, recompression,
and combinations of transformations affect AI-generated image detection performance, and can
transformation-aware training improve detector robustness?

## Repository Structure

```text
ai-image-robustness/
├── AGENTS.md
├── README.md
├── requirements.txt
├── .gitignore
├── data/ (raw/, processed/, splits/, metadata/) — NOT committed
├── notebooks/ (01–04)
├── src/ (dataset, preprocessing, model, train, transformations, evaluate, metrics)
├── models/ (baseline/, robust/)
├── results/ (metrics/, graphs/, confusion_matrices/, predictions/, experiment_logs/)
├── app/app.py
└── docs/ (methodology, experiment_log, results_summary)
```

See `AGENTS.md` for the full research methodology and contributor rules.

## Methodology

See `docs/methodology.md`. Status: TBD / scaffold only.

## Results

See `docs/results_summary.md` and `docs/experiment_log.md`. Status: NOT RUN — no results yet.

## How to Run

TBD — pending implementation.

## Streamlit App

TBD — pending implementation (`app/app.py` is a placeholder).

## Team Contributions

Per `AGENTS.md`:

- Person 1 — Data + Model Lead (`src/dataset.py`, `preprocessing.py`, `model.py`, `train.py`)
- Person 2 — Experiments + Evaluation + Application (`src/transformations.py`, `evaluate.py`, `metrics.py`, `results/`, `app/app.py`)
- Person 3 — Documentation + Presentation

## References

- Zhu et al., GenImage: A Million-Scale Benchmark for Detecting AI-Generated Images. <https://arxiv.org/abs/2306.08571>
- Grommelt et al., Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets. <https://arxiv.org/abs/2403.17608>
- Li et al., Detecting Compressed AI-Generated Images via Phase Spectrum Robustness, CVPR 2026.
