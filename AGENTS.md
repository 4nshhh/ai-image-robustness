# AGENTS.md

## 1. Project Identity

Project Title:

Robust Detection of AI-Generated Images Under Social Media-Style Compression and Transformations

Project Type:

Experimentation-Based Machine Learning / Computer Vision Project

Primary Research Question:

How do social-media-style transformations such as JPEG compression, resizing, recompression, and combinations of transformations affect AI-generated image detection performance, and can transformation-aware training improve detector robustness?

---

## 2. Project Goal

Build a reproducible binary image-classification system that distinguishes:

- Real images
- AI-generated images

The project systematically evaluates how detector performance changes when images are subjected to realistic image transformations.

The primary contribution is:

1. Controlled robustness experimentation
2. Dataset-bias awareness and mitigation
3. Comparison of clean-trained and transformation-aware models
4. Quantitative evaluation of robustness
5. A simple application demonstrating the final models

The Streamlit application is a demonstration layer, not the primary research contribution.

---

## 3. Core Research Pipeline

The overall pipeline is:

Dataset
    ↓
Dataset Audit
    ↓
Bias Analysis
    ↓
Controlled Dataset
    ↓
Train / Validation / Test Split
    ↓
Clean Baseline Training
    ↓
Clean Test Evaluation
    ↓
Transformation Experiments
    ↓
Robustness Analysis
    ↓
Transformation-Aware Training
    ↓
Robust Model Evaluation
    ↓
Standard vs Robust Comparison
    ↓
Error Analysis
    ↓
Streamlit Demonstration

Keep baseline experiments and robustness experiments clearly separated.

---

## 4. IMPORTANT AGENT RULES

This is an academic ML research project.

Do NOT treat this as a generic image-classification project.

The research methodology is more important than simply achieving high accuracy.

Always prioritize:

1. Experimental validity
2. Reproducibility
3. Dataset quality
4. Bias control
5. Prevention of data leakage
6. Correct train/validation/test separation
7. Controlled transformations
8. Consistent evaluation
9. Clear experiment tracking
10. Honest reporting of results

Do not optimize only for accuracy.

Never fabricate:

- metrics
- training results
- dataset statistics
- model performance
- experiment conclusions
- citations
- benchmark comparisons

If an experiment has not been run, clearly mark it as:

NOT RUN

or:

PENDING EXPERIMENT

Never present expected results as actual results.

---

## 5. Dataset

The preferred dataset is:

GenImage

However, the exact dataset/subset must NOT be assumed until the dataset audit is completed.

The initial target may be approximately:

- 4,000 Real images
- 4,000 AI-generated images

This is NOT a hard requirement.

Dataset quality and experimental validity are more important than reaching an arbitrary number of images.

The final dataset should preferably contain:

- balanced Real / AI classes
- multiple AI generators where feasible
- controlled image formats
- reasonably controlled image resolutions
- no corrupted images
- no obvious duplicates
- no train/test leakage

Do not download or process the entire GenImage dataset unless explicitly required.

Use a controlled subset when appropriate.

---

## 6. Dataset Bias Analysis

Dataset bias analysis is mandatory.

AI-generated image detectors can exploit unintended shortcuts such as:

- JPEG compression differences
- PNG vs JPEG differences
- image resolution differences
- file-size differences
- generator-specific artifacts
- preprocessing differences

Investigate whether Real and AI images differ systematically in:

- file format
- width
- height
- aspect ratio
- file size
- compression characteristics
- generator distribution

Do NOT assume that high baseline accuracy means the model learned genuine AI-generation artifacts.

Whenever possible, use the same preprocessing pipeline for both classes.

---

## 7. Data Leakage Prevention

Data leakage is a critical concern.

The dataset must be split into:

train
validation
test

BEFORE performing experimental transformations.

The test set must remain isolated.

Do NOT:

- train on test images
- tune model parameters using test performance
- create transformed versions of test images and use them for training
- allow duplicate or near-duplicate images across splits
- repeatedly optimize against the test set without acknowledging test leakage

If duplicate detection is possible, perform it before finalizing the split.

The test set should be treated as an evaluation set.

---

## 8. Repository Structure

Use the following project structure unless there is a strong technical reason to change it:

ai-image-robustness/
│
├── AGENTS.md
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   └── metadata/
│
├── notebooks/
│   └── 02_baseline_training.ipynb
│       (analysis/robust-training work lives in scripts/; placeholder
│       notebooks 01/03/04 were removed — see development log)
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── transformations.py
│   ├── evaluate.py
│   └── metrics.py
│
├── models/
│   ├── baseline/
│   └── robust/
│
├── results/
│   ├── metrics/
│   ├── graphs/
│   ├── confusion_matrices/
│   ├── predictions/
│   └── experiment_logs/
│
├── app/
│   └── app.py
│
└── docs/
    ├── DEVELOPMENT_LOG.md
    ├── EXPERIMENT_PROTOCOL.md
    ├── FINAL_REPORT_SOURCE.md
    ├── PRESENTATION_SOURCE.md
    ├── REFERENCES.md
    └── RESULTS_REFERENCE.md

Do not create unnecessary directories or duplicate functionality.

---

## 9. Data Files and Git

Large datasets must NOT be committed to GitHub.

The repository should contain:

- metadata
- split CSVs where appropriate
- scripts
- documentation

but not the full image dataset.

Use .gitignore for:

data/raw/
data/processed/
models/*.pth
models/*.pt
*.ckpt
__pycache__/
.ipynb_checkpoints/
.env

Never commit:

- passwords
- API keys
- private credentials
- huge datasets
- unnecessary model files
- environment files containing secrets

---

## 10. Metadata

Maintain a metadata CSV.

Recommended columns:

image_path
label
generator
format
width
height
aspect_ratio
file_size
split

Label mapping:

0 = Real
1 = AI-generated

The label mapping must remain consistent throughout the project.

Do not silently change label encoding.

---

## 11. Train / Validation / Test Split

Recommended starting split:

70% train
15% validation
15% test

The exact ratio may change if justified.

The split should be:

- reproducible
- stratified where appropriate
- documented
- saved to CSV

Use a fixed random seed.

Recommended starting seed:

SEED = 42

Do not randomly regenerate a new split every time the project runs.

---

## 12. Preprocessing

The baseline preprocessing pipeline should be consistent between Real and AI images.

Typical pipeline:

Image
    ↓
RGB conversion
    ↓
Resize
    ↓
Tensor conversion
    ↓
Normalization

Use preprocessing appropriate for the selected pretrained model.

Do not apply class-specific preprocessing.

Bad:

Real → JPEG conversion
AI   → PNG conversion

Good:

Real → same preprocessing
AI   → same preprocessing

---

## 13. Baseline Models

Initial candidate models:

1. ResNet-50
2. EfficientNet-B0

Use transfer learning from pretrained ImageNet weights when available.

The final classification layer must support binary classification.

Conceptually:

Backbone
    ↓
Global Pooling
    ↓
Classification Layer
    ↓
Real / AI

The models must be evaluated under comparable conditions.

Do not compare models using different:

- datasets
- splits
- preprocessing
- evaluation metrics
- training conditions

unless that difference is explicitly part of the experiment.

---

## 14. Primary Model Selection

After baseline experiments, select one primary model for deeper robustness experimentation.

Selection should consider:

- validation performance
- test performance
- F1 score
- AI-class recall
- ROC-AUC
- computational cost
- inference speed
- training stability

Do not automatically select the model with the highest accuracy.

Document the model-selection decision.

---

## 15. Transformations

The primary transformations investigated are:

### JPEG Compression

Suggested quality levels:

Q90
Q70
Q50
Q30

These levels may be adjusted after initial experimentation.

### Resizing

Investigate controlled downscaling.

Possible levels:

0.75x
0.50x
0.25x

The exact levels must be configurable.

### Recompression

Investigate repeated JPEG compression.

Example:

Original
    ↓
JPEG compression
    ↓
Save
    ↓
JPEG recompression

Recompression is different from applying JPEG compression once.

Preserve this distinction in the implementation.

### Combined Transformations

Examples:

Resize → JPEG
JPEG → Resize
Resize → JPEG → JPEG

Transformation order may matter and should be documented.

---

## 16. Transformation Implementation

All transformation functions should be implemented in:

src/transformations.py

Each transformation should be:

- configurable
- reusable
- independently testable
- deterministic when required

Prefer functions such as:

apply_jpeg_compression(...)
apply_resize(...)
apply_recompression(...)
apply_pipeline(...)

Avoid hardcoding experimental values throughout the codebase.

Keep experiment parameters centralized.

Example:

JPEG_QUALITY_LEVELS = [90, 70, 50, 30]

---

## 17. Transform Both Classes

When evaluating robustness, transformations should generally be applied to BOTH:

Real images
AI-generated images

Do not create an experiment where only AI images are compressed unless that is explicitly the research question.

Otherwise, the model may exploit transformation differences instead of AI-generation characteristics.

Good:

Real → JPEG Q50
AI   → JPEG Q50

Bad:

Real → Original
AI   → JPEG Q50

unless explicitly defined as a separate controlled experiment.

---

## 18. Clean Baseline Experiment

The first model should be trained on clean training data.

Example:

Training:
Clean Real + Clean AI

Testing:
Clean Real + Clean AI

This establishes the clean baseline.

Record:

- accuracy
- precision
- recall
- F1
- ROC-AUC
- confusion matrix
- AI recall
- AI false-negative rate

---

## 19. Robustness Experiment

After the clean baseline is established, evaluate the same trained model on transformed test data.

Example:

Model trained on clean data

Test:
Clean
JPEG Q90
JPEG Q70
JPEG Q50
JPEG Q30
Resize
Recompression
Combined transformations

Do NOT retrain the baseline model between these tests.

The purpose is to measure performance degradation caused by transformations.

---

## 20. Transformation-Aware Training

After baseline robustness experiments, train a robust model.

Transformation-aware training means the training data contains controlled transformations.

Example:

Training:
Real + AI
      +
controlled transformations

Testing:
Clean test
Transformed test
Unseen transformation levels if possible

The robust model must still solve:

Real vs AI

Do not change the classification task.

---

## 21. Seen vs Unseen Transformations

Where practical, distinguish between:

### Seen transformations

Transformation levels used during training.

Example:

Training:
JPEG Q90 + Q70

Testing:
Q90 + Q70

### Unseen transformations

Transformation levels not used during training.

Example:

Training:
JPEG Q90 + Q70

Testing:
JPEG Q50 + Q30

This provides a stronger test of generalization.

Do not claim generalization to unseen transformations unless the experiment was actually performed.

---

## 22. Evaluation Metrics

Important experiments should report:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

Also emphasize:

- AI-class Recall
- AI False Negative Rate
- Real-class Recall
- False Positive Rate

Because the project specifically investigates AI-generated image detection.

Do not rely only on accuracy.

---

## 23. Confusion Matrix

Label mapping:

0 = Real
1 = AI

Interpretation:

True Negative:
Real predicted as Real

False Positive:
Real predicted as AI

False Negative:
AI predicted as Real

True Positive:
AI predicted as AI

False negatives are especially important because they represent AI-generated images that evade detection.

---

## 24. Results Storage

Every major experiment should save its results.

Recommended structure:

results/
├── metrics/
├── graphs/
├── confusion_matrices/
├── predictions/
└── experiment_logs/

Example:

results/metrics/baseline_resnet50.csv
results/metrics/jpeg_q50.csv
results/metrics/robust_model.csv

Example graphs:

jpeg_quality_vs_f1.png
jpeg_quality_vs_ai_recall.png
baseline_vs_robust.png

Do not overwrite previous experiment results without a clear reason.

---

## 25. Experiment Naming

Use descriptive experiment names.

Examples:

baseline_resnet50_clean
baseline_efficientnetb0_clean
resnet50_jpeg_q90
resnet50_jpeg_q70
resnet50_jpeg_q50
resnet50_jpeg_q30
resnet50_resize_050
resnet50_recompression
resnet50_combined
resnet50_robust

Avoid:

test1
final2
new_model
best_model
experiment_final_final

---

## 26. Random Seeds

Use reproducible random seeds where possible.

Example:

SEED = 42

Set seeds for:

- Python
- NumPy
- PyTorch
- data splitting

Document when deterministic behavior is not possible.

---

## 27. Model Checkpoints

Use meaningful checkpoint names.

Examples:

models/baseline/resnet50_baseline.pth
models/baseline/efficientnetb0_baseline.pth
models/robust/resnet50_robust.pth

Each checkpoint must have associated information about:

- architecture
- preprocessing
- class mapping
- training configuration
- dataset split
- transformation configuration

Do not create a model checkpoint without knowing how its preprocessing works.

---

## 28. Code Organization

Write modular Python code.

Avoid putting the entire project inside notebooks.

Notebooks should primarily be used for:

- exploration
- visualization
- experiment analysis

Reusable functionality belongs in:

src/

Use:

dataset.py
preprocessing.py
model.py
train.py
transformations.py
evaluate.py
metrics.py

Do not duplicate functionality unnecessarily.

---

## 29. Notebook Rules

Use numbered notebooks:

01_dataset_analysis.ipynb
02_baseline_training.ipynb
03_transformation_experiments.ipynb
04_robust_training.ipynb

Each notebook should have a clear purpose.

Do not create dozens of random notebooks.

Important reusable logic should not exist only inside notebooks.

---

## 30. Dataset Analysis Notebook

01_dataset_analysis.ipynb should investigate:

- number of Real images
- number of AI images
- generator distribution
- image formats
- image resolutions
- aspect ratios
- file sizes
- corrupted images
- duplicate candidates
- class balance
- compression bias
- resolution bias

Useful plots include:

- Class distribution
- Resolution distribution
- Format distribution
- Generator distribution
- Width/height distribution
- File-size distribution

---

## 31. Baseline Training Notebook

02_baseline_training.ipynb should:

1. Load finalized splits
2. Load preprocessing
3. Train candidate models
4. Evaluate validation performance
5. Evaluate clean test performance
6. Save metrics
7. Save confusion matrices
8. Compare candidate models

---

## 32. Transformation Experiment Notebook

03_transformation_experiments.ipynb should:

1. Load the selected baseline checkpoint
2. Load the untouched test split
3. Apply transformations
4. Evaluate the same model
5. Store metrics
6. Generate robustness plots
7. Analyze failure cases

The baseline checkpoint must NOT be retrained for every transformation.

---

## 33. Robust Training Notebook

04_robust_training.ipynb should:

1. Define transformation-aware training configuration
2. Train robust model
3. Evaluate on clean test data
4. Evaluate on transformed test data
5. Compare against baseline
6. Save results
7. Generate comparison plots

---

## 34. Evaluation Code

Evaluation functionality belongs in:

src/evaluate.py
src/metrics.py

Evaluation should be reusable.

Do not duplicate metric calculation across multiple notebooks.

---

## 35. Error Analysis

Do not stop after computing metrics.

Investigate:

### False Positives

Real images predicted as AI.

### False Negatives

AI-generated images predicted as Real.

False negatives are especially important.

When possible, investigate whether errors are associated with:

- strong compression
- low resolution
- certain generators
- certain image content
- unusual aspect ratios
- transformation combinations

Do not make causal claims without evidence.

---

## 36. Streamlit Application

The application should be located at:

app/app.py

Basic flow:

Upload image
      ↓
Preview image
      ↓
Optional transformation
      ↓
Standard model prediction
      ↓
Robust model prediction
      ↓
Display:
    Real / AI
    Confidence
      ↓
Compare predictions

The application should use the same preprocessing and model-loading code as the experiments.

Do NOT duplicate model preprocessing manually inside the UI.

---

## 37. Application Design

Keep the UI simple.

Recommended interface:

AI Image Detection

Upload Image

[Image Preview]

Transformation:
- None
- JPEG Q90
- JPEG Q70
- JPEG Q50
- JPEG Q30
- Resize

Results:

Standard Model:
Prediction:
Confidence:

Robust Model:
Prediction:
Confidence:

If the robust model is unavailable, do not fake its output.

Display a clear message instead.

---

## 38. Model Loading

The application must load trained checkpoints.

Do not retrain models when Streamlit starts.

Use caching where appropriate.

The application should not require the full dataset for single-image inference.

---

## 39. Reproducibility

Document:

- Python version
- package versions
- dataset source
- dataset subset selection
- random seed
- image size
- batch size
- learning rate
- optimizer
- number of epochs
- model architecture
- preprocessing
- transformation configuration

Use:

requirements.txt

for dependencies.

---

## 40. Dependencies

Expected core dependencies:

torch
torchvision
numpy
pandas
Pillow
opencv-python
scikit-learn
matplotlib
tqdm
streamlit

Only add additional dependencies when necessary.

Do not introduce large frameworks without a clear reason.

---

## 41. Hardware

The project may be developed using:

- local Windows machine
- NVIDIA GPU
- Google Colab
- Kaggle

Code should detect CUDA when available.

Example:

device = "cuda" if torch.cuda.is_available() else "cpu"

Do not hardcode a specific GPU.

---

## 42. Configuration

Avoid hardcoding experimental settings throughout source files.

Important parameters include:

SEED
IMAGE_SIZE
BATCH_SIZE
LEARNING_RATE
NUM_EPOCHS
MODEL_NAME
NUM_CLASSES
JPEG_QUALITY_LEVELS
RESIZE_LEVELS

Keep experimental configuration centralized where practical.

---

## 43. Git Rules

Use Git regularly.

Prefer logically grouped commits.

Good examples:

feat: add dataset audit pipeline
feat: implement resnet50 baseline
feat: add jpeg transformation experiments
feat: add robustness evaluation
feat: add transformation-aware training
feat: add streamlit demo
docs: update experiment results

Avoid:

update
changes
final
final2
stuff

Never commit secrets or large datasets.

---

## 44. Before Modifying Existing Code

Before making a major modification:

1. Inspect the existing implementation.
2. Understand the current pipeline.
3. Reuse existing utilities where possible.
4. Avoid creating duplicate functionality.
5. Preserve existing interfaces unless there is a strong reason to change them.

Do not rewrite working modules unnecessarily.

---

## 45. Adding a New Experiment

Every experiment should answer:

What question does this experiment answer?

What variable is being changed?

What variables remain fixed?

What dataset split is used?

What model is used?

What metrics are measured?

Where are the results saved?

If these cannot be answered, the experiment is probably not sufficiently controlled.

---

## 46. Experimental Control

When comparing models or transformations, keep as many variables fixed as possible.

For example:

Same test set
Same preprocessing
Same metrics
Same label mapping
Same evaluation code

Only change the variable being studied.

---

## 47. Comparison Tables

Important results should eventually be summarized in tables.

Example:

| Model | Condition | Accuracy | Precision | Recall | F1 | ROC-AUC | AI Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| ResNet-50 | Clean | TBD | TBD | TBD | TBD | TBD | TBD |
| ResNet-50 | JPEG Q50 | TBD | TBD | TBD | TBD | TBD | TBD |
| ResNet-50 Robust | JPEG Q50 | TBD | TBD | TBD | TBD | TBD | TBD |

Never fill TBD with guessed values.

---

## 48. Results Interpretation

Distinguish between:

### Observation

What the experiment directly shows.

Example:

AI recall decreased from X to Y under JPEG Q30.

### Interpretation

A reasonable explanation supported by the experiment.

Example:

Detection performance becomes less reliable under stronger compression.

### Hypothesis

A possible explanation requiring further evidence.

Example:

Compression may be removing high-frequency artifacts useful for detection.

Do not present hypotheses as established facts.

---

## 49. Scientific Scope

Do not claim:

"Our model detects AI-generated images reliably in the real world."

unless supported by appropriate experiments.

Prefer:

"Our experiments show that the model maintains X performance under the evaluated transformation conditions."

Conclusions must remain within the scope of the experiments.

---

## 50. Literature-Informed Methodology

The project is informed by research on:

1. GenImage and large-scale AI-generated image detection
2. Dataset biases such as JPEG/compression and image-size shortcuts
3. Detection of compressed AI-generated images

These papers motivate the methodology.

Do not copy complex research architectures merely because they appear in the papers.

The project should remain feasible for a student experimentation project.

---

## 51. Primary References

### GenImage

Zhu et al.

GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image

https://arxiv.org/abs/2306.08571

### Dataset Bias

Grommelt et al.

Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets

https://arxiv.org/abs/2403.17608

### Compressed AI Image Detection

Li et al.

Detecting Compressed AI-Generated Images via Phase Spectrum Robustness

CVPR 2026.

https://openaccess.thecvf.com/content/CVPR2026/html/Li_Detecting_Compressed_AI-Generated_Images_via_Phase_Spectrum_Robustness_CVPR_2026_paper.html

Do not invent references.

If additional literature is introduced, clearly identify it as additional literature.

---

## 52. Team Roles

The project has three team members.

These roles represent PRIMARY OWNERSHIP and responsibility, not strict file permissions.

A team member may modify another person's files when required for:

- integration
- bug fixes
- refactoring
- consistent architecture
- shared functionality

Avoid unnecessary modifications to files primarily owned by another team member.

---

## 53. Person 1 — Data + Model Lead

Primary responsibilities:

- dataset acquisition
- dataset audit
- bias analysis
- metadata
- final subset selection
- train/validation/test split
- DataLoader
- preprocessing
- ResNet-50
- EfficientNet-B0
- training pipeline
- model checkpoints
- robust model training

Primary files:

src/dataset.py
src/preprocessing.py
src/model.py
src/train.py

---

## 54. Person 2 — Experiments + Evaluation + Application

Primary responsibilities:

- transformation implementation
- evaluation framework
- metrics
- robustness experiments
- transformation comparison
- error analysis
- graphs
- confusion matrices
- standard vs robust model comparison
- Streamlit application

Primary files:

src/transformations.py
src/evaluate.py
src/metrics.py
results/
app/app.py

Person 2 collaborates with Person 1 on:

- transformation-aware training
- robustness experiment design
- final model selection

---

## 55. Person 3 — Documentation + Presentation

Primary responsibilities:

- project documentation
- literature organization
- PPT
- report formatting
- methodology diagrams
- results presentation
- demo/video script
- final presentation

Person 3 should understand:

- dataset
- models
- transformations
- metrics
- final results

Person 3 must not invent technical results or claims.

---

## 56. Role Usage

When a team member asks the agent to implement something, they should identify their role when useful.

Example:

"I am Person 2. Implement the transformation framework."

The agent should prioritize the relevant ownership area while still inspecting the rest of the repository for integration requirements.

Roles are NOT strict access boundaries.

The agent may modify another person's files when necessary.

---

## 57. Team Integration Checkpoints

The implementation team should synchronize at three major checkpoints.

### Checkpoint 1 — Dataset

Agree on:

- final dataset
- labels
- metadata
- preprocessing
- train/validation/test split
- bias findings

### Checkpoint 2 — Baseline

Agree on:

- baseline models
- clean test results
- selected primary model
- preprocessing
- checkpoint

### Checkpoint 3 — Robustness

Agree on:

- transformations
- severity levels
- robustness results
- robust training setup
- standard vs robust comparison

Do not allow incompatible pipelines to develop independently.

---

## 58. Person 1 → Person 2 Handoff

Person 1 should provide:

- final train split
- final validation split
- final test split
- class mapping
- preprocessing configuration
- baseline model checkpoint
- model loading function
- selected primary model

Person 2 should be able to:

load checkpoint
+
load test split
+
apply transformation
+
evaluate

without rebuilding the training pipeline.

---

## 59. Person 2 → Person 1 Handoff

Person 2 should provide:

- transformation implementations
- recommended transformation levels
- transformation experiment results
- recommended robustness training configuration

Person 1 then implements transformation-aware training.

---

## 60. Final Model Comparison

The final comparison should ideally contain:

Standard Model
vs
Robust Model

under applicable conditions such as:

Clean
JPEG Q90
JPEG Q70
JPEG Q50
JPEG Q30
Resize
Recompression
Combined transformations

Only include conditions that were actually tested.

---

## 61. Final Research Questions

The final project should answer:

1. How well does the baseline detect AI-generated images on clean data?
2. How does JPEG compression affect performance?
3. How does resizing affect performance?
4. How does recompression affect performance?
5. What happens under combined transformations?
6. Does transformation-aware training improve robustness?
7. Does robustness training hurt clean-image performance?
8. Which model provides the best trade-off?
9. Which types of images remain difficult to classify?

These questions should guide the final analysis.

---

## 62. Do Not Overengineer

This is a student ML project.

Do NOT introduce unnecessary complexity such as:

- custom transformers
- complicated phase-spectrum architectures
- distributed training
- microservices
- databases
- unnecessary APIs
- excessive MLOps systems
- complex frontend frameworks

The core project is:

Dataset
+
Baseline Model
+
Controlled Transformations
+
Robustness Experiments
+
Transformation-Aware Training
+
Evaluation
+
Simple Application

Keep the implementation understandable.

---

## 63. Do Not Change the Research Question

Do not silently change the project into:

- generic deepfake detection
- face manipulation detection
- video deepfake detection
- GAN-only detection
- diffusion-only detection
- image generation
- watermark detection
- content moderation

unless explicitly instructed by the project team.

The research focus is:

Robust detection of AI-generated images under social-media-style transformations.

---

## 64. Do Not Create Fake Data

Do not generate synthetic training data simply to make the project appear complete.

If the actual dataset is unavailable, clearly report:

Dataset unavailable

and provide code that can run once the dataset is supplied.

Do not fabricate dataset statistics.

---

## 65. Do Not Fabricate Results

Never produce fake values such as:

Accuracy: 96.7%
F1: 95.8%

unless those values came from an actual experiment.

If code has not been executed, use:

TBD

If execution failed, report the actual error.

---

## 66. Error Handling

When code fails:

1. Read the actual error.
2. Identify the root cause.
3. Fix the smallest necessary component.
4. Re-run the relevant test.
5. Do not rewrite the entire project unless necessary.

Prefer targeted fixes.

---

## 67. Testing

At minimum, test:

### Dataset

- image loading
- corrupted image handling
- label mapping
- split integrity

### Transformations

- JPEG quality changes
- resize output dimensions
- recompression
- transformation order

### Model

- forward pass
- output dimensions
- checkpoint loading

### Evaluation

- metric calculation
- confusion matrix
- prediction labels

### Application

- model loading
- image upload
- preprocessing
- prediction

---

## 68. Code Style

Prefer:

- descriptive variable names
- type hints where useful
- docstrings for reusable functions
- small functions
- modular files
- clear comments

Avoid:

- unexplained magic numbers
- duplicated code
- giant functions
- unnecessary comments
- hardcoded absolute paths

Bad:

path = "C:/Users/Someone/Desktop/project/data/images"

Good:

from pathlib import Path

DATA_DIR = Path("data")

---

## 69. Path Handling

Use pathlib.

Prefer:

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

Avoid machine-specific absolute paths.

The project should work after cloning onto another machine with minimal changes.

---

## 70. Windows Compatibility

The project may be developed on Windows.

Avoid assuming Linux-only commands inside Python code.

Use cross-platform Python path handling.

README commands may provide Windows PowerShell equivalents where useful.

---

## 71. GPU Compatibility

Support:

CUDA
CPU

without requiring source-code modifications.

If CUDA is unavailable, fall back to CPU.

Do not assume a specific GPU model.

---

## 72. README Requirements

The final README should contain:

Project Overview
Research Question
Dataset
Dataset Bias Considerations
Methodology
Models
Transformations
Experiments
Results
How to Run
Streamlit App
Repository Structure
Team Contributions
References

Do not write final numerical results into README until experiments are complete.

---

## 73. Documentation

Whenever a major experiment is completed, update the relevant documentation.

At minimum record:

- experiment name
- model
- dataset split
- transformation
- transformation severity
- training configuration
- metrics
- observations

---

## 74. Experiment Log

Maintain the chronological development record:

docs/DEVELOPMENT_LOG.md

Per-experiment numbers live in the validated results reference:

docs/RESULTS_REFERENCE.md

with run-level detail under `results/experiment_logs/`.

Example entry (adapt field names to the actual completed experiment):

## Experiment: baseline_resnet50_clean

Model:
ResNet-50

Training data:
Clean training set

Test data:
Clean test set

Transformations:
None

Results:
Accuracy: TBD
Precision: TBD
Recall: TBD
F1: TBD
ROC-AUC: TBD
AI Recall: TBD

Observations:
TBD

Update this after real experiments.

---

## 75. Final Deliverables

The project should eventually contain:

### Code

src/
app/
notebooks/

### Models

baseline model
robust model

### Results

metrics
graphs
confusion matrices
error analysis

### Documentation

README
experiment log
methodology
results summary

### Application

Streamlit demo

---

## 76. Agent Workflow

When asked to implement something:

1. Inspect the repository.
2. Read relevant existing files.
3. Read and follow this AGENTS.md.
4. Identify the relevant team role.
5. Understand the existing pipeline.
6. Reuse existing architecture.
7. Make the smallest appropriate change.
8. Run relevant tests or checks.
9. Verify that the change does not introduce methodological problems.
10. Report exactly what was changed.
11. Report any limitations or unexecuted experiments.

Do not silently change the research methodology.

---

## 77. Before Major Architectural Changes

If a requested change would significantly alter:

- dataset methodology
- label definitions
- train/test split
- model architecture
- transformation methodology
- evaluation methodology
- research question

explain the impact before proceeding.

Do not silently make major methodological changes.

---

## 78. Final Principle

The project should optimize for:

Scientific validity
        >
Reproducibility
        >
Clear experimentation
        >
Interpretability
        >
Application quality
        >
Raw accuracy

The most important output is not merely a classifier.

The most important output is a defensible experimental study showing:

how AI-generated image detection behaves under social-media-style image transformations and whether transformation-aware training improves robustness.

Build the project so that another student can clone the repository, understand the methodology, reproduce the experiments, and verify the conclusions.

## Development Log — Mandatory

Maintain `docs/DEVELOPMENT_LOG.md` as the project's chronological development record.

Whenever a meaningful development activity occurs, update the development log as part of the same task.

Meaningful activities include:

- repository structure changes
- dataset decisions
- dataset acquisition or preparation
- dataset audits
- bias analysis
- preprocessing decisions
- model implementation
- training pipeline changes
- transformation implementation
- evaluation implementation
- experiment setup
- completed experiments
- model comparisons
- error analysis
- Streamlit/application development
- important bugs and their fixes
- dependency changes
- important architectural decisions
- changes to the research methodology
- significant Git/project workflow decisions

Do NOT log every trivial code edit.

The development log should explain:

1. What was done
2. Why it was done
3. What files/components were affected
4. Important implementation or methodological decisions
5. Tests or verification performed
6. Problems encountered and how they were solved
7. Current status

Use this format for meaningful development steps:

## Development Step: <Short Title>

### What We Did

<Describe the work actually performed.>

### Why

<Explain the reason for the decision or implementation.>

### Files / Components

<List important files or components affected.>

### Implementation Details

<Important technical details.>

### Verification

<Tests, checks, or validation actually performed.>

### Problems / Solutions

<Problems encountered and how they were solved. If none, say "None.">

### Status

<COMPLETED / IN PROGRESS / BLOCKED / NOT RUN>

Never invent development history, results, metrics, decisions, tests, or problems.

Only document actions that actually occurred.

If a task changes the methodology or architecture, explicitly record the reason.

If an experiment has not actually been executed, record it as NOT RUN or PENDING rather than inventing results.

Keep the log concise and useful for:

- final project report
- PPT preparation
- project demonstration
- viva/presentation questions
- understanding how the project was developed

---

## AI-Assisted Development Documentation

The project uses AI-assisted coding tools such as OpenCode.

The development log should document significant AI-assisted implementation when relevant, but should not record every individual prompt.

Clearly distinguish between:

### Team Decisions

Research methodology, experiment design, dataset decisions, model selection, transformation selection, evaluation strategy, interpretation of results, and other technical decisions made by the team.

### AI-Assisted Implementation

Code scaffolding, implementation assistance, refactoring, debugging assistance, testing assistance, and documentation assistance performed with the help of AI coding tools.

Do not claim that the team manually wrote code that was generated by an AI tool.

The team remains responsible for understanding, reviewing, testing, and approving the resulting implementation.

---

## Development Log Update Rule

At the end of every meaningful implementation task:

1. Determine whether the work is significant enough to document.
2. If yes, update `docs/DEVELOPMENT_LOG.md`.
3. Record only facts supported by the actual work performed.
4. Keep previous development entries intact.
5. Add the newest entry chronologically.
6. Do not rewrite historical entries unless correcting an actual factual error.
7. If the task produced no meaningful development change, do not add a log entry.

Whenever possible, update the development log in the same change/commit as the implementation it describes.