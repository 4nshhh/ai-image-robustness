# Development Log — ai-image-robustness

Chronological development record for the project. Only actions that actually
occurred are documented. No metrics, results, or dataset statistics are
reported unless produced by a real experiment.

Update rule: add an entry for every meaningful development milestone only.
Do NOT log trivial edits.

---

## Development Step: Project Initialization and Research Direction

### What We Did

Initialized the project with the research direction: robust detection of
AI-generated images under social-media-style compression and transformations.

### Why

To fix the research question before any code or data work, so all later
experiments stay in scope.

### Files / Components

- Research question defined in `AGENTS.md` and `README.md`.

### Implementation Details

Binary task: Real (0) vs AI-generated (1). Planned study: JPEG compression,
resizing, recompression, and combined transformations, plus
transformation-aware training.

### Verification

- `AGENTS.md` and `README.md` present in repository.

### Problems / Solutions

None.

### Status

COMPLETED (direction set; no data or code work implied).

---

## Development Step: AGENTS.md and Project Development Rules

### What We Did

Created `AGENTS.md` as the binding methodology and workflow specification
(commit `ef89054`), including experimental validity rules, bias/leakage
controls, repository structure, and team roles. The Development Log rules
section was appended afterward (currently an uncommitted working-tree change).

### Why

The methodology matters more than raw accuracy in this project, so the rules
had to exist before implementation began.

### Files / Components

- `AGENTS.md`

### Implementation Details

Key rules recorded: no fabricated results (`TBD` / `NOT RUN` for pending
work), mandatory bias analysis, test-set isolation, fixed label mapping
(0 = Real, 1 = AI), and the numbered-notebook / `src/` module layout.

### Verification

- `AGENTS.md` exists; committed in `ef89054`; working-tree extension verified
  via `git diff AGENTS.md`.

### Problems / Solutions

None.

### Status

COMPLETED (rules defined; log-section addition uncommitted).

---

## Development Step: Initial Repository Scaffolding

### What We Did

Created the initial project structure (commit `b6c0094 chore: initialize
project structure`): `data/`, `notebooks/`, `src/`, `models/`, `results/`,
`app/`, `docs/`, plus placeholder Python modules, notebooks,
`requirements.txt`, `.gitignore`, expanded `README.md`, and placeholder docs.
No dataset, training, transformation, evaluation, or app logic was included.

### Why

To give all three team members an immediately usable, methodology-compliant
layout before any implementation starts.

### Files / Components

- `data/raw|processed|splits|metadata/` (empty; raw/processed git-ignored)
- `notebooks/01_dataset_analysis.ipynb` through `04_robust_training.ipynb`
  (purpose headers only)
- `src/__init__.py`, `dataset.py`, `preprocessing.py`, `model.py`,
  `train.py`, `transformations.py`, `evaluate.py`, `metrics.py`
  (docstring-only placeholders at time of commit)
- `models/baseline/`, `models/robust/` (`.gitkeep` only)
- `results/metrics|graphs|confusion_matrices|predictions|experiment_logs/`
  (`.gitkeep` only)
- `app/app.py` (placeholder), `docs/methodology.md`,
  `docs/experiment_log.md`, `docs/results_summary.md` (`TBD` / `NOT RUN`)
- `requirements.txt`, `.gitignore`, `README.md`

### Implementation Details

Scaffolding only. Placeholder modules contain no logic. No data downloaded,
no fake data created, no results invented.

### Verification

- Directory tree inspected; all 4 notebooks parse as JSON; all Python
  placeholders parse successfully.

### Problems / Solutions

None.

### Status

COMPLETED.

---

## Development Step: Team-Role Organization

### What We Did

Adopted the primary-ownership roles defined in `AGENTS.md`. No code changes
in this step.

### Why

To keep dataset/model work, experiment/evaluation/app work, and documentation
consistent, with defined handoff checkpoints.

### Files / Components

- Person 1 — Data + Model Lead: `src/dataset.py`, `preprocessing.py`,
  `model.py`, `train.py`.
- Person 2 — Experiments + Evaluation + Application:
  `src/transformations.py`, `evaluate.py`, `metrics.py`, `results/`,
  `app/app.py`.
- Person 3 — Documentation + Presentation: `docs/`, report, PPT, demo script.

### Implementation Details

Roles are ownership, not access boundaries. Integration checkpoints agreed:
(1) dataset/splits/preprocessing, (2) baseline checkpoint selection,
(3) robustness results and robust-training setup.

### Verification

- Roles as documented in `AGENTS.md`; no implementation claimed.

### Problems / Solutions

None.

### Status

COMPLETED (organization defined; member work largely NOT STARTED — see
current status).

---

## Development Step: Current Status of Development

### What We Did

Recorded the honest project status as of this entry. No new implementation
in this step.

### Why

So the final report, PPT, and viva reflect what was actually done, with
unfinished work clearly marked.

### Files / Components

| Component | State |
|---|---|
| Dataset acquisition / audit / bias analysis | NOT STARTED |
| Metadata / splits | NOT STARTED |
| Preprocessing (`src/preprocessing.py`) | NOT STARTED (placeholder) |
| Models (`src/model.py`, `src/train.py`) | NOT STARTED (placeholders) |
| Transformations (`src/transformations.py`) | IN PROGRESS — implementation exists only as an uncommitted working-tree change (in-memory smoke test passed); not committed, not reviewed, not integrated |
| Evaluation / metrics (`src/evaluate.py`, `src/metrics.py`) | NOT STARTED (placeholders) |
| Baseline / robustness experiments | NOT RUN |
| Streamlit app (`app/app.py`) | NOT STARTED (placeholder) |
| Documentation (`docs/`, `README.md`) | Placeholders only; TBD / NOT RUN throughout |

### Implementation Details

Committed history: `dec9e94` (initial setup), `ef89054` (AGENTS.md),
`b6c0094` (scaffolding). Uncommitted working-tree changes: `AGENTS.md`
(log-rules section) and `src/transformations.py`. No experiment has been run;
no metrics exist. All result fields remain `TBD`.

### Verification

- `git log --oneline`, `git status --short`, and working-tree file inspection.

### Problems / Solutions

None.

### Status

IN PROGRESS (scaffolding done; project work ahead).

Historical snapshot: accurate when written; superseded by the entries below.

---

## Development Step: Transformation Layer Implementation (src/transformations.py)

### What We Did

Implemented the transformation layer in `src/transformations.py` (Person 2
scope), committed as `aeeb4b5 feat: add image transformation utilities`.
No other project file was modified.

### Why

Person 2 needed to work independently of the not-yet-started dataset/model
track. Controlled, reusable transformations are the prerequisite for all
robustness experiments.

### Files / Components

- `src/transformations.py` only (evidence: `git show --stat aeeb4b5` —
  1 file changed, +186/−4).

### Implementation Details

Pillow-based; every function accepts a PIL image and returns a new PIL
image; inputs are never modified in place; no class-specific logic
(applies equally to Real and AI images):

- `apply_jpeg_compression(image, quality)` — single JPEG encode/decode via
  `BytesIO`; `quality` validated as int 1–100.
- `apply_resize(image, scale, resample=BILINEAR)` — `round(w*scale)` x
  `round(h*scale)`, aspect ratio preserved; scale must be positive and finite.
- `apply_recompression(image, quality, passes=2)` — JPEG compression applied
  `passes` times in sequence; kept distinct from single compression.
- `apply_pipeline(image, transformations)` — ordered sequence of
  image-to-image callables; order preserved; empty sequence returns a copy.
- `_ensure_saveable_as_jpeg()` helper — converts RGBA/LA/PA/P to RGB so JPEG
  saving is safe; RGB/L/CMYK returned as a copy.
- Reference constants only (not hardcoded experiment config):
  `JPEG_QUALITY_LEVELS = [90, 70, 50, 30]`,
  `RESIZE_SCALES = [0.75, 0.50, 0.25]`.
- Full docstrings and type hints throughout.

### Verification

- In-memory smoke test on a generated 200x100 RGB image plus a 64x48 RGBA
  image, re-run against the committed state: JPEG Q70 output correct size;
  RGBA handled safely (RGB output, input untouched); resize 0.5 gives 100x50
  with aspect ratio preserved; recompression Q70x2 correct; sequential
  pipeline (resize -> JPEG) gives 100x50; original image bytes unchanged in
  every check. Result: SMOKE TEST PASSED.
- `src/transformations.py` function/constant definitions confirmed by
  inspection; all other `src/` modules still placeholders (see status entry).

### Problems / Solutions

None.

### Status

COMPLETED (implemented, tested, committed; not yet integrated with
dataset/evaluation; no experiment results).

---

## Development Step: Current Status of Development (Update)

### What We Did

Re-verified the log against the repository and Git history. No new
implementation in this step.

### Why

Two commits landed after the previous status snapshot, making parts of it
stale. Earlier entries are preserved as written (they were accurate at the
time); this entry is the current snapshot.

### Files / Components

| Component | State |
|---|---|
| AGENTS.md incl. Development Log rules | COMMITTED (`ff8c724`) |
| Repository scaffolding | COMPLETED (`b6c0094`) |
| `docs/DEVELOPMENT_LOG.md` | Created (`ff8c724`), updated by this entry |
| Transformations (`src/transformations.py`) | COMPLETED — committed (`aeeb4b5`), smoke-tested |
| Dataset / audit / bias analysis | NOT STARTED |
| Metadata / splits | NOT STARTED |
| Preprocessing, models, training (`preprocessing.py`, `model.py`, `train.py`) | NOT STARTED (placeholders) |
| Evaluation / metrics (`evaluate.py`, `metrics.py`) | NOT STARTED (placeholders) |
| Baseline / robustness experiments | NOT RUN |
| Streamlit app (`app/app.py`) | NOT STARTED (placeholder) |
| Results / metrics / graphs | None exist; all fields `TBD` |

### Implementation Details

Committed history: `dec9e94` (initial setup), `ef89054` (AGENTS.md),
`b6c0094` (scaffolding), `ff8c724` (log workflow + this log's creation),
`aeeb4b5` (transformation utilities). Working tree clean. No experiment has
been run; no dataset, model, training, or evaluation work exists. Nothing in
this log is invented — every claim traces to the commits and files above.

### Verification

- `git log --oneline`, `git status --short` (clean), `git show --stat`
  for `aeeb4b5` and `ff8c724`, placeholder grep over `src/`, function
  inspection of `src/transformations.py`, and a re-run smoke test.

### Problems / Solutions

None.

### Status

IN PROGRESS (transformation layer done; dataset/model/eval/app work ahead).

Historical snapshot: accurate when written; superseded by the entries below.

---

## Development Step: Evaluation and Metrics Foundation (src/metrics.py, src/evaluate.py)

### What We Did

Implemented the evaluation and metrics foundation (Person 2 scope):
`src/metrics.py` (reusable binary-classification metrics) and
`src/evaluate.py` (generic PyTorch model-on-DataLoader evaluation).
Uncommitted working-tree change; no other project file modified. No
experiments run, no dataset or trained model required or used.

### Why

Person 2 built this independently of the not-yet-started dataset/model
track, so that baseline, robustness, and robust-model evaluations later
share one code path, one label mapping, and one metric set.

### Files / Components

- `src/metrics.py` — accuracy, precision, recall, F1 (AI class, label 1,
  via scikit-learn, `zero_division=0`); ROC-AUC from AI-class
  probabilities (NaN when `y_true` has a single class); fixed-order
  confusion-matrix helper (`[[TN, FP], [FN, TP]]`, labels `[0, 1]`);
  `compute_classification_metrics(y_true, y_pred, y_prob=None)` returning a
  dict with accuracy/precision/recall/F1, ai_recall, ai_false_negative_rate,
  real_recall, false_positive_rate, roc_auc (`None` when `y_prob` omitted —
  never invented), confusion matrix plus tn/fp/fn/tp, and sample/support
  counts. Accepts lists, NumPy arrays, or compatible array-likes.
- `src/evaluate.py` — `evaluate_model(model, dataloader, device,
  threshold=0.5)`: eval mode under `torch.no_grad()` (original
  training state restored), collects true labels, predicted labels, and
  AI-class probabilities, then delegates to `compute_classification_metrics`.
  Output shapes handled: `[B]`/`[B, 1]` single logit via sigmoid,
  `[B, 2]` logits via softmax column 1; anything else raises `ValueError`.
  Never trains, never touches weights/data/transforms, no dataset or
  architecture assumptions, no checkpoint paths.

### Implementation Details

Design decisions: scikit-learn as the metric backend for standard
definitions; fixed label constants `LABEL_REAL = 0`, `LABEL_AI = 1`;
threshold configurable (default 0.5); device defaults to CUDA when
available else CPU; return structure `{"metrics", "y_true", "y_pred",
"y_prob", "threshold", "n_samples"}`.

### Verification

Tests executed with throwaway scripts in the system temp directory
(repo kept clean) using the `ml_clean` conda env
(torch 2.13.0, sklearn 1.7.2), because the base env has a broken
NumPy 2.5.2 / SciPy combination that prevents sklearn import:

- Metrics test on hand-checkable synthetic example (TN=3, FP=1, FN=1,
  TP=3): accuracy/precision/recall/F1 all 0.75 as hand-computed;
  confusion matrix `[[3,1],[1,3]]`; ROC-AUC 0.9375 (matches
  `sklearn.metrics.roc_auc_score` on the same inputs); AI recall 0.75,
  AI FNR 0.25, real recall 0.75, FPR 0.25; `roc_auc is None` when
  `y_prob` omitted; NumPy-array inputs accepted. PASSED.
- Evaluate test on a synthetic 8-sample DataLoader with three dummy
  classifiers (`[B]`, `[B,1]`, `[B,2]` outputs): all three ran,
  returned metrics with ROC-AUC present, collected 8 labels/predictions/
  probabilities in valid ranges; no-grad asserted inside `forward`;
  parameters bit-identical before/after; training mode restored.
  (Values acc=0.750, f1=0.750, auc=0.688 are dummy-model artifacts of
  random data, not project results.) PASSED.

### Problems / Solutions

- Base conda env cannot import sklearn (NumPy 2.5.2 vs SciPy built for
  NumPy 1.x). Solved by running tests under the existing `ml_clean`
  env. No project dependency changes made; `requirements.txt` untouched.

### Status

COMPLETED (implemented, tested; uncommitted; no integration yet — needs
Person 1's dataset/DataLoader, preprocessing, and trained checkpoints;
no experiments run).

---

## Development Step: Current Status of Development (Update 2)

### What We Did

Recorded the current snapshot after the evaluation/metrics work. No new
implementation in this step.

### Why

The previous snapshot predates `src/metrics.py` / `src/evaluate.py` and is
preserved as history; this entry is the current record for the final
report, PPT, demo, and viva.

### Files / Components

| Component | State |
|---|---|
| AGENTS.md incl. Development Log rules | Committed |
| Repository scaffolding | COMPLETED |
| Transformations (`src/transformations.py`) | COMPLETED — committed, smoke-tested |
| Evaluation / metrics (`src/metrics.py`, `src/evaluate.py`) | COMPLETED — implemented and tested, uncommitted |
| Dataset / audit / bias analysis | NOT STARTED |
| Metadata / splits | NOT STARTED |
| Preprocessing, models, training | NOT STARTED (placeholders) |
| Baseline / robustness / robust-training experiments | NOT RUN |
| Streamlit app (`app/app.py`) | NOT STARTED (placeholder) |
| Results / metrics / graphs | None exist; all fields `TBD` |

### Implementation Details

Person 2 track (`transformations.py`, `metrics.py`, `evaluate.py`) is
implemented and tested; Person 1 track (dataset, preprocessing, models,
training) has not started, so no end-to-end evaluation is possible yet.
No experiment has been run. Nothing in this log is invented.

### Verification

- Working-tree file inspection, `git status`, placeholder grep over
  `src/`, and the test runs documented in the entry above.

### Problems / Solutions

None (beyond the base-env NumPy/SciPy issue already recorded above).

### Status

IN PROGRESS (Person 2 foundation done; dataset/model/app/experiments ahead).

Historical snapshot: accurate when written; superseded by the entries below.

---

## Development Step: Experiment Config and Result Infrastructure (src/experiment_config.py, src/results.py)

### What We Did

Implemented the experiment result/configuration foundation on `ansh-branch`
(Person 2 scope): new `src/experiment_config.py` (declarative experiment
descriptions) and new `src/results.py` (storable experiment outcomes).
Uncommitted working-tree change, not yet committed. No actual experiments
run; no dataset loaded; no model trained.

### Why

Later robustness experiments need a shared vocabulary for "which model, on
which split, under which transformation, trained how" and a uniform way to
store outcomes, without depending on Person 1's unfinished dataset/model
track.

### Files / Components

- `src/experiment_config.py` — frozen dataclass `ExperimentConfig` with
  model_name, experiment_name, split (default `"test"`), transformation
  (default `"none"`), transformation_params (default `{}`), condition
  (`"clean"`/`"transformed"`), optional training_condition, seed (default
  42). Validation: non-empty names, condition restricted to the two values,
  params must be a dict, seed must be int, and clean ⟺
  `transformation="none"` with empty params (both directions enforced).
  Plus `is_clean` property and JSON-safe `to_dict()`.
- `src/results.py` — frozen dataclass `ExperimentResult` holding experiment/
  model names, transformation + params, condition/split/training_condition,
  accuracy/precision/recall/F1/AI recall/AI FNR/ROC-AUC, confusion matrix as
  immutable `((TN, FP), (FN, TP))` tuples, and n_samples.
  `from_metrics(config, metrics)` copies values from a
  `compute_classification_metrics` dict without recomputing anything;
  `to_dict()`/`from_dict()` give JSON-safe round-trips; `to_record()`
  flattens to one DataFrame-row/CSV-line dict (cm expanded to
  tn/fp/fn/tp, params as `param_<name>` columns). Missing ROC-AUC stays
  None, never invented.

### Implementation Details

Design decisions: frozen dataclasses (configs/results are values, not
mutable state); result layer organizes but never computes metrics
(no duplication of `src/metrics.py`); no file I/O in either module
(CSV/JSON writing comes later); compatible with existing
`transformations.py` / `metrics.py` / `evaluate.py` by construction
(`from_metrics` consumes their output shape directly).

### Verification

Throwaway in-memory test script (system temp dir, repo kept clean) run
under the `ml_clean` env, all on synthetic data:

1. Clean and transformed `ExperimentConfig` creation — OK.
2. Six invalid configs rejected (empty name, bad condition,
   clean-with-transformation, transformed-with-`none`, non-dict params,
   non-int seed) — OK.
3. `ExperimentResult.from_metrics` on synthetic 6-sample metrics —
   values copied exactly, n_samples preserved — OK.
4. Transformation params `{"quality": 50}` preserved through the result — OK.
5. `to_dict()` passes `json.dumps`; `from_dict` round-trip equals original — OK.
6. `to_record()` exposes tn/fp/fn/tp and `param_quality`; two records form
   a valid pandas DataFrame — OK.
7. Metrics dict without probabilities yields `roc_auc is None` — OK.

Result: ALL EXPERIMENT-INFRA TESTS PASSED. No project metrics fabricated;
all values synthetic or hand-checkable.

### Problems / Solutions

None.

### Status

COMPLETED (implemented, tested; uncommitted on `ansh-branch`; no
experiments run, no results claimed).

---

## Development Step: Current Status of Development (Update 3)

### What We Did

Recorded the current snapshot after the experiment-infrastructure work. No
new implementation in this step.

### Why

The previous snapshot predates `src/experiment_config.py` / `src/results.py`
and is preserved as history.

### Files / Components

| Component | State |
|---|---|
| Branch discipline | All work on `ansh-branch`; `main` untouched |
| Repository scaffolding | COMPLETED |
| Transformations (`src/transformations.py`) | COMPLETED — committed, smoke-tested |
| Evaluation / metrics (`src/metrics.py`, `src/evaluate.py`) | COMPLETED — committed (`f8c6c0d`) |
| Experiment config / results (`src/experiment_config.py`, `src/results.py`) | COMPLETED — implemented and tested, uncommitted |
| Dataset / audit / bias analysis | NOT STARTED |
| Metadata / splits | NOT STARTED |
| Preprocessing, models, training | NOT STARTED (placeholders) |
| Baseline / robustness / robust-training experiments | NOT RUN |
| Streamlit app (`app/app.py`) | NOT STARTED (placeholder) |
| Results / metrics / graphs | None exist; all fields `TBD` |

### Implementation Details

Person 2 foundation (transformations, metrics, evaluation, experiment
config/results) is implemented and tested; Person 1 track has not started,
so no end-to-end experiment is possible yet. Nothing in this log is invented.

### Verification

- `git branch --show-current` (`ansh-branch`), `git status`, working-tree
  file inspection, and the test run documented in the entry above.

### Problems / Solutions

None.

### Status

IN PROGRESS (Person 2 infrastructure done; dataset/model/app/experiments ahead).

Historical snapshot: accurate when written; superseded by the entries below.

---

## Development Step: Experiment Runner Orchestration (src/experiment_runner.py)

### What We Did

Implemented the reusable experiment orchestration layer on `ansh-branch`
(Person 2 scope): new `src/experiment_runner.py` with `run_experiment()`
and the `ExperimentOutcome` container. Uncommitted working-tree change,
not yet committed. No real experiments run; no dataset or checkpoint used.

### Why

The runner connects the already-built infrastructure
(`transformations.py`, `metrics.py`, `evaluate.py`,
`experiment_config.py`, `results.py`) so that one configured call will
execute an evaluation as soon as Person 1 provides a real model and
DataLoader — with no logic duplicated.

### Files / Components

- `src/experiment_runner.py` only. `run_experiment(config, model,
  dataloader, device, threshold, batch_transform, evaluate_fn)` calls the
  existing `evaluate_model()` by default, converts its metrics via the
  existing `ExperimentResult.from_metrics()` (no metric recalculation),
  and returns a frozen `ExperimentOutcome` holding config, result, and
  raw `y_true`/`y_pred`/`y_prob` arrays.

### Implementation Details

Key architectural decisions (transformation boundary): image-level PIL
transformations live in the dataset (Person 1's component, untouched
here), so transformed conditions take an already-transformed DataLoader
from the caller — no dataset interface invented, no in-place mutation.
An optional caller-supplied tensor-level `batch_transform` is applied to
cloned input batches through a generator passed to the existing
`evaluate_model` (no evaluation logic duplicated); an optional
`evaluate_fn` with the same contract allows future transformation-aware
evaluation strategies. No JPEG qualities, scales, passes, or combinations
hardcoded. Threshold validated in (0, 1); config type-checked.

### Verification

Throwaway synthetic test (temp dir, repo kept clean) under `ml_clean` env
on an 8-sample DataLoader with a dummy `[B, 2]` classifier:

1. Plain run executes and returns `ExperimentResult` — OK.
2. Delegation proved via a spy wrapping the real `evaluate_model`
   (called once, identical metrics) — OK.
3. Metadata preserved (names, clean condition, `transformation="none"`),
   metrics preserved, predictions/probabilities accessible — OK.
4. Model parameters bit-identical before/after — OK.
5. Transformed-condition run with tiny in-memory `batch*0.9+0.01`
   transform: params preserved, 3 batches seen, underlying dataset
   tensors unchanged — OK.

Result: ALL RUNNER TESTS PASSED. Reported values (e.g. acc=0.500) are
dummy artifacts of random data, not project results.

### Problems / Solutions

None.

### Status

COMPLETED (implemented, tested; uncommitted on `ansh-branch`; blocked on
Person 1's model/DataLoader for real runs; no experiments run).

---

## Development Step: Current Status of Development (Update 4)

### What We Did

Recorded the current snapshot after the runner work. No new implementation
in this step.

### Why

The previous snapshot predates `src/experiment_runner.py` and is preserved
as history.

### Files / Components

| Component | State |
|---|---|
| Branch discipline | All work on `ansh-branch`; `main` untouched |
| Transformations / metrics / evaluation | COMPLETED — committed, tested |
| Experiment config / results | COMPLETED — committed (`33b951c`) |
| Experiment runner (`src/experiment_runner.py`) | COMPLETED — implemented and tested, uncommitted |
| Dataset / splits / preprocessing / models / training | NOT STARTED (Person 1 placeholders) |
| Baseline / robustness / robust-training experiments | NOT RUN |
| Streamlit app (`app/app.py`) | NOT STARTED (placeholder) |
| Results / metrics / graphs | None exist; all fields `TBD` |

### Implementation Details

Person 2 infrastructure (transformations, metrics, evaluation, config,
results, runner) is complete and tested end-to-end on synthetic data;
real runs await Person 1's dataset/DataLoader, preprocessing, models, and
checkpoints. Nothing in this log is invented.

### Verification

- `git branch --show-current` (`ansh-branch`), `git status`, working-tree
  file inspection, and the test run documented in the entry above.

### Problems / Solutions

None.

### Status

IN PROGRESS (Person 2 infrastructure complete; dataset/model/app/experiments ahead).

Historical snapshot: accurate when written; superseded by the entries below.

---

## Development Step: Experiment Protocol Definition (docs/EXPERIMENT_PROTOCOL.md)

### What We Did

Defined the experiment protocol for the robustness study on `ansh-branch`
(Person 2 scope): new `docs/EXPERIMENT_PROTOCOL.md`, a planning/
specification document for the work that will run once Person 1 delivers
the dataset, DataLoader, model, and baseline checkpoint. Uncommitted
working-tree change. No Python source code modified; no experiments run.

### Why

Person 2's infrastructure (transformations, metrics, evaluation, config,
results, runner) is complete, so the next independent step was to fix
the study design — groups, controls, metrics, and interpretation rules —
before any data or checkpoints exist, preventing ad-hoc experimentation
later.

### Files / Components

- `docs/EXPERIMENT_PROTOCOL.md` only: research question and label mapping;
  prerequisites owned by Person 1; five experiment groups (clean baseline,
  JPEG Q90/70/50/30, resize 0.75/0.50/0.25, recompression defined as
  sequential encode→decode→encode cycles distinct from single
  compression, three justified combined pipelines C1–C3 with order
  recorded); fairness controls (same test set/checkpoint/preprocessing/
  threshold/metrics/code, both-class transforms, no leakage, A/B study
  separation); metric set with AI recall / AI FNR emphasized; robust
  training framed with levels deliberately left TBD; optional
  unseen-levels test marked OPTIONAL; result-storage requirements;
  bias-audit reminder without asserting biases exist; interpretation
  rules (observation vs interpretation vs hypothesis, no invented
  significance thresholds); 17-row experiment matrix, all
  PLANNED/PENDING.

### Implementation Details

Specification only — no code, no values decided beyond the proposed
levels from AGENTS.md, no training probabilities invented, no claim the
optional experiment will run.

### Verification

- Document written and inspected; `git status` confirms no source files
  touched; matrix statuses all PLANNED/PENDING, no COMPLETED entries.

### Problems / Solutions

None.

### Status

COMPLETED (protocol defined; uncommitted on `ansh-branch`; execution
blocked on Person 1 deliverables).

---

## Development Step: Current Status of Development (Update 5)

### What We Did

Recorded the current snapshot after the protocol work. No new
implementation in this step.

### Why

The previous snapshot predates `docs/EXPERIMENT_PROTOCOL.md` and is
preserved as history.

### Files / Components

| Component | State |
|---|---|
| Branch discipline | All work on `ansh-branch`; `main` untouched |
| Person 2 infrastructure (transformations, metrics, evaluation, config, results, runner) | COMPLETED — committed, tested |
| Experiment protocol (`docs/EXPERIMENT_PROTOCOL.md`) | COMPLETED — defined, uncommitted |
| Dataset / splits / preprocessing / models / training (Person 1) | NOT STARTED (placeholders) |
| All protocol experiments (17-row matrix) | PLANNED / PENDING — none run |
| Streamlit app (`app/app.py`) | NOT STARTED (placeholder) |
| Results / metrics / graphs | None exist; all fields `TBD` |

### Implementation Details

Everything Person 2 can do independently is specified and built; every
remaining step needs Person 1's dataset/DataLoader/preprocessing/model/
checkpoints. Nothing in this log is invented.

### Verification

- `git branch --show-current` (`ansh-branch`), `git status`, working-tree
  file inspection.

### Problems / Solutions

None.

### Status

IN PROGRESS (protocol defined; awaiting Person 1 deliverables).

---

## Development Step: Finalized Dataset Extraction and Verification (krish-branch)

### What We Did

Extracted and verified the finalized dataset ZIP on `krish-branch`.
All work stayed on `krish-branch`; `main` and teammates' branches untouched.
No training started; `src/preprocessing.py`, `src/model.py`, and training
code untouched; no images modified, resized, renamed, resampled, or
regenerated. The original ZIP was preserved in `data/downloads/`.

### Why

The finalized 8,000-image split is the Person 1 -> Person 2 handoff basis
(train/val/test must be fixed before any baseline or robustness experiment),
so its on-disk contents had to be verified against the expected counts
before use.

### Files / Components

- Source: `data/downloads/genimage_8000_final (1).zip`
  (1,888,200,231 bytes). NOTE: the file on disk carries the ` (1)` suffix;
  `data/downloads/genimage_8000_final.zip` (exact name from the task) does
  not exist. Extraction used the existing file as-is; it was not renamed,
  moved, or deleted.
- Destination: `data/splits/genimage_8000_split_v4/` with `train/`, `val/`,
  `test/`, `manifest.csv`, `split_metadata.json`.
- `.gitignore` extended (uncommitted): `data/downloads/` plus image
  extensions under `data/splits/**`, so the ZIP and extracted images can
  never be committed; `manifest.csv` / `split_metadata.json` stay trackable.

### Implementation Details

- Extraction via Python `zipfile` (`extractall` into the destination dir).
- Counts on disk: train/ai 2,800; train/real 2,800; val/ai 600;
  val/real 600; test/ai 586; test/real 600. Total images on disk: 7,986
  (expected 8,000; shortfall 14, all in `test/ai`).
- The ZIP itself contains exactly these files (7,986 images + manifest +
  metadata), so nothing was lost during extraction — the shortfall is in
  the source ZIP.
- `manifest.csv` exists with 8,000 rows and the expected declared
  distribution (incl. 600 `test/ai`), so 14 manifest rows reference files
  absent from the ZIP/disk: `test/ai/adm_00025.png`, `adm_00051.png`,
  `biggan_00016.png`, `glide_00016.png`, `glide_00044.png`,
  `midjourney_00031.png`, `midjourney_00060.png`, `sdv5_00019.png`,
  `sdv5_00048.png`, `wukong_00021.png`, `wukong_00029.png`,
  `wukong_00036.png`, `wukong_00041.png`, `wukong_00070.png`. No
  extra files on disk outside the manifest; no zero-byte or sub-1KB files;
  extensions only `.jpeg` (4,000, Real) and `.png` (3,986, AI).
- `split_metadata.json` exists and declares 8,000 total / test ai 600,
  which matches the manifest but NOT the 586 `test/ai` files present.
- Manifest `path` column uses the absolute prefix
  `/kaggle/working/genimage_8000_split_v4/...`; only the
  `split/label/filename` suffix matches this machine (portability note,
  not a data error).
- AI-assisted implementation: extraction/verification scripts were
  AI-generated throwaway files run from the repo root and deleted
  afterward; the repo contains no leftover verification scripts.

### Verification

- `git rev-parse --abbrev-ref HEAD` = `krish-branch` (before and after).
- ZIP entry listing: 7,988 file entries (7,986 images + manifest +
  metadata), per-class counts as above.
- Post-extraction disk counts re-checked per directory; manifest-vs-disk
  join on `split/label/filename` gives 14 missing, 0 extra.
- Image integrity: full PIL `verify()` + `load()` over all 7,986 present
  images — checked 7,986, ok 7,986, corrupted 0 (a 300-image stratified
  sample also passed 300/300 earlier). Full scan completed without timeout
  in the background run.
- Git safety: `git check-ignore` confirms the ZIP is ignored;
  `manifest.csv` / `split_metadata.json` remain visible to Git;
  `git status --short` shows only `M .gitignore` and untracked `data/`
  (images/ZIP hidden by ignore rules). Nothing committed (no commit was
  requested).
- Original ZIP still present in `data/downloads/` after extraction.

### Problems / Solutions

- ZIP filename mismatch (`genimage_8000_final (1).zip` vs expected
  `genimage_8000_final.zip`): proceeded with the existing file without
  renaming/deleting; flagged for the team to confirm canonical naming.
- 14 `test/ai` files missing relative to manifest/metadata: recorded
  exactly; NOT patched by inventing data. Awaiting team decision
  (re-supply the 14 files vs accept 586 and amend manifest/metadata).
- Full PIL scan initially timed out at the 120 s foreground limit;
  re-ran as a background job to completion (7,986/7,986 ok).

### Status

COMPLETED with a FAILED exact-count verification: structure, manifest
(8,000 rows), metadata file, and integrity of present images all pass,
but on-disk total is 7,986 vs expected 8,000 (`test/ai` 586 vs 600).
No training run; no model/preprocessing changes; awaiting next instruction.

---

## Development Step: Finalize 7,986-Image Dataset (Methodological Decision)

### What We Did

Accepted the 7,986 physically present images as the final dataset on
`krish-branch`. Pruned `manifest.csv` to the 7,986 present files only
(removed exactly the 14 absent `test/ai` rows; every other row
byte-identical, verified by multiset comparison against a backup).
Recomputed `split_metadata.json` from the pruned manifest (totals,
class counts, per-generator split counts) and added an `integrity_note`.
No image bytes touched; no preprocessing/model/training/notebook/
transformation code touched. AI-assisted implementation: throwaway scripts
in the system temp dir (repo kept clean); original 8,000-row manifest
backed up to temp before editing.

### Why

Exact recovery of the 14 missing `test/ai` files was investigated and
proven impossible from available sources: absent from the ZIP under
`test/ai/`, no other local archive/extraction/cache copy, Kaggle source
directory not accessible. Train/val same-named files are different images
(14/14 different SHA-256) and were explicitly rejected as substitutes.
The cross-split SHA-256 audit found zero exact-duplicate groups among the
7,986 present images, so the remaining dataset has no content-level
leakage. Leaving a manifest that declares 14 nonexistent files would be a
worse methodological error than formally accepting 7,986.

### Files / Components

- `data/splits/genimage_8000_split_v4/manifest.csv`: 8,000 → 7,986 rows
  (columns unchanged). Removed (all `test/ai`): adm_00025, adm_00051,
  biggan_00016, glide_00016, glide_00044, midjourney_00031,
  midjourney_00060, sdv5_00019, sdv5_00048, wukong_00021, wukong_00029,
  wukong_00036, wukong_00041, wukong_00070 (.png).
- `data/splits/genimage_8000_split_v4/split_metadata.json`: total 7,986
  (Real 4,000 / AI 3,986); test 1,186 (600 Real / 586 AI); per-generator
  test: adm 85, biggan 84, glide 83, midjourney 83, sdv5 84, vqdm 86,
  wukong 81; `split_ratio` kept as intended design; `integrity_note` added.
- `docs/experiment_log.md`: brief dataset-decision entry (test n = 1,186).

### Implementation Details

Final composition: train 5,600 (2,800/2,800), val 1,200 (600/600),
test 1,186 (600 Real / 586 AI). Manifest `path` values keep their
`/kaggle/working/...` prefix (portability note from the earlier entry).

### Verification

- Manifest 7,986 rows; every manifest file exists on disk; zero extra
  image files outside the manifest.
- Split/class counts: train 5,600; val 1,200; test 1,186; Real 4,000;
  AI 3,986; test Real 600; test AI 586.
- Full PIL verify+load over all 7,986 images: 0 corrupted.
- Full SHA-256 over all 7,986 images: 7,986 distinct hashes, 0 duplicates.
- `preprocessing.py`, `model.py`, `train.py`, notebooks, transformation
  code confirmed unmodified via `git status` (only the four intended
  files staged/committed).

### Problems / Solutions

- None in this step; the missing-14 problem was characterized in prior
  entries and resolved here by team decision, not by data fabrication.

### Status

COMPLETED (dataset finalized at 7,986; committed on `krish-branch`;
no experiments run yet).

---

## Development Step: Baseline ML Foundation (Person 1 Track)

### What We Did

Implemented the reusable Person 1 ML foundation on `krish-branch`
(Person 1 scope): `src/preprocessing.py`, `src/dataset.py`,
`src/model.py`, `src/train.py` (all previously docstring-only
placeholders), plus `tests/test_ml_foundation.py` (39-check suite, all
passing) and the `notebooks/02_baseline_training.ipynb` scaffold
(21 cells). No full training run; no experiment results exist.

### Why

Baseline training cannot start until the dataset, preprocessing, models,
and training loop share one tested code path with fixed label mapping
and no test-data contact. This unblocks the ResNet-50 / EfficientNet-B0
clean baselines while keeping Person 2's transformation/evaluation code
untouched.

### Files / Components

- `src/preprocessing.py` — `IMAGE_SIZE = 224` documented standard input
  for both models; fixed ImageNet mean/std (no statistics fitted on
  project data); `load_image` (PIL -> RGB, robust to L/P/RGBA JPEG+PNG);
  deterministic train core (direct resize) with optional mild
  `RandomHorizontalFlip` (`augment=True`, train only); eval pipeline with
  no randomness by construction (`get_transform` raises on val/test
  augmentation). No JPEG/robustness logic here by design.
- `src/dataset.py` — `GenImageDataset` over the finalized 7,986-row
  manifest (source of truth for split/label/generator); local paths
  resolved as `root/split/label/filename` via pathlib (manifest `path`
  column ignored: non-portable `/kaggle/working/...` prefix); explicit
  `LABEL_MAP = {"real": 0, "ai": 1}`; unknown labels raise instead of
  guessing; optional metadata return; `load_manifest` rejects missing
  files. Verified counts: train 5,600 (2,800/2,800), val 1,200
  (600/600), test 1,186 (600 Real / 586 AI).
- `src/model.py` — `build_model("resnet50" | "efficientnet_b0",
  pretrained=...)` with 2-class heads (`fc` / `classifier[1]`,
  `[B, 2]` logits for CrossEntropyLoss); modern `weights=` API with
  `pretrained=` fallback; `save/checkpoint` bundle carries model name,
  label map, and extras; no custom architectures.
- `src/train.py` — one shared loop for both models: `TrainConfig`
  dataclass (optimizer adam/adamw/sgd, lr, weight decay, epochs, seed 42,
  CPU/CUDA auto device), seeded DataLoaders (train/val only — no test
  loader exists in the module), train/val loss+accuracy history,
  best-validation-accuracy checkpointing, CLI flags. Test data cannot
  influence selection: it is never loaded here.
- `tests/test_ml_foundation.py` — stdlib-assert suite (no pytest):
  manifest counts/labels/existence, label-column-over-filename proof,
  RGB handling, tensor shapes, eval determinism, both model forward
  passes, checkpoint round-trip, and a 1-epoch CPU smoke run on 8
  train + 4 val images with checkpoint/history verification.
- `notebooks/02_baseline_training.ipynb` — 10-section scaffold
  (config, seed/device, loading, sanity checks, model, training,
  validation, checkpoint, test-eval placeholder, results placeholder);
  test evaluation explicitly deferred to notebook 03; zero fabricated
  values.

### Implementation Details

Team decisions: 224px direct-resize geometry shared by train/eval for
baseline interpretability; augmentation defaults OFF; ImageNet
normalization (pretrained weights); validation-accuracy checkpoint
selection; seed 42 with documented CUDA nondeterminism limits.
AI-assisted implementation: modules and tests were AI-scaffolded, then
reviewed, bug-fixed (see below), and test-verified by the team member.

### Verification

- `python tests/test_ml_foundation.py`: ALL 39 CHECKS PASSED on CPU
  (torch 2.14.0+cpu, no CUDA on this machine).
- Smoke run used exactly the 8/4 subset (asserted via accuracy
  fractions). Reported smoke values (e.g. loss ~0.68, acc 0.50) are
  random-initialization dummy artifacts, NOT project results.
- Notebook parses as JSON (nbformat 4, 21 cells).

### Problems / Solutions

- First background test run silently trained 1 epoch on the FULL train
  set on CPU: `fit()` built its own loaders and ignored the test's
  subset indices (fractional accuracies 0.5425/0.6183 exposed it — they
  are impossible on 8/4 samples). Fixed by adding `train_indices` /
  `val_indices` to `TrainConfig`, threading them into `fit()`, and
  adding the subset-fraction assertion. No dataset files were modified,
  no test data was involved, and all artifacts went to temp dirs.
  Resolved; suite re-run green.

### Status

COMPLETED (foundation implemented and tested; committed on
`krish-branch`; full 5,600-image training NOT run; Person 2 files
untouched).

---

## Development Step: GPU Readiness Review (Clean-Baseline Gate)

### What We Did

Reviewed the Person 1 foundation for training correctness (no rewrites:
code was already implemented and 39-check green), probed this machine
for CUDA, gated the test suite on GPU for all model/training tests,
fixed one real `.gitignore` safety gap, and prepared (but did NOT run)
the two clean-baseline configurations. All work on `krish-branch`.

### Why

Full baseline training is expensive and must only start on a correct,
leak-free pipeline with a proven GPU path. This step is the go/no-go
gate before any 5,600-image run.

### Files / Components

- Reviewed (read-only, no changes): `src/preprocessing.py`,
  `src/dataset.py`, `src/model.py`, `src/train.py`,
  `tests/test_ml_foundation.py`, `notebooks/02_baseline_training.ipynb`,
  `requirements.txt`, `AGENTS.md`. Findings: manifest is the sole label
  source (test file proves label-column-over-filename); `train.py`
  mentions "test" only in comments forbidding its use and has no test
  loader; heads replaced correctly (`fc` / `classifier[1]`, `[B, 2]`);
  checkpoint bundles carry architecture + label map; augmentation
  defaults OFF; eval deterministic.
- `tests/test_ml_foundation.py` (modified, was uncommitted): added
  `require_cuda()` — prints torch build, CUDA availability, GPU name,
  selected device, and STOPS with an explicit error instead of CPU
  fallback. `main()` runs dataset/preprocessing checks on CPU, then
  gates model/smoke tests on CUDA. Model tensors, checkpoint
  `map_location`, and smoke `TrainConfig(device="cuda")` are explicit.
- `.gitignore` (modified): fixed a real gap — `models/*.pth` does not
  match `models/baseline/*.pth`, so baseline checkpoints were NOT
  ignored. Now `models/**/*.pth` (+ `*.pt`), verified ignorable at any
  depth; metrics CSVs and docs stay trackable.

### Implementation Details

Baseline configs prepared (config-only, no code changes needed —
`python src/train.py --help` verified working):

- ResNet-50: `python src/train.py --model resnet50 --epochs 10
  --batch-size 32 --lr 1e-4 --weight-decay 1e-4 --optimizer adam
  --seed 42 --checkpoint-dir models/baseline` (ImageNet pretrained,
  clean train data, augmentation off, val-selected best checkpoint).
- EfficientNet-B0: identical command with `--model efficientnet_b0`
  (same split, same preprocessing, same protocol — fair comparison).

### Verification

- Environment: Python 3.11.9, torch 2.14.0+cpu, `cuda_available=False`
  (CPU-only build — no CUDA device, no GPU name/memory to report).
- GPU probe (temp dir, read-only): printed versions and STOPPED with
  exit 10 before any model code; nothing ran on CPU as a substitute.
- Test suite re-run: 25/25 CPU-allowed checks passed
  (dataset 15 + preprocessing 10), then the gate STOPPED at
  `require_cuda()` exactly as the standing GPU rule demands.
- CLI `--help` works for both models. `git check-ignore` confirms
  checkpoints/images/ZIP ignored, metrics/docs trackable. No `.pth`
  files exist in the repo.

### Problems / Solutions

- `.gitignore` checkpoint gap (above): fixed and verified. No
  implementation bugs found in `src/`; no metrics invented; smoke-test
  loss/accuracy values remain dummy artifacts, not results.

### Status

COMPLETED (review passed; GPU path enforced in tests but NO CUDA on
this machine, so ResNet-50/EfficientNet-B0 forward passes and GPU smoke
training are NOT RUN — pending a CUDA machine; full baseline training
NOT RUN).

---

## Development Step: RTX 3050 CUDA Validation + VRAM Assessment

### What We Did

Ran the previously CUDA-blocked ML foundation validation on the local
NVIDIA GeForce RTX 3050 Laptop GPU (torch 2.12.0+cu126, CUDA 12.6,
4.29 GB total VRAM, 3.46 GB free at probe time) on `krish-branch`.
Fixed one test-only device bug exposed by the GPU run, measured
per-batch-size VRAM for both models, and prepared (did NOT run) full
baseline commands. No dataset/manifest changes; no Person-2 code
touched; no checkpoints committed.

### Why

The prior gate entry left forward passes and smoke training as NOT RUN
for lack of CUDA. This machine unblocks that validation, and its 4 GB
VRAM limit needed an empirical batch-size verdict before any 10-epoch
baseline is launched.

### Files / Components

- `tests/test_ml_foundation.py` — one real fix: the checkpoint
  round-trip compared CUDA (`before`) vs CPU (reloaded) tensors, which
  passed on CPU-only runs but raised `Expected all tensors to be on the
  same device` on CUDA. Fix: `.to(device)` after reload plus a new
  `reloaded model params live on CUDA` assertion (suite now 40 checks).
- No `src/` changes needed; review of the committed pipeline still
  holds (manifest-only labels, no test loader, `[B, 2]` heads,
  val-selected checkpoints).

### Implementation Details

- Suite: `python tests/test_ml_foundation.py` → ALL 40 CHECKS PASSED,
  device `cuda` throughout the model/smoke sections; dataset and
  preprocessing sections ran on CPU as the rule allows. Zero tests ran
  model code on CPU. Smoke subset exactly as specified: 8 train
  (4 Real + 4 AI) + 4 val (2 + 2), 1 epoch, `device="cuda"`,
  checkpoint save/reload + history verified (values are random-init
  dummy artifacts, not results).
- Controlled VRAM probe (temp dir, single fwd+bwd+Adam step, fp32,
  random-init = same footprint as pretrained): ResNet-50 bs=32 →
  peak reserved 3.35 GB; bs=16 → 1.89 GB; bs=8 → 1.07 GB.
  EfficientNet-B0 bs=32 → 3.36 GB; bs=16 → 1.66 GB; bs=8 → 0.86 GB.
- Verdict: batch 32 leaves ~0.1 GB headroom on this 4 GB card —
  UNSAFE (display use, pin_memory, fragmentation can OOM it).
  Recommendation (not applied): run both full baselines at
  `--batch-size 16` to keep the comparison fair with margin.

### Verification

- `torch.cuda.is_available()` True; GPU name/VRAM printed by the
  suite gate and probe; `git status` clean except intended files;
  staged set will contain no images/ZIP/checkpoints (verified at
  commit); full 10-epoch training NOT run.

### Problems / Solutions

- CUDA-tensor comparison bug (above): fixed, suite green.
- `torch.cuda.reset_peak_memory_stats(0)` rejects the int ordinal on
  this build (`Invalid device argument`); passing a `torch.device`
  works. Temp-probe-only quirk, no repo impact.

### Status

COMPLETED (foundation CUDA-validated 40/40 on RTX 3050; batch-16
recommended for full baselines; full training NOT RUN).

---

## Development Step: Clean Baseline Test Evaluation (Held-Out Test)

### What We Did

Evaluated both best clean-baseline checkpoints on the finalized
held-out test split (1,186 images: 600 Real / 586 AI) on the RTX 3050
(CUDA, eval batch 32, threshold fixed at 0.5, deterministic eval
preprocessing). Saved per-image predictions, metrics JSON/CSV, and
confusion-matrix CSV/PNG under `results/`. No retraining, no tuning,
no test-based selection; training histories (untracked until now)
committed as the record behind the validation numbers. No checkpoints
committed (gitignored per AGENTS.md); no dataset/Person-2 changes.

### Why

The baselines were trained (reported best val: ResNet-50 0.8983 and
EfficientNet-B0 0.8950, both epoch 8 — confirmed from the history
JSONs, whose configs also confirm pretrained=true, batch 16,
augment off, seed 42, full train rows). The held-out test score is the
result that all later robustness comparisons build on.

### Files / Components

- Checkpoints used (read-only, NOT committed):
  `models/baseline/resnet50_best.pth` (epoch 8) and
  `models/baseline/efficientnet_b0_best.pth` (epoch 8). Verified:
  correct architecture names, `num_classes=2`, label map
  Real=0/AI=1, pretrained config, clean CUDA load, `[B, 2]` CUDA probe.
- `results/predictions/baseline_{resnet50,efficientnet_b0}_clean.csv`
  (1,186 rows each: split/label/generator/filename/true/pred/ai_prob).
- `results/metrics/baseline_{...}_clean.json` +
  `baseline_clean_test_summary.csv`.
- `results/confusion_matrices/baseline_{...}_clean.{csv,png}`.
- `models/baseline/{resnet50,efficientnet_b0}_history.json` (training
  record; weights stay untracked).

### Implementation Details

REAL measured test results (threshold 0.5, n = 1,186; NOT validation
numbers, NOT to be used for model selection):

- ResNet-50: acc 0.9081, prec 0.9409, rec 0.8686, F1 0.9033,
  ROC-AUC 0.9682; TN 568, FP 32, FN 77, TP 509; AI recall 0.8686,
  AI FNR 0.1314, Real recall 0.9467, FPR 0.0533.
- EfficientNet-B0: acc 0.9073, prec 0.9034, rec 0.9096, F1 0.9065,
  ROC-AUC 0.9649; TN 543, FP 57, FN 53, TP 533; AI recall 0.9096,
  AI FNR 0.0904, Real recall 0.9050, FPR 0.0950.

Observation (not a selection): both models land within 0.001 accuracy;
ResNet-50 leans precise (fewer Real false alarms, more missed AI),
EfficientNet-B0 leans sensitive (fewer missed AI, more false alarms).
Primary-model choice for notebook 03 is a team decision, still open.

### Verification

- Test counts asserted in-script (1,186; {0:600, 1:586}); CM cells
  sum to 1,186 and reconcile with class counts for both models.
- Eval script ran from system temp dir; repo holds only results/docs.
- `evaluate_model` (Person 2) consumed unchanged with default
  threshold; model code and weights untouched.

### Problems / Solutions

None in this step.

### Status

COMPLETED (clean test baselines measured and committed on
`krish-branch`; robustness experiments NOT run).

---

## Development Step: EfficientNet-B0 Baseline Verification on GPU (Person 2)

### What We Did

Independently re-verified Krish's clean EfficientNet-B0 baseline on
`ansh-branch` (Person 2 scope): new `scripts/verify_baseline.py`, executed
on the NVIDIA GeForce RTX 4050 Laptop GPU (torch 2.13.0+cu126) against the
local finalized dataset and the committed-untracked
`models/baseline/efficientnet_b0_best.pth` checkpoint. No retraining, no
threshold change (0.5), no test-set modification, no transformations.

### Why

The robustness study (protocol Sections 3.2–3.5) measures degradation
relative to this baseline, so Person 2 confirmed the reference numbers
reproduce on an independent machine/GPU before running any transformed
condition.

### Files / Components

- `scripts/verify_baseline.py` (new, uncommitted): enforces CUDA at
  startup (`RuntimeError`, no CPU fallback), prints torch/CUDA/GPU
  identity, loads `GenImageDataset(split="test",
  transform=get_transform("test"))` pinned to the finalized manifest,
  asserts test composition 1,186 (600 Real / 586 AI), loads the
  checkpoint via existing `src.model.load_checkpoint` (bundle verified:
  efficientnet_b0, 2 classes, Real=0/AI=1), asserts all 213 parameter
  tensors live on CUDA, evaluates via existing `src.evaluate.evaluate_model`
  (batch 32, num_workers=0, no_grad inside), compares six metrics against
  reference within 1e-4 (exit 1 on mismatch), and fails if `git status`
  shows tracked/staged images or checkpoints.
- No `src/` module modified.

### Implementation Details

One script bug fixed before the green run: first version asserted a
`manifest_path` attribute that `GenImageDataset` does not expose
(`AttributeError`); replaced with explicit manifest/root pinning plus a
per-row `test/` directory check and `class_counts()` (avoids loading
images twice).

### Verification

Actual run output: CUDA True, RTX 4050, test 1,186 (Real 600 / AI 586),
213/213 tensors on CUDA, evaluated 1,186, CM [[543, 57], [53, 533]],
accuracy 0.9073, precision 0.9034, recall/AI recall 0.9096, F1 0.9065,
ROC-AUC 0.9649, AI FNR 0.0904 — every reference metric matched within
1e-4 (4-decimal prints identical). `git status` showed only
`?? scripts/`; no dataset images or `.pth` files tracked/staged.

### Problems / Solutions

- Script attribute bug (above): fixed, re-run green. No data, model, or
  environment issues.

### Status

COMPLETED (baseline independently VERIFIED on GPU; uncommitted;
robustness experiments still NOT run — cleared to start with JPEG sweep
after primary-model confirmation, already given: EfficientNet-B0).

---

## Development Step: Experiment 1 — JPEG Compression Robustness (EfficientNet-B0)

### What We Did

Ran the first protocol experiment on `ansh-branch` (Person 2 scope):
same 1,186-image test set + same EfficientNet-B0 checkpoint as the
verified baseline, under clean / JPEG Q90 / Q70 / Q50 / Q30, on the
RTX 4050 (CUDA, batch 32, threshold 0.5, no retraining). New
`scripts/run_jpeg_robustness.py` (uncommitted); no `src/` module
modified; baseline files untouched.

### Why

Protocol group 3.2: measure detection degradation under strengthening
JPEG compression, the most common social-media transformation.

### Files / Components

- `scripts/run_jpeg_robustness.py` — per-condition `JpegTestDataset`
  wrapper applying existing `apply_jpeg_compression` to the PIL image
  of every row (both classes, originals only read) before the unchanged
  clean preprocessing; existing `evaluate_model` + `ExperimentConfig` +
  `ExperimentResult.from_metrics`; asserts n=1,186, 600/586 classes,
  prediction/manifest order agreement, clean-within-1e-4, weight and
  file immutability; fails on git-tracked images/checkpoints.
- Outputs (new, uncommitted):
  `results/metrics/jpeg_robustness_efficientnet_b0.csv` (5 rows),
  `results/predictions/jpeg_robustness_efficientnet_b0.csv` (5,930 rows:
  filename/split/generator/true/pred/ai_prob/condition/quality),
  `results/confusion_matrices/jpeg_robustness_efficientnet_b0.csv`,
  5 per-condition JSON logs in `results/experiment_logs/`.

### Implementation Details

ACTUAL MEASURED RESULTS (n=1,186 each; AI F1 = F1 of class 1):

| Condition | Acc | Prec | AI Rec | AI F1 | AUC | AI FNR |
|---|---|---|---|---|---|---|
| clean | 0.9073 | 0.9034 | 0.9096 | 0.9065 | 0.9649 | 0.0904 |
| jpeg_q90 | 0.8727 | 0.8877 | 0.8498 | 0.8684 | 0.9432 | 0.1502 |
| jpeg_q70 | 0.8272 | 0.8848 | 0.7474 | 0.8104 | 0.9066 | 0.2526 |
| jpeg_q50 | 0.7934 | 0.8764 | 0.6775 | 0.7642 | 0.8755 | 0.3225 |
| jpeg_q30 | 0.7428 | 0.8808 | 0.5546 | 0.6806 | 0.8442 | 0.4454 |

Observation (not causal claim): monotonic degradation with stronger
compression, concentrated in AI recall (0.9096 → 0.5546; FNR 0.0904 →
0.4454) while precision stays ~0.88 — the detector misses more AI images
rather than false-alarming more Real ones.

### Verification

- Every condition n=1,186 with 600 Real / 586 AI (asserted in-run and
  re-checked in the predictions CSV: 10/10 condition×class groups exact).
- Clean reproduces verified baseline within 1e-4 (all six metrics).
- Model parameters bit-identical before/after; test files size+mtime
  identical; `y_true` matches manifest labels in order (no drift).
- `git status`: only new result files + `scripts/`; baseline results,
  checkpoints, dataset untouched; nothing committed.

### Problems / Solutions

None in the run.

### Status

COMPLETED (Experiment 1 measured and saved; uncommitted; resize /
recompression / combined / robust-training NOT started per task scope).

---

## Development Step: Experiment 2 — Resize Robustness (EfficientNet-B0)

### What We Did

Ran protocol group 3.3 on `ansh-branch` (Person 2 scope): same 1,186-image
test set + same EfficientNet-B0 checkpoint as the verified baseline, under
clean / resize 0.75 / 0.50 / 0.25, on the RTX 4050 (CUDA, batch 32,
threshold 0.5, no retraining). New `scripts/run_resize_robustness.py`
(uncommitted); existing `apply_resize` reused with default BILINEAR for
all conditions; no `src/` modified; baseline and JPEG files untouched.

### Why

Measure detection degradation under platform-style downscaling, applied as
true information loss (PIL downscale before the unchanged 224x224 model
preprocessing), not as a preprocessing variant.

### Files / Components

- `scripts/run_resize_robustness.py` — per-condition wrapper applying
  `apply_resize(image, scale)` to every row (both classes, originals only
  read) before clean preprocessing; existing `evaluate_model` +
  `ExperimentConfig(transformation="resize", params={"scale"})` +
  `ExperimentResult.from_metrics`; asserts n=1,186, 600/586, order
  agreement, y_true identical across conditions, clean-within-1e-4, weight
  and file immutability.
- Outputs (new, uncommitted):
  `results/metrics/resize_robustness_efficientnet_b0.csv` (4 rows),
  `results/predictions/resize_robustness_efficientnet_b0.csv` (4,744 rows),
  `results/confusion_matrices/resize_robustness_efficientnet_b0.csv`,
  4 per-condition JSON logs in `results/experiment_logs/`.

### Implementation Details

ACTUAL MEASURED RESULTS (n=1,186 each):

| Condition | Acc | Prec | AI Rec | AI F1 | AUC | AI FNR | CM [[TN,FP],[FN,TP]] |
|---|---|---|---|---|---|---|---|
| clean | 0.9073 | 0.9034 | 0.9096 | 0.9065 | 0.9649 | 0.0904 | [[543,57],[53,533]] |
| resize_075 | 0.8685 | 0.8125 | 0.9539 | 0.8776 | 0.9560 | 0.0461 | [[471,129],[27,559]] |
| resize_050 | 0.7707 | 0.6938 | 0.9590 | 0.8052 | 0.9273 | 0.0410 | [[352,248],[24,562]] |
| resize_025 | 0.5371 | 0.5180 | 0.9061 | 0.6592 | 0.6680 | 0.0939 | [[106,494],[55,531]] |

Observation (not a causal claim): resize degrades accuracy through the
OPPOSITE error pattern from JPEG — precision collapses (Real FP 57 →
129 → 248 → 494) while AI recall stays high (~0.91–0.96); at 0.25 the
model predicts nearly everything as AI (AUC 0.668). JPEG destroyed AI
recall; downscaling destroys Real precision. The asymmetry is flagged
for error analysis, not explained here.

### Verification

- Every condition n=1,186, 600/586 (asserted in-run; all 8
  condition×class groups re-verified in predictions CSV).
- y_true identical across all four conditions (asserted).
- Clean reproduces baseline within 1e-4; params bit-identical; test
  files size+mtime identical; all-CUDA run.
- `git status`: only new resize files + script; JPEG/baseline/src
  untouched; nothing committed.

### Problems / Solutions

None in the run. The inverted error pattern vs JPEG was unexpected but
is a measured observation, reported without causal claims.

### Status

COMPLETED (Experiment 2 measured and saved; uncommitted; recompression /
combined / robust-training NOT started per task scope).

---

## Development Step: Experiment 3 — Recompression Robustness (EfficientNet-B0)

### What We Did

Ran protocol group 3.4 on `ansh-branch` (Person 2 scope): same 1,186-image
test set + same EfficientNet-B0 checkpoint, under clean / recompression
Q90/Q70/Q50/Q30 with passes=2 (exactly what existing
`apply_recompression` implements: two sequential same-quality JPEG
encode→decode→encode cycles), on the RTX 4050 (CUDA, batch 32, threshold
0.5, no retraining). New `scripts/run_recompression_robustness.py`
(uncommitted); no duplicate recompression logic; no `src/` modified;
baseline, JPEG, and resize files untouched.

### Why

Uploads are often compressed more than once across platforms; the protocol
requires recompression measured as its own operation, distinct from single
compression.

### Files / Components

- `scripts/run_recompression_robustness.py` — per-condition wrapper applying
  `apply_recompression(image, quality, passes=2)` to every row (both
  classes, originals only read) before clean preprocessing; existing
  `evaluate_model` + `ExperimentConfig(transformation="recompression",
  params={"quality","passes"})` + `ExperimentResult.from_metrics`; asserts
  n=1,186, 600/586, order agreement, y_true identical, clean-within-1e-4,
  weight and file immutability.
- Outputs (new, uncommitted):
  `results/metrics/recompression_robustness_efficientnet_b0.csv` (5 rows),
  `results/predictions/recompression_robustness_efficientnet_b0.csv`
  (5,930 rows with quality+passes columns),
  `results/confusion_matrices/recompression_robustness_efficientnet_b0.csv`,
  5 per-condition JSON logs in `results/experiment_logs/`.

### Implementation Details

ACTUAL MEASURED RESULTS (n=1,186 each):

| Condition | Acc | Prec | AI Rec | AI F1 | AUC | AI FNR | CM [[TN,FP],[FN,TP]] |
|---|---|---|---|---|---|---|---|
| clean | 0.9073 | 0.9034 | 0.9096 | 0.9065 | 0.9649 | 0.0904 | [[543,57],[53,533]] |
| recomp_q90_p2 | 0.8727 | 0.8891 | 0.8481 | 0.8681 | 0.9428 | 0.1519 | [[538,62],[89,497]] |
| recomp_q70_p2 | 0.8272 | 0.8880 | 0.7440 | 0.8097 | 0.9062 | 0.2560 | [[545,55],[150,436]] |
| recomp_q50_p2 | 0.7917 | 0.8775 | 0.6724 | 0.7614 | 0.8750 | 0.3276 | [[545,55],[192,394]] |
| recomp_q30_p2 | 0.7428 | 0.8828 | 0.5529 | 0.6800 | 0.8439 | 0.4471 | [[557,43],[262,324]] |

Observation (not a causal claim): recompression tracks single JPEG almost
exactly (Q90/Q70/Q30 accuracies identical to Experiment 1 to 4 decimals;
Q50 within 0.002) with the same AI-recall-collapse pattern and stable
precision — the second pass adds essentially no further damage at these
levels. Like JPEG and unlike resize, Real precision is preserved while AI
images increasingly evade detection.

### Verification

- Every condition n=1,186, 600/586 (asserted in-run; all 10
  condition×class groups re-verified in predictions CSV).
- y_true identical across conditions; clean within 1e-4; params
  bit-identical; test files size+mtime identical; all-CUDA run.
- `git status`: only new recompression files + script + log edit;
  baseline/JPEG/resize/src untouched; nothing committed.

### Problems / Solutions

None in the run.

### Status

COMPLETED (Experiment 3 measured and saved; uncommitted; combined /
robust-training NOT started per task scope).

---

## Development Step: Experiment 4 — Combined Transformation Robustness (EfficientNet-B0)

### What We Did

Ran protocol group 3.5 on `ansh-branch` (Person 2 scope): same 1,186-image
test set + same EfficientNet-B0 checkpoint, under clean, C1 =
Resize 0.50 → JPEG Q50, C2 = JPEG Q50 → Resize 0.50, C3 = Resize 0.50 →
JPEG Q50 → JPEG Q50, on the RTX 4050 (CUDA, batch 32, threshold 0.5, no
retraining). New `scripts/run_combined_robustness.py` (uncommitted);
pipelines expressed with existing `apply_pipeline` (order preserved; C3
holds two sequential JPEG passes); no `src/` modified; baseline, JPEG,
resize, recompression files untouched.

### Why

Protocol order-effect study: C1 vs C2 share parameters with opposite order;
C3 bridges the single-transformation results (resize 0.50, JPEG Q50,
recompression Q50 P2).

### Files / Components

- `scripts/run_combined_robustness.py` — per-condition wrapper applying the
  ordered pipeline to every row (both classes, originals only read) before
  clean preprocessing; research transforms strictly precede the single
  final 224x224 preprocessing; existing `evaluate_model` +
  `ExperimentConfig(transformation="combined", params={pipeline, scale,
  quality, passes})` + `ExperimentResult.from_metrics`; asserts n=1,186,
  600/586, order agreement, y_true identical, clean-within-1e-4, weight
  and file immutability.
- Outputs (new, uncommitted):
  `results/metrics/combined_robustness_efficientnet_b0.csv` (4 rows),
  `results/predictions/combined_robustness_efficientnet_b0.csv` (4,744 rows
  with pipeline/scale/quality/passes columns),
  `results/confusion_matrices/combined_robustness_efficientnet_b0.csv`,
  4 per-condition JSON logs in `results/experiment_logs/`.

### Implementation Details

ACTUAL MEASURED RESULTS (n=1,186 each):

| Condition | Acc | Prec | AI Rec | AI F1 | AUC | AI FNR | CM [[TN,FP],[FN,TP]] |
|---|---|---|---|---|---|---|---|
| clean | 0.9073 | 0.9034 | 0.9096 | 0.9065 | 0.9649 | 0.0904 | [[543,57],[53,533]] |
| C1 resize→jpeg | 0.7336 | 0.8516 | 0.5580 | 0.6742 | 0.8638 | 0.4420 | [[543,57],[259,327]] |
| C2 jpeg→resize | 0.7715 | 0.7191 | 0.8823 | 0.7923 | 0.8784 | 0.1177 | [[398,202],[69,517]] |
| C3 resize→jpeg→jpeg | 0.7310 | 0.8560 | 0.5478 | 0.6681 | 0.8648 | 0.4522 | [[546,54],[265,321]] |

Observation (not a causal claim): transformation ORDER matters
measurably — C1 and C2 share parameters yet fail oppositely: C1 shows the
JPEG pattern (AI recall collapse, Real side untouched: TN/FP identical to
clean at 543/57), C2 shows the resize pattern (Real precision collapse,
FP 202, AI recall preserved at 0.8823). The LAST transformation in the
pipeline dominates the error mode. C3 ≈ C1 (extra JPEG pass adds ~nothing),
consistent with Experiment 3's finding.

### Verification

- Every condition n=1,186, 600/586 (asserted; all 8 groups re-verified).
- y_true identical; clean within 1e-4; params bit-identical; files
  size+mtime identical; all-CUDA; each condition rebuilt from original
  source images (fresh PIL read per row per condition).
- `git status`: only new combined files + script + log edit; all prior
  outputs/src untouched; nothing committed.

### Problems / Solutions

None in the run.

### Status

COMPLETED (Experiment 4 measured and saved; uncommitted; robust-training /
error-analysis / Streamlit NOT started per task scope).

---

## Development Step: Experiment 5 — Robust Training Implementation (scripts/train_robust.py)

### What We Did

Implemented (NOT run) the transformation-aware fine-tuning entry point on
`ansh-branch` (Person 2 scope): new `scripts/train_robust.py` following the
approved Phase A decisions and the Section 6 protocol rules. No training
loop executed; no test data accessed; no `src/` modified.

### Why

Robust training needs a dedicated path: existing `fit()` always builds a
fresh model (no checkpoint init) and only offers flip augmentation, so the
approved policy (baseline init + stochastic robustness transforms) cannot
run through it without damage. The new script reuses `set_seed`,
`build_optimizer`, `train_one_epoch`, `validate`, `GenImageDataset`,
`load_checkpoint`, `get_transform`, and all four transformation functions.

### Files / Components

- `scripts/train_robust.py` (new, uncommitted): fixed policy constants
  (clean 40 / JPEG 20 Q{90,70,50,30} / resize 15 {0.75,0.50} / recomp 15
  Q{90,70} p2 / combined 10 {C1,C2}; unseen reserved: 0.25, recomp
  Q50/Q30, C3); `sample_augmentation(rng)` + `RobustTrainTransform`
  (fresh PIL copy, per-worker `Random` streams seeded from (42, wid),
  base = clean train preprocessing, augment OFF); entry asserts train
  5,600 (2800/2800) and val 1,200 (600/600) from the manifest, loads
  `efficientnet_b0_best.pth` (arch + Linear(...,2) head verified, all
  params trainable), enforces CUDA (`RuntimeError`, no fallback), runs
  10 epochs (adam, lr/weight-decay 1e-4, batch 16, num_workers 0),
  selects best clean-val accuracy (tie → lower val loss, then earlier),
  writes `models/robust/efficientnet_b0_robust_best.pth` + history +
  config/policy JSONs, and refuses to overwrite existing outputs without
  `--overwrite`. No test split is ever constructed.

### Implementation Details

Decisions (all from the approved spec, none improvised): fine-tune all
params from the baseline checkpoint; policy probabilities sum to 1.0
(asserted in code); combined C1/C2 use scale 0.50 + Q50 in exact order;
val preprocessing deterministic clean; gitignored `.pth`, trackable JSONs.

### Verification (permitted checks only; temp scripts, repo kept clean)

- `py_compile` + import: OK.
- 40k-draw sampler test (seed 42): clean 0.4029, JPEG 0.1961, resize
  0.1504, recomp 0.1505, C1 0.0496/C2 0.0505; all levels uniform — OK.
- Same-seed identical sequences; wrapper preserves source size, outputs
  [3,224,224] — OK.
- C1/C2 pixel-identical to manual ordered application; `clean` maps to
  None; sampler contains no label branching — OK.
- No `split="test"` / test-manifest pattern in the entry point — OK.
- Checkpoint compatibility: bundle efficientnet_b0, Linear head
  out_features 2, all params trainable — OK.
- Training loop NOT executed; test images NOT accessed; baseline
  files/results untouched.

### Problems / Solutions

- Shell lacked `conda`/`python` on PATH in this session; used the full
  `anaconda3\Scripts\conda.exe` path. No repo impact. (Pillow
  `getdata` deprecation warnings appeared in the temp test only.)

### Status

COMPLETED (implementation + checks; uncommitted; training NOT started;
test evaluation NOT run).

Launch later with: `conda run -n ml_clean python scripts/train_robust.py`

---

## Development Step: Experiment 6 — Baseline vs Robust Evaluation (EfficientNet-B0)

### What We Did

Implemented and ran the head-to-head evaluation on `ansh-branch`
(Person 2 scope): new `scripts/run_robust_evaluation.py` evaluating the
frozen baseline checkpoint AND the epoch-8 robust checkpoint
(`models/robust/efficientnet_b0_robust_best.pth`, trained 10 epochs per
the committed history: best clean-val 0.9008) on the same 1,186-image
test set under all 15 protocol conditions (clean + JPEG x4 + resize x3 +
recompression x4 + combined x3). Eval only; no retraining, no tuning.

### Why

The primary research question: does transformation-aware training improve
robustness while preserving clean performance? Both models share test
images, order, labels, preprocessing, threshold 0.5, code, and metrics.

### Files / Components

- `scripts/run_robust_evaluation.py` (new, uncommitted): per-model loop
  over 15 conditions via existing `evaluate_model` (batch 32, CUDA,
  no_grad) + `ExperimentConfig`/`ExperimentResult.from_metrics`; asserts
  n=1,186, 600/586, cross-model y_true identity, finite [0,1] probs,
  baseline-clean-within-1e-4, param and file immutability.
- Outputs (new, uncommitted): `results/metrics/robust_evaluation_… .csv`
  (30 rows), `results/predictions/…` (35,580 rows with example_id/model/
  ai_prob), `results/confusion_matrices/…` (30 rows),
  `results/metrics/baseline_vs_robust_….csv` (15 rows with pp deltas),
  30 per-model-condition JSON logs. Baseline experiment files untouched.

### Implementation Details

ACTUAL RESULTS — robust AI-recall gain (pp) vs baseline, per condition:
clean +1.02; JPEG Q90 +6.14, Q70 +15.02, Q50 +21.33, Q30 +32.08;
resize 0.75 −0.68, 0.50 −0.85, 0.25 +4.27; recomp Q90 +6.66, Q70 +15.02,
Q50 +21.67, Q30 +32.25; C1 +35.67, C2 +4.95, C3 +36.69.
Robust clean cost: accuracy −1.77pp (0.9073→0.8895), precision −3.82pp,
AUC −0.58pp; robust clean AI recall +1.02pp (FNR 0.0904→0.0802).
Robust worst condition: resize_025 acc 0.6492 (baseline 0.5371) —
improved but still the weakest absolute score; unseen-by-training
conditions (0.25, recomp Q50/Q30, C3) all improved substantially.

Observation (not causal proof): transformation-aware training converts
the JPEG-family AI-recall collapse into a largely preserved recall at a
small clean-accuracy cost; resize_025 remains hard for both models. The
Real-JPEG / AI-PNG format confound (Section 9) still tempers any claim
that compression alone caused the baseline degradation.

### Verification

- 15 conditions × 2 models = 30 rows; every pair n=1,186, 600/586.
- y_true identical across models and conditions; probs finite in [0,1].
- Baseline clean reproduces reference within 1e-4; params bit-identical
  for both models; source images/checkpoints unmodified (size+mtime).
- Two script bugs fixed before the green save (comparison/print dicts
  used long metric names instead of the `ai_f1`/`ai_fnr`/`fpr` row keys;
  eval outputs themselves were unaffected). One redundant re-run timed
  out on machine slowness; on-disk artifacts are from the fully validated
  run (identical deterministic code path).
- `git status`: only new Experiment 6 files + script + log edit; all
  prior results/src/checkpoints untouched; nothing committed.

### Problems / Solutions

- Key-name mismatches (above): fixed, outputs verified row-counted
  (30/15/35,580/30 JSONs). No data/model issues.

### Status

COMPLETED (evaluation measured and saved; uncommitted; error-analysis /
Streamlit NOT started per task scope).

---

## Development Step: Experiment 7 — Visualization and Analysis Summary

### What We Did

Added and ran `scripts/plot_robustness_results.py` on `ansh-branch`
(Person 2 scope): validates the saved Experiment 6 CSVs, renders 7
report figures into `results/graphs/`, and generates
`results/metrics/robustness_analysis_summary.md` computed from the CSVs.
No inference, no training, no test access, no `src/` changes.

### Why

Decision-ready figures and a report summary derived reproducibly from the
validated numbers, with all deltas computed (percentage points) rather
than hardcoded.

### Files / Components

- `scripts/plot_robustness_results.py` (new, uncommitted): schema/row/
  finiteness/n=1186/CM-sum/delta-consistency validation (exit 1 on any
  failure); grouped bars (AI recall, accuracy, AI FNR across 15
  conditions); AI recall vs JPEG quality and vs resize scale with clean
  references; 2×4 confusion-matrix heatmaps (clean, JPEG Q30, resize
  0.25, C3) per model; all at 150 dpi with labels/legends.
- `results/graphs/`: 7 PNGs (ai_recall/accuracy/ai_fnr_all_conditions,
  ai_recall_vs_jpeg_quality, ai_recall_vs_resize_scale,
  confusion_matrices_key_conditions_{baseline,robust}).
- `results/metrics/robustness_analysis_summary.md`: clean trade-off
  (−1.77pp acc, +1.02pp AI recall), top-5 AI-recall gains (C3 +36.69 …
  recomp Q50 +21.67), regressions (resize_050 −0.85, resize_075 −0.68),
  unseen-condition results, and limits incl. the Real-JPEG/AI-PNG
  format confound.

### Implementation Details

Key findings (computed, not claimed): robust training helps everywhere
except negligible resize_075/050 AI-recall dips; unseen conditions all
improve; resize_025 stays weakest absolute for both models.

### Verification

- Script output: source validation OK; all 8 paths exist and non-empty
  (44–61 KB PNGs, 1.6 KB md). Command:
  `conda run -n ml_clean python scripts/plot_robustness_results.py`.
- Two script bugs fixed before the green run (comparison-column filter
  matching `delta_pp_*`; DataFrame-vs-Series `.items()` in regressions
  loop; tight_layout warning on colorbar figures). Source CSVs untouched.

### Problems / Solutions

- None in data; script bugs above fixed and re-run green.

### Status

COMPLETED (figures + summary generated; uncommitted).

---

## Development Step: Experiment 8 — Dataset Format and Metadata Bias Audit

### What We Did

Added and ran `scripts/audit_dataset_bias.py` on `ansh-branch`
(Person 2 scope): read-only probe of all 7,986 manifest images for
extension, decoded format, geometry, color mode, and JPEG quantization
presence, per split × class, plus a labeled DIAGNOSTIC metadata
classifier (logistic regression on 7 trivial features; fit on train,
C selected on val, test reported once without tuning). No data, split,
model, or result modified.

### Why

The Real-JPEG / AI-PNG confound was asserted but never quantified; every
robustness claim needs measured shortcut evidence behind it.

### Files / Components

- `scripts/audit_dataset_bias.py` (new, uncommitted).
- `results/metrics/dataset_bias_audit.csv` (6 split×class rows).
- `results/experiment_logs/dataset_bias_audit.json` (full tables, seed 42).

### Implementation Details

ACTUAL FINDINGS (descriptive only, not causal):

- Format separation is PERFECT in all splits: 100% Real = `.jpeg`/JPEG
  (4,000, all carry quantization tables), 100% AI = `.png`/PNG (3,986,
  zero quantization tables). Extension or decoded format alone is a
  perfect label predictor.
- Mode separators: RGBA occurs only in AI (~14% of AI: 400/2800 train);
  grayscale L only in Real (~1.4%). Geometry: AI images are always
  square (W==H, 128–1024px generator canvases); Real are variable
  rectangles (e.g. train mean ~470×405, range 63–3872px).
- Diagnostic classifier: val accuracy 1.0000 (all C), held-out test
  1.0000 — trivial metadata separates the classes perfectly.
- Zero unreadable files (7,986/7,986 probed).

What this does NOT show: that any model uses these shortcuts, or that
compression effects are reducible to format. It bounds interpretation:
clean accuracy and JPEG-family degradations are entangled with a perfect
format signal, and the report must say so.

### Verification

- Command: `conda run -n ml_clean python scripts/audit_dataset_bias.py`
  (EXIT True). Manifest 7,986 rows; outputs exist and non-empty.
- `git status`: only the script + 2 outputs (+ this log edit); frozen
  split, checkpoints, prior results untouched; nothing committed.

### Problems / Solutions

- None in data. Geometry/format gaps are findings, not errors.

### Status

COMPLETED (audit measured and saved; uncommitted).

---

## Development Step: Experiment 9 — Format-Normalization Diagnostic

### What We Did

Added and ran `scripts/run_format_diagnostic.py` on `ansh-branch`
(Person 2 scope): frozen baseline + robust EfficientNet-B0 checkpoints on
the same 1,186-image test set under clean, lossless-PNG re-encode, and
JPEG Q95 with 4:4:4 subsampling (script-local BytesIO helpers; decoded-RGB
copies re-encoded BEFORE unchanged 224x224 preprocessing). Eval only,
threshold 0.5, CUDA, no tuning, no modifications to data or checkpoints.

### Why

Experiment 8 proved perfect Real-JPEG / AI-PNG separation in metadata; this
diagnostic measures checkpoint response when inputs are normalized to a
common container at (near-)lossless fidelity.

### Files / Components

- `scripts/run_format_diagnostic.py` (new, uncommitted).
- `results/metrics/format_diagnostic_efficientnet_b0.csv` (6 rows),
  `results/metrics/baseline_vs_robust_format_efficientnet_b0.csv` (3 rows),
  `results/predictions/…` (7,116 rows), `results/confusion_matrices/…`
  (6 rows), 6 per-model-condition JSON logs.

### Implementation Details

ACTUAL RESULTS (n=1,186 each; descriptive only):

| Model | Condition | Acc | AI Rec | AI FNR | AUC | CM |
|---|---|---|---|---|---|---|
| baseline | clean | 0.9073 | 0.9096 | 0.0904 | 0.9649 | [[543,57],[53,533]] |
| baseline | png | 0.9073 | 0.9096 | 0.0904 | 0.9649 | identical |
| baseline | jpeg_q95 | 0.8988 | 0.8857 | 0.1143 | 0.9601 | [[547,53],[67,519]] |
| robust | clean | 0.8895 | 0.9198 | 0.0802 | 0.9591 | [[516,84],[47,539]] |
| robust | png | 0.8895 | 0.9198 | 0.0802 | 0.9591 | identical |
| robust | jpeg_q95 | 0.8862 | 0.9096 | 0.0904 | 0.9560 | [[518,82],[53,533]] |

Key measured finding: lossless PNG re-encoding changes NOTHING
(bit-identical predictions — expected, since the pipeline already decodes
to RGB; the container alone carries no signal to these classifiers).
JPEG Q95 costs little (baseline AI recall −2.39pp, robust −0.90pp).
This does NOT identify shortcut use and does NOT normalize away bias:
heavy-compression degradations (Exp 1/3) reflect lossy pixel damage, not
container identity — a hypothesis consistent with, but not proven by,
this diagnostic.

### Verification

- Command: `conda run -n ml_clean python scripts/run_format_diagnostic.py`
  (EXIT True). 6 rows / 3 comparison rows / 7,116 predictions / 6 logs;
  n=1,186, 600/586, y_true identical, finite probs, baseline clean within
  1e-4, params bit-identical, files untouched, all-CUDA.
- `git status`: only new Experiment 9 files + script (+ this log edit);
  prior outputs/src/checkpoints untouched; nothing committed.

### Problems / Solutions

- None.

### Status

COMPLETED (diagnostic measured and saved; uncommitted).

---

## Development Step: Experiment 10 — Streamlit Research Demo

### What We Did

Implemented `app/app.py` on `ansh-branch` (Person 2 scope): Streamlit demo
loading both frozen EfficientNet-B0 checkpoints (cached per session),
upload + RGB decode, optional JPEG/resize input demos via existing
helpers, side-by-side predicted class + raw AI score, research disclaimer.
Also updated the README Streamlit section with run instructions. No
checkpoints, data, or results modified.

### Why

Demonstration layer for the finished baseline-vs-robust comparison, per
AGENTS.md application requirements.

### Files / Components

- `app/app.py` (rewrote scaffold): `load_checkpoint` + `get_transform`
  reuse (same RGB/224/ImageNet path, label map asserted, `[1,2]` logits
  asserted); `st.cache_resource` session-once loading with actionable
  missing-checkpoint errors; JPEG Q90–Q30 / resize 0.75–0.25 demos with
  input-only warning; eval-mode no-grad inference on CUDA-if-available;
  invalid-file and inference-failure handling; disclaimer (predictions ≠
  proof, format bias known, unseen sources unestablished).
- `README.md`: Streamlit section only (run command + checkpoint note).

### Implementation Details

Design: predicted class (threshold 0.5) displayed separately from the raw
AI score, explicitly labeled NOT calibrated confidence.

### Verification

- Headless checks (temp script, repo kept clean; Streamlit 1.54.0):
  import OK; both checkpoints load on CUDA in eval mode; RGB preprocess
  yields (3,224,224); baseline pred=1 score 0.9327 / robust pred=1 score
  0.8278 on a synthetic image (smoke values, not results); resize+jpeg
  demo path OK; garbage upload raises UnidentifiedImageError (caught).
  ALL APP CHECKS PASSED.
- `streamlit run app/app.py --server.headless true` boots; health
  endpoint returns ok. Full browser interaction NOT tested (headless
  environment) — stated limitation.
- `git status` scope: app + README (+ this log edit); checkpoints/data/
  results untouched; nothing committed.

### Problems / Solutions

- None in code. Bare-mode ScriptRunContext warning in headless test is
  expected Streamlit behavior, not a defect.

### Status

COMPLETED (demo implemented and smoke-tested; uncommitted).

---

## Development Step: Dashboard Refactor — AI Image Robustness Lab (replaces prediction demo)

### What We Did

Permanently replaced the Experiment 10 image-prediction demo with a
read-only research dashboard on `ansh-branch` (Person 2 scope): rewrote
`app/app.py` as AI Image Robustness Lab (Overview, Transformation
Experiments, Baseline vs Robust, Dataset Bias & Format, Methodology &
Limitations tabs); updated the README Streamlit section (purpose, run,
artifacts, deployment, limits). No other files touched.

### Why

Team decision: the public artifact communicates completed results; it must
not classify arbitrary uploads or imply reliable authenticity detection.

### Files / Components

- `app/app.py` (rewritten): zero torch/checkpoint/upload/inference code
  (verified by search — sole `.pth` mention is a prose path); cached
  read-only loads of the 30-row eval CSV, comparison CSVs (incl. format),
  bias audit CSV/JSON, summary md, split metadata, manifest generators,
  robust-training config/history; schema + delta-consistency validation
  with visitor-friendly errors; grouped bars, sweeps, pp-delta tables,
  side-by-side comparison with direction-aware notes, per-model CM tables,
  bias/format/methodology sections with observation/interpretation/unproven
  separation.
- `README.md`: Streamlit section rewritten for the dashboard.
- `requirements.txt`: unchanged (torch retained for research scripts;
  streamlit/matplotlib/pandas already listed — no env rewrite).

### Implementation Details

Preflight found: app entry `app/app.py` (prediction demo); all result
artifacts present (only prior uncommitted changes were README/app/dev-log
from Experiment 10 — preserved and built upon); no deployment config
exists (Streamlit Community Cloud assumed; entry `app/app.py`); total
predictions CSVs ~7 MB stay out of the dashboard's load path (metrics +
small JSONs only); checkpoints/raw images never needed by the app.

### Verification

- Headless `load_all()`: 30/15/6/6 rows, no torch imported — OK.
- Full `AppTest.from_file(...).run(timeout=120)`: 5 tabs, 4 metric cards,
  5 dataframes, zero exceptions, zero error elements — OK.
- `streamlit run --server.headless` boots; health endpoint ok. Browser
  interaction and live deployment NOT tested — stated limitations.
- `git status` scope: app + README (+ this log edit); all research
  artifacts, checkpoints, splits untouched; nothing committed.

### Problems / Solutions

- None in dashboard code (written with the corrected validation patterns
  from the start). AppTest default 3 s timeout was environmental — reran
  with timeout=120 and passed. No source-data inconsistencies found.

### Status

COMPLETED (dashboard implemented and tested headless; uncommitted; not
deployed).

---

## Development Step: Dashboard Redesign — Publication-Quality Research Interface

### What We Did

Redesigned `app/app.py` on `ansh-branch` (Person 2 scope) from the crowded
tab prototype into a spacious scientific dashboard; added
`.streamlit/config.toml` (light theme, teal accent). No research code,
results, or protocols touched.

### Why

Faculty/reviewer-facing polish: readable charts, compact filters, no
repository internals in the UI, research terminology throughout.

### Files / Components

- `app/app.py` (rewritten presentation layer; data loading/validation
  logic preserved): sidebar radio navigation (Overview, Transformation
  Robustness, Model Comparison, Dataset Bias & Format, Methodology &
  Limitations); human-readable labels everywhere (`AI Detection Recall`,
  `25% Resize`, `Resize → JPEG → JPEG`, model names — verified by search,
  raw identifiers remain code-only); single-select filters (metric /
  family / model / condition) with immediate useful defaults; large
  matplotlib charts (line sweeps for ordered JPEG/resize/recompression
  parameters, grouped or single bars otherwise, horizontal bars on the
  comparison page — never stacked); percentages in UI with pp deltas;
  direction-aware metric notes; CM tables labeled Actual/Predicted;
  bias page split into observed/interpretation/unproven; methodology in
  expandable sections.
- `.streamlit/config.toml` (new): light theme + accent color.
- `README.md`, `requirements.txt`: unchanged (dashboard description still
  accurate; no new dependencies).

### Implementation Details

Decisions: sidebar over tabs (reliable, obvious active state, small-screen
friendly); matplotlib over native charts (label/legend/size control);
line charts only for genuinely ordered numeric axes; combined-family and
all-condition views use grouped bars; single-model views show one series
without empty comparisons.

### Verification

- `git diff --check`: clean. Scope: only `app/app.py` + new theme file.
- Headless `load_all()`: 30/15/6/6 rows, no torch — OK (unchanged logic).
- `AppTest`: all 5 sidebar sections render, zero exceptions/errors;
  Transformation filters (metric/family/model) and comparison condition
  filter update without errors — OK. (One harness quirk: AppTest widget
  state with `format_func` needs a fresh session per interaction; app
  code unaffected.)
- `use_container_width` deprecation fixed (`width="stretch"`).
- Headless server boot, health ok. Browser interaction and live
  deployment NOT tested — stated limitations.
- No torch/checkpoint/upload/inference code in the app (search-verified);
  saved artifacts byte-identical (only app + theme + this log edit
  modified); nothing committed.

### Problems / Solutions

- None in app code. Multi-line `python -c` breaks `conda run` on this
  machine — temp-file scripts used for harness tests instead.

### Status

COMPLETED (redesign implemented and validated headless; uncommitted; not
deployed).

---

## Development Step: Dashboard Polish — Navigation Icons and Chart Readability

### What We Did

Focused polish of `app/app.py` on `ansh-branch` (Person 2 scope):
native Material icons on all five sidebar items (radio options starting
with `:material/home|tune|balance|database|menu_book:`, extracted natively
by Streamlit's button-group widget) and matching icons on all five page
headings via a shared `section_heading()` helper; categorical charts
converted to horizontal grouped bars with dynamic height
(`bar_height(n)`), short chart labels, and readable y-axis rows; line
sweeps kept for ordered JPEG/resize/recompression axes. Added
`.streamlit/config.toml` in the prior step (already present). No layout,
theme, filters, terminology, data logic, or results changed.

### Why

Icons give each section a scannable visual anchor; horizontal bars fix the
overlapping x-axis labels that made the all-conditions chart unreadable.

### Files / Components

- `app/app.py` only (+ this log edit): `SECTIONS`/`NAV_OPTIONS`/
  `NAV_TO_KEY` tables, `CHART_LABEL` short forms (full descriptions stay
  in tables/captions), horizontal `grouped_bars` + `bar_height`,
  single-model horizontal bars, comparison page short labels.
- `.streamlit/config.toml`: unchanged from prior step.

### Implementation Details

Icon names verified against the installed Streamlit Material-icon set
before use (all five present; nearby alternatives like
`sliders_horizontal`/`scan_search` absent and avoided). Line charts touch
only genuinely ordered numeric axes; combined/all-condition views use
grouped (never stacked) bars; single-model views plot one series.
Sweeps keep evaluation order (JPEG 90→30, scale 0.75→0.25).

### Verification

- `git diff --check`: clean. Scope: `app/app.py` + log edit only.
- AppTest: all 5 icon-labeled sections render, zero exceptions/errors;
  7 filter/chart paths (all-conditions, JPEG/Resize/Recompression sweeps,
  single-model, combined, clean, comparison condition) update cleanly.
- Headless server boot, health ok. Browser icon rendering and live
  deployment NOT tested — stated limitations (shortcodes are validated
  names; unsupported names would show literal text, so names were
  pre-verified).
- Research artifacts byte-identical; no torch/training/inference code;
  nothing committed.

### Problems / Solutions

- Three batch heading edits duplicated `def` lines (syntax errors); all
  found by `ast.parse` and repaired, 5 page functions confirmed.
- Follow-up fix (same session): robust clean-accuracy KPIs on Overview and
  Model Comparison formatted the raw rate without ×100 (`0.89%` instead of
  `88.95%`); baseline cards were correct. Fixed both to `* 100:.2f}%`,
  audited all other displays (tables use `:.2%`, deltas use pp — all
  correct, none double-scaled), and verified rendered KPIs read
  `90.73%` / `88.95%` via AppTest.
- AppTest `selectbox.set_value` needs internal option values (not indices)
  with `format_func` widgets — harness quirk, app unaffected.

### Status

COMPLETED (polish implemented and validated headless; uncommitted; not
deployed).

---

## Development Step: Documentation Overhaul — README, Report/PPT Sources, Results Reference

### What We Did

Documentation-only work on `main` (ansh-branch fully merged via PR #3, tree
clean, so documentation proceeded on the merged history): rewrote
`README.md` (15 sections: RQ, findings table, dashboard, structure, setup,
dataset, models, transforms, results links, reproducibility, limitations,
references, team roles); created `docs/FINAL_REPORT_SOURCE.md` (15-section
academic report source with observation/interpretation/hypothesis labels),
`docs/PRESENTATION_SOURCE.md` (11-slide plan with messages, visuals,
speaker notes, timings), `docs/RESULTS_REFERENCE.md` (condition/model/
source table for every metric + reconciliation note), `docs/REFERENCES.md`
(3 project-cited works, verification flags). No code, data, checkpoints,
results, or configs touched.

### Why

Repository needed reviewer-ready docs and an evidence-locked foundation for
the final report and slides.

### Files / Components

- `README.md` (rewritten), `docs/FINAL_REPORT_SOURCE.md`,
  `docs/PRESENTATION_SOURCE.md`, `docs/RESULTS_REFERENCE.md`,
  `docs/REFERENCES.md` (new).

### Implementation Details

All metrics re-verified against artifacts during writing (both baseline
JSONs, full 15-row comparison CSV, robust training config/history,
bias audit JSON, split metadata; robust clean precision/F1 confirmed as
0.8652/0.8916 from the eval CSV, not derived). Deltas stated strictly as
percentage points. Related work beyond the 3 cited papers explicitly
marked open; Li et al. author/venue flagged for verification.

### Verification

- Every relative link checked against the repo (all resolve; no absolute
  local paths in new docs).
- `git diff --check` clean; `git status` shows only README + 4 new docs;
  no application, research, data, or checkpoint file modified.
- Stale scaffolds `docs/methodology.md`, `docs/results_summary.md`, and the
  baseline entry of `docs/experiment_log.md` (still TBD/NOT RUN) left
  untouched; RESULTS_REFERENCE explicitly supersedes them for numbers.

### Problems / Solutions

- None. Missing material honestly flagged: dashboard screenshot,
  related-work survey, significance testing.

### Status

COMPLETED (docs written and validated; uncommitted).

---

## Development Step: Documentation Consolidation for Final Submission

### What We Did

Documentation-only consolidation on `main`: updated
`docs/EXPERIMENT_PROTOCOL.md` to the completed-project state; deleted
`docs/methodology.md`, `docs/results_summary.md`, `docs/experiment_log.md`
after preserving their unique content; repointed links in README,
FINAL_REPORT_SOURCE, RESULTS_REFERENCE, and notebook 02. No code, data,
checkpoints, results, or configs touched.

### Why

`docs/` held stale TBD/NOT RUN scaffolds competing with the authoritative
protocol/results reference; final submission needs one consistent story.

### Files / Components

- `docs/EXPERIMENT_PROTOCOL.md`: preamble + §6/§7/§9.2/matrix now record
  executed robust training (policy, epoch 8 / val 0.9008, `models/robust/`
  checkpoint), designed unseen-level coverage, completed file-size-gap audit
  note, and a corrected "15 conditions" count (an earlier draft said 16).
- Deleted: `methodology.md` (scaffold RQ+pipeline, duplicated elsewhere,
  zero references), `results_summary.md` (stale TBD table, superseded, zero
  references), `experiment_log.md` (its unique 7,986-dataset decision
  survives in `split_metadata.json` integrity_note, this log, and
  RESULTS_REFERENCE; its baseline entry was stale NOT RUN).
- Links repointed to authoritative homes (RESULTS_REFERENCE,
  split_metadata.json); notebook 02 cell edit verified as valid JSON.
- `AGENTS.md` structure listing + §74-style log-rule reference left
  untouched (binding doc, out of scope) — noted, not fixed.

### Implementation Details

Decisions: delete rather than archive (history preserved in Git);
no new document created; pp-vs-percent discipline kept; format confound
and no-generalization statements unchanged.

### Verification

- `git status` shows only doc changes + 3 deletions; `git diff --check`
  clean; repo-wide search confirms no live references to deleted files
  outside AGENTS.md history and this log's own historical notes.
- Relative links re-checked (all resolve); notebook JSON parses.

### Problems / Solutions

- None. Open: AGENTS.md docs-listing now slightly stale (accepted, reported).

### Status

COMPLETED (consolidation done; uncommitted).
