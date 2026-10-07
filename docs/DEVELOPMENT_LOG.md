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
