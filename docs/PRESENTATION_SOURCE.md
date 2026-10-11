# Final Presentation Blueprint --- AI Image Robustness Lab

**11 content slides · approximately 16 minutes**

This document contains the slide text, visual instructions, and a
speaking script for the final presentation. The reported metrics were
checked against `docs/RESULTS_REFERENCE.md` and
`results/metrics/baseline_vs_robust_efficientnet_b0.csv`. Rates are
percentages; model differences are reported in **percentage points
(pp)**. AI recall is the percentage of AI-generated images correctly
detected (class 1 = AI, class 0 = Real).

All referenced graphs are in `results/graphs/`. The architecture diagram
needs to be created, and the optional dashboard screenshot needs to be
captured from the running app.

------------------------------------------------------------------------

## Slide 1 --- Title and main result (1 min)

**Main point:** Social-media-style transformations can make AI-generated
images harder to detect. Training with examples of those transformations
helped the model recover much of its lost recall.

**On-slide text** - **Robust Detection of AI-Generated Images Under
Social Media-Style Compression and Transformations** - Krish --- data,
models, and training · Ansh --- experiments, evaluation, and dashboard ·
Person 3 --- documentation and presentation - **Main result:** Under
JPEG quality 30, AI recall rose from 55.46% to 87.54% after
transformation-aware training. Clean accuracy decreased by 1.77
percentage points.

**Visual:** No visual is needed on the title slide. An Overview
screenshot from the running dashboard is optional; capture it from the
actual app rather than recreating it.

**Speaker notes:** "We wanted to find out what happens when an AI-image
detector meets the kinds of changes images go through on social
platforms. We tested compression, resizing, and combinations of both.
Then we trained the model with a planned mix of transformed images and
checked whether its results improved."

**Transition:** "Let's start with the problem we wanted to measure."

------------------------------------------------------------------------

## Slide 2 --- The problem and research question (1.5 min)

**Main point:** A detector can perform well on clean images but struggle
after images are compressed or resized.

**On-slide text** - **The problem: clean images versus real upload
conditions** - Social platforms may recompress or downscale uploaded
images. - These changes can affect how reliably a detector identifies
AI-generated images. - **Research question:** How do JPEG compression,
resizing, repeated compression, and combined transformations affect
detection? Can training with transformed images make the detector more
reliable?

**Visual:** No separate visual is necessary; introduce the problem
verbally.

**Speaker notes:** "Our clean test result was around 91% AI recall. But
with JPEG quality set to 30, recall fell to 55.46%, meaning the model
missed many more AI-generated images. We wanted to measure that drop
carefully and find out how much could be recovered through training."

**Transition:** "We turned that question into three specific
objectives."

------------------------------------------------------------------------

## Slide 3 --- Objectives and approach (1 min)

**Main point:** We compared the same two models on the same test images
across 15 conditions.

**On-slide text** 1. Measure how detection changes across 15 fixed test
conditions. 2. Compare the baseline model with a model trained to handle
common image transformations. 3. Check whether the improvements also
appear in conditions excluded from robust training. - **Fair
comparison:** same test images, a fixed 0.5 decision threshold, and the
same evaluation metrics.

**Visual:** A simple three-step strip is enough: **Measure baseline →
Train for transformations → Evaluate both models**.

**Speaker notes:** "We kept the test set fixed so the baseline and
robust model could be compared directly. The test set contains 1,186
images, and we did not use it to train the model or select a checkpoint.
We evaluated both models using the same threshold and metrics."

**Transition:** "Here is the dataset behind those comparisons."

------------------------------------------------------------------------

## Slide 4 --- Dataset and evaluation design (1.5 min)

**Main point:** The project uses a fixed Tiny-GenImage subset with
separate training, validation, and test splits.

**On-slide text** - **Tiny-GenImage subset: 7,986 images** - Train:
5,600 images --- 2,800 Real and 2,800 AI - Validation: 1,200 images ---
600 Real and 600 AI - Test: 1,186 images --- 600 Real and 586 AI -
Images cover seven AI generators: ADM, BigGAN, GLIDE, Midjourney, SDv5,
VQDM, and Wukong. - Fourteen source files could not be recovered and
were excluded without replacement. - A SHA-256 check found no duplicate
images across the splits. - The test split stayed unchanged throughout
the experiments.

**Visual:** Create a compact split-count table using
`data/splits/genimage_8000_split_v4/split_metadata.json`. Keep the
counts easy to read.

**Speaker notes:** "The test split has 600 Real images and 586 AI
images, so we use those actual counts rather than assuming both classes
have 600. We fixed the split before running the experiments and kept it
separate from training and model selection."

**Transition:** "Next, this is how each image moves through the system."

------------------------------------------------------------------------

## Slide 5 --- System architecture and experiment setup (1.5 min)

**Main point:** Every result comes from the same evaluation pipeline;
only the model and image condition change.

**On-slide text** - **Pipeline:** Fixed data split → RGB image → resize
to 224 × 224 → ImageNet normalization → EfficientNet-B0 → threshold of
0.5 → evaluation metrics. - Metrics: accuracy, precision, AI recall, F1,
ROC-AUC, AI false-negative rate, and confusion-matrix counts. - We
selected EfficientNet-B0 as the main model because it had higher AI
recall than ResNet-50: **90.96% versus 86.86%** on the clean test set. -
We tested 15 conditions: clean images, JPEG quality 90--30, resize
scales 0.75/0.50/0.25, two-pass recompression, and three ordered
combinations. - The robust model was fine-tuned from the baseline for 10
epochs; the best checkpoint was selected using clean validation
performance at epoch 8.

**Visual:** Create a pipeline diagram with boxes and arrows from the
description above. Do not add results or steps that are not in the
project.

**Speaker notes:** "Both models use the same image preprocessing and
evaluation steps. ResNet-50 had slightly higher clean accuracy---90.81%
compared with 90.73% for EfficientNet-B0---but it missed more AI images.
Since detecting AI images is the main task here, we chose
EfficientNet-B0 for the robustness experiments."

**Transition:** "Before changing the training, we first measured where
the baseline struggled."

------------------------------------------------------------------------

## Slide 6 --- What happened to the baseline? (2 min)

**Main point:** Compression mainly caused missed AI images, while
aggressive resizing produced many false alarms.

**On-slide text** - **Clean test:** 90.73% accuracy and 90.96% AI
recall. - **JPEG quality 30:** AI recall dropped to 55.46%. - **Resize
to 25% of the original dimensions:** accuracy fell to 53.71%, while AI
recall remained at 90.61%. - At that resize level, false positives rose
from 57 to 494: many Real images were classified as AI. - Recompression
results were close to those from a single JPEG pass; the combined
transformations showed that the order of operations also mattered.

**Visual:** Main graph: `results/graphs/ai_recall_vs_jpeg_quality.png`.
Point out the baseline curve at JPEG Q30. Supporting graph:
`results/graphs/ai_recall_vs_resize_scale.png`.

**Speaker notes:** "The two transformations caused different problems.
JPEG compression made the model miss more AI images. Strong resizing led
it to label many Real images as AI. That is why accuracy alone does not
tell the whole story---we also need to look at recall and the confusion
matrix."

**Transition:** "We then changed the training process to include a
planned mix of transformations."

------------------------------------------------------------------------

## Slide 7 --- Training the model for transformed images (1.5 min)

**Main point:** The robust model trained on a fixed mix of clean and
transformed images, with several conditions kept out for a separate
test.

**On-slide text** - Each training image was assigned one of these
options, with both classes treated the same way: - 40% clean - 20% JPEG
compression at quality 90, 70, 50, or 30 - 15% resizing to 75% or 50% -
15% two-pass recompression at quality 90 or 70 - 10% combined
transformations C1 or C2 - **Held out from training:** resize 25%,
recompression at quality 50 and 30, and combination C3. - The model was
fine-tuned for 10 epochs, and checkpoint selection used validation data
only.

**Visual:** A compact percentage table or bar showing the five training
categories. Use `models/robust/robust_training_config.json` for the
exact policy.

**Speaker notes:** "We did not train on every condition. We held out
four conditions so we could check whether improvements carried over to
transformations the model had not seen during training. The test set was
not used to make training decisions."

**Transition:** "Let's compare the baseline and robust model directly."

------------------------------------------------------------------------

## Slide 8 --- Baseline versus robust model (2 min)

**Main point:** Transformation-aware training greatly improved recall in
most conditions, with a small reduction in clean accuracy.

**On-slide text** - **Clean images:** accuracy changed from 90.73% to
88.95% (−1.77 pp); AI recall rose from 90.96% to 91.98% (+1.02 pp). -
**JPEG quality 30:** AI recall improved from 55.46% to 87.54% (+32.08
pp). - **Combined condition C3:** AI recall improved from 54.78% to
91.47% (+36.69 pp). - **Resize 25%:** still difficult, with robust-model
accuracy at 64.92%. - AI recall improved in 13 of the 15 tested
conditions; the two small decreases were at resize scales 0.75 and 0.50.

**Visual:**
`results/graphs/ai_recall_all_conditions_baseline_vs_robust.png`.
Highlight the gains at JPEG Q30 and C3, and make the two resize
decreases visible rather than hiding them.

**Speaker notes:** "The robust model gave up 1.77 percentage points of
clean accuracy, while its AI recall increased by more than 32 points at
JPEG quality 30. The largest gain was in C3, a combined condition we had
kept out of training. The improvement was broad, but it was not
identical in every condition."

**Transition:** "Let's take a closer look at the conditions that were
held out."

------------------------------------------------------------------------

## Slide 9 --- Results on conditions held out from training (1.5 min)

**Main point:** The robust model also improved on all four conditions
that were excluded from its training data.

**On-slide text** - We held out four conditions during training: resize
25%, recompression Q50, recompression Q30, and C3. - **Resize 25%:** AI
recall rose from 90.61% to 94.88% (+4.27 pp); accuracy rose from 53.71%
to 64.92%. - **Recompression Q50:** AI recall rose from 67.24% to 88.91%
(+21.67 pp). - **Recompression Q30:** AI recall rose from 55.29% to
87.54% (+32.25 pp). - **C3:** AI recall rose from 54.78% to 91.47%
(+36.69 pp).

**Visual:** A four-row table using
`results/metrics/baseline_vs_robust_efficientnet_b0.csv`, with columns
for condition, baseline AI recall, robust AI recall, and change in
percentage points.

**Speaker notes:** "These results are important because the four
conditions were excluded from robust training. For C3, AI recall
increased from 54.78% to 91.47%. That suggests the training benefits
carried over to this unseen transformation pipeline. Resize 25% also
improved, although its accuracy remained much lower than the clean-image
result."

**Transition:** "One more result stood out: the order of transformations
changed the errors we saw."

------------------------------------------------------------------------

## Slide 10 --- Why transformation order matters (1.5 min)

**Main point:** Applying the same transformations in a different order
produced very different detection results.

**On-slide text** - **C1 --- Resize to 50% → JPEG Q50:** baseline AI
recall was 55.80%. The Real-image counts matched the clean baseline: 543
true negatives and 57 false positives. - **C2 --- JPEG Q50 → resize to
50%:** baseline AI recall was 88.23%, while false positives on Real
images rose to 202. - **C3 --- Resize → JPEG Q50 → JPEG Q50:** baseline
AI recall was 54.78%, close to C1. - **Finding:** The order of
transformations changes the model's failure pattern.

**Visual:** Show the C1 and C2 confusion-matrix counts side by side
using `results/metrics/combined_robustness_efficientnet_b0.csv`. Keep
the operation order visible in each heading.

**Speaker notes:** "C1 and C2 use the same two operations, but in a
different order, and their results are very different. C1 mainly misses
AI images, while C2 has more false positives on Real images. We have
measured this pattern, but we have not established the mechanism behind
it. So our conclusion is that order matters for the result---not that we
have proved why."

**Transition:** "To finish, here is what the experiments let us
conclude."

------------------------------------------------------------------------

## Slide 11 --- Conclusion and implications (1 min)

**Main point:** Training with representative transformations improved AI
recall in most tested conditions, including the four held-out
conditions.

**On-slide text** - JPEG compression caused more missed AI images, while
strong resizing produced many false positives. - Transformation-aware
training improved AI recall across JPEG, recompression, combined, and
held-out conditions. - The improvement came with a measured
clean-accuracy decrease of 1.77 percentage points. - Results depend on
the dataset and the tested conditions; this experiment does not
establish a universal detector.

**Visual:** No additional visual is needed. Finish with the three
takeaways: **test transformed images · report recall and false positives
· record transformation order**.

**Speaker notes:** "Our experiments show that clean-image performance
does not fully predict performance after compression and resizing.
Training with a planned set of transformations improved recall in most
of our tests, including conditions excluded from training, with a small
clean-accuracy cost. We also found that transformation order can change
the type of errors the model makes. Thank you---we're happy to take
questions."

------------------------------------------------------------------------

## Appendix --- Terminology and references (backup slide)

-   **AI recall:** percentage of AI-generated images correctly detected;
    higher is better.
-   **AI false-negative rate (FNR):** percentage of AI-generated images
    missed; lower is better.
-   **Percentage points (pp):** the difference between two percentages.
-   **C1/C2/C3:** the three ordered combined pipelines defined on Slide
    5.
-   **Class labels:** 0 = Real, 1 = AI-generated. In confusion matrices,
    rows are true labels and columns are predicted labels.
-   **References** (as listed in `docs/REFERENCES.md`):
    -   Zhu et al., *GenImage* (arXiv:2306.08571) --- benchmark.
    -   Grommelt et al., *Fake-or-JPEG* (arXiv:2403.17608) ---
        evaluation methodology.
    -   Li et al., phase-spectrum robustness (CVPR 2026; verify
        bibliographic details before presenting).
