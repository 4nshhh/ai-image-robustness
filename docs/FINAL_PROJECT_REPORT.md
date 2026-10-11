# Robust Detection of AI-Generated Images Under Social Media-Style Compression and Transformations

## A Controlled Robustness Study with Transformation-Aware Training

**AIML Course Project --- Final Report**

**Team responsibilities:** Krish --- data, models, and training; Ansh
--- experiments, evaluation, and dashboard; third team member ---
documentation and presentation.\
**Class labels:** 0 = Real, 1 = AI-generated.

------------------------------------------------------------------------

## Abstract

AI-generated images are increasingly shared through platforms that
resize and recompress uploaded content. These changes can affect the
visual details used by an image detector, so performance on clean images
may not fully describe how well a detector works after an image has been
processed. This project studies the effect of JPEG compression,
resizing, repeated JPEG compression, and combinations of these
transformations on a binary Real-versus-AI image classifier. We
evaluated EfficientNet-B0 and ResNet-50 on a fixed Tiny-GenImage subset,
then used EfficientNet-B0 for a controlled robustness study.

The test set contains 1,186 images: 600 Real and 586 AI-generated. We
evaluated the baseline model across 15 image conditions and then
compared it with a version fine-tuned using a predefined mixture of
clean and transformed training images. On clean test images, the
baseline EfficientNet-B0 achieved 90.73% accuracy and 90.96% AI recall.
Under JPEG quality 30, AI recall fell to 55.46%. Resizing caused a
different problem: at 25% scale, accuracy fell to 53.71% while AI recall
remained at 90.61%, indicating a substantial increase in false positives
on Real images. Transformation-aware training improved AI recall in 13
of the 15 tested conditions. At JPEG quality 30, recall increased to
87.54%; for the held-out C3 pipeline, it increased from 54.78% to
91.47%. The robust model's clean accuracy was 88.95%, a decrease of 1.77
percentage points from the baseline. These results show that training
with representative image transformations can improve detection
performance across many of the tested conditions, although the size and
direction of the effect depend on the transformation applied.

------------------------------------------------------------------------

## 1. Introduction

AI-generated images can now be created by a wide range of
image-generation systems and shared in the same places as photographs.
Once an image is uploaded to a social platform or passed between
services, it may be resized or recompressed. These operations can alter
fine visual details, edges, and textures that a detector may use when
distinguishing real images from generated ones.

A detector's performance on an original image therefore does not
necessarily tell us how it will perform after common image-processing
operations. A model may identify an AI-generated image correctly in its
original form but miss it after compression. Other transformations may
change the kinds of errors the detector makes, such as incorrectly
classifying Real images as AI-generated.

This project investigates those effects through a controlled set of
experiments. We first measure two established convolutional neural
network architectures on clean images. We then apply defined
transformations to the same test set and record how each condition
changes the model's predictions. Finally, we fine-tune the selected
model with a fixed mixture of clean and transformed training images and
evaluate whether this training improves performance on both familiar and
held-out transformation conditions.

The central focus is robustness: how the detector behaves when its
inputs have been altered, and how much of the measured performance loss
can be recovered through transformation-aware training.

## 2. Problem Statement

The task is to classify each image as either Real or AI-generated. Real
images use label 0, while AI-generated images use label 1. The
classifier is evaluated on clean images and on images processed through
a fixed set of transformations designed to represent compression and
resizing operations.

The study addresses two related questions. First, how do individual
transformations and their order affect the detector's accuracy and error
pattern? Second, can training with a planned mixture of transformed
images improve performance under those conditions without substantially
reducing performance on clean images?

## 3. Research Question and Objectives

The main research question is:

**How do JPEG compression, resizing, recompression, and combined
transformations affect AI-generated image detection, and can
transformation-aware training improve robustness?**

The project has four practical objectives:

1.  Measure baseline detection performance on clean images and under 15
    fixed test conditions.
2.  Compare EfficientNet-B0 and ResNet-50 on the clean test set and
    select a primary model for the robustness experiments.
3.  Fine-tune the selected model using a predefined transformation-aware
    training policy.
4.  Compare the baseline and fine-tuned model on the same test
    conditions, including transformation conditions excluded from robust
    training.

The experiment uses the same test images, preprocessing pipeline,
decision threshold, and metric definitions for the baseline and robust
model. This makes the comparisons direct and keeps the evaluation
procedure consistent across conditions.

## 4. Related Work

The project draws on research into large-scale evaluation of
AI-generated image detection and the effect of image compression on
detection performance.

Zhu and colleagues introduced **GenImage**, a large-scale benchmark for
detecting AI-generated images. The dataset used in this project is a
small subset of the broader GenImage resource. Its generator diversity
provides a starting point for evaluating a binary detector across images
produced by several generation systems.

Grommelt and colleagues studied common biases in generated-image
detection datasets in *Fake or JPEG? Revealing Common Biases in
Generated Image Detection Datasets*. This work highlights the importance
of understanding how compression and image processing relate to
evaluation outcomes. In the present project, compression is treated as
an experimental condition: its quality level and order relative to
resizing are specified so that the resulting performance changes can be
measured.

The project also draws motivation from research on robustness for
compressed AI-generated images, including work on phase-spectrum
robustness. The implementation here takes a straightforward fine-tuning
approach. Instead of introducing a specialized architecture, it tests
whether a fixed mixture of common transformations during training
improves the robustness of an existing image classifier.

These works provide context for the experiment. The present study
focuses specifically on a controlled comparison of clean inputs,
transformed inputs, and a transformation-aware fine-tuned model. Full
bibliographic details are listed in the References section.

## 5. Dataset and Split Methodology

### 5.1 Dataset

The experiments use a subset of Tiny-GenImage containing 7,986 images.
The images include Real examples and AI-generated examples from seven
generators: ADM, BigGAN, GLIDE, Midjourney, SDv5, VQDM, and Wukong.

Fourteen AI test files could not be recovered from the source export.
They were excluded without replacement, leaving the final split sizes
shown below. The dataset manifest records each image's split, label, and
generator.

### 5.2 Train, validation, and test splits

The dataset was divided using seed 42. The training split contains 5,600
images, the validation split contains 1,200 images, and the frozen test
split contains 1,186 images.

  Split          Real images   AI-generated images       Total
  ------------ ------------- --------------------- -----------
  Training             2,800                 2,800       5,600
  Validation             600                   600       1,200
  Test                   600                   586       1,186
  **Total**        **4,000**             **3,986**   **7,986**

A SHA-256 integrity check found no duplicate images across the splits.
The test split was kept separate from model training, hyperparameter
decisions, and checkpoint selection. Every transformation condition uses
the same test images and ground-truth labels, allowing the model's
behaviour to be compared across conditions.

## 6. System Architecture

The system follows a shared evaluation pipeline. Images are selected
from the fixed dataset split, passed through the specified research
transformation when required, and then processed using the same input
pipeline. The model produces two-class scores, which are converted into
predictions using a threshold of 0.5. The predictions are compared with
the ground-truth labels to calculate the evaluation metrics.

**Figure 1. System architecture and experiment pipeline**

![System architecture and experiment
pipeline](../results/graphs/system_architecture_pipeline.png)

The two baseline architectures evaluated on clean images were
EfficientNet-B0 and ResNet-50. EfficientNet-B0 was selected for the
robustness study because its clean AI recall was higher, while its F1
score was similar to that of ResNet-50. The robust model was created by
fine-tuning the EfficientNet-B0 baseline checkpoint with the planned
transformation mixture.

The same metric calculation process is used for every condition. This
shared pipeline is important because it reduces the chance that
differences between results are caused by changes in preprocessing or
evaluation code instead of by the image transformation or model being
tested.

## 7. Models and Preprocessing

### 7.1 Model architectures

The initial comparison used EfficientNet-B0 and ResNet-50, each
configured for two-class classification. Both models predict one of the
two labels: Real or AI-generated.

EfficientNet-B0 achieved 90.73% clean test accuracy, 90.34% precision,
90.96% AI recall, and an F1 score of 90.65%. ResNet-50 achieved 90.81%
accuracy and 94.09% precision, but its AI recall was 86.86%.
EfficientNet-B0 was chosen as the primary model because it detected a
larger share of AI-generated images while maintaining a similar F1
score.

### 7.2 Preprocessing

Images are converted to RGB, resized to 224 × 224 pixels, converted to
tensors, and normalized using ImageNet normalization values. The same
final preprocessing is applied after each research transformation. The
decision threshold is fixed at 0.5 for the reported classification
metrics.

### 7.3 Training setup

The baseline training setup uses Adam with a learning rate of (1
`\times 10`{=tex}\^{-4}), weight decay of (1 `\times 10`{=tex}\^{-4}), a
batch size of 16, and 10 epochs. The best baseline checkpoint is
selected using validation accuracy. The test set is not used for
training or checkpoint selection.

## 8. Image Transformation Conditions

The robustness experiments apply transformations to fresh copies of the
test images. The original files remain unchanged. Both Real and
AI-generated images receive the same type of transformation within a
condition.

### 8.1 JPEG compression

JPEG compression is tested at quality levels 90, 70, 50, and 30. Each
condition uses a single encode-and-decode pass. Lower quality settings
introduce stronger compression, allowing performance to be compared as
image quality decreases.

### 8.2 Resizing

The resize conditions scale each image to 75%, 50%, or 25% of its
original dimensions while preserving its aspect ratio. These conditions
measure how reducing image dimensions affects detection.

### 8.3 Recompression

Recompression consists of two sequential JPEG passes at the same quality
setting. It is evaluated at quality levels 90, 70, 50, and 30. This is
treated separately from single-pass JPEG compression because an image
can pass through more than one encoding stage during sharing or editing.

### 8.4 Combined transformations

Three combined pipelines examine the effect of applying transformations
in different sequences.

  -----------------------------------------------------------------------
  Condition               Operations              Purpose
  ----------------------- ----------------------- -----------------------
  C1                      Resize to 50% → JPEG    Resize followed by
                          Q50                     compression

  C2                      JPEG Q50 → resize to    Compression followed by
                          50%                     resize

  C3                      Resize to 50% → JPEG    Resize followed by two
                          Q50 → JPEG Q50          compression passes
  -----------------------------------------------------------------------

The order is recorded as part of each condition. C1 and C2 use the same
two operations in different sequences, while C3 adds a second JPEG pass
after resizing. Across the experiment, the clean condition and the
transformed conditions form the 15-condition robustness evaluation.

## 9. Transformation-Aware Training Strategy

The robust model was initialized from the trained EfficientNet-B0
baseline checkpoint. All model parameters were trainable during
fine-tuning. The training policy was defined before the robust model was
evaluated and used seed 42.

For each training image, the policy selected one of five categories:

  Training category            Share of training draws Available settings
  -------------------------- ------------------------- --------------------------------
  Clean image                                      40% No research transformation
  JPEG compression                                 20% Quality 90, 70, 50, or 30
  Resizing                                         15% Scale 0.75 or 0.50
  Recompression                                    15% Two passes at quality 90 or 70
  Combined transformations                         10% C1 or C2

The transformations were applied to both classes using the same policy.
The model was fine-tuned for 10 epochs with Adam, a learning rate of (1
`\times 10`{=tex}\^{-4}), weight decay of (1 `\times 10`{=tex}\^{-4}),
and batch size 16.

Four conditions were deliberately excluded from robust training: resize
scale 0.25, recompression at quality 50, recompression at quality 30,
and C3. These conditions were kept for evaluation to test whether
improvements extended to settings not included in the training mixture.

Checkpoint selection used clean validation performance. Epoch 8 was
selected as the best checkpoint, with validation accuracy of 0.9008. The
test set did not influence the training policy or checkpoint selection.

## 10. Evaluation Metrics and Experimental Protocol

### 10.1 Metrics

The experiment reports several complementary metrics:

-   **Accuracy:** the fraction of all images classified correctly.
-   **Precision:** the fraction of images predicted as AI that are
    actually AI-generated.
-   **AI recall:** the fraction of AI-generated images correctly
    detected.
-   **F1 score:** the harmonic mean of precision and recall.
-   **ROC-AUC:** the area under the receiver operating characteristic
    curve, calculated from AI-class probabilities.
-   **AI false-negative rate (FNR):** the fraction of AI-generated
    images classified as Real.
-   **Real recall and false-positive rate:** measures that help show how
    often Real images are correctly identified or incorrectly flagged as
    AI.
-   **Confusion-matrix counts:** true negatives, false positives, false
    negatives, and true positives.

In each confusion matrix, rows represent the actual class and columns
represent the predicted class. With class 0 representing Real and class
1 representing AI-generated, the matrix is written as (\[\[TN, FP\],
\[FN, TP\]\]).

### 10.2 Comparison procedure

Each condition uses the same 1,186 test images: 600 Real and 586
AI-generated. The ground-truth label order remains unchanged across
conditions. A clean control is included in the evaluation process and
reproduces the baseline within a tolerance of (10\^{-4}).

The baseline and robust model are compared condition by condition.
Differences between percentages are reported in **percentage points
(pp)**. For example, a change from 55% to 87% recall is a gain of 32
percentage points, not a 32% relative increase.

## 11. Results

### 11.1 Clean test performance

The initial model comparison provides the reference point for the
robustness study.

  Model               Accuracy   Precision   AI recall   F1 score   ROC-AUC
  ----------------- ---------- ----------- ----------- ---------- ---------
  EfficientNet-B0       90.73%      90.34%      90.96%     90.65%    0.9649
  ResNet-50             90.81%      94.09%      86.86%     90.33%    0.9682

ResNet-50 achieved slightly higher overall accuracy and precision on
clean images. EfficientNet-B0 detected a larger proportion of
AI-generated images, with AI recall of 90.96% compared with 86.86%.
Since the project focuses on reliably identifying AI-generated content,
EfficientNet-B0 was selected for the subsequent robustness experiments.

### 11.2 JPEG compression results

**Figure 2. AI recall across JPEG quality settings**

![AI recall vs JPEG
quality](../results/graphs/ai_recall_vs_jpeg_quality.png)

The baseline model's AI recall decreased as JPEG quality was reduced.
Recall was 84.98% at Q90, 74.74% at Q70, 67.75% at Q50, and 55.46% at
Q30. At the lowest tested quality, the baseline missed a substantial
share of AI-generated images.

The robust model maintained higher recall across the same conditions. At
Q90, recall increased from 84.98% to 91.13%. At Q70, it increased from
74.74% to 89.76%. At Q50, it increased from 67.75% to 89.08%. The
largest gain among the single-pass JPEG conditions occurred at Q30,
where recall increased from 55.46% to 87.54%, a gain of 32.08 percentage
points.

These results show that the transformation-aware training policy
substantially improved recall under the tested JPEG settings,
particularly at stronger compression levels.

### 11.3 Resizing results

**Figure 3. AI recall across resize scales**

![AI recall vs resize
scale](../results/graphs/ai_recall_vs_resize_scale.png)

The resize experiments produced a different pattern. Baseline AI recall
was 95.39% at scale 0.75, 95.90% at scale 0.50, and 90.61% at scale
0.25. However, accuracy fell from 86.85% to 77.07% and then to 53.71% as
the scale decreased.

The confusion counts show why. At scale 0.25, the baseline classified
494 Real images as AI, compared with 57 false positives on the clean
test set. The model continued to detect many AI-generated images, but
its predictions for Real images deteriorated sharply.

The robust model achieved AI recall of 94.71%, 95.05%, and 94.88% at
scales 0.75, 0.50, and 0.25. Its accuracy at those scales was 87.44%,
85.33%, and 64.92%, respectively. The strongest resize condition
therefore improved substantially, although its accuracy remained below
the clean-image result.

### 11.4 Overall baseline-versus-robust comparison

**Figure 4. AI recall across all 15 test conditions**

![Baseline vs robust AI recall, all
conditions](../results/graphs/ai_recall_all_conditions_baseline_vs_robust.png)

The robust model achieved higher AI recall than the baseline in 13 of
the 15 conditions. The two decreases were small and occurred at resize
scales 0.75 and 0.50. The results show that the training policy improved
recall across a wide range of the tested transformations, with the
strongest gains appearing in several JPEG-family and combined
conditions.

The change on clean images provides important context for these gains.
Baseline accuracy was 90.73%, compared with 88.95% for the robust model,
a decrease of 1.77 percentage points. Clean AI recall rose from 90.96%
to 91.98%, an increase of 1.02 percentage points. Clean precision
decreased by 3.82 percentage points, while ROC-AUC decreased by 0.58
percentage points. The effect of robust training is therefore best
understood by considering both its improved transformed-image recall and
its modest decrease in clean accuracy.

### 11.5 Recompression results

For the baseline model, two-pass recompression produced AI-recall values
close to those from single-pass JPEG compression, with differences of
approximately 0.2 percentage points. The robust model improved recall
under all tested recompression settings.

  ------------------------------------------------------------------------
  Recompression   Baseline AI recall   Robust AI recall             Change
  quality                                               
  --------------- ------------------ ------------------ ------------------
  Q90                         84.81%             91.47%           +6.66 pp

  Q70                         74.40%             89.42%          +15.02 pp

  Q50                         67.24%             88.91%          +21.67 pp

  Q30                         55.29%             87.54%          +32.25 pp
  ------------------------------------------------------------------------

The gains increased as the compression quality decreased. This follows
the pattern observed in the single-pass JPEG experiment and shows that
the robust model also handled the tested repeated-compression conditions
more reliably.

### 11.6 Combined transformation results

The combined pipelines show that the sequence of operations is important
to the observed error pattern.

  Condition                    Baseline AI recall   Robust AI recall      Change
  -------------------------- -------------------- ------------------ -----------
  C1: resize → JPEG                        55.80%             91.47%   +35.67 pp
  C2: JPEG → resize                        88.23%             93.17%    +4.95 pp
  C3: resize → JPEG → JPEG                 54.78%             91.47%   +36.69 pp

The baseline recall for C1 was 55.80%, while C2 reached 88.23%, despite
using the same two operations in the opposite order. C3 produced a
baseline result close to C1. After transformation-aware training, recall
improved across all three pipelines, with the largest gain occurring in
C3.

### 11.7 Confusion-matrix analysis

**Figure 5. Robust-model confusion matrices for selected conditions**

![Key confusion matrices, robust
model](../results/graphs/confusion_matrices_key_conditions_robust.png)

The confusion matrices help distinguish cases where a model misses
AI-generated images from cases where it incorrectly flags Real images.
On clean images, the baseline confusion matrix is
(\[\[543,57\],\[53,533\]\]). Under JPEG Q30, the baseline matrix becomes
(\[\[556,44\],\[261,325\]\]), showing a sharp increase in AI false
negatives. At resize scale 0.25, the matrix is
(\[\[106,494\],\[55,531\]\]), showing that the dominant error is the
large number of Real images classified as AI.

These counts support the metric-level findings: JPEG compression mainly
increased missed AI images, while strong resizing mainly increased false
positives on Real images. Reporting only accuracy would not reveal these
two different patterns as clearly.

### 11.8 Held-out transformation results

Four conditions were excluded from robust training: resize scale 0.25,
recompression Q50, recompression Q30, and C3. Their results are
important because they show how the robust model performed when
evaluated on conditions that were not part of its training mixture.

  Held-out condition     Baseline AI recall   Robust AI recall      Change
  -------------------- -------------------- ------------------ -----------
  Resize 0.25                        90.61%             94.88%    +4.27 pp
  Recompression Q50                  67.24%             88.91%   +21.67 pp
  Recompression Q30                  55.29%             87.54%   +32.25 pp
  C3                                 54.78%             91.47%   +36.69 pp

All four held-out conditions improved. C3 showed the largest gain,
rising from 54.78% to 91.47% AI recall. Recompression Q30 also improved
by more than 32 percentage points. Resize scale 0.25 improved in recall
and accuracy, although the robust model's accuracy of 64.92% indicates
that this condition remained difficult.

These results suggest that the improvements extended beyond the exact
settings used during robust training. They do not, by themselves,
establish a specific mechanism for why the model generalizes to these
conditions.

## 12. Discussion

The experiments show that image transformations can affect a detector in
different ways. JPEG compression caused a strong decline in AI recall as
quality decreased. Strong resizing caused accuracy to fall largely
because Real images were increasingly classified as AI. This difference
is visible in the confusion matrices and reinforces the importance of
reporting multiple metrics.

Transformation order also changed the observed performance. C1 and C2
contain the same two operations, but reversing their order produced
substantially different baseline AI recall. The results establish an
order-dependent pattern for these tested pipelines. They do not
establish the exact internal mechanism responsible for that pattern.

Transformation-aware fine-tuning improved AI recall in 13 of 15
conditions and improved all four conditions that were excluded from
training. The largest gains appeared under JPEG Q30, recompression Q30,
and C3. The clean test results show a trade-off: the robust model had
slightly lower accuracy and precision, while clean AI recall increased.
These metrics should be considered together when judging the practical
value of the model.

Resize scale 0.25 remained challenging in absolute accuracy even after
fine-tuning. The robust model's 64.92% accuracy was an improvement over
the baseline's 53.71%, but the remaining error rate indicates that
further work would be needed to improve performance under severe
downscaling.

## 13. Conclusion and Practical Implications

This project evaluated an AI-generated image detector under clean
conditions and a fixed set of compression, resizing, recompression, and
combined transformations. The EfficientNet-B0 baseline achieved 90.73%
clean accuracy and 90.96% AI recall. Under JPEG Q30, AI recall fell to
55.46%, while resize scale 0.25 reduced accuracy to 53.71% as false
positives on Real images increased.

Fine-tuning EfficientNet-B0 with a predefined mixture of clean and
transformed training images improved AI recall in 13 of the 15
conditions. Recall under JPEG Q30 increased to 87.54%, and recall under
the held-out C3 pipeline increased from 54.78% to 91.47%. The robust
model's clean accuracy was 88.95%, 1.77 percentage points below the
baseline, while clean AI recall increased to 91.98%.

The findings support three practical recommendations. First, detectors
should be evaluated on transformed images as well as clean inputs.
Second, recall and false-positive behaviour should be considered
alongside accuracy, because different transformations can produce
different error patterns. Third, combined transformations should be
described in their actual order, since the sequence of operations can
materially change the result.

The conclusions apply to the dataset and conditions evaluated in this
study. They demonstrate measurable improvements from
transformation-aware training within this experimental setup, not a
guarantee of performance on every image source or platform.

## 14. Future Work

Several extensions could build on these experiments. Confidence
intervals and significance testing would help quantify uncertainty in
the measured differences. Additional training approaches, including
frequency-aware or multi-scale methods, could be explored to improve
performance under severe resizing. Evaluation using images processed by
real social platforms would help determine how laboratory
transformations compare with actual upload pipelines. Testing on
additional generators and datasets would also provide a broader picture
of how well the observed improvements transfer to new image sources.

## References

1.  Zhu, B., et al. *GenImage: A Million-Scale Benchmark for Detecting
    AI-Generated Images*. arXiv:2306.08571.
    https://arxiv.org/abs/2306.08571
2.  Grommelt, P., et al. *Fake or JPEG? Revealing Common Biases in
    Generated Image Detection Datasets*. arXiv:2403.17608.
    https://arxiv.org/abs/2403.17608
3.  Li, et al. *Detecting Compressed AI-Generated Images via Phase
    Spectrum Robustness*. CVPR 2026.
    https://openaccess.thecvf.com/content/CVPR2026/html/Li_Detecting_Compressed_AI-Generated_Images_via_Phase_Spectrum_Robustness_CVPR_2026_paper.html

------------------------------------------------------------------------

## Appendix A --- AI Recall Across All Conditions

The following values are taken from the saved baseline-versus-robust
comparison.

  Condition             Baseline AI recall   Robust AI recall      Change
  ------------------- -------------------- ------------------ -----------
  Clean                             90.96%             91.98%    +1.02 pp
  JPEG Q90                          84.98%             91.13%    +6.14 pp
  JPEG Q70                          74.74%             89.76%   +15.02 pp
  JPEG Q50                          67.75%             89.08%   +21.33 pp
  JPEG Q30                          55.46%             87.54%   +32.08 pp
  Resize 0.75                       95.39%             94.71%    −0.68 pp
  Resize 0.50                       95.90%             95.05%    −0.85 pp
  Resize 0.25                       90.61%             94.88%    +4.27 pp
  Recompression Q90                 84.81%             91.47%    +6.66 pp
  Recompression Q70                 74.40%             89.42%   +15.02 pp
  Recompression Q50                 67.24%             88.91%   +21.67 pp
  Recompression Q30                 55.29%             87.54%   +32.25 pp
  C1                                55.80%             91.47%   +35.67 pp
  C2                                88.23%             93.17%    +4.95 pp
  C3                                54.78%             91.47%   +36.69 pp

## Appendix B --- Training Configuration Summary

The robust model was fine-tuned from the baseline EfficientNet-B0
checkpoint using Adam, learning rate (1 `\times 10`{=tex}\^{-4}), weight
decay (1 `\times 10`{=tex}\^{-4}), batch size 16, and 10 epochs. The
random seed was 42. Training used 40% clean examples, 20% JPEG
compression, 15% resizing, 15% two-pass recompression, and 10% combined
C1/C2 transformations. Resize 0.25, recompression Q50/Q30, and C3 were
held out. The selected checkpoint was epoch 8, based on clean validation
accuracy of 0.9008.

## Appendix C --- Selected Confusion Matrices

Confusion matrices use rows for actual labels (\[Real, AI\]) and columns
for predicted labels (\[Real, AI\]).

  Condition               Confusion matrix
  ----------------------- ------------------------------
  Baseline, clean         (\[\[543,57\],\[53,533\]\])
  Baseline, JPEG Q30      (\[\[556,44\],\[261,325\]\])
  Baseline, resize 0.25   (\[\[106,494\],\[55,531\]\])
  Baseline, C1            (\[\[543,57\],\[259,327\]\])
  Baseline, C2            (\[\[398,202\],\[69,517\]\])
  Robust, C3              (\[\[443,157\],\[50,536\]\])
