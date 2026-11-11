# Images and graphs for the presentation

[Home](../README.md)

**Suggested short presentation: figures 1 → 2 → 3 → 4.** The remaining figures are useful backups. Click a preview or “Open full-size image” to open the original PNG; use GitHub’s download control to save it for slides.

All figures below are existing saved experiment outputs. No training or scores were changed during the November 11 cleanup.

## Jump to a figure

| Order | Figure | Use |
| --- | --- | --- |
| 1 | [Model comparison](#figure-1) | Main presentation |
| 2 | [What a prediction looks like](#figure-2) | Main presentation |
| 3 | [Does it hold across participants?](#figure-3) | Main presentation |
| 4 | [Why the annotation issue matters](#figure-4) | Main presentation |
| 5 | [Baseline methods and sensor channels](#figure-5) | Backup |
| 6 | [Original held-out recording](#figure-6) | Backup |
| 7 | [CNN training and checkpoint selection](#figure-7) | Backup |
| 8 | [Training within each development fold](#figure-8) | Backup |
| 9 | [Examples requiring reference review](#figure-9) | Backup |

<a id="figure-1"></a>
## 1. Model comparison

[![Model comparison](../experiments/cnn_v1/rf_vs_cnn.png)](../experiments/cnn_v1/rf_vs_cnn.png)

[Open full-size image](../experiments/cnn_v1/rf_vs_cnn.png)

**What to say:** On the same 27 pilot test participants, event F1 is 67.3% for the CNN versus 50.2% for the RF at IoU ≥ 0.30.

**Keep in mind:** These are annotation-relative timing scores, not accuracy or dysphagia diagnosis. Context length changes along with architecture.

<a id="figure-2"></a>
## 2. What a prediction looks like

[![What a prediction looks like](../experiments/cnn_v1/example_rf_vs_cnn.png)](../experiments/cnn_v1/example_rf_vs_cnn.png)

[Open full-size image](../experiments/cnn_v1/example_rf_vs_cnn.png)

**What to say:** Walk through the recorded signals, reference intervals and model detections on a common timeline.

**Keep in mind:** One illustrative recording does not establish overall performance or confirm unlabelled swallows.

<a id="figure-3"></a>
## 3. Does it hold across participants?

[![Does it hold across participants?](../experiments/grouped_cv_v1/grouped_validation.png)](../experiments/grouped_cv_v1/grouped_validation.png)

[Open full-size image](../experiments/grouped_cv_v1/grouped_validation.png)

**What to say:** Three development folds give pooled event F1 of 68.0% for the CNN and 52.2% for the RF. The original 27 pilot test participants are excluded.

**Keep in mind:** Internal validation on the same dataset, not an independent clinical or wearable evaluation.

<a id="figure-4"></a>
## 4. Why the annotation issue matters

[![Why the annotation issue matters](../label_review/unlabelled_regions.png)](../label_review/unlabelled_regions.png)

[Open full-size image](../label_review/unlabelled_regions.png)

**What to say:** This locates detections that do not overlap a supplied annotation. Reference coverage needs clarification before treating all of them as false swallows.

**Keep in mind:** No-overlap detections can be errors or unlabelled events. This plot cannot tell which.

<a id="figure-5"></a>
## 5. Baseline methods and sensor channels

[![Baseline methods and sensor channels](../results/model_comparison.png)](../results/model_comparison.png)

[Open full-size image](../results/model_comparison.png)

**What to say:** Compare the energy baseline and the microphone, acceleration and combined random-forest models.

**Keep in mind:** The combined RF was selected using validation data. Do not pick a model just because it looks best on test results.

<a id="figure-6"></a>
## 6. Original held-out recording

[![Original held-out recording](../results/held_out_example.png)](../results/held_out_example.png)

[Open full-size image](../results/held_out_example.png)

**What to say:** Show the original baseline example if asked how event intervals are compared with annotations.

**Keep in mind:** An illustrative plot, not a separately validated demonstration on our wearable.

<a id="figure-7"></a>
## 7. CNN training and checkpoint selection

[![CNN training and checkpoint selection](../experiments/cnn_v1/training_curves.png)](../experiments/cnn_v1/training_curves.png)

[Open full-size image](../experiments/cnn_v1/training_curves.png)

**What to say:** Explain training loss, validation event F1 and the selected checkpoint.

**Keep in mind:** Checkpoint and threshold selection use validation participants, not pilot-test performance.

<a id="figure-8"></a>
## 8. Training within each development fold

[![Training within each development fold](../experiments/grouped_cv_v1/inner_validation_curves.png)](../experiments/grouped_cv_v1/inner_validation_curves.png)

[Open full-size image](../experiments/grouped_cv_v1/inner_validation_curves.png)

**What to say:** Each outer fold has a separate inner-validation set for model selection.

**Keep in mind:** Inner-validation curves are not the outer-assessment results.

<a id="figure-9"></a>
## 9. Examples requiring reference review

[![Examples requiring reference review](../label_review/review_examples.png)](../label_review/review_examples.png)

[Open full-size image](../label_review/review_examples.png)

**What to say:** Use these cases to explain why synchronized reference evidence is needed to resolve errors.

**Keep in mind:** Review decisions remain unknown. Waveforms and model predictions are not clinical ground truth.

## A simple closing point

“The public-data experiments show that swallow-timing detection is feasible. Our next step is to validate the recording hardware and evaluate independently labelled recordings from our actual sensors.”


