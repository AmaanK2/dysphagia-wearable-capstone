# Presentation figures

[Home](../README.md) · Every image here is a saved experiment output.

## How to read the metrics

- **Precision:** Of the events the model flagged as swallows, what fraction matched a labelled swallow?
- **Recall:** Of all labelled swallows, what fraction did the model find?
- **F1:** A score that balances precision and recall. It drops when either is low.

For example, if the model flags 5 events and 3 match out of 4 labelled swallows, precision is **3/5 = 60%**, recall is **3/4 = 75%**, and F1 is **about 67%**.

## Jump to a figure

| Order | Figure | Use |
| --- | --- | --- |
| W1 | [Wearable + public data: how much weight?](#figure-w1) | Main presentation |
| W2 | [Making public recordings look like ours](#figure-w2) | Backup |
| 1 | [Model comparison](#figure-1) | Main presentation |
| 2 | [What a prediction looks like](#figure-2) | Main presentation |
| 3 | [Does it hold across participants?](#figure-3) | Main presentation |
| 4 | [Why the annotation issue matters](#figure-4) | Main presentation |
| 5 | [Baseline methods and sensor channels](#figure-5) | Backup |
| 6 | [Original held-out recording](#figure-6) | Backup |
| 7 | [CNN training and checkpoint selection](#figure-7) | Backup |
| 8 | [Training within each development fold](#figure-8) | Backup |
| 9 | [Examples requiring reference review](#figure-9) | Backup |

<a id="figure-w1"></a>
## W1. Wearable + public data: how much weight?

[![F1 vs public share](../experiments/mixed_public_wearable/share_sweep.png)](../experiments/mixed_public_wearable/share_sweep.png)

Each point is event F1 on a wearable participant the model never trained on. Training only on our recordings gives 64–69% F1. Adding the adapted public data at 20–65% of the training weight raises it to about 80–86%. Above ~80% public weight, accuracy falls again.

<a id="figure-w2"></a>
## W2. Making public recordings look like ours

[![Feature shift before and after adaptation](../experiments/mixed_public_wearable/adaptation_check.png)](../experiments/mixed_public_wearable/adaptation_check.png)

Bars show how far each detector feature of the public data sits from our wearable data, in wearable standard deviations. Grey is the raw public data and blue is after adaptation (resampling, unit scaling, equalisation). The green band (±0.2) counts as a negligible difference.

<a id="figure-1"></a>
## 1. Model comparison

[![Model comparison](../public_baseline/experiments/cnn_v1/rf_vs_cnn.png)](../public_baseline/experiments/cnn_v1/rf_vs_cnn.png)

[Open full-size image](../public_baseline/experiments/cnn_v1/rf_vs_cnn.png)

On the same 27 held-out participants, the CNN scored 67.3% event F1 versus 50.2% for the random forest, using the supplied annotations and an overlap threshold of 0.30.

<a id="figure-2"></a>
## 2. What a prediction looks like

[![What a prediction looks like](../public_baseline/experiments/cnn_v1/example_rf_vs_cnn.png)](../public_baseline/experiments/cnn_v1/example_rf_vs_cnn.png)

[Open full-size image](../public_baseline/experiments/cnn_v1/example_rf_vs_cnn.png)

This recording shows where each model detected swallows alongside the annotated intervals.

<a id="figure-3"></a>
## 3. Does it hold across participants?

[![Does it hold across participants?](../public_baseline/experiments/grouped_cv_v1/grouped_validation.png)](../public_baseline/experiments/grouped_cv_v1/grouped_validation.png)

[Open full-size image](../public_baseline/experiments/grouped_cv_v1/grouped_validation.png)

Across three development folds, pooled event F1 was 68.0% for the CNN and 52.2% for the random forest. The original 27 test participants were held out of this analysis.

<a id="figure-4"></a>
## 4. Why the annotation issue matters

[![Why the annotation issue matters](../public_baseline/label_review/unlabelled_regions.png)](../public_baseline/label_review/unlabelled_regions.png)

[Open full-size image](../public_baseline/label_review/unlabelled_regions.png)

This plot shows where detections fell outside the supplied annotations. Some may be model errors, while others may be events that were never labelled.

<a id="figure-5"></a>
## 5. Baseline methods and sensor channels

[![Baseline methods and sensor channels](../public_baseline/results/model_comparison.png)](../public_baseline/results/model_comparison.png)

[Open full-size image](../public_baseline/results/model_comparison.png)

The baseline compares a simple energy threshold with random forests trained on microphone data, acceleration data, or both.

<a id="figure-6"></a>
## 6. Original held-out recording

[![Original held-out recording](../public_baseline/results/held_out_example.png)](../public_baseline/results/held_out_example.png)

[Open full-size image](../public_baseline/results/held_out_example.png)

An example from the held-out set shows how the baseline detections line up with the annotated swallows.

<a id="figure-7"></a>
## 7. CNN training and checkpoint selection

[![CNN training and checkpoint selection](../public_baseline/experiments/cnn_v1/training_curves.png)](../public_baseline/experiments/cnn_v1/training_curves.png)

[Open full-size image](../public_baseline/experiments/cnn_v1/training_curves.png)

Training loss and validation F1 across epochs show how the CNN checkpoint was selected.

<a id="figure-8"></a>
## 8. Training within each development fold

[![Training within each development fold](../public_baseline/experiments/grouped_cv_v1/inner_validation_curves.png)](../public_baseline/experiments/grouped_cv_v1/inner_validation_curves.png)

[Open full-size image](../public_baseline/experiments/grouped_cv_v1/inner_validation_curves.png)

Each fold used a separate validation set to select its checkpoint and threshold.

<a id="figure-9"></a>
## 9. Examples requiring reference review

[![Examples requiring reference review](../public_baseline/label_review/review_examples.png)](../public_baseline/label_review/review_examples.png)

[Open full-size image](../public_baseline/label_review/review_examples.png)

These examples show where synchronized reference recordings would help clarify what happened.

