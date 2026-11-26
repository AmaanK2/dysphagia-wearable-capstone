# Random forest vs gradient boosting (swallow detector)

Both models use the same 29 features, the same public/wearable weighting and the same test. Each participant is held out in turn, and in the nested results the alert threshold is also chosen without the test participant. Only the classifier changes. 154 reviewed swallows from 3 participants.

## Random forest settings

| Model | Our recordings only | + 35% adapted public data |
| --- | ---: | ---: |
| Gradient boosting (shipped: 300 trees, ≤15 leaves, learning rate 0.05) | 76.2% | **86.2%** |
| Random forest, 300 trees, depth 14, ≥5 windows per leaf | **85.4%** | 81.5% |
| Random forest, fully grown, ≥20 windows per leaf | 81.9% | 81.8% |
| Random forest, fully grown, ≥40 windows per leaf | 79.6% | 82.1% |
| Random forest, half the features per split | 80.4% | 83.1% |

Nested event F1. Raw output: `forest_settings.log`.

## Best of each, with and without the mic/IMU agreement check

| Model | F1 | Precision | Recall | False alerts/min | P01 / P02 / P03 |
| --- | ---: | ---: | ---: | ---: | --- |
| Boosting + 35% public | 86.2% | 81.5% | 91.6% | 1.14 | 85.5 / 81.0 / 89.9% |
| Boosting + 35% public, with check | 86.0% | 81.0% | 91.6% | 1.18 | 85.7 / 81.0 / 89.1% |
| **Random forest, our data only** | **85.4%** | 82.0% | 89.0% | 1.07 | 83.2 / 83.1 / 88.5% |
| Random forest, our data only, with check | 82.9% | 79.6% | 86.4% | 1.21 | 82.4 / 84.2 / 82.5% |
| Average of both models | 85.8% | 81.8% | 90.3% | 1.11 | 85.0 / 80.0 / 90.1% |
| Average of both models, with check | 85.9% | 83.0% | 89.0% | 1.00 | 85.7 / 82.1 / 88.4% |

## What this means

- **The two approaches tie.** Boosting needs the public data to reach ~86%; a random forest reaches ~85% from our recordings alone. With 3 participants, a 1-point gap is noise.
- **Why the forest does better on little data.** Each of its trees sees a random sample of windows and features, and averaging 300 independent trees overfits less. Boosting keeps correcting its own errors, so with only two training participants it partly learns their quirks (76%) until the public data adds variety.
- **Why public data hurts the forest.** Its deep trees also fit whatever mismatch remains between the public sensor and ours.
- **Averaging the two doesn't help.** The models make similar mistakes, so the average lands between them.
- **The mic/IMU agreement check only helps under the looser test.** Under nested evaluation it's neutral for boosting and costs the forest 2.5 points, so the forest ships without it.

## In the app

`data_collection/models/mixed/mixed_forest_wearable_only.json` appears in the Detector list as **Random forest · wearable data only**. It has 300 trees (3.7 MB, ~0.06 ms per window), alert threshold 0.7 and a 1.6 s cooldown. With its threshold tuned on the same pooled folds (the looser test), it scores 86.9% F1 and 0.82 false alerts/min; the shipped boosting model scores 89.7% on that test. The recommended detector stays **Mixed 35/65 · trees**, because it is marginally ahead and its earlier version is the only one tested live on new sessions.

Rebuild it: `python data_collection/train_mixed.py --mode export --model forest --no-gate --export mixed_forest_wearable_only=0`
