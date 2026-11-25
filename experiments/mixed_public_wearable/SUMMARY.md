# Wearable + adapted public data

We adapted the public Pittsburgh recordings ([Zenodo 4539695](https://zenodo.org/records/4539695)) so they look like Swallow Studio recordings. Then we trained wearable detectors on a weighted mix of the two datasets. Every score below comes from **held-out wearable participants**. The public data is only used for training.

![F1 vs public share](share_sweep.png)

## Bottom line

| Model (file in `data_collection/models/mixed/`) | Public / wearable weight | LOPO event F1 | Precision | Recall | Extra alerts / min |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous wearable-only (same protocol, logistic) | 0 / 100 | 68.7% | 59% | 81% | 2.61 |
| `mixed_50_50.json` (boosted trees + sensor gate) | 50 / 50 | 84.9% | 84% | 86% | 0.78 |
| `mixed_80_20.json` (boosted trees + sensor gate) | 80 / 20 | 83.9% | 82% | 86% | 0.89 |
| **`mixed_optimal_35_65.json`** (boosted trees + sensor gate) | **35 / 65** | **87.4%** | **85%** | **89%** | **0.72** |

LOPO means leave one wearable participant out (P01, P02, P03), using 85 reviewed swallows. In these rows the alert threshold and cooldown were chosen on the same pooled folds, so the numbers are **optimistic**. A threshold chosen with the test person fully held out gives lower scores:
- **About 77–80% F1** at the 20–65% public splits.
- **76.8%** when the share itself is also chosen without the test person (`evaluation_share_selection_trees.json`).

The previous nested wearable-only result was 57–64%. Treat ~77–80% as the realistic expectation for a new wearer.

Each model has a `*_logistic.json` twin. It uses the same data and weighting with the original quadratic logistic detector (82.1% / 81.3% / 83.7% F1). Trees were chosen because they score higher, and they are the first models to detect P01 reasonably well (P01 F1 rises from about 20–29% to 62–67%).

## What the adaptation does (`data_collection/adapt_public.py`)

The comparison of the raw datasets (from `check_adaptation.py` and the analysis in this run):

| | Public (raw) | Wearable |
| --- | --- | --- |
| Rates | 4 kHz for all channels | mic 44.1 kHz, IMU **~904 Hz** (nominal 833 Hz) |
| Units | voltage-like, swallow mic RMS ≈ 0.017 | mic PCM counts, swallow RMS ≈ 88; accel m/s², swallow RMS ≈ 0.27 |
| Axes | anatomical A-P / S-I / M-L, gravity removed | native X/Y/Z with gravity mostly on X (along the neck) |
| Swallow/background RMS | mic 2.3×, accel 1.8× | mic 5.1×, accel 3.4× |
| Swallow label duration | median 0.80 s | median 0.84 s |

Steps applied to every public recording:
1. **Resample:** mic to 44.1 kHz, rounded and clipped to 16-bit PCM. Accel anti-aliased and resampled to the measured 904.6 Hz.
2. **Axes and gravity:** map S-I→X, M-L→Y, A-P→Z, then add the wearable's median gravity vector. The detector's IMU features combine all three axes, so this mostly matters for future per-axis models.
3. **Units and spectral shape:** both are calibrated in the detector's own feature space. The process iterates until:
   - the median swallow-window RMS matches the wearable (mic ×2987, accel ×1.89), and
   - each sensor's band shares match. A smooth equaliser applies mic +3.5/−2.6/−1.5/+0.6 dB and accel −2.1/+2.1/+0.8/−0.8 dB across the detector's four bands.
4. **No added noise:** the public background is already noisier than the wearable's.
5. **No label changes:** swallow durations already agree.

**Result:** on public recordings not used for calibration, the mean feature mismatch fell from 0.63 to **0.12** wearable SDs on swallow windows and from 0.47 to **0.15** on background windows (`adaptation_check.png`).
- **Remaining shift:** how sharply a swallow's energy rises relative to the preceding half-second. Public clips are short, with swallows close together.
- **Does adaptation matter?** Training on *unadapted* public data at the same splits gave 74–80% (fixed threshold) and 64–73% (nested) F1, with 2–3× the false alarms of adapted data at 20–50% (nested).

## How the weighting works

`public_share` is the fraction of the total training loss contributed by public windows. Within each dataset, swallow and background windows are class-balanced. So at 50/50, the 85 wearable swallows carry the same total weight as ~1,600 public swallows. Hard-negative emphasis (background windows that look like swallows, plus reviewed talking/cough) is applied within each dataset, so it doesn't change the split.

## Why 35/65 is the "optimal" split

- **A plateau, not a peak.** Every split from 20% to 65% public scores within a few points. The differences are a handful of events out of 85 swallows.
- **35% wins on two measures.** It is the best split for boosted trees (fixed threshold) and the best logistic split under fully nested evaluation.
- **Above ~65% the model gets fragile.** A model dominated by public data gives wearable swallows lower scores. A threshold picked from two people then misses many swallows on the third (logistic nested F1 drops to 64–68%).
- **Choosing the split without the test person** picks 50% for trees and 65% for logistic, consistently inside the plateau.
- **Theory agrees.** In domain-adaptation theory (Ben-David et al., 2010, *A theory of learning from different domains*), the best weight on the target domain grows with source–target divergence. It is well above its raw sample proportion. Here, naive pooling would give your data about 5% of the weight; the data says roughly 50–80%.

## Other experiments

- **Model variants** (`evaluation_variants.json`): the quadratic logistic settings were already near the best. A linear model fails under nested thresholds. Removing hard-negative weighting costs 4–15 F1 points.
- **Sensor-agreement gate:** requiring the mic-only and IMU-only checks to reach 0.4 added 1–4 F1 points at every split. All exported models include it.
- **Runtime:** in a rolling replay, a tree model scores a 100 ms tick in 0.35 ms (median), against 0.25 ms for logistic.

## Limits

- **Three participants and 85 dry swallows.** No result here should be read as precise to better than about ±5 F1 points. P01's microphone dropouts still dominate P01's errors.
- **Different swallow types.** Public swallows are clinical bolus swallows from patients; wearable swallows are dry swallows from healthy people.
- **Optimistic final thresholds.** The exported models choose their threshold on the same three people that measure them. Confirm on new participants before trusting the numbers.
- **Missing sessions.** Only 9 of the 12 sessions in `LIVE_DETECTION_RESULTS.md` are in Git (P03 S03–S05 are missing). Re-running with them is a direct improvement.

## Reproduce

From the repository root, in an environment with `data_collection/requirements.txt` plus `scikit-learn`, `pandas` and `matplotlib`, after `python download_data.py`:

```sh
python data_collection/check_adaptation.py
python data_collection/train_mixed.py                                   # logistic sweep
python data_collection/train_mixed.py --model trees                     # trees sweep
python data_collection/train_mixed.py --mode control --shares 0.2 0.35 0.5 0.8
python data_collection/train_mixed.py --mode variants --shares 0.35 0.5
python data_collection/train_mixed.py --model trees --mode select-share --shares 0 0.2 0.35 0.5 0.65 0.8
python data_collection/train_mixed.py --model trees --mode export \
  --export mixed_50_50=0.5 mixed_80_20=0.8 mixed_optimal_35_65=0.35
python experiments/plot_mixed_sweep.py
python -m unittest discover -s data_collection -p "test_mixed.py"
```

Adapted public features are cached in `_cache/adapted_public/` (about 1 minute to build per wearable-participant subset). To use a model in Swallow Studio, copy it to `data_collection/models/wearable.json`. Its loader now accepts tree models as well as logistic ones.
