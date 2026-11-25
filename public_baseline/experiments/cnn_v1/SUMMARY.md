# PyTorch experiment 1

Interpretation update: the linked study reports unlabelled events and imaging pauses. Unannotated time is not verified negative ground truth. All reported scores are relative to supplied annotations; see label_review/README.md.

One prespecified CNN run; the existing random forest and pilot test participant split were preserved.

CNN parameters: 36,681. Selected epoch: 6 of 11 completed epochs.
The model learns from filtered waveform sequences with about 2.6 seconds of symmetric context. The baseline uses 0.5-second feature windows. This comparison changes architecture and context together.

| Metric | Random forest | PyTorch CNN |
| --- | ---: | ---: |
| Event precision, IoU >= 0.30 | 37.6% | 61.5% |
| Event recall, IoU >= 0.30 | 75.4% | 74.2% |
| Event F1, IoU >= 0.30 | 50.2% | 67.3% |
| Event F1, IoU >= 0.50 | 31.1% | 56.1% |
| Extra events / recording minute | 11.70 | 4.35 |

CNN window accuracy: 87.7%. This is different from event F1 and is not the primary measure.

## Interpretation limits

- This is swallow timing detection, not normal/abnormal, dysphagia or aspiration classification.
- Labels and unannotated background retain the uncertainties described in the baseline audit.
- The pilot test results had already been inspected for the random forest. CNN checkpoint, normalization and threshold use training/validation data only; this is not a fresh blinded external study.
- One random seed and one participant split are insufficient to claim a generally superior architecture. CNN changes both model and context; isolate these factors in later ablations.
- Both preprocessing and CNN are offline/noncausal. Clinical clips do not establish continuous home-monitoring performance.
- Keep these results fixed. Use grouped cross-validation within development participants for subsequent choices, then obtain external data for stronger validation.
