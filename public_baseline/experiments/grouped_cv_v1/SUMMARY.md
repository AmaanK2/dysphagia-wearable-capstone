# Grouped development validation

Interpretation update: the linked study reports unlabelled events and imaging pauses. Unannotated time is not verified negative ground truth. All reported scores are relative to supplied annotations; see label_review/README.md.

Three outer folds across 148 development participant IDs. Each outer assessment participant appears exactly once; the original 27 pilot test participants are excluded entirely.
Within each fold, 20% of the remaining participants form a separate inner-validation set. Both models select thresholds there; the CNN also selects its checkpoint there. Normalization is fitted only to inner-training participants.

| Pooled outer-assessment metric | Random forest | CNN |
| --- | ---: | ---: |
| Event precision | 41.9% | 60.3% |
| Event recall | 69.3% | 77.9% |
| Event F1 | 52.2% | 68.0% |
| Strict event F1, IoU >= 0.50 | 32.9% | 58.9% |
| Extra detections / recording minute | 9.48 | 5.05 |

## Fold-level F1, IoU >= 0.30

| Fold | RF | CNN |
| --- | ---: | ---: |
| 1 | 53.9% | 70.3% |
| 2 | 53.5% | 67.9% |
| 3 | 48.4% | 65.6% |

## CNN error review

- 680 extra detections: no_annotation_overlap.
- 96 extra detections: overlaps_annotation_boundary_or_duplicate.

These categories describe overlap with annotations, not the physiological cause of an error. Missing/incomplete labels remain a possible issue. Review cnn_error_review.csv alongside the raw signals.

## Limits

- Exploratory internal validation on the same public dataset, not external clinical validation.
- CNN changes both temporal context and architecture compared with RF. Matched-context ablations are still needed.
- This is swallow timing detection, not dysphagia or aspiration prediction.
- Original pilot results and weights are unchanged. No outer-assessment results were used to change these runs.
- Cluster bootstrap intervals reflect assessment-participant variability for these fitted folds; they do not capture all retraining uncertainty.
- Do not select the best outer fold as a production model. Fold checkpoints exist for reproducibility; the original CNN remains the single-split candidate.
