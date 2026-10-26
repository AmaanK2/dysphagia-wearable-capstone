# Project status

## Completed

- Participant-separated random forest baseline and PyTorch CNN for swallow timing.
- Three-fold participant-grouped development evaluation, with pooled event F1 of 52.2% for random forest and 68.0% for CNN at IoU 0.3.
- Annotation coverage investigation and a review queue. All proposed new labels remain unverified.
- Separate STVS activity benchmark with checksum checks, duplicate/conflicting-label audit, validation-based model selection and a supplied external test partition.
- Package layout, tests and setup instructions for teammates.

Event F1 balances detected annotated swallows and extra detections. It is not the percentage of patients diagnosed correctly. Activity accuracy counts correct labels on pre-cut clips and answers a different question.

## Next useful engineering work

1. Review error plots and label coverage with the supervisor before deciding what unlabelled time means.
2. Confirm prototype sensors, acquisition rate, channels and mounting. Save timestamped raw recordings and acquisition settings before attempting real-time inference.
3. Establish a permitted collection and annotation protocol, including rest, speech, cough, throat clearing and motion. Validate on unseen participants and sessions.
4. Add acquisition and visualization against recorded files first, then hardware. Evaluate continuous swallow detection, false alarms per minute and latency on representative data with verified coverage.
5. Consider CNN-LSTM or other temporal models only under the same participant-separated evaluation. A larger model alone does not resolve missing labels or sensor mismatch.

No hardware acquisition, clinical risk classifier or validated wearable performance is implemented yet. The current timing pipeline is offline; its filters and CNN use future context.
