# Session 4 quality note

This is the original 120-second V1 mixed-activity recording and its observer/review annotations. `Mic.csv` is stored in GitHub as the lossless `Mic.csv.gz` because the original CSV exceeds GitHub's per-file limit. After checkout, run `gzip -dk Mic.csv.gz` in this folder to restore `Mic.csv`. The decompressed file's SHA-256 is `815818d6e7bdac0b1eaeb934ff42184f4f43545e92c9c743352115eb21797029`, matching the local original byte for byte.

- Microphone and IMU sample indices are continuous for the full two minutes; no clipped microphone samples were found.
- The microphone repeatedly approaches a narrow, approximately 6-count RMS background for multi-second spans, including about 22–37 s, 44–53 s, and 103–120 s. The IMU continues during those spans. Inspect signal quality before using those intervals as two-sensor observations.
- All six reviewed labels have microphone activity: four dry swallows and two talking intervals. These are observer labels and have not been independently clinically verified.
- `Session.json` reports `complete` because the digital streams continued to arrive; it does not certify analog signal quality or label validity.
