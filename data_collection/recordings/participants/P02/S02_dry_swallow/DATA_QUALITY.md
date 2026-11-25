# P02 Session 2 quality note

This is the original 180-second V1 dry-swallow recording and its observer/review annotations. The raw microphone CSV is stored on GitHub as the lossless `Mic.csv.gz` because the uncompressed file exceeds GitHub's per-file limit. Run `gzip -dk Mic.csv.gz` in this folder after checkout to restore `Mic.csv`. The decompressed file's SHA-256 is `d86ce793527469768b0e1cf8afb6c1b64769fc52c4793cf0ef4c70adf8ec590d`.

- The session completed. Microphone and IMU sample indices are continuous for all three minutes.
- The microphone has 256 samples at or near 16-bit full scale, concentrated around 81 s and 103 s. Those short spans are clipped and should be inspected before analysis. No sustained microphone noise-floor interval was found.
- `Events.csv` contains ten dry-swallow reviewed labels. These observer labels have not been independently clinically verified.
- `Session.json` status describes digital capture completion, not validated sensor placement or label quality.
