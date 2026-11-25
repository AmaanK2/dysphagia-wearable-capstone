# P02 Session 1 quality note

This is the original 120-second V1 mixed-activity recording and its observer/review annotations. The raw microphone CSV is stored on GitHub as the lossless `Mic.csv.gz` because the uncompressed file exceeds GitHub's per-file limit. Run `gzip -dk Mic.csv.gz` in this folder after checkout to restore `Mic.csv`. The decompressed file's SHA-256 is `5582ef93f392f352231cc2dbe3c75b73bae75d96c0e587fb7be762629b734777`.

- The session completed. Microphone and IMU sample indices are continuous for the full two minutes.
- No clipped microphone samples or sustained microphone noise-floor intervals were found.
- `Events.csv` contains eight dry-swallow, one talking, and one cough reviewed labels. These observer labels have not been independently clinically verified.
- `Session.json` status describes digital capture completion, not validated sensor placement or label quality.
