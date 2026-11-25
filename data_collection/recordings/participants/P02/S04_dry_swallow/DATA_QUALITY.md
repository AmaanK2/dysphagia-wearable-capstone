# P02 Session 4 quality note

This is the original 180-second V1 dry-swallow recording and its observer/review annotations. The raw microphone CSV is stored on GitHub as the lossless `Mic.csv.gz` because the uncompressed file exceeds GitHub's per-file limit. Run `gzip -dk Mic.csv.gz` in this folder after checkout to restore `Mic.csv`. The decompressed file's SHA-256 is `dc9f028708b31559a83942c470ec02076bbd67f89b6dfb63da02dd2d587a82c4`.

- The session completed. Microphone and IMU sample indices are continuous for all three minutes.
- No clipped microphone samples or sustained microphone noise-floor intervals were found. The IMU has a smaller axis range than the other P02 sessions; inspect placement and motion context before interpreting that difference.
- `Events.csv` contains 14 dry-swallow reviewed labels. These observer labels have not been independently clinically verified.
- `Session.json` status describes digital capture completion, not validated sensor placement or label quality.
