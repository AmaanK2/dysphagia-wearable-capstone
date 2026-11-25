# Public-dataset label review

**Problem:** the study behind Zenodo 4539695 reports that some swallows went unlabelled while X-ray imaging was paused ([Khalifa et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7251089/)). So unlabelled time in the public data is not guaranteed to be "no swallow". Some of the models' "extra detections" may be real, unlabelled swallows. Our public-dataset scores measure agreement with the supplied labels, not true clinical accuracy.

## Files

| File | Contents |
| --- | --- |
| `review_queue.csv` | 776 extra detections and 334 missed labels from grouped validation. Review columns are blank. |
| `unlabelled_regions.png`, `review_examples.png` | Where unmatched detections fall, plus example cases |
| `audit_summary.json`, `selected_review_cases.json` | Machine-readable summary and the plotted cases |
| `verified_coverage_TEMPLATE.csv` | Empty template for recording verified label coverage |

Regenerate with `python audit_annotations.py` from `public_baseline/`. It won't overwrite a queue that already has review decisions.

## Open questions for the dataset authors

1. Are the labels complete per file? What do header-only label files mean?
2. Are imaging-pause masks or the synchronised video available?
3. Why do the archive counts differ from the paper?
