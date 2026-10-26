# Annotation audit and reference review

## Important finding

The study linked by Zenodo 4539695 describes periods when X-ray imaging was paused but acoustic/vibration recording continued. It reports swallows missing from video labels and subsequent review against video and session logs. The paper does not supply a mapping here from those coverage problems to the released files.

Source: Khalifa, Coyle & Sejdic (2020), https://pmc.ncbi.nlm.nih.gov/articles/PMC7251089/ (Results, paragraphs discussing imaging pauses and manual validation; DOI 10.1038/s41598-020-65492-1).

Consequently, our assumption that every unannotated time point is non-swallow background is not verified. A no-overlap detection could be an actual model error, an unlabelled swallow, or an event without usable reference evidence. Do not assume the model is right either. Existing metrics remain reproducible comparisons with supplied labels; they do not establish true clinical precision or accuracy.

## What was done

- Preserved original metrics and all model weights. No metric was recomputed by removing difficult cases.
- Created a reference-review queue from grouped out-of-fold development errors; original pilot test participants are excluded.
- Set every review decision to `unknown`. No waveform-based or model-based pseudo-labels were accepted as truth.
- Categorized the 680 no-overlap detections by location, not physiological cause.
- Prepared an empty coverage template rather than inventing valid observation periods.

## Files

- review_queue.csv: 776 extra detections and 334 missed annotations, with identifiers, timing, model score and blank reference-review fields.
- verified_coverage_TEMPLATE.csv: header-only template for independently established observation coverage. It contains no verified intervals.
- unlabelled_regions.png: locations of no-overlap detections.
- review_examples.png: deterministic illustrative cases. These cannot confirm unlabelled swallows.
- audit_summary.json: machine-readable findings and current review status.

## How to use the queue

Match each interval to synchronized reference evidence using the exact recording identifier and clip-relative timing. Record the source, timing convention, reviewer and decision. Suggested decisions are `confirmed_swallow`, `confirmed_non_swallow`, `unverifiable` and `unknown`; do not fill them based solely on a model prediction or a waveform plot. Any reference protocol and adjudication should be agreed with the clinical stakeholder.

Missing annotations and incorrect boundaries must be versioned separately from original labels, with reasons and provenance. A reviewed queue is not an unbiased new test set because it was selected from model errors. Use it for development and diagnostic analysis; construct a separately selected, comprehensively reviewed dataset for final evaluation.

Run `python audit_annotations.py` to regenerate an untouched queue and plots. The script refuses to overwrite a queue containing review decisions, reference fields or notes. Preserve and version that work first. Raw signals must be available at data/raw/signals.zip when running outside the original workspace.

## Questions to resolve with Dr. Zheng or the dataset authors

1. Are the released annotations complete for each file? What do header-only label files mean?
2. Are valid VFSS observation intervals or imaging-pause masks available, aligned with the released signal clips?
3. Can synchronized video/session logs be accessed for reference review? What is the clip timing convention?
4. Why do the downloadable archive counts differ from the paper/record description? Is this a subset or a particular release?
5. Are any separately linked clinical outcome labels available, such as airway-protection ratings? The current CSVs contain no such field.

## Next modeling decision

Do not tune the detector to suppress every unannotated-region detection until coverage and label meaning are clarified. Continue to report annotation-relative pilot metrics with this caveat. Once verified negatives/coverage are available, version them, define a prospective evaluation protocol and retrain using development data only. The separate throat-activity dataset can support its own swallow/cough/speech/throat-clear experiment, but cannot establish missing labels in this dataset.
