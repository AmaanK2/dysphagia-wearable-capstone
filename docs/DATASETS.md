# What is the data?

## HRCA: swallow timing

[Zenodo 4539695](https://zenodo.org/records/4539695) contains neck microphone and three-axis accelerometer recordings sampled at 4,000 Hz. Arrays have four channels. Label files contain swallow boundaries referenced to 60-frame/s videofluoroscopy.

The usable set in our audit contains 1,276 recordings and 1,862 annotated swallow events. Some recordings contain multiple events. We split by participant so recordings from one person cannot appear in both fitting and assessment sets. The grouped development evaluation excludes the original pilot test participants.

The associated [paper](https://doi.org/10.1038/s41598-020-65492-1) explains that signals continued while imaging could be paused. Unlabelled signal time is therefore not verified negative ground truth. Read [the label review](LABEL_REVIEW.md) before interpreting extra detections. No additional labels have been verified by our team.

## STVS: four activities

[Zenodo 12630148](https://zenodo.org/records/12630148) contains presegmented vibration clips: cough, speak, swallow and throat clearing. Each row is **4,000 samples at 6,400 Hz**, or **0.625 seconds**, from the z-axis of an IIM-42652 sensor in a custom throat mounting.

The release has 8,208 training clips and 792 test clips. [The authors](https://github.com/yhsong06/Throat-related-events-classification) describe 12 training participants and 20 separate test participants. Per-row participant IDs are absent from the arrays, so our inner training/validation split is clip-based, not participant-independent. Identical waveform hashes stay together. Conflicting training labels are excluded and recorded in `audit.json`; original archives are unchanged. Exact train/test overlap causes training to stop.

Our lightweight random forest/CNN comparison is not a reproduction of the paper's multi-network ensemble. See [the paper](https://doi.org/10.1038/s41746-024-01417-w) for its methods.

## Why we keep them separate

HRCA teaches temporal localization using four channels. STVS tests activity recognition on one vibration channel in already segmented clips. Neither provides a clinical dysphagia label for our classifier. Different rates, sensors, placement, mounting and acquisition electronics mean we must collect compatible data from our prototype and test transfer. EMG is not included in these two pipelines.

We do not concatenate these datasets or compare their accuracy and event F1 as if they were the same task.
