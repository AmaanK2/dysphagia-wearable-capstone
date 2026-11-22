# Session 2 quality note

This is the original 60-second V1 recording and its observer/review annotations. The raw CSV and session metadata are unchanged.

- Microphone and IMU sample indices are continuous for the full recording; no clipping was found in the microphone samples.
- The microphone has visible activity through about 36.9 s, then falls to an approximately 6-count RMS background from 37.0 s to the end. The IMU continues recording.
- The five reviewed swallow labels from 9.70 to 25.01 s are within the microphone-active portion. The marker presses at 41.81, 46.39, and 51.30 s occur after the microphone dropout and should not be treated as complete two-sensor observations.
- `Session.json` reports `complete` because the digital streams continued to arrive. That status does not certify analog signal quality or clinical label validity.
