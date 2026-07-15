# Notes on End-of-Turn Detection

- We extract 19 causal audio and metadata features, including pause sequence indices, cumulative pause counts, and utterance duration history.
- The model leverages frame-level pitch contours (F0 mean, std, max, min, and final voiced slope) to detect falling intonation typical of statement ends.
- Temporal-spectral features like zero-crossing rate and spectral centroid trend capture changes in brightness and speaking rate right before a pause.
- Short-time RMS energy (mean, std, and slope in the final 50 frames) is used to track energy decay trailing into the pause.
- Random Forest works best for English, while HistGradientBoosting generalizes best on Hindi data, handling non-linear decision boundaries.
- The model occasionally triggers false cutoffs on long hesitation pauses (e.g. "uh", "um" equivalents) that mimic turn endings acoustically.
- Very short voiced regions can cause noisy pitch estimation, occasionally misclassifying queries or rising-intonation continuations.
- With one more day, we would implement voicing detection (unvoiced vs. voiced segments) to isolate final syllables and compute lengthening ratios.
- We would also explore language-specific hyperparameter tuning for gradient boosting and collect a wider range of speaker pitch averages to normalize F0.
