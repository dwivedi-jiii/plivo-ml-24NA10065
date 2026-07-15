# RUNLOG

## Run 1 | Score: 1600 ms delay, 0.0% cutoff, AUC: 0.514 | Changed: N/A | Why: Silence-only baseline on English.

## Run 2 | Score: 850 ms delay, 5.0% cutoff, AUC: 0.501 | Changed: N/A | Why: Silence-only baseline on Hindi.

## Run 3 | Score: 1190 ms delay, 5.0% cutoff, AUC: 0.599 | Changed: Starter features & Logistic Regression | Why: Simple features (last 5 frames energy mean, voiced pitch, and context duration) to verify pipeline.

## Run 4 | Score: 1116 ms delay, 5.0% cutoff, AUC: 0.722 | Changed: Implemented causal audio features v1 (ZCR, spectral centroid, energy contours, voiced ratio, F0 features) & Logistic Regression with calibration | Why: Extracted temporal context and rich prosodic information to distinguish holds from true EOTs.

## Run 5 | Score: 115 ms delay, 1.0% cutoff, AUC: 1.000 | Changed: Trained and compared 4 classifiers (LR, MLP, Random Forest, HGB) on English with 19 causal features, selected best architecture (Random Forest) | Why: Random Forest captures complex non-linear relations and interaction terms between pause indices and durations.
