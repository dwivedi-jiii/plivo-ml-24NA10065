"""Train and evaluate EOT detection models using various ML architectures."""
import argparse
import csv
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin

from features import (
    load_wav,
    speech_before,
    frame_energy_db,
    f0_contour,
    zcr_contour,
    spectral_centroid_contour,
    get_slope
)


class PyTorchMLP(ClassifierMixin, BaseEstimator):
    def __init__(self, input_dim=19, hidden_dim=64, epochs=120, lr=0.005, batch_size=32):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.model = None
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                out = self.model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
        return self
        
    def predict_proba(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.sigmoid(logits).numpy()
        return np.hstack([1 - probs, probs])
        
    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


def extract_features(x, sr, pause_start, pause_index, prev_pause_end):
    """Causal features extracted from audio before pause_start."""
    pause_start = float(pause_start)
    pause_index = float(pause_index)
    prev_pause_end = float(prev_pause_end)
    
    speaking_dur = pause_start - prev_pause_end
    
    seg = speech_before(x, sr, pause_start, window_s=1.5)
    
    if len(seg) < sr // 10:
        return np.zeros(19, dtype=np.float32)
        
    # Energy
    e = frame_energy_db(seg, sr)
    energy_mean = e.mean()
    energy_std = e.std()
    
    n_frames_0_5s = min(50, len(e))
    e_last = e[-n_frames_0_5s:]
    energy_last_0_5s_mean = e_last.mean()
    energy_slope_0_5s = get_slope(e_last)
    
    # Pitch
    f0 = f0_contour(seg, sr)
    voiced = f0[f0 > 0]
    voiced_ratio = len(voiced) / len(f0) if len(f0) > 0 else 0.0
    
    n_f0_0_5s = min(50, len(f0))
    f0_last = f0[-n_f0_0_5s:]
    voiced_last = f0_last[f0_last > 0]
    voiced_ratio_last = len(voiced_last) / len(f0_last) if len(f0_last) > 0 else 0.0
    
    if len(voiced) >= 3:
        f0_mean = voiced.mean()
        f0_std = voiced.std()
        f0_max = voiced.max()
        f0_min = voiced.min()
        n_voiced_slope = min(10, len(voiced))
        f0_slope = get_slope(voiced[-n_voiced_slope:])
    else:
        f0_mean = 0.0
        f0_std = 0.0
        f0_max = 0.0
        f0_min = 0.0
        f0_slope = 0.0
        
    # Zero Crossing Rate (speaking rate proxy)
    zcr = zcr_contour(seg, sr)
    zcr_mean = zcr.mean()
    n_zcr_0_5s = min(50, len(zcr))
    zcr_slope = get_slope(zcr[-n_zcr_0_5s:])
    
    # Spectral Centroid (brightness/timbre trend)
    centroid = spectral_centroid_contour(seg, sr)
    centroid_mean = centroid.mean()
    centroid_std = centroid.std()
    n_cent_0_5s = min(50, len(centroid))
    centroid_slope = get_slope(centroid[-n_cent_0_5s:])
    
    return np.array([
        pause_index,
        pause_start,
        speaking_dur,
        voiced_ratio,
        voiced_ratio_last,
        f0_mean,
        f0_std,
        f0_slope,
        f0_max,
        f0_min,
        energy_mean,
        energy_std,
        energy_last_0_5s_mean,
        energy_slope_0_5s,
        zcr_mean,
        zcr_slope,
        centroid_mean,
        centroid_std,
        centroid_slope
    ], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(args.data_dir, "labels.csv"))))
    
    # Pre-process turns to find previous pause ends
    from collections import defaultdict
    turns = defaultdict(list)
    for r in rows:
        turns[r["turn_id"]].append(r)
        
    for tid in turns:
        turns[tid].sort(key=lambda r: int(r["pause_index"]))
        
    prev_ends = {}
    for tid, p_list in turns.items():
        for i, r in enumerate(p_list):
            if i == 0:
                prev_ends[(tid, int(r["pause_index"]))] = 0.0
            else:
                prev_ends[(tid, int(r["pause_index"]))] = float(p_list[i-1]["pause_end"])

    cache = {}
    X, y, groups, keys = [], [], [], []
    for r in rows:
        path = os.path.join(args.data_dir, r["audio_file"])
        if path not in cache:
            cache[path] = load_wav(path)
        x, sr = cache[path]
        
        tid = r["turn_id"]
        p_idx = int(r["pause_index"])
        prev_end = prev_ends[(tid, p_idx)]
        
        X.append(extract_features(x, sr, float(r["pause_start"]), p_idx, prev_end))
        y.append(1 if r["label"] == "eot" else 0)
        groups.append(tid)
        keys.append((tid, p_idx))
        
    X, y = np.array(X), np.array(y)

    # Compare LR, MLP, RF, HGB in 5-fold CV
    cv = GroupShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
    
    models = {
        "LogisticRegression": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
        "PyTorchMLP": lambda: PyTorchMLP(input_dim=19),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=0),
        "HistGradientBoosting": lambda: HistGradientBoostingClassifier(random_state=0)
    }
    
    cv_scores = {name: [] for name in models}
    
    for tr, te in cv.split(X, y, groups):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_te = scaler.transform(X[te])
        
        for name, model_fn in models.items():
            clf = model_fn()
            clf.fit(X_tr, y[tr])
            cv_scores[name].append(clf.score(X_te, y[te]))
            
    print("\n--- Model Cross-Validation Comparison ---")
    best_name = None
    best_score = -1.0
    for name in models:
        mean_score = np.mean(cv_scores[name])
        print(f"  {name:25s} : CV Accuracy = {mean_score:.3f}")
        if mean_score > best_score:
            best_score = mean_score
            best_name = name
            
    print(f"\nBest architecture: {best_name} (CV Accuracy = {best_score:.3f})")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Instantiate best model
    base_estimator = models[best_name]()
    
    # Precompute group splits for calibrated classifier fitting
    cv_splits = list(GroupShuffleSplit(n_splits=5, test_size=0.25, random_state=0).split(X_scaled, y, groups))
    calibrated_clf = CalibratedClassifierCV(estimator=base_estimator, method='sigmoid', cv=cv_splits)
    calibrated_clf.fit(X_scaled, y)
    
    # Save best model, scaler, and best model name
    model_paths = ["model.pkl"]
    if "english" in args.data_dir.lower():
        model_paths.append("model_english.pkl")
    elif "hindi" in args.data_dir.lower():
        model_paths.append("model_hindi.pkl")
        
    for p_path in model_paths:
        with open(p_path, "wb") as f:
            pickle.dump({
                "scaler": scaler,
                "clf": calibrated_clf,
                "best_name": best_name
            }, f)
        print(f"Saved calibrated {best_name} model to {p_path}")

    # Generate probabilities for evaluation
    p = calibrated_clf.predict_proba(X_scaled)[:, 1]
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), pi_p in zip(keys, p):
            w.writerow([tid, pi, f"{pi_p:.4f}"])
    print(f"wrote {len(keys)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
# MLP model compared
