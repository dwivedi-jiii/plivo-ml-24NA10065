"""Inference script to load saved models and run predictions on unseen datasets."""
import argparse
import csv
import os
import pickle
import numpy as np

from features import (
    load_wav,
    speech_before,
    frame_energy_db,
    f0_contour,
    zcr_contour,
    spectral_centroid_contour,
    get_slope
)


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

    # Load appropriate model
    model_name = "model.pkl"
    if "english" in args.data_dir.lower():
        model_name = "model_english.pkl"
    elif "hindi" in args.data_dir.lower():
        model_name = "model_hindi.pkl"
        
    if not os.path.exists(model_name):
        model_name = "model.pkl"
        
    if not os.path.exists(model_name):
        raise SystemExit(f"No trained model file found (checked {model_name} and model.pkl)")
        
    with open(model_name, "rb") as f:
        m = pickle.load(f)
        
    scaler = m["scaler"]
    clf = m["clf"]
    print(f"Loaded calibrated model: {m.get('best_name', 'Model')} from {model_name}")

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
    X, keys = [], []
    for r in rows:
        path = os.path.join(args.data_dir, r["audio_file"])
        if path not in cache:
            cache[path] = load_wav(path)
        x, sr = cache[path]
        
        tid = r["turn_id"]
        p_idx = int(r["pause_index"])
        prev_end = prev_ends[(tid, p_idx)]
        
        X.append(extract_features(x, sr, float(r["pause_start"]), p_idx, prev_end))
        keys.append((tid, p_idx))
        
    X = np.array(X)
    X_scaled = scaler.transform(X)
    
    p = clf.predict_proba(X_scaled)[:, 1]
    
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), pi_p in zip(keys, p):
            w.writerow([tid, pi, f"{pi_p:.4f}"])
    print(f"wrote {len(keys)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
