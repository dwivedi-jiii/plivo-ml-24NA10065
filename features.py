"""Audio utilities for the EOT assignment.

These are UTILITIES, not features. Turning them into informative features
(slopes, ratios, statistics over time) is your job.

Causality reminder: for a pause at `pause_start`, you may only touch
audio[0 : pause_start]. Note that `pause_end` is FUTURE information for a
hold pause — using it (e.g., pause duration) in features is a violation.
"""
import numpy as np
import soundfile as sf

FRAME_MS = 25
HOP_MS = 10


def load_wav(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def speech_before(x, sr, pause_start, window_s=1.5):
    """The last `window_s` seconds of audio strictly before the pause."""
    end = int(pause_start * sr)
    start = max(0, end - int(window_s * sr))
    return x[start:end]


def frames(x, sr, frame_ms=FRAME_MS, hop_ms=HOP_MS):
    fl = int(sr * frame_ms / 1000)
    hp = int(sr * hop_ms / 1000)
    if len(x) < fl:
        return np.empty((0, fl), dtype=np.float32)
    n = 1 + (len(x) - fl) // hp
    idx = np.arange(fl)[None, :] + hp * np.arange(n)[:, None]
    return x[idx]


def frame_energy_db(x, sr):
    """Short-time energy per frame, in dB."""
    fr = frames(x, sr)
    if len(fr) == 0:
        return np.array([-120.0], dtype=np.float32)
    rms = np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


def autocorr_f0(frame, sr, fmin=60.0, fmax=400.0, voicing_thresh=0.30):
    """Fundamental frequency of one frame via autocorrelation.

    Returns 0.0 for unvoiced/silent frames.
    """
    frame = frame - np.mean(frame)
    if np.max(np.abs(frame)) < 1e-4:
        return 0.0
    ac = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    lo = int(sr / fmax)
    hi = min(int(sr / fmin), len(ac) - 1)
    if hi <= lo:
        return 0.0
    lag = lo + int(np.argmax(ac[lo:hi]))
    if ac[lag] < voicing_thresh:
        return 0.0
    return float(sr / lag)


def f0_contour(x, sr, frame_ms=40, hop_ms=HOP_MS):
    """Per-frame F0 (Hz), 0.0 where unvoiced. Longer frames help pitch."""
    fr = frames(x, sr, frame_ms=frame_ms, hop_ms=hop_ms)
    return np.array([autocorr_f0(f, sr) for f in fr], dtype=np.float32)


def zcr_contour(x, sr, frame_ms=FRAME_MS, hop_ms=HOP_MS):
    """Per-frame Zero Crossing Rate."""
    fr = frames(x, sr, frame_ms=frame_ms, hop_ms=hop_ms)
    if len(fr) == 0:
        return np.array([0.0], dtype=np.float32)
    signs = np.sign(fr)
    # Count sign changes
    zcr = np.mean(np.abs(np.diff(signs, axis=1)) > 0, axis=1)
    return zcr


def spectral_centroid_contour(x, sr, frame_ms=FRAME_MS, hop_ms=HOP_MS):
    """Per-frame Spectral Centroid in Hz."""
    fr = frames(x, sr, frame_ms=frame_ms, hop_ms=hop_ms)
    if len(fr) == 0:
        return np.array([0.0], dtype=np.float32)
    n_fft = fr.shape[1]
    window = np.hanning(n_fft)
    fr_win = fr * window
    spec = np.abs(np.fft.rfft(fr_win, axis=1))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    sum_spec = np.sum(spec, axis=1)
    sum_spec[sum_spec == 0] = 1e-12
    centroid = np.sum(spec * freqs[None, :], axis=1) / sum_spec
    return centroid


def get_slope(y):
    """Simple linear regression slope of 1D array y."""
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n)
    x_mean = (n - 1) / 2.0
    y_mean = np.mean(y)
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sum((x - x_mean) ** 2)
    if den == 0:
        return 0.0
    return float(num / den)
# pitch slope and energy decay added
