# main.py
# Record a sample, preprocess, window, extract features, infer

import json
import pickle
import numpy as np
import sys

# ── Config ────────────────────────────────────────────
MODEL_FILE   = "cnn_model.keras"
INPUT_FILE   = sys.argv[1] if len(sys.argv) > 1 else "1.json"
WINDOW_SIZE  = 80
STRIDE       = 40
N_CHANNELS   = 11
MAX_DELTA    = 0.05
DOWNSAMPLE   = 5
ACC_THRESH   = 0.1
FLEX_THRESH  = 0.05

# ── Preprocessing ─────────────────────────────────────
def filter_flex(frames):
    last_valid = [None] * 5
    for f in frames:
        for ch in range(5):
            v = f['flex'][ch]
            if last_valid[ch] is not None and abs(v - last_valid[ch]) > MAX_DELTA:
                f['flex'][ch] = last_valid[ch]
            else:
                last_valid[ch] = v
    return frames

def downsample_frames(frames, group=DOWNSAMPLE):
    out = []
    for i in range(0, len(frames) - group + 1, group):
        g = frames[i:i + group]
        mid = g[group // 2]
        out.append({
            'ts':   mid['ts'],
            'acc':  [sum(f['acc'][ch]  for f in g) / group for ch in range(3)],
            'gyr':  [sum(f['gyr'][ch]  for f in g) / group for ch in range(3)],
            'flex': [sum(f['flex'][ch] for f in g) / group for ch in range(5)]
        })
    return out

# ── Windowing ─────────────────────────────────────────
def extract_windows(frames):
    windows = []
    for start in range(0, len(frames) - WINDOW_SIZE + 1, STRIDE):
        window = frames[start:start + WINDOW_SIZE]
        sig = np.array([
            f['acc'] + f['gyr'] + f['flex']
            for f in window
        ], dtype=np.float32)
        windows.append(sig)
    return np.array(windows)

# ── Activity detection ────────────────────────────────
def is_active(window):
    acc_mag    = np.sqrt(np.sum(window[:, 0:3]**2, axis=1))
    flex_delta = np.max(window[:, 6:11]) - np.min(window[:, 6:11])
    return np.std(acc_mag) > ACC_THRESH or flex_delta > FLEX_THRESH

# ── Feature extraction ────────────────────────────────
def extract_features(X):
    feats = []
    for window in X:
        f = []
        flex     = window[:, 6:11].copy()
        flex_min = flex.min(axis=0)
        flex_max = flex.max(axis=0)
        flex     = (flex - flex_min) / (flex_max - flex_min + 1e-8)
        window   = window.copy()
        window[:, 6:11] = flex

        acc  = window[:, 0:3]
        gyr  = window[:, 3:6]
        flex = window[:, 6:11]

        for ch in range(window.shape[1]):
            ch_data = window[:, ch]
            f.append(np.mean(ch_data))
            f.append(np.std(ch_data))
            f.append(np.max(ch_data))
            f.append(np.min(ch_data))
            f.append(np.sqrt(np.mean(ch_data**2)))
            f.append(np.sum(np.abs(np.diff(np.sign(ch_data)))) / 2)
            f.append(np.max(ch_data) - np.min(ch_data))
            f.append(np.sum(np.abs(ch_data)))

        jerk = np.diff(acc, axis=0)
        for ch in range(3):
            f.append(np.mean(np.abs(jerk[:, ch])))
            f.append(np.std(jerk[:, ch]))
            f.append(np.max(np.abs(jerk[:, ch])))

        ang_jerk = np.diff(gyr, axis=0)
        for ch in range(3):
            f.append(np.mean(np.abs(ang_jerk[:, ch])))
            f.append(np.std(ang_jerk[:, ch]))
            f.append(np.max(np.abs(ang_jerk[:, ch])))

        flex_diff = np.diff(flex, axis=0)
        for ch in range(5):
            f.append(np.mean(np.abs(flex_diff[:, ch])))
            f.append(np.max(np.abs(flex_diff[:, ch])))
            f.append(np.std(flex_diff[:, ch]))

        for ch in range(5):
            f.append(np.argmax(flex[:, ch]) / WINDOW_SIZE)
            f.append(np.argmin(flex[:, ch]) / WINDOW_SIZE)

        for i in range(5):
            for j in range(i+1, 5):
                corr = np.corrcoef(flex[:, i], flex[:, j])[0, 1]
                f.append(0.0 if np.isnan(corr) else corr)

        ax, ay, az = acc[:, 0], acc[:, 1], acc[:, 2]
        roll  = np.arctan2(ay, az)
        pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
        f.append(np.mean(roll));  f.append(np.std(roll))
        f.append(np.mean(pitch)); f.append(np.std(pitch))

        f.append(np.sum(np.abs(acc)) / WINDOW_SIZE)
        f.append(np.sum(np.abs(gyr)) / WINDOW_SIZE)

        for ch in range(N_CHANNELS):
            fft_vals = np.abs(np.fft.rfft(window[:, ch]))
            freqs    = np.fft.rfftfreq(WINDOW_SIZE)
            f.append(freqs[np.argmax(fft_vals)])
            f.append(np.sum(fft_vals**2))

        acc_mag = np.sqrt(np.sum(acc**2, axis=1))
        gyr_mag = np.sqrt(np.sum(gyr**2, axis=1))
        for ch in range(5):
            corr_acc = np.corrcoef(flex[:, ch], acc_mag)[0, 1]
            corr_gyr = np.corrcoef(flex[:, ch], gyr_mag)[0, 1]
            f.append(0.0 if np.isnan(corr_acc) else corr_acc)
            f.append(0.0 if np.isnan(corr_gyr) else corr_gyr)

        acc_onset  = np.argmax(np.abs(np.diff(acc_mag))    > np.std(acc_mag))    / WINDOW_SIZE
        gyr_onset  = np.argmax(np.abs(np.diff(gyr_mag))    > np.std(gyr_mag))    / WINDOW_SIZE
        flex_onset = np.argmax(np.abs(np.diff(flex[:, 0])) > np.std(flex[:, 0])) / WINDOW_SIZE
        f.append(acc_onset);  f.append(gyr_onset);  f.append(flex_onset)
        f.append(acc_onset - flex_onset)
        f.append(gyr_onset  - flex_onset)

        feats.append(f)
    return np.array(feats, dtype=np.float32)

# ── Main ──────────────────────────────────────────────
def main():
    # load model
    with open(MODEL_FILE, 'rb') as f:
        bundle = pickle.load(f)
    svm     = bundle['svm']
    scaler  = bundle['scaler']
    classes = bundle['classes']

    # load sample
    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]

    for trial in data:
        label  = trial.get('label', 'unknown')
        frames = trial['frames']
        print(f"\nTrial: {label} | {len(frames)} raw frames")

        # preprocess
        frames = filter_flex(frames)
        frames = downsample_frames(frames)
        print(f"After downsample: {len(frames)} frames")

        if len(frames) < WINDOW_SIZE:
            print("  Too short to window — skipping")
            continue

        # window
        windows = extract_windows(frames)
        print(f"Windows: {len(windows)}")

        # filter inactive windows
        active = [w for w in windows if is_active(w)]
        print(f"Active windows: {len(active)}")

        if not active:
            print("  No active windows detected — idle sample")
            continue

        # feature extract
        X = extract_features(np.array(active))
        X = scaler.transform(X)

        # infer — majority vote across all active windows
        preds  = svm.predict(X)
        probas = svm.predict_proba(X)

        # vote
        votes = {}
        for p in preds:
            votes[classes[p]] = votes.get(classes[p], 0) + 1
        winner     = max(votes, key=votes.get)
        confidence = votes[winner] / len(preds) * 100
        avg_proba  = probas.mean(axis=0)
        top_prob   = avg_proba.max() * 100

        print(f"\n  ► Predicted: {winner}")
        print(f"    Vote confidence: {confidence:.1f}% ({votes[winner]}/{len(preds)} windows)")
        print(f"    Avg probability: {top_prob:.1f}%")
        if label != 'unknown':
            print(f"    True label:  {label}")
            print(f"    Correct:     {'✓' if winner == label else '✗'}")

if __name__ == '__main__':
    main()