# feature_extract.py
# Windows trials, extracts rich features, saves numpy arrays for train.py

import json
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ── Config ────────────────────────────────────────────
WINDOW_SIZE = 80
STRIDE      = 40
N_CHANNELS  = 11  # ax ay az gx gy gz f0 f1 f2 f3 f4

# ── Windowing ─────────────────────────────────────────
def extract_windows(records):
    X, y = [], []
    for trial in records:
        frames = trial['frames']
        label  = trial['label']
        if len(frames) < WINDOW_SIZE:
            print(f"  Skipping short trial: {label} | {trial['position']} | {len(frames)} frames")
            continue
        for start in range(0, len(frames) - WINDOW_SIZE + 1, STRIDE):
            window = frames[start:start + WINDOW_SIZE]
            sig = np.array([
                f['acc'] + f['gyr'] + f['flex']
                for f in window
            ], dtype=np.float32)
            X.append(sig)
            y.append(label)
    return np.array(X), np.array(y)

# ── Feature extraction ────────────────────────────────
def extract_features(X):
    feats = []
    for window in X:
        f = []
        acc  = window[:, 0:3]   # ax ay az
        gyr  = window[:, 3:6]   # gx gy gz
        flex = window[:, 6:11]  # f0 f1 f2 f3 f4

        # ── Per channel stats (all 11 channels) ───────
        for ch in range(window.shape[1]):
            ch_data = window[:, ch]
            f.append(np.mean(ch_data))
            f.append(np.std(ch_data))
            f.append(np.max(ch_data))
            f.append(np.min(ch_data))
            f.append(np.sqrt(np.mean(ch_data**2)))                   # RMS
            f.append(np.sum(np.abs(np.diff(np.sign(ch_data)))) / 2)  # zero crossings
            f.append(np.max(ch_data) - np.min(ch_data))              # peak to peak
            f.append(np.sum(np.abs(ch_data)))                        # SMA per channel

        # ── Jerk — derivative of acceleration ─────────
        jerk = np.diff(acc, axis=0)
        for ch in range(3):
            f.append(np.mean(np.abs(jerk[:, ch])))
            f.append(np.std(jerk[:, ch]))
            f.append(np.max(np.abs(jerk[:, ch])))

        # ── Angular jerk — derivative of gyro ─────────
        ang_jerk = np.diff(gyr, axis=0)
        for ch in range(3):
            f.append(np.mean(np.abs(ang_jerk[:, ch])))
            f.append(np.std(ang_jerk[:, ch]))
            f.append(np.max(np.abs(ang_jerk[:, ch])))

        # ── Flex derivative per finger ─────────────────
        flex_diff = np.diff(flex, axis=0)
        for ch in range(5):
            f.append(np.mean(np.abs(flex_diff[:, ch])))  # mean flex rate
            f.append(np.max(np.abs(flex_diff[:, ch])))   # max flex rate
            f.append(np.std(flex_diff[:, ch]))            # std flex rate

        # ── Time to peak per finger ────────────────────
        for ch in range(5):
            f.append(np.argmax(flex[:, ch]) / WINDOW_SIZE)  # normalized time to peak
            f.append(np.argmin(flex[:, ch]) / WINDOW_SIZE)  # normalized time to min

        # ── Finger correlation ─────────────────────────
        for i in range(5):
            for j in range(i+1, 5):
                corr = np.corrcoef(flex[:, i], flex[:, j])[0, 1]
                f.append(0.0 if np.isnan(corr) else corr)

        # ── Tilt angles from accelerometer ────────────
        ax, ay, az = acc[:, 0], acc[:, 1], acc[:, 2]
        roll  = np.arctan2(ay, az)
        pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
        f.append(np.mean(roll));  f.append(np.std(roll))
        f.append(np.mean(pitch)); f.append(np.std(pitch))

        # ── SMA across acc and gyro axes ──────────────
        f.append(np.sum(np.abs(acc)) / WINDOW_SIZE)   # acc SMA
        f.append(np.sum(np.abs(gyr)) / WINDOW_SIZE)   # gyro SMA

        # ── Dominant frequency + spectral energy ──────
        for ch in range(N_CHANNELS):
            fft_vals = np.abs(np.fft.rfft(window[:, ch]))
            freqs    = np.fft.rfftfreq(WINDOW_SIZE)
            f.append(freqs[np.argmax(fft_vals)])        # dominant frequency
            f.append(np.sum(fft_vals**2))               # spectral energy

        # ── Flex-IMU correlation ───────────────────────
        acc_mag = np.sqrt(np.sum(acc**2, axis=1))
        gyr_mag = np.sqrt(np.sum(gyr**2, axis=1))
        for ch in range(5):
            corr_acc = np.corrcoef(flex[:, ch], acc_mag)[0, 1]
            corr_gyr = np.corrcoef(flex[:, ch], gyr_mag)[0, 1]
            f.append(0.0 if np.isnan(corr_acc) else corr_acc)
            f.append(0.0 if np.isnan(corr_gyr) else corr_gyr)

        # ── Onset timing — which sensor moves first ────
        acc_onset  = np.argmax(np.abs(np.diff(acc_mag))  > np.std(acc_mag))  / WINDOW_SIZE
        gyr_onset  = np.argmax(np.abs(np.diff(gyr_mag))  > np.std(gyr_mag))  / WINDOW_SIZE
        flex_onset = np.argmax(np.abs(np.diff(flex[:, 0])) > np.std(flex[:, 0])) / WINDOW_SIZE
        f.append(acc_onset)
        f.append(gyr_onset)
        f.append(flex_onset)
        f.append(acc_onset - flex_onset)   # did wrist or fingers move first
        f.append(gyr_onset - flex_onset)

        feats.append(f)

    return np.array(feats, dtype=np.float32)

# ── Load ──────────────────────────────────────────────
def load_json(fname):
    with open(fname) as f:
        data = json.load(f)
    return [data] if isinstance(data, dict) else data

# ── Main ──────────────────────────────────────────────
def main():
    splits = {
        'train': load_json('train.json'),
        'val':   load_json('val.json'),
        'test':  load_json('test.json'),
    }

    # fit label encoder on train only
    le = LabelEncoder()
    _, y_train_raw = extract_windows(splits['train'])
    le.fit(y_train_raw)
    print(f"Classes ({len(le.classes_)}): {list(le.classes_)}")
    np.save('label_classes.npy', le.classes_)

    for split, records in splits.items():
        print(f"\n── {split} ──")
        X_seq, y_raw = extract_windows(records)
        X_feat = extract_features(X_seq)
        y = le.transform(y_raw)

        np.save(f'X_{split}_seq.npy',  X_seq)
        np.save(f'X_{split}_feat.npy', X_feat)
        np.save(f'y_{split}.npy',      y)

        print(f"  Windows:    {len(X_seq)}")
        print(f"  Seq shape:  {X_seq.shape}")
        print(f"  Feat shape: {X_feat.shape}")
        print(f"  Saved X_{split}_seq.npy, X_{split}_feat.npy, y_{split}.npy")

    print(f"\nTotal features per window: {X_feat.shape[1]}")
    print("Done. Ready to run train.py")

if __name__ == "__main__":
    main()