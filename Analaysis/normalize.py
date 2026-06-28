# normalize_persons.py
import json
import numpy as np
import glob

for fname in glob.glob("person*.json"):
    with open(fname) as f:
        data = json.load(f)

    # compute per-person flex min/max across ALL frames
    all_flex = [[] for _ in range(5)]
    for trial in data:
        for frame in trial['frames']:
            for ch in range(5):
                all_flex[ch].append(frame['flex'][ch])

    flex_min = [np.min(all_flex[ch]) for ch in range(5)]
    flex_max = [np.max(all_flex[ch]) for ch in range(5)]
    print(f"{fname}: min={[f'{v:.3f}' for v in flex_min]} max={[f'{v:.3f}' for v in flex_max]}")

    # normalize each frame's flex against this person's range
    for trial in data:
        for frame in trial['frames']:
            for ch in range(5):
                frame['flex'][ch] = (frame['flex'][ch] - flex_min[ch]) / (flex_max[ch] - flex_min[ch] + 1e-8)

    out = fname.replace('.json', '_normalized.json')
    with open(out, 'w') as f:
        json.dump(data, f)
    print(f"  Saved {out}\n")