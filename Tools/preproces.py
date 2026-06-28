import json
import sys

def filter_flex(frames, max_delta=0.05):
    last_valid = [None] * 5
    for f in frames:
        for ch in range(5):
            v = f['flex'][ch]
            if last_valid[ch] is not None and abs(v - last_valid[ch]) > max_delta:
                f['flex'][ch] = last_valid[ch]  # reject and replace
            else:
                last_valid[ch] = v  # accept
    return frames

def downsample_frames(frames, group=5):
    downsampled = []
    for i in range(0, len(frames) - group + 1, group):
        g = frames[i:i + group]
        mid = g[2]
        new_frame = {
            'ts':   mid['ts'],
            'acc':  [sum(f['acc'][ch]  for f in g) / group for ch in range(3)],
            'gyr':  [sum(f['gyr'][ch]  for f in g) / group for ch in range(3)],
            'flex': [sum(f['flex'][ch] for f in g) / group for ch in range(5)]
        }
        downsampled.append(new_frame)
    return downsampled

def preprocess(input_file, output_file):
    with open(input_file) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]

    for trial in data:
        before = len(trial['frames'])
        trial['frames'] = filter_flex(trial['frames'])       # 1. filter
        trial['frames'] = downsample_frames(trial['frames']) # 2. downsample
        after = len(trial['frames'])
        print(f"  [{trial['label']} | {trial['position']} | trial {trial['trial']}] "
              f"→ {before} → {after} frames")

    with open(output_file, 'w') as f:
        json.dump(data, f)
    print(f"\nDone → saved to {output_file}")

if __name__ == "__main__":
    input_file  = sys.argv[1] if len(sys.argv) > 1 else "6.json"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "Charles3_pre.json"
    preprocess(input_file, output_file)