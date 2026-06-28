# merge_dataset.py
# Merges person files into train/test splits.
# Usage:
#   python merge_dataset.py --train person1.json person2.json ... --test person9.json person10.json
#   --split 0.9 controls within-subject train/test ratio (default 90/10)

import json
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--train', nargs='+', required=True, help='Person files for training')
parser.add_argument('--test',  nargs='+', default=[],   help='Person files for cross-subject test (all goes to test)')
parser.add_argument('--split', type=float, default=0.9, help='Train ratio for within-subject split (default 0.9)')
parser.add_argument('--seed',  type=int,   default=42)
args = parser.parse_args()

random.seed(args.seed)

train_trials = []
val_trials   = []
test_trials  = []

# within-subject split from --train files
for fname in args.train:
    with open(fname) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    random.shuffle(data)
    cut = int(len(data) * args.split)
    train_trials.extend(data[:cut])
    val_trials.extend(data[cut:])
    print(f"  [train] {fname}: {len(data)} trials → {cut} train, {len(data)-cut} val")

# cross-subject test — all goes to test
for fname in args.test:
    with open(fname) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    test_trials.extend(data)
    print(f"  [test]  {fname}: {len(data)} trials → all test")

with open("train.json", 'w') as f:
    json.dump(train_trials, f)
with open("val.json", 'w') as f:
    json.dump(val_trials, f)
with open("test.json", 'w') as f:
    json.dump(test_trials, f)

print(f"\nDone.")
print(f"  train.json: {len(train_trials)} trials")
print(f"  val.json:   {len(val_trials)} trials")
print(f"  test.json:  {len(test_trials)} trials")