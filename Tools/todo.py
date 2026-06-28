# swap_labels.py
# Swaps two gesture labels in a JSON file.
# Usage: python swap_labels.py data.json Gigem HornsDown

import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('file', help='JSON file')
parser.add_argument('label_a', help='First gesture label')
parser.add_argument('label_b', help='Second gesture label')
args = parser.parse_args()

with open(args.file) as f:
    data = json.load(f)
if isinstance(data, dict):
    data = [data]

count_a = count_b = 0
for t in data:
    if t['label'] == args.label_a:
        t['label'] = args.label_b
        count_a += 1
    elif t['label'] == args.label_b:
        t['label'] = args.label_a
        count_b += 1

with open(args.file, 'w') as f:
    json.dump(data, f)

print(f"Swapped {count_a} {args.label_a}→{args.label_b} and {count_b} {args.label_b}→{args.label_a} → {args.file}")