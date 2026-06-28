# merge_person.py
# Merges multiple raw session files for one person into a single person file.
# Usage:
#   python merge_person.py --person person1 --folder ./sessions/
#   python merge_person.py --person person1 raw1.json raw2.json raw3.json

import json
import argparse
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument('--person', required=True, help='Person ID e.g. person1')
parser.add_argument('--folder', help='Folder containing raw JSON files')
parser.add_argument('files', nargs='*', help='Individual raw JSON files')
args = parser.parse_args()

# collect files from folder or individual args
if args.folder:
    files = glob.glob(os.path.join(args.folder, "*.json"))
else:
    files = args.files

if not files:
    print("No files found.")
    exit(1)

all_trials = []
for fname in files:
    with open(fname) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    for t in data:
        t['person'] = args.person
    all_trials.extend(data)
    print(f"  {fname}: {len(data)} trials")

out = f"{args.person}.json"
with open(out, 'w') as f:
    json.dump(all_trials, f)

print(f"\nMerged {len(all_trials)} trials → {out}")