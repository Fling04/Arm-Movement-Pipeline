# cleanup.py
# Delete trials from a JSON by gesture, position, and trial number.
# Usage: python cleanup.py data.json --delete Gigem early 1 --delete HornsDown middle 2

import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('file', help='JSON file to clean')
parser.add_argument('--delete', nargs=3, action='append', metavar=('GESTURE', 'POSITION', 'TRIAL'),
                    help='e.g. --delete Gigem early 1')
args = parser.parse_args()

with open(args.file) as f:
    data = json.load(f)
if isinstance(data, dict):
    data = [data]

before = len(data)
to_delete = set((g, p, int(n)) for g, p, n in args.delete) if args.delete else set()

filtered = []
for t in data:
    key = (t['label'], t['position'], t['trial'])
    if key in to_delete:
        print(f"  Deleted: {t['label']} | {t['position']} | trial {t['trial']}")
    else:
        filtered.append(t)

with open(args.file, 'w') as f:
    json.dump(filtered, f)

print(f"\nRemoved {before - len(filtered)} trials. {len(filtered)} remaining → {args.file}")