import sys

import pandas as pd

path, task = sys.argv[1], sys.argv[2]
col = sys.argv[3] if len(sys.argv) > 3 else "report"
limit = int(sys.argv[4]) if len(sys.argv) > 4 else 6000
df = pd.read_csv(path)
row = df[df.task == task].iloc[0]
if col == "cols":
    for c in df.columns:
        v = str(row[c])
        print(f"--- {c} ({len(v)} chars): {v[:200]!r}")
else:
    print(str(row[col])[:limit])
