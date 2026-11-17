import glob
import gzip
import os
import pickle
import sys

import pandas as pd

from python_scripts.paths import REPORT_FILENAME

csv = sys.argv[1]
df = pd.read_csv(csv)
print(df[["task", "success", "n_steps", "n_invalid", "n_attempts"]].to_string(index=False))


def walk(obj, seen):
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if type(obj).__name__ in ("WorldModelDecision", "EnvironmentStepRecord"):
        yield obj
        return
    if isinstance(obj, dict):
        for x in obj.values():
            yield from walk(x, seen)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            yield from walk(x, seen)
    elif hasattr(obj, "__dict__"):
        for x in vars(obj).values():
            yield from walk(x, seen)


for dirs in df["session_dirs"]:
    for d in eval(dirs) if isinstance(dirs, str) and dirs.startswith("[") else [dirs]:
        path = os.path.join(d, REPORT_FILENAME)
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rb") as f:
            report = pickle.load(f)
        found = list(walk(report, set()))
        print("  env step records:", sum(type(x).__name__ == "EnvironmentStepRecord" for x in found))
        decs = [x for x in found if type(x).__name__ == "WorldModelDecision" and x.choice is not None]
        exact = sum(x.predicted_pixel_error == 0.0 for x in decs)
        print(f"{os.path.basename(d.rstrip('/'))}: decisions {len(decs)}  exact prediction {exact}  "
              f"n_candidates {sorted({len(x.candidates) for x in decs})}  "
              f"labels {decs[0].candidates if decs else None}")
