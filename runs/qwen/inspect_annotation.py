import json
import pickle
import sys

import numpy as np

d = sys.argv[1]
ann = json.load(open(f"{d}/trajectory_annotation.json"))
with open(f"{d}/trajectory_annotation.pkl", "rb") as f:
    pk = pickle.load(f)
print("labels:", len(ann))
k = next(iter(pk))
v = pk[k]
print("per-label value: list of", len(v), "trajectories")
for i, x in enumerate(v[0]):
    print(f"  field[{i}] {type(x).__name__} shape={getattr(x, 'shape', None)} "
          f"len={len(x) if hasattr(x, '__len__') else '-'} sample={str(x)[:120] if not hasattr(x, 'shape') else ''}")
n_traj = [len(v) for v in pk.values()]
steps = [len(t[0]) for v in pk.values() for t in v]
rew = [float(np.sum(t[3])) for v in pk.values() for t in v]
inits = {t[4] for v in pk.values() for t in v}
print("trajectories per label: min", min(n_traj), "median", int(np.median(n_traj)), "max", max(n_traj),
      "| total", sum(n_traj))
print("frames per trajectory: min", min(steps), "median", int(np.median(steps)), "max", max(steps))
print("return per trajectory: min", round(min(rew), 3), "median", round(float(np.median(rew)), 3), "max", round(max(rew), 3))
print("distinct init states:", len(inits))
