import glob
import os
import sys

import pandas as pd

model = sys.argv[1] if len(sys.argv) > 1 else "gemma-4-31b-it"
rows = []
for d in sorted(glob.glob("results/benchmark/*/")):
    game = os.path.basename(d.rstrip("/"))
    b = f"{d}subgoal_single_visual_low_level_{model}.csv"
    c = f"{d}info_subgoal_retrieval_single_visual_low_level_{model}_curiosity_only.csv"
    if not (os.path.exists(b) and os.path.exists(c)):
        continue
    B = pd.read_csv(b, usecols=["task", "success", "n_steps"])
    C = pd.read_csv(c, usecols=["task", "success", "n_steps", "selected_entry_ids", "n_supervisor_calls"])
    for X in (B, C):
        X["success"] = X["success"].astype(str).str.lower().eq("true")
    m = B.merge(C, on="task", suffixes=("_b", "_c"))
    lost = m[m.success_b & ~m.success_c]
    gained = m[~m.success_b & m.success_c]
    rows.append(dict(game=game, n=len(m), lost=len(lost), gained=len(gained),
                     both=int((m.success_b & m.success_c).sum()),
                     steps_lost_c=lost.n_steps_c.mean(), steps_lost_b=lost.n_steps_b.mean()))
    if len(sys.argv) > 2 and game == sys.argv[2]:
        pd.set_option("display.width", 250)
        pd.set_option("display.max_colwidth", 120)
        print(lost[["task", "n_steps_b", "n_steps_c", "selected_entry_ids"]].to_string())
r = pd.DataFrame(rows)
print(r.round(1).to_string(index=False))
print("total lost", r.lost.sum(), "gained", r.gained.sum())
