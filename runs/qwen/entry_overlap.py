import glob
import json
import os

import pandas as pd

M = "gemma-4-31b-it"
tot_lost_entries = tot_shared = 0
per = []
for d in sorted(glob.glob("results/benchmark/*/")):
    game = os.path.basename(d.rstrip("/"))
    b = f"{d}subgoal_single_visual_low_level_{M}.csv"
    c = f"{d}info_subgoal_retrieval_single_visual_low_level_{M}_curiosity_only.csv"
    if not (os.path.exists(b) and os.path.exists(c)):
        continue
    B = pd.read_csv(b, usecols=["task", "success"])
    C = pd.read_csv(c, usecols=["task", "success", "selected_entry_ids"])
    for X in (B, C):
        X["success"] = X["success"].astype(str).str.lower().eq("true")
    m = B.merge(C, on="task", suffixes=("_b", "_c"))
    ids = m.selected_entry_ids.apply(lambda s: set(json.loads(s)) if isinstance(s, str) else set())
    lost = ids[m.success_b & ~m.success_c]
    won = ids[m.success_c]
    won_pool = set().union(*won) if len(won) else set()
    for s in lost:
        tot_lost_entries += len(s)
        tot_shared += len(s & won_pool)
    if len(lost):
        lost_pool = set().union(*lost)
        per.append(dict(game=game, lost_tasks=len(lost), distinct_entries_on_lost=len(lost_pool),
                        also_served_on_solved=len(lost_pool & won_pool)))
print(pd.DataFrame(per).to_string(index=False))
print(f"\nentry servings on lost tasks: {tot_lost_entries}; of these, entry also served on a task the guide run solved: "
      f"{tot_shared} ({100 * tot_shared / max(tot_lost_entries, 1):.0f}%)")
