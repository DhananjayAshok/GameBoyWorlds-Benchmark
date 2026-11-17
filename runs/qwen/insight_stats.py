import glob
import json
import os
import re

import pandas as pd

model = "gemma-4-31b-it"
CONFLICT = re.compile(r"conflict|contradict|sources (state|conflict|differ)|some sources|others state|one record", re.I)
BUTTON = re.compile(r"\b(press(ing)?|A button|B button|START|SELECT)\b")
rows = []
for d in sorted(glob.glob("results/benchmark/*/")):
    game = os.path.basename(d.rstrip("/"))
    b = f"{d}subgoal_single_visual_low_level_{model}.csv"
    c = f"{d}info_subgoal_retrieval_single_visual_low_level_{model}_curiosity_only.csv"
    if not (os.path.exists(b) and os.path.exists(c)):
        continue
    B = pd.read_csv(b, usecols=["task", "success", "n_steps", "n_attempts"])
    C = pd.read_csv(c, usecols=["task", "success", "n_steps", "n_attempts", "insights_block",
                                "n_insights_candidate", "n_insights_kept", "n_insights_distilled", "step_log"])
    for X in (B, C):
        X["success"] = X["success"].astype(str).str.lower().eq("true")
    m = B.merge(C, on="task", suffixes=("_b", "_c"))
    for _, r in m.iterrows():
        blk = str(r.insights_block) if isinstance(r.insights_block, str) else ""
        log = json.loads(r.step_log) if isinstance(r.step_log, str) else []
        worst = max((len(s["attempts"]) for s in log), default=0)
        outcome = ("lost" if r.success_b and not r.success_c else
                   "gained" if r.success_c and not r.success_b else
                   "both" if r.success_b else "neither")
        rows.append(dict(game=game, outcome=outcome,
                         served=int(bool(blk.strip())),
                         n_lines=blk.count("\n- ") + (1 if blk.strip().startswith("-") else 0),
                         conflict=int(bool(CONFLICT.search(blk))),
                         buttons=len(BUTTON.findall(blk)),
                         kept=r.n_insights_kept, cand=r.n_insights_candidate,
                         attempts_c=r.n_attempts_c, attempts_b=r.n_attempts_b,
                         worst_step_attempts=worst))
df = pd.DataFrame(rows)
g = df.groupby("outcome")
print(g.agg(n=("game", "size"), served=("served", "mean"), lines=("n_lines", "mean"),
            conflict=("conflict", "mean"), button_mentions=("buttons", "mean"),
            kept=("kept", "mean"), candidates=("cand", "mean"),
            attempts_cur=("attempts_c", "mean"), attempts_base=("attempts_b", "mean"),
            worst_step_attempts=("worst_step_attempts", "mean")).round(2).to_string())
print()
print(df.groupby("game").agg(conflict=("conflict", "mean"), lines=("n_lines", "mean")).round(2).to_string())
