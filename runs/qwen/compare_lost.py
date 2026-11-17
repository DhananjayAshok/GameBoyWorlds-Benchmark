import json
import sys

import pandas as pd

game = sys.argv[1]
tasks = sys.argv[2:]
model = "gemma-4-31b-it"
C = pd.read_csv(f"results/benchmark/{game}/info_subgoal_retrieval_single_visual_low_level_{model}_curiosity_only.csv")
B = pd.read_csv(f"results/benchmark/{game}/subgoal_single_visual_low_level_{model}.csv")
for t in tasks:
    c = C[C.task == t].iloc[0]
    b = B[B.task == t].iloc[0]
    print("=" * 100)
    print("TASK:", t, "| baseline steps", b.n_steps, "| curiosity steps", c.n_steps,
          "| kept", c.n_insights_kept, "/", c.n_insights_candidate, "| replans", c.n_replans)
    print("SERVED:\n" + str(c.insights_block))
    print("CUR PLAN:", str(c.original_plan).replace("[STEP]", " || "))
    print("BASE PLAN:", str(b.original_plan).replace("[STEP]", " || "))
    log = json.loads(c.step_log)
    for s in log:
        atts = s["attempts"]
        cleared = sum(str(a.get("cleared")) == "True" for a in atts)
        diag = [a.get("diagnosis") for a in atts if a.get("diagnosis")]
        print(f"  STEP ({len(atts)} attempts, cleared {cleared}): {s['step'][:140]}")
        for d in diag[:2]:
            print("     diag:", str(d)[:220])
