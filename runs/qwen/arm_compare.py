import os

import pandas as pd

M = "gemma-4-31b-it"
ARMS = {
    "baseline": "subgoal_single_visual_low_level_{m}",
    "curiosity": "info_subgoal_retrieval_single_visual_low_level_{m}_curiosity_only",
    "zeroshot": "info_subgoal_retrieval_single_visual_low_level_{m}_zeroshot_only",
    "both": "info_subgoal_retrieval_single_visual_low_level_{m}_both",
    "parametric": "info_subgoal_parametric_single_visual_low_level_{m}",
}
rows = []
for game in sorted(os.listdir("results/benchmark")):
    row = {"game": game}
    for arm, stem in ARMS.items():
        p = f"results/benchmark/{game}/{stem.format(m=M)}.csv"
        if os.path.exists(p):
            df = pd.read_csv(p, usecols=["success"])
            row[arm] = df["success"].astype(str).str.lower().eq("true").mean() * 100
            row[arm + "_n"] = len(df)
    if "baseline" in row and "curiosity" in row:
        rows.append(row)
r = pd.DataFrame(rows).set_index("game")
arms = [a for a in ARMS if a in r]
d = r[arms].copy()
for a in arms[1:]:
    d["d_" + a] = d[a] - d["baseline"]
pd.set_option("display.width", 250)
print(d.round(1).to_string())
print()
for a in arms[1:]:
    sub = d.dropna(subset=[a])
    print(f"{a:11s} games={len(sub):2d}  mean delta={sub['d_'+a].mean():+.1f}  "
          f"worse={int((sub['d_'+a] < 0).sum())}  better={int((sub['d_'+a] > 0).sum())}")
both = d.dropna(subset=["parametric"])
print("\ncorr(d_curiosity, d_parametric) over", len(both), "games:",
      round(both["d_curiosity"].corr(both["d_parametric"]), 2))
z = d.dropna(subset=["zeroshot"])
print("corr(d_curiosity, d_zeroshot) over", len(z), "games:", round(z["d_curiosity"].corr(z["d_zeroshot"]), 2))
print(r[[c for c in r.columns if c.endswith("_n")]].to_string())
