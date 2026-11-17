import glob
import os

import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)

CORE = ["bomberman_quest", "bomberman_pocket", "deja_vu_1", "deja_vu_2",
        "legend_of_zelda_links_awakening", "legend_of_zelda_the_oracle_of_seasons",
        "pokemon_red", "pokemon_crystal", "sword_of_hope_1", "sword_of_hope_2"]
FULL = {"pokemon_red": 49}
P = "subgoal_single_visual_low_level_"
rows = []
for game in CORE:
    for path in glob.glob(f"results/benchmark/{game}/{P}*.csv"):
        model = os.path.basename(path)[len(P):-4]
        df = pd.read_csv(path, usecols=["success"])
        sr = df["success"].astype(str).str.lower().eq("true").mean() * 100
        rows.append(dict(game=game, model=model, sr=sr, n=len(df), full=len(df) >= FULL.get(game, 50)))
r = pd.DataFrame(rows)
r["cell"] = r.sr.round(1).astype(str) + r.apply(lambda x: "" if x.full else f" ({x.n})", axis=1)
p = r.pivot_table(index="game", columns="model", values="cell", aggfunc="first").reindex(CORE).fillna("-")
srp = r[r.full].pivot_table(index="game", columns="model", values="sr", aggfunc="first").reindex(CORE)
p.loc["MEAN (complete games)"] = [f"{srp[m].mean():.1f} [{srp[m].notna().sum()}/10]" if m in srp else "-" for m in p.columns]
print(p.to_string())
