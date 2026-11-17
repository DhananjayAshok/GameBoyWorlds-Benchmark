import glob
import os

import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 500)

CORE = ["bomberman_quest", "bomberman_pocket", "deja_vu_1", "deja_vu_2",
        "legend_of_zelda_links_awakening", "legend_of_zelda_the_oracle_of_seasons",
        "pokemon_red", "pokemon_crystal", "sword_of_hope_1", "sword_of_hope_2"]


def load(path):
    df = pd.read_csv(path, usecols=["success", "n_steps"])
    s = df["success"].astype(str).str.lower().eq("true")
    return len(df), s.mean() * 100, int(df["n_steps"].max())


rows = []
for path in glob.glob("results/benchmark/*/*.csv"):
    game = os.path.basename(os.path.dirname(path))
    stem = os.path.basename(path)[:-4]
    for prefix, arm in (("info_subgoal_retrieval_single_visual_low_level_", "curiosity"),
                        ("subgoal_single_visual_low_level_", "baseline")):
        if not stem.startswith(prefix):
            continue
        model = stem[len(prefix):]
        if arm == "curiosity":
            if not model.endswith("_curiosity_only"):
                continue
            model = model[:-len("_curiosity_only")]
        if "_" in model.replace("-", ""):
            continue
        n, sr, mx = load(path)
        rows.append(dict(game=game, model=model, arm=arm, n=n, sr=sr, max_steps_seen=mx))

r = pd.DataFrame(rows)
cur_models = sorted(r[r.arm == "curiosity"].model.unique())
r = r[r.model.isin(cur_models)]
r["cell"] = r.sr.round(1).astype(str) + " (" + r.n.astype(str) + ", <=" + r.max_steps_seen.astype(str) + ")"
p = r.pivot_table(index="game", columns=["model", "arm"], values="cell", aggfunc="first")
order = [g for g in CORE if g in p.index] + [g for g in p.index if g not in CORE]
print("success % (n tasks, max n_steps seen)")
print(p.reindex(order).fillna("-").to_string())

piv = r.pivot_table(index="game", columns=["model", "arm"], values="sr", aggfunc="first")
print("\nmean over games where both arms exist:")
for m in cur_models:
    if (m, "curiosity") not in piv or (m, "baseline") not in piv:
        continue
    both = piv[[(m, "curiosity"), (m, "baseline")]].dropna()
    core = both[both.index.isin(CORE)]
    print(f"  {m}: all {len(both)} games  curiosity {both.iloc[:,0].mean():.1f}  baseline {both.iloc[:,1].mean():.1f}"
          f" | core {len(core)} games  curiosity {core.iloc[:,0].mean():.1f}  baseline {core.iloc[:,1].mean():.1f}")
