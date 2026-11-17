import csv
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

CORE = ["bomberman_quest", "bomberman_pocket", "deja_vu_1", "deja_vu_2",
        "legend_of_zelda_links_awakening", "legend_of_zelda_the_oracle_of_seasons",
        "pokemon_red", "pokemon_crystal", "sword_of_hope_1", "sword_of_hope_2"]
M = "gemma-4-31b-it"
csv.field_size_limit(sys.maxsize)
rng = np.random.default_rng(0)
B = 20000


def load(game, stem):
    df = pd.read_csv(f"results/benchmark/{game}/{stem}.csv", usecols=["task", "success"], engine="python")
    df["s"] = df["success"].astype(str).str.lower().eq("true").astype(int)
    df["k"] = df.groupby("task").cumcount()
    return df.set_index(["task", "k"])["s"]


allb, allc, gid = [], [], []
rows = []
for gi, g in enumerate(CORE):
    b_full = load(g, f"subgoal_single_visual_low_level_{M}")
    c_full = load(g, f"subgoal_oracle_world_model_low_level_{M}")
    t = b_full.index.intersection(c_full.index)
    b, c = b_full[t].values, c_full[t].values
    n = len(t)
    d = c - b
    boot = d[rng.integers(0, n, (B, n))].mean(1) * 100
    lo, hi = np.percentile(boot, [2.5, 97.5])
    gain, loss = int(((c == 1) & (b == 0)).sum()), int(((c == 0) & (b == 1)).sum())
    p = binomtest(gain, gain + loss, 0.5).pvalue if gain + loss else 1.0
    rows.append(dict(game=g, n_paired=n, base_full=b_full.mean() * 100, oracle_full=c_full.mean() * 100,
                     diff=d.mean() * 100, lo=lo, hi=hi, gained=gain, lost=loss, p=p))
    allb.append(b); allc.append(c); gid.append(np.full(n, gi))

r = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(r.round(3).to_string(index=False))

b, c, gid = np.concatenate(allb), np.concatenate(allc), np.concatenate(gid)
d = c - b
groups = [np.where(gid == i)[0] for i in range(len(CORE))]
boots = [np.mean([d[rng.choice(ix, len(ix))].mean() for ix in groups]) * 100 for _ in range(B)]
lo, hi = np.percentile(boots, [2.5, 97.5])
gain, loss = int(((c == 1) & (b == 0)).sum()), int(((c == 0) & (b == 1)).sum())
print(f"\nmacro mean diff {r['diff'].mean():.2f}  stratified paired bootstrap 95% CI [{lo:.2f}, {hi:.2f}]"
      f"  P(diff<=0)={np.mean(np.array(boots) <= 0):.4f}")
print(f"macro full-set means: baseline {r['base_full'].mean():.2f}  oracle {r['oracle_full'].mean():.2f}")
print(f"pooled tasks {len(d)}  gained {gain} lost {loss}  sign test p={binomtest(gain, gain + loss, 0.5).pvalue:.4f}")
print(f"games won {int((r['diff'] > 0).sum())} lost {int((r['diff'] < 0).sum())}")
