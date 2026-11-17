import csv
import os
import sys

import pandas as pd

csv.field_size_limit(sys.maxsize)

GAMES = ["bomberman_quest", "bomberman_pocket", "deja_vu_1", "deja_vu_2",
         "legend_of_zelda_links_awakening", "legend_of_zelda_the_oracle_of_seasons",
         "pokemon_red", "pokemon_crystal", "sword_of_hope_1", "sword_of_hope_2"]
MODELS = {"Qwen3-VL-2B": "qwen3-vl-2b-instruct", "Qwen3-VL-4B": "qwen3-vl-4b-instruct",
          "Qwen3-VL-8B": "qwen3-vl-8b-instruct", "Qwen3-VL-32B": "qwen3-vl-32b-instruct"}
ARMS = {
    "baseline": "subgoal_single_visual_low_level_{m}.csv",
    "world modelling": None,
    "autonomous skill acquisition": None,
    "curiosity based document guide": "info_subgoal_retrieval_single_visual_low_level_{m}_curiosity_only.csv",
}


def n_tasks(game):
    ref = f"results/benchmark/{game}/subgoal_single_visual_low_level_gemma-4-31b-it.csv"
    return len(pd.read_csv(ref, usecols=["task"], engine="python"))


def score(path, total):
    if not os.path.exists(path):
        return ""
    d = pd.read_csv(path, usecols=["success"], engine="python", on_bad_lines="skip")
    if len(d) < total:
        return ""
    return round(100 * d.success.astype(float).mean(), 1)


rows = []
for label, m in MODELS.items():
    for g in GAMES:
        total = n_tasks(g)
        row = {"model": label, "game": g}
        for arm, pattern in ARMS.items():
            row[arm] = "" if pattern is None else score(f"results/benchmark/{g}/{pattern.format(m=m)}", total)
        rows.append(row)

out = "results/rebuttal_qwen.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(open(out).read())
