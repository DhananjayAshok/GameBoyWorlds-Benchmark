import pandas as pd

M = "gemma-4-31b-it"
for g in ["deja_vu_2", "legend_of_zelda_links_awakening", "legend_of_zelda_the_oracle_of_seasons"]:
    b = pd.read_csv(f"results/benchmark/{g}/subgoal_single_visual_low_level_{M}.csv", usecols=["task"]).task.tolist()
    c = pd.read_csv(f"results/benchmark/{g}/info_subgoal_retrieval_single_visual_low_level_{M}_curiosity_only.csv", usecols=["task"]).task.tolist()
    diff = [i for i in range(50) if b[i] != c[i]]
    print(f"== {g}: {len(diff)} rows differ by position; {len(set(b) & set(c))} shared task strings")
    for i in diff[:4]:
        print(f"  [{i}] base: {b[i][:110]}\n       cur:  {c[i][:110]}")
