import sys

import pandas as pd

df = pd.read_csv(sys.argv[1])
col = [c for c in df.columns if c.startswith("actual_answer_")][0]
start, end = int(sys.argv[2]), int(sys.argv[3])
print("leaked_reasoning:", int(df[col].str.contains("Reasoning:", na=False).sum()))
for i in range(start, min(end, len(df))):
    r = df.iloc[i]
    print(f"[{i}] {r['game']} | Q: {r['question']}\n  EXP: {r['expected_answer']}\n  ANS: {r[col]}")
