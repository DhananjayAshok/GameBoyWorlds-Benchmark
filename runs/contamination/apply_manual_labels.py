import sys

import pandas as pd

LABELS = {
    "gemini-3.6-flash": dict(
        official_wrong={69, 229},
        official_partial={82, 88, 285},
        fan_right={103, 108, 109, 119, 131, 143, 152, 176, 195, 199,
                   301, 302, 303, 308, 309, 310, 311, 314, 315, 316, 318, 319, 320, 352, 365,
                   375, 376, 381, 393, 395, 399},
        fan_partial={134, 137, 141, 151, 169, 189,
                     313, 380, 387, 392, 396},
    ),
    "gemini-3.1-pro-preview": dict(
        official_wrong={64, 207},
        official_partial={82, 208, 285},
        fan_right={103, 108, 109, 110, 114, 117, 119, 122, 130, 131, 140, 143, 155, 159, 165,
                   170, 176, 178, 195, 199,
                   300, 303, 308, 309, 310, 311, 314, 315, 316, 318, 319, 339, 344, 347, 351,
                   352, 361, 366, 372, 375, 381, 386, 393, 395, 399},
        fan_partial={142, 150, 151, 152, 189,
                     312, 325, 336, 354, 356, 374, 377, 392, 396, 398},
    ),
}

path = sys.argv[1]
df = pd.read_csv(path)
col = [c for c in df.columns if c.startswith("actual_answer_") and "_evaluated_" not in c][0]
sets = LABELS[col[len("actual_answer_"):]]

labels = []
for i in range(len(df)):
    official = i < 100 or 200 <= i < 300
    if official:
        lab = ("Wrong" if i in sets["official_wrong"]
               else "Partially Right" if i in sets["official_partial"] else "Right")
    else:
        lab = ("Right" if i in sets["fan_right"]
               else "Partially Right" if i in sets["fan_partial"] else "Wrong")
    labels.append(lab)
df[f"{col}_evaluated_claude_annotation"] = labels
df.to_csv(path, index=False)

df["right"] = [l == "Right" for l in labels]
df["credit"] = [1.0 if l == "Right" else 0.5 if l == "Partially Right" else 0.0 for l in labels]
df["official"] = df["game"].isin(["Pokémon Red", "Pokémon Crystal"])
print(df.groupby("game", sort=False)[["right", "credit"]].mean().round(3).to_string())
print(df.groupby("official")[["right", "credit"]].mean().round(3).to_string())
print("overall strict", round(df["right"].mean(), 4), "partial", round(df["credit"].mean(), 4))
print(pd.Series(labels).value_counts().to_string())
