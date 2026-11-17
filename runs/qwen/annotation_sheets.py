import json
import os
import pickle
import random
import sys

import numpy as np
from PIL import Image, ImageDraw

d, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
os.makedirs(out, exist_ok=True)
ann = json.load(open(f"{d}/trajectory_annotation.json"))
with open(f"{d}/trajectory_annotation.pkl", "rb") as f:
    pk = pickle.load(f)
keys = sorted(pk, key=lambda k: int(k))
random.Random(seed).shuffle(keys)
for k in keys[:n]:
    obs, _, hla, rew, init = pk[k][0]
    idx = sorted({0, len(obs) // 3, 2 * len(obs) // 3, len(obs) - 1})
    tiles = [Image.fromarray(np.asarray(obs[i]).astype(np.uint8)).convert("L").resize((320, 288), Image.NEAREST)
             for i in idx]
    sheet = Image.new("L", (330 * len(tiles), 288 + 40), 255)
    for j, t in enumerate(tiles):
        sheet.paste(t, (330 * j, 40))
    draw = ImageDraw.Draw(sheet)
    acts = ",".join(str(a[0]) for a in np.asarray(hla).reshape(len(hla), -1))
    draw.text((5, 5), f"#{k} [{init}] label: {ann[str(k)]}  frames {idx} of {len(obs)}", fill=0)
    draw.text((5, 22), f"actions: {acts[:150]}", fill=0)
    sheet.save(f"{out}/seg_{k}.png")
    print(f"{out}/seg_{k}.png\t{ann[str(k)]}\t{init}\t{len(obs)} frames\treturn {float(np.sum(rew)):.2f}")
