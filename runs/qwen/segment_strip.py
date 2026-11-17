import json
import os
import pickle
import sys

import numpy as np
from PIL import Image, ImageDraw

NAMES = ["DOWN", "LEFT", "RIGHT", "UP", "A", "B", "START"]
d, out = sys.argv[1], sys.argv[2]
keys = sys.argv[3:]
os.makedirs(out, exist_ok=True)
ann = json.load(open(f"{d}/trajectory_annotation.json"))
with open(f"{d}/trajectory_annotation.pkl", "rb") as f:
    pk = pickle.load(f)
for k in keys:
    obs, acts, hla, rew, init = pk[int(k) if int(k) in pk else k][0]
    acts = [NAMES[int(a)] if 0 <= int(a) < len(NAMES) else str(int(a)) for a in np.asarray(acts).reshape(-1)]
    n = len(obs)
    idx = list(range(n)) if n <= 12 else sorted({int(round(x)) for x in np.linspace(0, n - 1, 12)})
    W, H = 160, 144
    cols = 6
    rows = (len(idx) + cols - 1) // cols
    sheet = Image.new("L", (cols * (W + 6), rows * (H + 22) + 20), 255)
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 4), f"#{k} [{init}] label: {ann[str(k)]}", fill=0)
    for j, i in enumerate(idx):
        x, y = (j % cols) * (W + 6), 20 + (j // cols) * (H + 22)
        sheet.paste(Image.fromarray(np.asarray(obs[i]).astype(np.uint8)).convert("L"), (x, y))
        nxt = acts[i] if i < len(acts) else "end"
        draw.text((x + 2, y + H + 4), f"f{i} -> {nxt}", fill=0)
    sheet.save(f"{out}/strip_{k}.png")
    print(f"{out}/strip_{k}.png  {ann[str(k)]}  actions: {' '.join(acts)}")
