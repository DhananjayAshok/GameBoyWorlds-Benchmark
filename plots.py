import math
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.image import BboxImage
from matplotlib.legend_handler import HandlerBase
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import Bbox, TransformedBbox
from PIL import Image, ImageDraw, ImageFont

from python_scripts import paths
from utils import load_parameters, log_error, log_info

PARAMETERS = load_parameters()
RESULTS_DIR = PARAMETERS["results_dir"]

ARM = {"supervisor": "subgoal", "executor": "single_visual", "controller_variant": "low_level"}

LOGOS = os.path.join(RESULTS_DIR, "figures", "fancy", "logos")
OUT = os.path.join(RESULTS_DIR, "figures")
CONTAMINATION_DIR = os.path.join(RESULTS_DIR, "contamination")

GAMES = [
    "bomberman_pocket", "bomberman_quest",
    "deja_vu_1", "deja_vu_2",
    "legend_of_zelda_links_awakening", "legend_of_zelda_the_oracle_of_seasons",
    "pokemon_red", "pokemon_crystal",
    "sword_of_hope_1", "sword_of_hope_2",
]

FRONTIER_MODELS = [
    ("gemini-3.6-flash", "Gemini 3.6 Flash", "gemini", "#1a73e8"),
    ("claude-haiku-4.5", "Claude Haiku 4.5", "anthropic", "#d97757"),
    ("gpt-5-mini", "GPT-5 mini", "openai", "#10a37f"),
    ("gemma-4-31b-it", "Gemma 4 31B", "google", "#8ab4f8"),
    ("qwen3-vl-32b-instruct", "Qwen3-VL 32B", "qwen", "#615ced"),
]
SERIES = [
    ("bomberman_", "Bomberman", "bomberman"),
    ("deja_vu_", "Déjà Vu", "magnifier"),
    ("legend_of_zelda_", "Zelda", "hylian_shield"),
    ("pokemon_", "Pokémon", "pokeball"),
    ("sword_of_hope_", "Sword of Hope", "sword"),
]
SERIES_COLOUR = "#9a9a9a"
SHORT = {
    "bomberman_pocket": "Po", "bomberman_quest": "Qu",
    "deja_vu_1": "1", "deja_vu_2": "2",
    "legend_of_zelda_links_awakening": "LA", "legend_of_zelda_the_oracle_of_seasons": "OS",
    "pokemon_red": "R", "pokemon_crystal": "C",
    "sword_of_hope_1": "1", "sword_of_hope_2": "2",
}
DIV_RED_BLUE = ["#7a1414", "#b52a2a", "#e34948", "#f2a19f", "#f0efec", "#9ec5f4", "#3987e5", "#256abf", "#0d366b"]

CAT_CANON = {"dialog": "dialogue", "interact": "interaction", "inspect": "interaction", "take": "interaction",
             "use": "interaction", "navigaton": "navigation", "catch_pokemon": "combat", "inventory": "menu"}
CAT_DROP = {"hard", "reasoning", "progression", "survival", "dialogue"}
CATEGORIES = ["navigation", "interaction", "combat", "menu"]
CAT_COLOUR = {"navigation": "#1f5fa8", "interaction": "#3f3f9f", "combat": "#0f7489", "menu": "#6b3f99"}
CAT_HATCH = {"navigation": "", "interaction": "//", "combat": "..", "menu": "\\\\"}

TREEMAP_MODEL = "gemini-3.6-flash"

GAIN_METHODS = [
    ("wm", "WM", "globe", "#2E6FD9"),
    ("pae", "PAE", "recycle", "#7D4CC4"),
    ("guide", "Guide", "doc", "#0F9D76"),
]
GAIN_RATES = {
    "bomberman_pocket":                      {"base": 72.0, "guide": 88.0, "wm": 74.0, "pae": 76.0},
    "bomberman_quest":                       {"base": 38.0, "guide": 36.0, "wm": 36.0, "pae": 16.0},
    "deja_vu_1":                             {"base": 14.0, "guide": 10.0, "wm": 12.0, "pae": 10.0},
    "deja_vu_2":                             {"base": 8.0, "guide": 6.0, "wm": 6.0, "pae": 2.0},
    "legend_of_zelda_links_awakening":       {"base": 38.0, "guide": 40.0, "wm": 40.0, "pae": 28.0},
    "legend_of_zelda_the_oracle_of_seasons": {"base": 40.0, "guide": 44.0, "wm": 38.0, "pae": 14.0},
    "pokemon_red":                           {"base": 38.8, "guide": 36.7, "wm": 39.0, "pae": 8.2},
    "pokemon_crystal":                       {"base": 36.0, "guide": 30.0, "wm": 34.0, "pae": 4.0},
    "sword_of_hope_1":                       {"base": 56.0, "guide": 52.0, "wm": 48.0, "pae": 36.0},
    "sword_of_hope_2":                       {"base": 66.0, "guide": 72.0, "wm": 66.0, "pae": 32.0},
}

CONTAMINATION_CSV = os.path.join(CONTAMINATION_DIR, "pokemon_contamination_benchmark.csv")
CONTAMINATION_MODELS = [
    ("gemini-3.5-flash-lite", "Gemini 3.5 Flash-Lite", "gemini", "#1a73e8"),
    ("chatgpt_gpt-5.6-sol", "GPT-5.6-sol", "openai", "#10a37f"),
    ("Claude-Code-Best-No-Internet", "Claude Opus 5", "anthropic", "#d97757"),
]
CONTAMINATION_GAMES = [
    ("Pokémon Red", "R", "Red", ["#ff5a4e", "#c8161d"]),
    ("Pokémon Crystal", "C", "Crystal", ["#9ff0ff", "#2aa7d6", "#1b5fa8"]),
]
CONTAMINATION_GAME_GROUPS = [
    ("Classic", ["Pokémon Red", "Pokémon Crystal"]),
]

FAILURE_MODES = [
    ("Navigation", 66, "#1f5fa8"),
    ("Decisiveness", 50, "#3f3f9f"),
    ("Perception", 37, "#6b3f99"),
    ("Repetition", 20, "#0f7489"),
    ("Coordination", 18, "#2a4b7c"),
    ("Planning", 16, "#7a5bb5"),
    ("Familiarity", 13, "#1b8aa8"),
]

FAILURE_MODE_COUNTS = {
    "Navigation":   [51, 19, 3, 2],
    "Decisiveness": [9, 32, 6, 6],
    "Perception":   [11, 25, 1, 3],
    "Repetition":   [7, 5, 3, 8],
    "Coordination": [4, 10, 3, 1],
    "Planning":     [6, 5, 3, 6],
    "Familiarity":  [7, 3, 5, 3],
}

LABEL = {m[0]: m[1] for m in FRONTIER_MODELS}
COLOUR = {m[0]: m[3] for m in FRONTIER_MODELS}


def set_style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"],
                         "mathtext.fontset": "stix"})


def load_results(keys, games=GAMES):
    frames = []
    for g in games:
        series = pd.read_csv(paths.benchmark_series_csv(PARAMETERS, game=g))
        t = series[series["game"] == g].reset_index(drop=True)
        for k in keys:
            path = paths.benchmark_csv(PARAMETERS, game=g, model=k, **ARM)
            paths.require(path, "benchmark", PARAMETERS, g)
            r = pd.read_csv(path, usecols=["task", "success"])
            if len(r) != len(t):
                log_error(
                    f"{path} has {len(r)} row(s) but '{g}' has {len(t)} benchmark task(s) in "
                    f"{paths.benchmark_series_csv(PARAMETERS, game=g)}. This arm is a partial "
                    "sweep; finish it or drop the game from GAMES before plotting.",
                    PARAMETERS,
                )
            r["game"], r["model"] = g, k
            r["success"] = r["success"].astype(str).str.strip().str.lower() == "true"
            r["task_category"] = t["task_category"].values
            frames.append(r)
    return pd.concat(frames, ignore_index=True)


def task_categories(s):
    out = []
    for c in re.split(r"[,+]", str(s).lower().replace('"', "")):
        c = CAT_CANON.get(c.strip(), c.strip())
        if c and c not in CAT_DROP and c not in out:
            out.append(c)
    return out


def squarify(values, x, y, w, h):
    total = sum(values)
    sizes = [v * w * h / total for v in values]
    rects = []

    def worst(row, side):
        s = sum(row)
        return max(side * side * max(row) / (s * s), s * s / (side * side * min(row)))

    while sizes:
        side = min(w, h)
        row = [sizes[0]]
        while len(row) < len(sizes) and worst(row + [sizes[len(row)]], side) <= worst(row, side):
            row.append(sizes[len(row)])
        s = sum(row)
        if w >= h:
            cw, cy = s / h, y
            for r in row:
                rects.append((x, cy, cw, r / cw))
                cy += r / cw
            x, w = x + cw, w - cw
        else:
            ch, cx = s / w, x
            for r in row:
                rects.append((cx, y, r / ch, ch))
                cx += r / ch
            y, h = y + ch, h - ch
        sizes = sizes[len(row):]
    return rects


def shade(colour, f):
    r, g, b = to_rgb(colour)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def darken(colour, f=0.55):
    return tuple(v * f for v in to_rgb(colour))


def band(ax, x0, t0, b0, x1, t1, b1, **kw):
    t = np.linspace(0, 1, 200)
    s = 3 * t ** 2 - 2 * t ** 3
    ax.fill_between(x0 + (x1 - x0) * t, t0 + (t1 - t0) * s, b0 + (b1 - b0) * s, lw=0, **kw)


def _quad(p0, c, p1, n=24):
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * c[0] + t * t * p1[0],
             (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * c[1] + t * t * p1[1]) for t in np.linspace(0, 1, n)]


def _gradient(px, stops):
    t = np.linspace(0, 1, px)[None, :, None] * 0.5 + np.linspace(0, 1, px)[:, None, None] * 0.5
    cols = np.array([to_rgb(c) for c in stops])
    pos = np.linspace(0, 1, len(stops))
    rgb = np.stack([np.interp(t[..., 0], pos, cols[:, i]) for i in range(3)], -1)
    return Image.fromarray((rgb * 255).astype(np.uint8)).convert("RGBA")


def draw_gemini(px):
    th = np.linspace(0, 2 * np.pi, 400)
    c, s = np.cos(th), np.sin(th)
    pts = [(px / 2 + 0.48 * px * np.sign(a) * abs(a) ** 3.6, px / 2 + 0.48 * px * np.sign(b) * abs(b) ** 3.6)
           for a, b in zip(c, s)]
    mask = Image.new("L", (px, px), 0)
    ImageDraw.Draw(mask).polygon(pts, fill=255)
    img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    img.paste(_gradient(px, ["#1c7df2", "#6a6ee8", "#a36cd9"]), (0, 0), mask)
    return img


def draw_pokeball(px):
    img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    m, w = px * 0.04, px * 0.055
    box = [m, m, px - m, px - m]
    d.pieslice(box, 180, 360, fill="#e3350d")
    d.pieslice(box, 0, 180, fill="white")
    d.ellipse(box, outline="#222", width=int(w))
    d.rectangle([m, px / 2 - w / 2, px - m, px / 2 + w / 2], fill="#222")
    r = px * 0.16
    d.ellipse([px / 2 - r, px / 2 - r, px / 2 + r, px / 2 + r], fill="white", outline="#222", width=int(w))
    r = px * 0.07
    d.ellipse([px / 2 - r, px / 2 - r, px / 2 + r, px / 2 + r], fill="white", outline="#222", width=int(w * 0.5))
    return img


def draw_hylian_shield(px):
    img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    def shield(inset):
        l, r, t, mid, bot = 0.12 + inset, 0.88 - inset, 0.04 + inset, 0.5, 0.96 - inset * 1.4
        pts = _quad((l, t + 0.04), (l, t), (l + 0.08, t))
        pts += [(r - 0.08, t)] + _quad((r - 0.08, t), (r, t), (r, t + 0.04))[1:]
        pts += [(r, 0.52)] + _quad((r, 0.52), (r, 0.8), (mid, bot))[1:]
        pts += _quad((mid, bot), (l, 0.8), (l, 0.52))[1:]
        return [(x * px, y * px) for x, y in pts]

    d.polygon(shield(0), fill="#b8bcc4")
    d.polygon(shield(0.06), fill="#1f4fa8")
    h, cx, top = 0.13 * px, px / 2, 0.14 * px
    for ox, oy in [(0, 0), (-h / 2, h), (h / 2, h)]:
        d.polygon([(cx + ox, top + oy), (cx + ox - h / 2, top + oy + h), (cx + ox + h / 2, top + oy + h)], fill="#f2c230")
    wy = 0.58 * px
    d.polygon([(cx, wy - 0.05 * px), (cx - 0.28 * px, wy - 0.12 * px), (cx - 0.2 * px, wy + 0.02 * px),
               (cx, wy + 0.12 * px), (cx + 0.2 * px, wy + 0.02 * px), (cx + 0.28 * px, wy - 0.12 * px)], fill="#d8262e")
    return img


def draw_bomberman(px):
    img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    ow = int(px * 0.035)
    d.line([(px / 2, px * 0.2), (px / 2, px * 0.1)], fill="#222", width=int(px * 0.04))
    r = px * 0.075
    d.ellipse([px / 2 - r, px * 0.1 - r, px / 2 + r, px * 0.1 + r], fill="#f25c9a", outline="#222", width=ow)
    d.ellipse([px * 0.1, px * 0.18, px * 0.9, px * 0.96], fill="white", outline="#222", width=ow)
    d.rounded_rectangle([px * 0.22, px * 0.4, px * 0.78, px * 0.8], radius=px * 0.14, fill="#f7b9a0",
                        outline="#222", width=ow)
    for ex in (0.4, 0.6):
        d.ellipse([px * (ex - 0.04), px * 0.48, px * (ex + 0.04), px * 0.68], fill="#222")
    return img


def draw_magnifier(px):
    img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.line([(px * 0.58, px * 0.58), (px * 0.9, px * 0.9)], fill="#6b4226", width=int(px * 0.13))
    r, c = px * 0.3, px * 0.4
    d.ellipse([c - r, c - r, c + r, c + r], fill="#d6ecf7", outline="#333", width=int(px * 0.08))
    d.arc([c - r * 0.65, c - r * 0.65, c + r * 0.65, c + r * 0.65], 190, 260, fill="white", width=int(px * 0.05))
    return img


def draw_sword(px):
    img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx, bw = px / 2, px * 0.075
    d.polygon([(cx, px * 0.03), (cx + bw, px * 0.13), (cx + bw, px * 0.66), (cx - bw, px * 0.66), (cx - bw, px * 0.13)],
              fill="#cfd6de", outline="#555", width=int(px * 0.02))
    d.line([(cx, px * 0.12), (cx, px * 0.64)], fill="#8c96a3", width=int(px * 0.02))
    d.rounded_rectangle([px * 0.24, px * 0.64, px * 0.76, px * 0.73], radius=px * 0.04, fill="#e0a526", outline="#6b4a0e",
                        width=int(px * 0.02))
    d.rectangle([cx - bw * 0.7, px * 0.73, cx + bw * 0.7, px * 0.9], fill="#7a4a24")
    r = px * 0.06
    d.ellipse([cx - r, px * 0.9 - r, cx + r, px * 0.9 + r], fill="#e0a526", outline="#6b4a0e", width=int(px * 0.02))
    return img


DRAWN_LOGOS = {"gemini": draw_gemini, "pokeball": draw_pokeball, "hylian_shield": draw_hylian_shield,
               "bomberman": draw_bomberman, "magnifier": draw_magnifier, "sword": draw_sword}


def load_logo(logo, s):
    path = os.path.join(LOGOS, f"{logo}.png")
    if os.path.exists(path):
        return Image.open(path).convert("RGBA").resize((s, s), Image.LANCZOS)
    if logo not in DRAWN_LOGOS:
        log_error(
            f"Logo {logo!r} has neither a file at {path} nor a drawing in DRAWN_LOGOS "
            f"(have: {sorted(DRAWN_LOGOS)}).",
            PARAMETERS,
        )
    return DRAWN_LOGOS[logo](s * 4).resize((s, s), Image.LANCZOS)


def make_badge(colour, logo, px=256, frac=0.58):
    img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    w = int(px * 0.08)
    draw.ellipse([w // 2, w // 2, px - w // 2 - 1, px - w // 2 - 1], fill=(255, 255, 255, 255), outline=colour, width=w)
    s = int(px * frac)
    lg = load_logo(logo, s)
    img.alpha_composite(lg, ((px - s) // 2, (px - s) // 2))
    return np.asarray(img) / 255.0


class ImageHandler(HandlerBase):
    def __init__(self, arr, scale=2.1):
        super().__init__()
        self.arr, self.scale = arr, scale

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        s = height * self.scale
        bb = Bbox.from_bounds(-xdescent + width / 2 - s / 2, -ydescent + height / 2 - s / 2, s, s)
        img = BboxImage(TransformedBbox(bb, trans))
        img.set_data(self.arr)
        return [img]


def logo_legend(fig, badges, keys, labels, **kw):
    handles = [Patch(label=l) for l in labels]
    handler_map = {h: ImageHandler(badges[k]) for h, k in zip(handles, keys) if k is not None}
    leg = fig.legend(handles=handles, handler_map=handler_map, handlelength=1.6, frameon=True, fancybox=False,
                     edgecolor="#888", borderpad=0.6, handletextpad=0.3, **kw)
    rows = [r for col in leg._legend_handle_box.get_children() for r in col.get_children()]
    for row, k, text in zip(rows, keys, leg.get_texts()):
        if k is None:
            row._children = row._children[1:]
            text.set_fontweight("bold")
    return leg


def img_at(ax, arr, xy, zoom, xycoords="data", align=(0.5, 0.5)):
    ab = AnnotationBbox(OffsetImage(arr, zoom=zoom), xy, xycoords=xycoords, frameon=False,
                        box_alignment=align, annotation_clip=False, pad=0)
    ax.add_artist(ab)


def save(fig, name, dpi=170):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    fig.savefig(path[:-4] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    log_info(f"saved {path}", PARAMETERS)


def frontier_data():
    data = load_results([m[0] for m in FRONTIER_MODELS])
    badges = {k: make_badge(c, logo) for k, _, logo, c in FRONTIER_MODELS}
    order = list(data.groupby("model")["success"].mean().sort_values(ascending=False).index)
    return data, badges, order


def frontier_leaderboard():
    data, badges, order = frontier_data()
    plt.rcParams.update({"font.size": 38.4})
    agg = data.groupby("model")["success"].agg(["sum", "count"]).reindex(order[::-1])
    agg["rate"] = 100 * agg["sum"] / agg["count"]
    fig, ax = plt.subplots(figsize=(12, 1.4 * len(agg) + 0.6))
    top = 0
    for y, (k, r) in enumerate(agg.iterrows()):
        top = max(top, r["rate"])
        ax.barh(y, r["rate"], color=COLOUR[k], height=0.68)
        img_at(ax, badges[k], (-0.012, y), zoom=0.24, xycoords=("axes fraction", "data"), align=(1, 0.5))
        ax.text(r["rate"] + 1.2, y, f"{r['rate']:.1f}%", va="center", ha="left", fontsize=48, fontweight="bold",
                color=COLOUR[k])
    ax.set_yticks([])
    ax.set_xlim(0, top + 12)
    ax.set_xlabel("Success Rate (%)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    save(fig, "frontier_leaderboard.png")
    log_info(agg.to_string(), PARAMETERS)


def frontier_heatmap():
    data, badges, order = frontier_data()
    series_badges = {p: make_badge(SERIES_COLOUR, logo, frac=0.8) for p, _, logo in SERIES}
    plt.rcParams.update({"font.size": 30})
    rates = data.groupby(["model", "game"])["success"].mean().unstack() * 100
    cmap = LinearSegmentedColormap.from_list("div_red_blue", DIV_RED_BLUE)
    norm = Normalize(0, 100)

    xs, x, last = [], 0.0, None
    for g in GAMES:
        s = next(p for p, *_ in SERIES if g.startswith(p))
        if last is not None and s != last:
            x += 0.25
        xs.append(x)
        last, x = s, x + 1

    fig, ax = plt.subplots(figsize=(15, 1.05 * len(order) + 1.6))
    for y, k in enumerate(order):
        for xc, g in zip(xs, GAMES):
            v = rates.loc[k, g]
            ax.add_patch(Rectangle((xc, y), 1, 1, facecolor=cmap(norm(v)), edgecolor="white", lw=3))
            ax.text(xc + 0.5, y + 0.5, f"{v:.0f}", ha="center", va="center", fontsize=33, color="white" if abs(v - 50) > 22 else "#1a1a1a")
        img_at(ax, badges[k], (-0.006, y + 0.5), zoom=0.19, xycoords=("axes fraction", "data"), align=(1, 0.5))

    ax.set_xlim(0, x)
    ax.set_ylim(len(order), 0)
    ax.set_yticks([])
    ax.set_xticks([xc + 0.5 for xc in xs])
    ax.set_xticklabels([SHORT[g] for g in GAMES])
    ax.tick_params(axis="x", length=0, pad=6)
    for prefix, *_ in SERIES:
        members = [xc + 0.5 for xc, g in zip(xs, GAMES) if g.startswith(prefix)]
        ax.plot([members[0] - 0.45, members[-1] + 0.45], [-0.135, -0.135], transform=ax.get_xaxis_transform(),
                color="#555", lw=2, clip_on=False)
        img_at(ax, series_badges[prefix], (sum(members) / len(members), -0.16), zoom=0.21,
               xycoords=("data", "axes fraction"), align=(0.5, 1))
    for s in ax.spines.values():
        s.set_visible(False)

    logo_legend(fig, series_badges, [p for p, *_ in SERIES], [n for _, n, _ in SERIES], bbox_transform=ax.transAxes,
                loc="upper center", bbox_to_anchor=(0.5, -0.33), ncol=3, fontsize=38, columnspacing=1.2)
    save(fig, "frontier_heatmap.png")
    log_info(rates.reindex(index=order, columns=GAMES).round(1).to_string(), PARAMETERS)


def frontier_model_legend():
    _, badges, order = frontier_data()
    fig = plt.figure(figsize=(1, 1))
    logo_legend(fig, badges, order, [LABEL[k] for k in order], loc="center", ncol=len(order), fontsize=40,
                columnspacing=1.6)
    save(fig, "frontier_model_legend.png")


TREEMAP_W, TREEMAP_H, TREEMAP_HEADER = 16, 9, 1.5


def treemap_canvas(caption):
    fig = plt.figure(figsize=(TREEMAP_W, TREEMAP_H + TREEMAP_HEADER))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, TREEMAP_W)
    ax.set_ylim(TREEMAP_H, -TREEMAP_HEADER)
    ax.axis("off")
    ax.add_patch(Rectangle((0, -TREEMAP_HEADER), TREEMAP_W, TREEMAP_HEADER, facecolor="#1a1a1a", lw=0))
    ax.text(TREEMAP_W / 2, -TREEMAP_HEADER / 2, caption, ha="center", va="center", color="white", fontsize=52,
            fontweight="bold")
    return fig, ax


def treemap_font(lines, w, h, cap=52.8):
    return min(cap, 0.8 * w * 72 / (0.55 * max(map(len, lines))), 0.7 * h * 72 / (1.25 * len(lines)))


def failure_mode_treemap():
    total = sum(n for _, n, _ in FAILURE_MODES)
    fig, ax = treemap_canvas("Failure modes, sized by prevalence")
    rects = squarify([n for _, n, _ in FAILURE_MODES], 0, 0, TREEMAP_W, TREEMAP_H)
    for (name, n, c), (x, y, rw, rh) in zip(FAILURE_MODES, rects):
        ax.add_patch(Rectangle((x, y), rw, rh, facecolor=c, edgecolor="white", lw=5))
        lines = {"Coordination": ["Co-", "ordination"]}.get(name, [name]) + [f"{100 * n / total:.0f}%"]
        scale = {"Coordination": 1.4, "Planning": 1.25, "Familiarity": 1.25}.get(name, 1)
        ax.text(x + rw / 2, y + rh / 2, "\n".join(lines), ha="center", va="center", color="white",
                fontsize=scale * treemap_font(lines, rw, rh), fontweight="bold", linespacing=1.15)
    save(fig, "failure_mode_treemap.png")


def task_category_treemap():
    data = load_results([TREEMAP_MODEL])
    data = data.assign(cat=data["task_category"].apply(task_categories)).explode("cat").dropna(subset=["cat"])
    agg = data.groupby("cat")["success"].agg(["sum", "count"]).sort_values("count", ascending=False)
    fig, ax = treemap_canvas("Task success rates, sized by frequency")
    for (cat, r), (x, y, rw, rh) in zip(agg.iterrows(), squarify(list(agg["count"]), 0, 0, TREEMAP_W, TREEMAP_H)):
        c, rate = CAT_COLOUR[cat], r["sum"] / r["count"]
        ax.add_patch(Rectangle((x, y), rw, rh * rate, facecolor=shade(c, 0.78), lw=0))
        ax.add_patch(Rectangle((x, y + rh * rate), rw, rh * (1 - rate), facecolor=c, edgecolor=shade(c, 0.3),
                               hatch="//", lw=0))
        ax.add_patch(Rectangle((x, y), rw, rh, fill=False, edgecolor="white", lw=6))
        lines = [cat.capitalize(), f"{100 * rate:.0f}%"]
        ax.text(x + rw / 2, y + rh / 2, "\n".join(lines), ha="center", va="center",
                fontsize=treemap_font(lines, rw, rh), fontweight="bold",
                color="#1a1a1a", linespacing=1.15,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "none", "alpha": 0.85})
    save(fig, "task_category_treemap.png")
    log_info(agg.assign(rate=(100 * agg["sum"] / agg["count"]).round(1)).to_string(), PARAMETERS)


def failure_mode_alluvial(categories=CATEGORIES, name="failure_mode_alluvial.png"):
    gap, left_w, right_w = 5, 0.16, 0.2
    flow_font, mode_font, label_cols, label_col_w = 26.4, 26.4, 8, 0.085
    modes = list(FAILURE_MODE_COUNTS)
    m = np.array([FAILURE_MODE_COUNTS[k] for k in modes], dtype=float)[:, [CATEGORIES.index(c) for c in categories]]
    cat_tot, mode_tot, total = m.sum(0), m.sum(1), m.sum()
    left_h = cat_tot.sum() + gap * (len(categories) - 1)
    right_h = mode_tot.sum() + gap * (len(modes) - 1)

    plt.rcParams.update({"font.size": 30})
    fig, ax = plt.subplots(figsize=(15, 11))
    ax.set_xlim(0, 1)
    ax.set_ylim(right_h, 0)
    ax.axis("off")
    left_top = (right_h - left_h) / 2 + np.concatenate([[0], np.cumsum(cat_tot + gap)[:-1]])
    right_top = np.concatenate([[0], np.cumsum(mode_tot + gap)[:-1]])
    left_cur, right_cur = left_top.copy(), right_top.copy()
    xl, xr = left_w, 1 - right_w
    chip_h = flow_font * 1.45 * right_h / (ax.get_position().height * fig.get_figheight() * 72)

    strips = []
    for j, cat in enumerate(categories):
        c = CAT_COLOUR[cat]
        tex = {"hatch": CAT_HATCH[cat], "edgecolor": darken(c)}
        for i in range(len(modes)):
            v = m[i, j]
            if v == 0:
                continue
            band(ax, xl, left_cur[j], left_cur[j] + v, xr, right_cur[i], right_cur[i] + v, facecolor=c, alpha=0.55,
                 **tex)
            strips.append((left_cur[j] + v / 2, right_cur[i] + v / 2, f"{100 * v / cat_tot[j]:.0f}%"))
            left_cur[j] += v
            right_cur[i] += v

    placed = {}
    for y0, y1, text in sorted(strips):
        for col in range(label_cols):
            x = xl + 0.035 + col * label_col_w
            t = (x - xl) / (xr - xl)
            y = y0 + (y1 - y0) * (3 * t ** 2 - 2 * t ** 3)
            if all(abs(y - py) >= chip_h for py in placed.get(col, [])):
                placed.setdefault(col, []).append(y)
                ax.text(x, y, text, ha="center", va="center", fontsize=flow_font, fontweight="bold", color="#1a1a1a",
                        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.85})
                break

    for j, cat in enumerate(categories):
        c = CAT_COLOUR[cat]
        ax.add_patch(Rectangle((0, left_top[j]), left_w, cat_tot[j], facecolor=c, lw=0, hatch=CAT_HATCH[cat],
                               edgecolor=darken(c)))
        ax.add_patch(Rectangle((0, left_top[j]), left_w, cat_tot[j], fill=False, edgecolor="#333", lw=1.5))
        ax.text(left_w / 2, left_top[j] + cat_tot[j] / 2, cat.capitalize(), ha="center", va="center", fontsize=24,
                fontweight="bold", color="white")

    for i, k in enumerate(modes):
        big = mode_tot[i] / total > 0.12
        ax.add_patch(Rectangle((xr, right_top[i]), right_w, mode_tot[i], facecolor="#f2f2f2", edgecolor="#333",
                               lw=1.5))
        ax.text(xr + right_w / 2, right_top[i] + mode_tot[i] / 2,
                f"{k}{chr(10) if big else ' '}{100 * mode_tot[i] / total:.0f}%", ha="center", va="center",
                fontsize=mode_font * (1 if big else 17 / 22), linespacing=1.05)
    save(fig, name)


def failure_mode_alluvial_nav_int():
    failure_mode_alluvial(["navigation", "interaction"], "failure_mode_alluvial_nav_int.png")


def draw_globe(px, colour):
    img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    w, m, c = int(px * 0.07), px * 0.06, px / 2
    d.ellipse([m, m, px - m, px - m], outline=colour, width=w)
    d.ellipse([c - px * 0.2, m, c + px * 0.2, px - m], outline=colour, width=int(w * 0.8))
    d.line([(m, c), (px - m, c)], fill=colour, width=int(w * 0.8))
    for dy in (-0.52, 0.52):
        y = c + dy * (c - m)
        hw = (c - m) * np.sqrt(1 - dy ** 2)
        d.line([(c - hw, y), (c + hw, y)], fill=colour, width=int(w * 0.8))
    return img


def draw_recycle(px, colour):
    img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c, r, w = px / 2, px * 0.34, int(px * 0.12)
    for i in range(3):
        a0 = -80 + 120 * i
        a1 = a0 + 78
        d.arc([c - r, c - r, c + r, c + r], a0, a1, fill=colour, width=w)
        th = np.radians(a1)
        rad, tan = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
        p = np.array([c, c]) + (r - w / 2) * rad
        hl, hw = px * 0.22, px * 0.17
        d.polygon([tuple(p + hl * tan), tuple(p + hw * rad), tuple(p - hw * rad)], fill=colour)
    return img


def draw_doc(px, colour):
    img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    w = int(px * 0.07)
    x0, y0, x1, y1, f = px * 0.18, px * 0.06, px * 0.82, px * 0.94, px * 0.22
    d.line([(x0, y0), (x1 - f, y0), (x1, y0 + f), (x1, y1), (x0, y1), (x0, y0)], fill=colour, width=w, joint="curve")
    d.line([(x1 - f, y0), (x1 - f, y0 + f), (x1, y0 + f)], fill=colour, width=int(w * 0.8))
    for k, fy in enumerate((0.42, 0.58, 0.74)):
        d.line([(x0 + px * 0.12, px * fy), (x1 - px * (0.2 if k == 0 else 0.12), px * fy)], fill=colour,
               width=int(w * 0.8))
    return img


for _key, _, _icon, _colour in GAIN_METHODS:
    DRAWN_LOGOS[_icon] = lambda px, fn=globals()[f"draw_{_icon}"], c=_colour: fn(px, c)


def fmt_gain(v):
    v = round(v)
    return f"+{v}%" if v > 0 else ("0%" if v == 0 else f"−{-v}%")


def self_improvement_gain():
    dot_in, vf = 0.36, 28.6
    lane_in, game_gap, series_gap = 0.62, 0.2, 0.4
    left, right, top, bottom, plot_h = 1.9, 0.3, 0.45, 3.9, 7.6
    ylim = (-38, 20)
    gains = {k: {g: GAIN_RATES[g][k] - GAIN_RATES[g]["base"] for g in GAMES} for k, *_ in GAIN_METHODS}
    method_badges = {k: make_badge(c, icon, frac=0.62) for k, _, icon, c in GAIN_METHODS}
    series_badges = {p: make_badge(SERIES_COLOUR, logo, frac=0.8) for p, _, logo in SERIES}

    col_w = lane_in * len(GAIN_METHODS)
    xs, x = [], 0.0
    for gi, g in enumerate(GAMES):
        if gi:
            x += series_gap if gi % 2 == 0 else game_gap
        xs.append(x)
        x += col_w
    plot_w = x
    fig_w, fig_h = left + plot_w + right, bottom + plot_h + top
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([left / fig_w, bottom / fig_h, plot_w / fig_w, plot_h / fig_h])
    ax.set_xlim(-0.3, plot_w)
    ax.set_ylim(*ylim)
    pp_per_in = (ylim[1] - ylim[0]) / plot_h
    label_pp = vf / 72 * 1.2 * pp_per_in
    dot_pp = dot_in * pp_per_in
    off = dot_pp / 2 + 0.3

    ax.axhline(0, color="#333", lw=2, zorder=1)
    dots = [(x0 + lane_in * (li + 0.5), gains[k][g], k) for g, x0 in zip(GAMES, xs)
            for li, (k, *_) in enumerate(GAIN_METHODS)]
    placed = []

    def box(cx, v, s, text):
        w = len(text) * vf * 0.55 / 72 + 0.06
        if s == "right":
            x0 = cx + dot_in / 2 + 0.04
            return x0, x0 + w, v - label_pp / 2, v + label_pp / 2
        y0 = v + off if s == "above" else v - off - label_pp
        return cx - w / 2, cx + w / 2, y0, y0 + label_pp

    def clashes(b, own, zero=True):
        if zero and b[2] < 0 < b[3] and own[1] != 0:
            return True
        if any(b[0] < p[1] and p[0] < b[1] and b[2] < p[3] and p[2] < b[3] for p in placed):
            return True
        rx, ry = dot_in / 2, dot_pp / 2
        return any(d is not own and b[0] < d[0] + rx and d[0] - rx < b[1] and b[2] < d[1] + ry and d[1] - ry < b[3]
                   for d in dots)

    for d in dots:
        cx, v, k = d
        text = fmt_gain(v)
        pref = "above" if v >= 0 else "below"
        other = "below" if pref == "above" else "above"
        s = next((s for s in (pref, other, "right") if not clashes(box(cx, v, s, text), d)), None) or \
            next((s for s in (pref, other) if not clashes(box(cx, v, s, text), d, zero=False)), pref)
        b = box(cx, v, s, text)
        placed.append(b)
        img_at(ax, method_badges[k], (cx, v), zoom=dot_in * 72 / 256)
        tx, ty, ha, va = {"above": (cx, v + off, "center", "bottom"), "below": (cx, v - off, "center", "top"),
                          "right": (b[0], v, "left", "center")}[s]
        ax.text(tx, ty, text, ha=ha, va=va, fontsize=vf, fontweight="bold", color="#1a1a1a", zorder=5)

    for gi, (g, x0) in enumerate(zip(GAMES, xs)):
        ax.text(x0 + col_w / 2, -0.03, SHORT[g], transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=33.8)
        if gi % 2 == 0:
            sep = x0 + col_w + game_gap / 2
            ax.axvspan(x0 - (series_gap / 2 if gi else 0.3), sep, color="#eef3fa", zorder=-1, lw=0)
            if gi:
                ax.axvline(x0 - series_gap / 2, color="#1f5fa8", lw=2.2, zorder=0)
            prefix = next(p for p, *_ in SERIES if g.startswith(p))
            img_at(ax, series_badges[prefix], (sep, -0.125), zoom=0.247, xycoords=("data", "axes fraction"),
                   align=(0.5, 1))

    ax.set_xticks([])
    ax.set_yticks(np.arange(-35, 20, 5))
    ax.tick_params(axis="y", labelsize=31.2)
    ax.set_ylabel("Task completion gain (pp)", fontsize=33.8)
    ax.grid(axis="y", color="#eee", lw=1)
    ax.set_axisbelow(True)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)

    legends = [
        logo_legend(fig, method_badges, [k for k, *_ in GAIN_METHODS], [n for _, n, *_ in GAIN_METHODS],
                    loc="lower left", ncol=len(GAIN_METHODS), fontsize=31.2, columnspacing=0.9),
        logo_legend(fig, series_badges, [p for p, *_ in SERIES], [n for _, n, _ in SERIES],
                    loc="lower left", ncol=len(SERIES), fontsize=31.2, columnspacing=0.9),
    ]
    fig.canvas.draw()
    gap = 0.4 / fig_w
    widths = [lg.get_window_extent().width / fig.bbox.width for lg in legends]
    x = (left + plot_w / 2) / fig_w - (sum(widths) + gap) / 2
    for lg, lw in zip(legends, widths):
        lg.set_bbox_to_anchor((x, 0.02), transform=fig.transFigure)
        x += lw + gap
    save(fig, "self_improvement_gain.png")


def letter_badge(letter, stops, px=256):
    s = px * 4
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([s * 0.02, s * 0.02, s * 0.98, s * 0.98], fill=stops[0])
    font = ImageFont.truetype(findfont(FontProperties(family="serif", weight="bold")), int(s * 0.58))
    d.text((s / 2, s / 2), letter, font=font, fill="#1a1a1a", anchor="mm")
    return np.asarray(img.resize((px, px), Image.LANCZOS)) / 255.0


def contamination_accuracy():
    if not os.path.exists(CONTAMINATION_CSV):
        log_error(f"Contamination benchmark CSV not found: {CONTAMINATION_CSV}", PARAMETERS)
    d = pd.read_csv(CONTAMINATION_CSV)
    out = {}
    for key, *_ in CONTAMINATION_MODELS:
        col = f"actual_answer_{key}_evaluated_human_annotation"
        out[key] = (d.assign(ok=d[col].str.strip() == "Right").groupby("game")["ok"].mean() * 100).to_dict()
    return out


def contamination_dotplot():
    w, left, right, top, bottom = 21.0, 1.5, 5.4, 0.1, 1.5
    dot_in, tier_gap, vf = 0.7, 0.06, 31.2
    row_h = dot_in + 2 * vf / 72 * 1.15 + 0.08
    step = dot_in + tier_gap
    xlim = (-3, 106)
    acc = contamination_accuracy()
    model_badges = {k: make_badge(c, logo) for k, _, logo, c in CONTAMINATION_MODELS}
    game_badges = {g: letter_badge(letter, stops) for g, letter, _, stops in CONTAMINATION_GAMES}
    plot_w = w - left - right
    near = dot_in * (xlim[1] - xlim[0]) / plot_w * 1.05

    layouts = []
    for key, *_ in CONTAMINATION_MODELS:
        vals = [acc[key][g] for g, *_ in CONTAMINATION_GAMES]
        order = list(np.argsort(vals))
        tier = [0] * len(vals)
        for n, i in enumerate(order):
            t = 0
            while any(tier[j] == t and abs(vals[i] - vals[j]) < near for j in order[:n]):
                t += 1
            tier[i] = t
        layouts.append((vals, tier))
    heights = [row_h + max(t) * step for _, t in layouts]
    total = sum(heights)

    fig_h = total + top + bottom
    fig = plt.figure(figsize=(w, fig_h))
    ax = fig.add_axes([left / w, bottom / fig_h, plot_w / w, total / fig_h])
    ax.set_xlim(*xlim)
    ax.set_ylim(total, 0)

    y0 = 0.0
    for (key, *_), (vals, tier), h in zip(CONTAMINATION_MODELS, layouts, heights):
        base = y0 + h / 2 + max(tier) * step / 2
        ax.plot([0, 100], [base, base], color="#c8c8c8", lw=3, solid_capstyle="round", zorder=1)
        img_at(ax, model_badges[key], (-0.012, base), zoom=0.29, xycoords=("axes fraction", "data"), align=(1, 0.5))
        for i, ((g, *_), v, t) in enumerate(zip(CONTAMINATION_GAMES, vals, tier)):
            yc = base - t * step
            img_at(ax, game_badges[g], (v, yc), zoom=dot_in * 72 / 256)
            highest = max((j for j in range(len(vals)) if abs(vals[j] - v) < near), key=lambda j: tier[j])
            kw = {"ha": "center", "fontsize": vf, "fontweight": "bold", "color": "#1a1a1a", "zorder": 5}
            if i == highest:
                ax.text(v, yc - dot_in / 2 - 0.04, f"{v:.0f}%", va="bottom", **kw)
            elif round(v) != round(vals[highest]):
                ax.text(v, yc + dot_in / 2 + 0.04, f"{v:.0f}%", va="top", **kw)
        y0 += h

    ax.set_yticks([])
    ax.set_xticks(np.arange(0, 101, 20))
    ax.tick_params(axis="x", labelsize=33.8)
    ax.set_xlabel("Accuracy (%)", fontsize=36.4)
    ax.grid(axis="x", color="#eee", lw=1)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    lx, ly = (left + plot_w + 0.3) / w, (bottom + total) / fig_h
    logo_legend(fig, model_badges, [k for k, *_ in CONTAMINATION_MODELS], [n for _, n, *_ in CONTAMINATION_MODELS],
                loc="upper left", bbox_to_anchor=(lx, ly), ncol=1, fontsize=vf, labelspacing=0.8)
    game_keys, game_labels = [], []
    for heading, games in CONTAMINATION_GAME_GROUPS:
        game_keys += [None] + games
        game_labels += [heading] + games
    logo_legend(fig, game_badges, game_keys, game_labels, loc="upper left", bbox_to_anchor=(lx, ly - 2.9 / fig_h), ncol=1, fontsize=vf, labelspacing=0.8)
    save(fig, "contamination_dotplot.png")
    log_info(pd.DataFrame(acc).round(1).to_string(), PARAMETERS)


PLOTS = {
    "frontier_leaderboard": frontier_leaderboard,
    "frontier_heatmap": frontier_heatmap,
    "frontier_model_legend": frontier_model_legend,
    "task_category_treemap": task_category_treemap,
    "failure_mode_treemap": failure_mode_treemap,
    "self_improvement_gain": self_improvement_gain,
    "contamination_dotplot": contamination_dotplot,
    "failure_mode_alluvial": failure_mode_alluvial,
    "failure_mode_alluvial_nav_int": failure_mode_alluvial_nav_int,
}


def main():
    set_style()
    os.makedirs(OUT, exist_ok=True)
    requested = sys.argv[1:] or list(PLOTS)
    unknown = [name for name in requested if name not in PLOTS]
    if unknown:
        log_error(f"No such plot(s): {unknown}. Available: {sorted(PLOTS)}", PARAMETERS)
    for name in requested:
        PLOTS[name]()


if __name__ == "__main__":
    main()
