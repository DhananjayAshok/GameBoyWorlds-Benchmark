"""
Write a run's people registry out as a dataset: one folder of plain files (JSON lines, CSV, PNG) that needs none of this
code to read.

  python debug.py --game pokemon_red strategist npc-export --name RUN [--episode N] [--output DIR]

  <output>/                      default: results/npc_dataset/<game>/<run>/
    npcs.jsonl                   one person per line: names, kind, description, look, where and when seen
    looks.jsonl                  one look per line: its views (facing, first seen, picture files)
    sightings.csv                one row per sighting, with the square in the merged map and the area's name
    conversations.jsonl          one conversation per line: speaker, place, text, summary, outcome, picture files
    signs.jsonl                  signs and notices that were read
    images/looks/<look>/view<k>_x1.png and _x4.png       the sprite pictures
    images/context/<person>.png                           the screen the person was first seen on (red box on them)
    images/context/<conversation>_before.png              the screen just before a conversation began
    images/dialogue/<conversation>_<i>.png                the text boxes as they were on screen

A person's id in the dataset is ``<game>/<run>/<npc id>``, so two runs can be merged without clashes. Everything is read
from the saved registry and map; nothing is recomputed and no model is called.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image

from execution.perception.pokemon.grid import cell_box, draw_box
from execution.strategist.pokemon.global_map import GlobalMap
from execution.strategist.pokemon.npc_model import display_name
from execution.strategist.pokemon.npcs import NPCRegistry


def _grey(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    return array[:, :, 0] if array.ndim == 3 and array.shape[2] == 1 else array


def _save(array: np.ndarray, path: str, scale: int = 1) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    array = _grey(array).astype(np.uint8)
    if scale != 1:
        array = cv2.resize(array, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    Image.fromarray(array).save(path)
    return path


def _rel(path: str, root: str) -> str:
    return os.path.relpath(path, root)


def _area(gmap: Optional[GlobalMap], segment: int) -> str:
    try:
        return gmap.place_label(segment) if gmap is not None else ""
    except Exception:
        return ""


def _jsonl(path: str, rows) -> int:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def export_registry(reg: NPCRegistry, gmap: Optional[GlobalMap], out: str, game: str, run: str) -> Dict[str, int]:
    """Write every file of the dataset under ``out``. Returns how many rows of each kind were written."""
    os.makedirs(out, exist_ok=True)
    resolve = (lambda seg, sq: gmap.resolve(seg, sq)) if gmap is not None else (lambda seg, sq: (seg, sq))

    looks = []
    for look in reg.looks.values():
        views = []
        for k, view in enumerate(look.views):
            files = [_rel(_save(view.pixels, os.path.join(out, "images", "looks", look.id, f"view{k}_x{x}.png"), x), out)
                     for x in (1, 4)]
            views.append({"index": k, "facing": view.facing, "mirrored_of": view.mirrored_of,
                          "first_seen": list(view.first_seen), "times_seen": view.n_seen, "files": files})
        looks.append({"look": look.id, "kind_hint": look.kind_hint, "views": views,
                      "links": [list(link) for link in look.links]})

    people = []
    rows = []
    for npc in sorted(reg.live_npcs(), key=lambda n: n.id):
        seen = reg.last_seen(gmap, npc) if gmap is not None else None
        context = None
        if npc.context is not None and npc.context_cell is not None:
            shown = np.asarray(draw_box(_grey(npc.context), *cell_box(*npc.context_cell)))   # RGB, red box
            context = _rel(_save(shown, os.path.join(out, "images", "context", f"{npc.id}.png")), out)
        areas = sorted({resolve(s.segment, s.square)[0] for s in npc.sightings})
        people.append({
            "dataset_id": f"{game}/{run}/{npc.id}", "id": npc.id, "name": display_name(npc), "label": npc.name,
            "real_name": npc.real_name, "name_type": npc.name_type, "kind": npc.kind,
            "description": npc.description, "look": reg.look_of(npc).id,
            "hidden_object": reg.look_of(npc).kind_hint == "object", "aliases": npc.aliases,
            "first_seen": {"episode": npc.created[0], "map_step": npc.created[1]},
            "last_seen": ({"episode": seen[2].episode, "segment": seen[0], "square": list(seen[1]),
                           "area": _area(gmap, seen[0])} if seen else None),
            "areas_seen": [{"segment": si, "name": _area(gmap, si)} for si in areas],
            "n_sightings": len(npc.sightings), "conversations": list(npc.conversations),
            "talked_to": any(reg.conversations[c].text or reg.conversations[c].crops
                             for c in npc.conversations if c in reg.conversations),
            "context_image": context, "needs": npc.needs})
        for s in npc.sightings:
            seg, square = resolve(s.segment, s.square)
            rows.append({"dataset_id": f"{game}/{run}/{npc.id}", "episode": s.episode, "map_step": s.map_step,
                         "segment": seg, "area": _area(gmap, seg), "x": square[0], "y": square[1], "view": s.view,
                         "facing": reg.sighting_facing(reg.look_of(npc), s), "turned_mirror": s.flipped,
                         "raw_segment": s.segment, "raw_x": s.square[0], "raw_y": s.square[1]})
    with open(os.path.join(out, "sightings.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["dataset_id"])
        writer.writeheader()
        writer.writerows(rows)

    visit_of = {c.id: (k, same) for npc in reg.live_npcs() for k, c, same in reg.visits(npc)}
    conversations = []
    for c in sorted(reg.conversations.values(), key=lambda c: (c.episode, c.map_step_start)):
        seg, square = resolve(c.segment, c.square)
        crops = [_rel(_save(crop, os.path.join(out, "images", "dialogue", f"{c.id}_{i}.png")), out)
                 for i, crop in enumerate(c.crops)]
        before = (_rel(_save(c.context, os.path.join(out, "images", "context", f"{c.id}_before.png")), out)
                  if c.context is not None else None)
        conversations.append({
            "dataset_id": f"{game}/{run}/{c.id}", "id": c.id,
            "npc": f"{game}/{run}/{reg.get(c.npc).id}" if c.npc in reg.npcs else None,
            "sign": c.sign, "episode": c.episode, "map_step_start": c.map_step_start, "map_step_end": c.map_step_end,
            "segment": seg, "area": _area(gmap, seg), "player_square": list(square),
            "target_square": list(c.target) if c.target else None, "trigger": c.trigger,
            "attribution": getattr(c, "how", "seen"),
            "attribution_confidence": getattr(c, "confidence", "high"), "text": c.text,
            "summary": c.summary, "speaker_name": c.speaker_name, "name_type": c.name_type, "outcome": c.outcome,
            "led_to": c.led_to, "complete": c.complete, "dialogue_images": crops, "context_image": before})

    signs = [{"id": s.id, "segment": resolve(s.segment, s.square)[0], "square": list(resolve(s.segment, s.square)[1]),
              "area": _area(gmap, resolve(s.segment, s.square)[0]), "text": s.text, "conversations": s.conversations}
             for s in reg.signs.values()]

    counts = {"npcs": _jsonl(os.path.join(out, "npcs.jsonl"), people),
              "looks": _jsonl(os.path.join(out, "looks.jsonl"), looks),
              "sightings": len(rows),
              "conversations": _jsonl(os.path.join(out, "conversations.jsonl"), conversations),
              "signs": _jsonl(os.path.join(out, "signs.jsonl"), signs)}
    return counts
