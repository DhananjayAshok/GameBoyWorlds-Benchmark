from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional, Tuple

import numpy as np

from execution.perception.pokemon.grid import GRID_SKIP

from .model import DIRECTIONS, WALKABLE

if TYPE_CHECKING:
    from .model import Cell, GlobalMap


def segment_bounds(gmap: GlobalMap, segment: int) -> Optional[Tuple[int, int, int, int]]:
    seg = gmap.segments[segment]
    if not seg:
        return None
    xs = [g[0] for g in seg]; ys = [g[1] for g in seg]
    return min(xs) - 1, max(xs) + 1, min(ys) - 1, max(ys) + 1


def render_segment(gmap: GlobalMap, segment: Optional[int] = None, trail: bool = True,
                   upto_step: Optional[int] = None, bounds: Optional[Tuple[int, int, int, int]] = None,
                   highlight: Iterable[Cell] = (), trail_from: Optional[int] = None,
                   doors: bool = False) -> Optional[np.ndarray]:
    import cv2

    si = gmap.segment if segment is None else segment
    seg = {g: c for g, c in gmap.segments[si].items() if upto_step is None or c.first_seen <= upto_step}
    bounds = bounds or segment_bounds(gmap, si)
    if not seg or bounds is None:
        return None
    x0, x1, y0, y1 = bounds
    w, h = (x1 - x0 + 1) * GRID_SKIP, (y1 - y0 + 1) * GRID_SKIP
    canvas = np.full((h, w, 3), 60, np.uint8)
    canvas[::4, :, :] = 90
    def box(g):
        return (g[0] - x0) * GRID_SKIP, (y1 - g[1]) * GRID_SKIP
    def outline(g, colour, width):
        px, py = box(g)
        cv2.rectangle(canvas, (px, py), (px + GRID_SKIP - 1, py + GRID_SKIP - 1), colour, width)
    for g, c in seg.items():
        if c.pixels is None or not (x0 <= g[0] <= x1 and y0 <= g[1] <= y1):
            continue
        px, py = box(g)
        canvas[py:py + GRID_SKIP, px:px + GRID_SKIP] = cv2.cvtColor(c.pixels.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if si == gmap.segment or upto_step is not None:
        for g, c in seg.items():
            if c.kind == WALKABLE and any((g[0] + dx, g[1] + dy) not in seg for dx, dy in DIRECTIONS.values()):
                outline(g, (0, 200, 0), 1)
    for g in highlight:
        outline(g, (255, 0, 255), 1)
    if doors:
        for i, door in enumerate(gmap.door_list(si)):
            colour = (255, 255, 0) if door["leads_to"] is not None else (0, 200, 255)
            for g in door["cells"]:
                if x0 <= g[0] <= x1 and y0 <= g[1] <= y1:
                    outline(g, colour, 2)
            gx, gy = box(door["cells"][0])
            cv2.putText(canvas, str(i), (gx + 1, gy + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1)
    steps = [(t, p) for t, s, p in gmap.trail
             if s == si and (upto_step is None or t <= upto_step) and (trail_from is None or t > trail_from)]
    if trail:
        pts = [p for _, p in steps]
        for a, b in zip(pts, pts[1:]):
            (ax, ay), (bx, by) = box(a), box(b)
            cv2.line(canvas, (ax + 8, ay + 8), (bx + 8, by + 8), (255, 120, 0), 1)
    if upto_step is not None:
        if steps:
            outline(steps[-1][1], (0, 0, 255), 2)
    elif si == gmap.segment:
        outline(gmap.pos, (0, 0, 255), 2)
    return canvas


def render_overlay(gmap: GlobalMap, source: int, target: int, offset: Cell) -> Optional[np.ndarray]:
    import cv2

    src = {(g[0] + offset[0], g[1] + offset[1]): c for g, c in gmap.segments[source].items()}
    dst = gmap.segments[target]
    both = list(src) + list(dst)
    if not both:
        return None
    x0, x1 = min(g[0] for g in both), max(g[0] for g in both)
    y0, y1 = min(g[1] for g in both), max(g[1] for g in both)
    canvas = np.full(((y1 - y0 + 1) * GRID_SKIP, (x1 - x0 + 1) * GRID_SKIP, 3), 60, np.uint8)
    canvas[::4, :, :] = 90
    for layer in (dst, src):
        for g, c in layer.items():
            if c.pixels is not None:
                px, py = (g[0] - x0) * GRID_SKIP, (y1 - g[1]) * GRID_SKIP
                piece = cv2.cvtColor(c.pixels.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                if g not in dst:
                    piece = cv2.addWeighted(piece, 0.6, np.full_like(piece, (255, 120, 0)), 0.4, 0)
                canvas[py:py + GRID_SKIP, px:px + GRID_SKIP] = piece
    for g in set(src) & set(dst):
        a, b = src[g], dst[g]
        if a.pixels is not None and a.pixels.min() == a.pixels.max() and b.hashes == a.hashes:
            continue
        px, py = (g[0] - x0) * GRID_SKIP, (y1 - g[1]) * GRID_SKIP
        colour = (0, 200, 0) if a.hashes == b.hashes else (0, 0, 255)
        cv2.rectangle(canvas, (px, py), (px + GRID_SKIP - 1, py + GRID_SKIP - 1), colour, 1)
    return canvas
