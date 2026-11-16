"""
The NPC registry's people on the global map, the way doors are: drawn on the map pictures and numbered on the text
map, with a list under it. Everything here works on what the global map's own functions return (``render_segment``,
``prompt_block``), so the map's code is the same with the registry on or off; without the registry nothing here is
called.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from execution.perception.pokemon.grid import GRID_SKIP
from execution.strategist.pokemon.global_map import (DIRECTIONS, LEGEND, OCCUPIED, SCREEN_UNIT, GlobalMap, MapCell,
                                                     ascii_window, door_lines, glyph, render_segment, sector,
                                                     segment_bounds)

Cell = Tuple[int, int]

#: People are drawn in yellow (BGR): not used by the green, blue, red, cyan, orange or magenta marks; Pokemon in light
#: blue, people who may have moved in grey; squares that may hold someone not seen whole yet get a magenta ring.
NPC_COLOUR = (0, 255, 255)
POKEMON_COLOUR = (255, 200, 120)
STALE_COLOUR = (160, 160, 160)
MAYBE_COLOUR = (255, 0, 255)
#: Map pictures for the report and the people legend: text this size per unit of ``scale``.
LABEL_FONT = 0.33
#: The legend when the NPC registry's people are on the map: they get numbers, so the tile table's guesses get o.
PEOPLE_LEGEND = (LEGEND.replace("n person", "o maybe a person, not confirmed")
                 + "   1-9 a person (listed under the map)   + a person who may have moved")
MAX_PEOPLE_LINES = 8
#: When the map did not place the episode's opening screen, the people are told as remembered: "held" (a text box or
#: a menu, no fade: the player stands where the map last placed them, so positions still hold) or "away" (a fade, or
#: the map lost the player: they may be somewhere else, so no positions).
WHERE_NOTES = {
    "held": "Nobody can be recognised on the screen now (it is not free roam): this is the list as remembered, "
            "measured from where you stand.",
    "away": "The screen now could not be placed on the map (you may have gone through a door): this is the list as "
            "remembered for the area the map last had you in, without positions.",
}
#: The text map's first line, as ``prompt_block`` writes it: the window follows it.
MAP_HEADER = "MAP OF EVERYTHING YOU HAVE SEEN IN THIS AREA (north is up, one character per step):"


def _label(canvas: np.ndarray, text: str, x: int, y: int, colour, scale: int) -> None:
    """``text`` in ``colour`` on a black box whose bottom-left corner is at (x, y), kept inside the picture."""
    import cv2

    font = LABEL_FONT * scale
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font, 1)
    x = max(0, min(x, canvas.shape[1] - w - 2))
    y = max(h + 2, min(y, canvas.shape[0] - 1))
    cv2.rectangle(canvas, (x, y - h - 2), (x + w + 1, y + 1), (0, 0, 0), -1)
    cv2.putText(canvas, text, (x + 1, y - 1), cv2.FONT_HERSHEY_SIMPLEX, font, colour, 1, cv2.LINE_AA)


def _dashed_box(canvas: np.ndarray, a: Cell, b: Cell, colour, dash: int, width: int = 1) -> None:
    import cv2

    (x0, y0), (x1, y1) = a, b
    for x in range(x0, x1 + 1, 2 * dash):
        cv2.line(canvas, (x, y0), (min(x + dash - 1, x1), y0), colour, width)
        cv2.line(canvas, (x, y1), (min(x + dash - 1, x1), y1), colour, width)
    for y in range(y0, y1 + 1, 2 * dash):
        cv2.line(canvas, (x0, y), (x0, min(y + dash - 1, y1)), colour, width)
        cv2.line(canvas, (x1, y), (x1, min(y + dash - 1, y1)), colour, width)


def _wrap(text: str, width: int, font: float) -> List[str]:
    """``text`` cut into lines that fit ``width`` pixels."""
    import cv2

    lines, line = [], ""
    for word in text.split(" "):
        trial = f"{line} {word}" if line else word
        if line and cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, font, 1)[0][0] > width:
            lines.append(line)
            line = "    " + word
        else:
            line = trial
    return lines + [line]


def person_tag(mark: Any) -> str:
    """The label of a person on the map pictures: P (person) or K (a Pokemon) and the stable number; ? if they may
    have moved."""
    return f"{'K' if mark.kind == 'pokemon' else 'P'}{mark.number}{'?' if mark.stale else ''}"


def people_legend(marks: Iterable[Any]) -> List[str]:
    """One line per person drawn on a map picture."""
    out = []
    for m in marks:
        talked = {"yes": f"talked to x{m.times_talked}" + (" (same words)" if m.same_words else ""),
                  "probably": f"probably talked to x{m.times_talked}", "no": "not talked to"}[m.talked]
        bits = [talked, f"faces {m.facing}" if m.facing else "", f"last seen ep {m.last_seen[0]}",
                "may have moved" if m.stale else "", f"seen on {len(m.earlier) + 1}+ squares" if m.earlier else ""]
        out.append(f"{person_tag(m)} {m.name}: " + ", ".join(b for b in bits if b))
    return out


def _colour(m: Any):
    return STALE_COLOUR if m.stale else POKEMON_COLOUR if m.kind == "pokemon" else NPC_COLOUR


def render_with_people(gmap: GlobalMap, segment: Optional[int] = None, people: Iterable[Any] = (),
                       maybe: Iterable[Cell] = (), scale: int = 1, legend: bool = False,
                       **render_kwargs: Any) -> Optional[np.ndarray]:
    """
    The global map's own picture (``render_segment`` with ``render_kwargs``), enlarged ``scale`` times, with the
    registry's people marks (``NPCRegistry.map_people``) drawn on it: a yellow box labelled P and the person's number
    (solid once talked to here, dashed if not, thin if only probably), a tick on the side the person faces, faded dots
    on earlier squares, grey with a ? if they may have moved. ``maybe`` are squares that may hold someone not seen
    whole yet (a magenta ring). ``legend`` adds a strip naming each person.
    """
    import cv2

    canvas = render_segment(gmap, segment, **render_kwargs)
    if canvas is None:
        return None
    si = gmap.segment if segment is None else segment
    x0, x1, y0, y1 = render_kwargs.get("bounds") or segment_bounds(gmap, si)   # the bounds the picture used
    if scale != 1:
        canvas = cv2.resize(canvas, (canvas.shape[1] * scale, canvas.shape[0] * scale),
                            interpolation=cv2.INTER_NEAREST)
    size = GRID_SKIP * scale

    def inside(g):
        return x0 <= g[0] <= x1 and y0 <= g[1] <= y1
    def box(g):
        return (g[0] - x0) * size, (y1 - g[1]) * size
    def outline(g, colour, width, inset=0):
        px, py = box(g)
        cv2.rectangle(canvas, (px + inset, py + inset), (px + size - 1 - inset, py + size - 1 - inset), colour, width)
    def centre(g):
        return box(g)[0] + size // 2, box(g)[1] + size // 2

    for g in maybe:
        if inside(g):
            gx, gy = box(g)
            cv2.circle(canvas, (gx + size // 2, gy + size // 2), size // 2, MAYBE_COLOUR, 1)
            _label(canvas, "?", gx + size // 2, gy + 2 * scale + 6, MAYBE_COLOUR, scale)
    people = list(people)
    for m in people:
        colour = _colour(m)
        for k, g in enumerate(m.earlier):                       # where the person stood before, fading out
            if inside(g):
                faded = canvas.copy()
                cv2.circle(faded, centre(g), max(2, size // 6), colour, -1)
                cv2.addWeighted(faded, (0.5, 0.3, 0.15)[min(k, 2)], canvas, 1 - (0.5, 0.3, 0.15)[min(k, 2)], 0, canvas)
        if m.earlier and inside(m.earlier[0]) and inside(m.square):
            cv2.line(canvas, centre(m.earlier[0]), centre(m.square), colour, 1, cv2.LINE_4)
        if m.spread is not None:
            (ax, ay), (bx, by) = box((m.spread[0], m.spread[3])), box((m.spread[2], m.spread[1]))
            _dashed_box(canvas, (ax, ay), (bx + size - 1, by + size - 1), colour, 2 * scale)
        if not inside(m.square):
            continue
        px, py = box(m.square)
        inset = scale
        if m.talked == "yes":
            outline(m.square, colour, 2, inset)
        elif m.talked == "probably":
            outline(m.square, colour, 1, inset)
        else:
            _dashed_box(canvas, (px + inset, py + inset), (px + size - 1 - inset, py + size - 1 - inset), colour,
                        2 * scale, scale)
        if m.facing in DIRECTIONS:                                # a tick on the side the person faces
            fx, fy = SCREEN_UNIT[m.facing]
            cx, cy, r = px + size // 2, py + size // 2, size // 2
            tip = (cx + fx * (r + 2 * scale), cy + fy * (r + 2 * scale))
            side = (fy * 3 * scale, fx * 3 * scale)
            base = (cx + fx * (r - 2 * scale), cy + fy * (r - 2 * scale))
            pts = np.array([tip, (base[0] + side[0], base[1] + side[1]), (base[0] - side[0], base[1] - side[1])],
                           np.int32)
            cv2.fillPoly(canvas, [pts], colour)
        above = py - 1 if py >= size // 2 else py + size + size // 2
        _label(canvas, person_tag(m), px, above, colour, scale)
    if legend and people:
        rows = []
        for m, text in zip(people, people_legend(people)):
            rows += [(line, _colour(m)) for line in _wrap(text, canvas.shape[1] - 8, LABEL_FONT * scale)]
        line_h = 14 * scale
        strip = np.zeros((line_h * len(rows) + 6, canvas.shape[1], 3), np.uint8)
        for i, (text, colour) in enumerate(rows):
            cv2.putText(strip, text, (4, line_h * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT * scale, colour, 1,
                        cv2.LINE_AA)
        canvas = np.vstack([canvas, strip])
    return canvas


# -- the text map ---------------------------------------------------------------------------------

def people_glyph(c: Optional[MapCell]) -> str:
    """``glyph`` with the registry on: a square the tile table took for a person prints ``o`` (``#`` if it is also
    an obstacle, a statue), since the people the registry knows are numbered over it."""
    if c is not None and c.kind == OCCUPIED:
        return "#" if "obstacle" in set(c.tags) else "o"
    return glyph(c)


def people_marks(gmap: GlobalMap, people: Iterable[Any]) -> Tuple[List[Any], Dict[Cell, str]]:
    """The people of this area nearest first, and what the text map prints on their squares: their number in the
    list (1-9, * beyond), or + for someone who may have moved."""
    px, py = gmap.pos
    ordered = sorted(people, key=lambda m: (abs(m.square[0] - px) + abs(m.square[1] - py), m.number))
    grid = {}
    for i, m in enumerate(ordered):
        grid.setdefault(m.square, "+" if m.stale else str(i + 1) if i < 9 else "*")
    return ordered, grid


def people_window(gmap: GlobalMap, grid: Dict[Cell, str], rx: int = 12, ry: int = 7) -> str:
    """``ascii_window`` with the people of ``grid`` (from ``people_marks``) on their squares."""
    seg, (px, py) = gmap.current, gmap.pos
    rows = []
    for y in range(py + ry, py - ry - 1, -1):
        rows.append("".join("@" if (x, y) == (px, py) else grid.get((x, y)) or people_glyph(seg.get((x, y)))
                            for x in range(px - rx, px + rx + 1)))
    return "\n".join(rows)


def people_lines(gmap: GlobalMap, ordered: List[Any], where: str = "here") -> List[str]:
    """The people list under the text map, in the words of ``door_lines``; ``ordered`` from ``people_marks``.
    ``where``: "here" (the screen now was placed), else a key of ``WHERE_NOTES``."""
    if not ordered:
        return []
    px, py = gmap.pos
    lines = ["People in this area (x = steps to the right, y = steps up, measured from you; the quoted names are "
             "labels the player does not know):"]
    if where in WHERE_NOTES:
        lines.append(WHERE_NOTES[where])
    for i, m in enumerate(ordered[:MAX_PEOPLE_LINES]):
        dx, dy = m.square[0] - px, m.square[1] - py
        steps = abs(dx) + abs(dy)
        what = ", ".join(x for x in (m.kind if m.kind != "unknown" else "", f"faces {m.facing}" if m.facing else "")
                         if x)
        if m.talked == "no":
            talked = "NOT talked to yet"
        else:
            talked = (f"{'probably ' if m.talked == 'probably' else ''}talked to here {m.times_talked} time(s)"
                      + (", the same words each time: nothing new to hear" if m.same_words else ""))
        at = "" if where == "away" else f' at ({dx}, {dy}), {steps} steps {sector(dx, dy) if steps else "here"}'
        lines.append(f'- {i + 1} "{m.name}"{f" ({what})" if what else ""}{at}, {m.when}: {talked}.')
    if len(ordered) > MAX_PEOPLE_LINES:
        lines.append(f"...and {len(ordered) - MAX_PEOPLE_LINES} more.")
    lines.append(f"Not yet talked to here: {sum(m.talked == 'no' for m in ordered)} of {len(ordered)}.")
    return lines


def with_people(text: str, gmap: GlobalMap, people: Iterable[Any], where: str = "here") -> str:
    """
    The map section ``prompt_block`` wrote (``text``, for ``gmap`` as it is now) with the registry's people of this
    area (``NPCRegistry.map_people``): numbered on the text map, the people legend, and the people list after the
    doors. A part of ``text`` that is not where it is looked for is left as it is; the list then goes at the end.
    ``where`` as in ``people_lines``.
    """
    ordered, grid = people_marks(gmap, people)
    lines = text.split("\n")
    plain = ascii_window(gmap).split("\n")
    if MAP_HEADER in lines:
        at = lines.index(MAP_HEADER) + 1
        if lines[at:at + len(plain)] == plain:
            lines[at:at + len(plain)] = people_window(gmap, grid).split("\n")
    if "Legend: " + LEGEND in lines:
        lines[lines.index("Legend: " + LEGEND)] = "Legend: " + PEOPLE_LEGEND
    added = people_lines(gmap, ordered, where)
    if added:
        doors = door_lines(gmap)
        squares = next((i for i, line in enumerate(lines) if line.startswith("Squares mapped here: ")), None)
        if squares is not None and lines[squares + 1:squares + 1 + len(doors)] == doors:
            lines[squares + 1 + len(doors):squares + 1 + len(doors)] = added
        else:
            lines += added
    return "\n".join(lines)
