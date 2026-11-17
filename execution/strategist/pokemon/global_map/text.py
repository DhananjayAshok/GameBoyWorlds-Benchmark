from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .model import BLOCKED, LEDGE, OCCUPIED, UNKNOWN, WALKABLE, is_door
from .render import segment_bounds
from .routes import plan_route, render_plan

if TYPE_CHECKING:
    from .model import Cell, GlobalMap, MapCell


LEGEND = ('@ you   . walkable   # blocked   = ledge (jump down only)   D door/stairs   " tall grass   '
          '~ water   n person   % not yet recognised   ? never seen')
SECTORS = ("east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east")


def glyph(c: Optional[MapCell]) -> str:
    if c is None:
        return "?"
    tags = set(c.tags)
    if c.kind == OCCUPIED:
        return "n"
    if is_door(c):
        return "D"
    if "water" in tags:
        return "~"
    if c.kind == WALKABLE and "tall grass" in tags:
        return '"'
    return {WALKABLE: ".", BLOCKED: "#", LEDGE: "=", UNKNOWN: "%"}.get(c.kind, "?")


def ascii_window(gmap: GlobalMap, rx: int = 12, ry: int = 7) -> str:
    seg, (px, py) = gmap.current, gmap.pos
    rows = []
    for y in range(py + ry, py - ry - 1, -1):
        rows.append("".join("@" if (x, y) == (px, py) else glyph(seg.get((x, y)))
                            for x in range(px - rx, px + rx + 1)))
    return "\n".join(rows)


def sector(dx: int, dy: int) -> str:
    import math
    return SECTORS[int(round(math.atan2(dy, dx) / (math.pi / 4))) % 8]


def frontier_targets(gmap: GlobalMap, local_kinds: Optional[Dict[Cell, str]] = None,
                     limit: int = 8) -> Tuple[List[Dict[str, Any]], Counter]:
    # dx, dy are relative to the player: x right, y up
    from execution.perception.pokemon.navigation import shortest_path, in_bounds

    px, py = gmap.pos
    mass: Counter = Counter()
    nearest: Dict[str, Dict[str, Any]] = {}
    for g in gmap.frontier():
        dx, dy = g[0] - px, g[1] - py
        if (dx, dy) == (0, 0):
            continue
        on_screen = local_kinds is not None and in_bounds((dx, dy))
        if on_screen and shortest_path(local_kinds, (0, 0), [(dx, dy)]) is None:
            continue
        t = {"dx": dx, "dy": dy, "steps": abs(dx) + abs(dy), "sector": sector(dx, dy), "on_screen": on_screen}
        mass[t["sector"]] += 1
        if t["sector"] not in nearest or t["steps"] < nearest[t["sector"]]["steps"]:
            nearest[t["sector"]] = t
    out = sorted(nearest.values(), key=lambda t: t["steps"])
    return out[:limit], mass


def segment_ascii(gmap: GlobalMap, segment: int, mark: Optional[Cell] = None) -> str:
    seg, bounds = gmap.segments[segment], segment_bounds(gmap, segment)
    if bounds is None:
        return "(empty)"
    x0, x1, y0, y1 = bounds
    return "\n".join("".join("X" if (x, y) == mark else glyph(seg.get((x, y))) for x in range(x0, x1 + 1))
                     for y in range(y1, y0 - 1, -1))


def _door_text(gmap: GlobalMap, segment: int) -> str:
    leads = [gmap.area_label(d["leads_to"]) for d in gmap.door_list(segment) if d["leads_to"] is not None]
    unentered = sum(1 for d in gmap.door_list(segment) if d["leads_to"] is None)
    parts = [f"one to {name}" for name in leads] + ([f"{unentered} not entered yet"] if unentered else [])
    return ", ".join(parts) or "none known"


def route_lines(gmap: GlobalMap) -> List[str]:
    lines = []
    for r in gmap.routes:
        plan = plan_route(gmap, r["targets"])
        lines.append(f"Route you looked up to \"{r['request']}\":\n" + render_plan(gmap, r["targets"], plan))
    return lines


MAX_DOOR_LINES = 8


def door_lines(gmap: GlobalMap) -> List[str]:
    px, py = gmap.pos
    doors = []
    for door in gmap.door_list():
        dx, dy = min(((g[0] - px, g[1] - py) for g in door["cells"]), key=lambda d: abs(d[0]) + abs(d[1]))
        doors.append((abs(dx) + abs(dy), dx, dy, door["leads_to"]))
    if not doors:
        return []
    lines = ["Doors and stairs in this area (x = steps to the right, y = steps up, measured from you):"]
    for steps, dx, dy, leads_to in sorted(doors, key=lambda d: d[:3])[:MAX_DOOR_LINES]:
        behind = (f"leads to {gmap.area_label(leads_to)}"
                  + (", fully explored" if gmap.info(leads_to).explored is not None else "")
                  if leads_to is not None else "NOT entered yet")
        lines.append(f"- ({dx}, {dy}), {steps} steps {sector(dx, dy) if steps else 'here'}: {behind}")
    return lines


def elsewhere_lines(gmap: GlobalMap, groups: List[Dict[str, Any]]) -> List[str]:
    lines = []
    for i, g in enumerate(groups, 1):
        areas = []
        for si, plan in g["areas"]:
            n = len(gmap.frontier(si))
            edge = f"{n} unexplored edge square{'s' if n != 1 else ''}"
            if plan is None:
                areas.append(f"{gmap.area_label(si)} ({edge})")
                continue
            chain = " -> ".join([gmap.area_label(plan["hops"][0]["from"])] + [gmap.area_label(h["to"])
                                                                              for h in plan["hops"]])
            areas.append(f"{gmap.area_label(si)} ({edge}; {chain}, {len(plan['hops'])} door(s))")
        if g["plan"] is None:
            lines.append(f"{i}. No way there is known from here yet: " + "; ".join(areas) + ".")
            continue
        lines.append(f"{i}. " + "; ".join(areas) + ".")
        lines.append("   " + render_plan(gmap, [g["target"]], g["plan"]).splitlines()[-1])
    return lines
