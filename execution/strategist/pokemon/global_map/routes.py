from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

from .model import DIRECTIONS, LEDGE, OCCUPIED, UNKNOWN, WALKABLE

if TYPE_CHECKING:
    from .model import Cell, GlobalMap


# Cost of stepping onto a square of each kind; walls and squares never seen cannot be crossed.
STEP_COSTS = {WALKABLE: 1, UNKNOWN: 3, OCCUPIED: 5}
# A door the player cannot reach over mapped squares is costed at this many times its straight-line distance.
UNMAPPED_PENALTY = 2
# A straight line between two consecutive recorded positions counts as walked, up to this many squares long.
MAX_WALK_FILL = 8
# A door's departure square that is missing or a wall moves to the nearest walkable square within this distance.
SNAP_RADIUS = 2
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


def walked_squares(gmap: GlobalMap, segment: int) -> set:
    points = [p for _, s, p in gmap.trail if s == segment]
    walked = set(points) | set(gmap.visits[segment])
    for a, b in zip(points, points[1:]):
        if (a[0] == b[0]) != (a[1] == b[1]) and abs(a[0] - b[0]) + abs(a[1] - b[1]) <= MAX_WALK_FILL:
            sx, sy = (b[0] > a[0]) - (b[0] < a[0]), (b[1] > a[1]) - (b[1] < a[1])
            walked |= {(a[0] + sx * i, a[1] + sy * i) for i in range(abs(b[0] - a[0]) + abs(b[1] - a[1]))}
    return walked


def _snap(gmap: GlobalMap, segment: int, cell: Cell, walked: set) -> Optional[Cell]:
    seg = gmap.segments[segment]

    def usable(q: Cell) -> bool:
        return q in walked or (q in seg and seg[q].kind in STEP_COSTS)
    if usable(cell):
        return cell
    near = [(max(abs(ox), abs(oy)), abs(ox) + abs(oy), (cell[0] + ox, cell[1] + oy))
            for ox in range(-SNAP_RADIUS, SNAP_RADIUS + 1) for oy in range(-SNAP_RADIUS, SNAP_RADIUS + 1)]
    return next((q for _, _, q in sorted(near) if usable(q)), None)


def path_in_segment(gmap: GlobalMap, segment: int, start: Cell,
                    goal: Cell, walked: Optional[set] = None) -> Optional[Tuple[int, List[Tuple[str, str]]]]:
    import heapq

    seg = gmap.segments[segment]
    walked = walked_squares(gmap, segment) if walked is None else walked
    best: Dict[Cell, int] = {start: 0}
    previous: Dict[Cell, Optional[Tuple[Cell, str, str]]] = {start: None}
    heap = [(0, start)]
    while heap:
        cost, cell = heapq.heappop(heap)
        if cell == goal:
            moves = []
            while previous[cell] is not None:
                parent, direction, move = previous[cell]
                moves.append((direction, move))
                cell = parent
            return cost, moves[::-1]
        if cost > best[cell]:
            continue
        for direction, (dx, dy) in DIRECTIONS.items():
            nxt, move = (cell[0] + dx, cell[1] + dy), "walk"
            c = seg.get(nxt)
            if nxt == goal or nxt in walked:
                step = 1
            elif c is None:
                continue
            elif c.kind == LEDGE and direction == "down":
                nxt, move, step = (nxt[0], nxt[1] - 1), "jump", 2
                landing = seg.get(nxt)
                if nxt != goal and nxt not in walked and (landing is None or landing.kind not in STEP_COSTS):
                    continue
            elif c.kind in STEP_COSTS:
                step = STEP_COSTS[c.kind]
            else:
                continue
            if cost + step < best.get(nxt, cost + step + 1):
                best[nxt], previous[nxt] = cost + step, (cell, direction, move)
                heapq.heappush(heap, (cost + step, nxt))
    return None


def compress_moves(moves: List[Tuple[str, str]]) -> str:
    runs: List[List[Any]] = []
    for direction, move in moves:
        if move == "walk" and runs and runs[-1][:2] == [direction, "walk"]:
            runs[-1][2] += 1
        else:
            runs.append([direction, move, 1])
    return ", ".join(f"{d} {n}" if m == "walk" else f"jump {d} over the ledge" for d, m, n in runs)


def _exits(gmap: GlobalMap) -> Dict[int, List[Dict[str, Any]]]:
    exits: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for (_, a, pa, b, pb), direction in zip(gmap.warps, gmap.warp_moves):
        if a == b or not gmap.segments[a] or not gmap.segments[b]:
            continue
        if not any(e["depart"] == pa and e["to"] == b for e in exits[a]):
            exits[a].append({"from": a, "depart": pa, "direction": direction, "to": b, "arrive": pb})
    reverse = [{"from": e["to"], "depart": e["arrive"], "direction": OPPOSITE.get(e["direction"]), "to": a,
                "arrive": e["depart"]}
               for a, out in exits.items() for e in out if not any(x["to"] == a for x in exits.get(e["to"], []))]
    for e in reverse:
        exits[e["from"]].append(e)
    return exits


def plan_route(gmap: GlobalMap, targets: Iterable[int]) -> Optional[Dict[str, Any]]:
    import heapq
    import itertools

    targets = {gmap.resolve(t, (0, 0))[0] for t in targets}
    exits, order = _exits(gmap), itertools.count()
    walked = {si: walked_squares(gmap, si) for si in exits}
    heap = [(0, next(order), gmap.segment, gmap.pos, [])]
    settled = set()
    while heap:
        cost, _, segment, cell, hops = heapq.heappop(heap)
        if (segment, cell) in settled:
            continue
        settled.add((segment, cell))
        if segment in targets:
            return {"hops": hops, "cost": cost}
        for e in exits.get(segment, []):
            goal = _snap(gmap, segment, e["depart"], walked[segment])
            found = path_in_segment(gmap, segment, cell, goal, walked[segment]) if goal is not None else None
            if found is not None:
                e = dict(e, depart=goal)
            if found is None:
                walk, moves = UNMAPPED_PENALTY * (abs(e["depart"][0] - cell[0]) + abs(e["depart"][1] - cell[1])), None
            else:
                walk, moves = found
            heapq.heappush(heap, (cost + walk + 1, next(order), e["to"], e["arrive"],
                                  hops + [dict(e, start=cell, moves=moves)]))
    return None


def render_plan(gmap: GlobalMap, targets: Iterable[int], plan: Optional[Dict[str, Any]]) -> str:
    names = " or ".join(gmap.area_label(t) for t in sorted({gmap.resolve(t, (0, 0))[0] for t in targets}))
    if plan is None:
        return f"No way into {names} is known from here yet."
    hops = plan["hops"]
    if not hops:
        return f"You are already in {names}."
    areas = [gmap.area_label(hops[0]["from"])] + [gmap.area_label(h["to"]) for h in hops]
    lines = [f"Areas to pass through, in order: {' -> '.join(areas)} ({len(hops)} door(s))."]
    first = hops[0]
    dx, dy = first["depart"][0] - first["start"][0], first["depart"][1] - first["start"][1]
    through = (f"walk {first['direction']} through the door into {gmap.area_label(first['to'])}"
               if first["direction"] else f"go through the door into {gmap.area_label(first['to'])}")
    if first["moves"] is None:
        lines.append(f"Next: the way into {gmap.area_label(first['to'])} is from ({dx}, {dy}), but the squares between "
                     f"you and it are not all mapped yet; head that way, then {through}.")
    elif not first["moves"]:
        lines.append(f"Next: you are standing where you need to be; {through}.")
    else:
        lines.append(f"Next: go to ({dx}, {dy}) (x = steps right, y = steps up, from you), "
                     f"{len(first['moves'])} moves: {compress_moves(first['moves'])}; then {through}.")
    return "\n".join(lines)
