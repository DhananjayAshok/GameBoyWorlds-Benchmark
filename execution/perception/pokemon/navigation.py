import re
from collections import deque
from typing import Collection, Dict, Iterable, List, Optional, Tuple

from gameboy_worlds.interface.pokemon.actions import HARD_MAX_STEPS

from execution.perception.pokemon.grid import LEFT_COL, TOP_ROW

Cell = Tuple[int, int]

WALKABLE = "walkable"
BLOCKED = "blocked"
LEDGE = "ledge"

WALKABLE_TAGS = {
    "ground", "ground decoration", "tall grass", "entrance", "staircase", "elevator",
    "special ground tile", "player",
}
LEDGE_TAGS = {"ledge"}
QUARTERS = ("top_left", "top_right", "bottom_left", "bottom_right")

X_RANGE = range(LEFT_COL, LEFT_COL + 10)
Y_RANGE = range(TOP_ROW - 8, TOP_ROW + 1)
DIRECTIONS: Dict[str, Cell] = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}
MAX_STEPS_PER_MOVE = HARD_MAX_STEPS
COORDINATE_PATTERN = re.compile(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")


def in_bounds(cell: Cell) -> bool:
    return cell[0] in X_RANGE and cell[1] in Y_RANGE


def cell_kinds(identified, ledge_patterns: Optional[Collection[Tuple[Optional[str], ...]]] = None) -> Dict[Cell, str]:
    """
    Walkable, ledge or blocked for every on-screen cell. ``ledge_patterns`` are cells (their four
    quarter hashes, in QUARTERS order) the player has been seen hopping down over: the tile labels
    can call a ledge an obstacle, and a cell once crossed is a ledge whatever its labels say.
    """
    tags: Dict[Cell, set] = {}
    hashes: Dict[Cell, Dict[str, Optional[str]]] = {}
    for (x, y, quarter), tile in identified.items():
        tags.setdefault((x, y), set()).add(tile.tag)
        hashes.setdefault((x, y), {})[quarter] = tile.hash
    kinds = {}
    for cell, cell_tags in tags.items():
        if cell_tags <= WALKABLE_TAGS:
            kinds[cell] = WALKABLE
        elif cell_tags <= WALKABLE_TAGS | LEDGE_TAGS:
            kinds[cell] = LEDGE
        elif ledge_patterns and tuple(hashes[cell].get(q) for q in QUARTERS) in ledge_patterns:
            kinds[cell] = LEDGE
        else:
            kinds[cell] = BLOCKED
    return kinds


def neighbours(cell: Cell) -> List[Tuple[str, Cell]]:
    return [(d, (cell[0] + dx, cell[1] + dy)) for d, (dx, dy) in DIRECTIONS.items()]


def shortest_path(kinds: Dict[Cell, str], start: Cell, goals: Iterable[Cell]) -> Optional[List[Tuple[str, str, Cell]]]:
    goals = set(goals)
    if start in goals:
        return []
    previous: Dict[Cell, Tuple[Cell, str, str]] = {start: None}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        for direction, nxt in neighbours(cell):
            if not in_bounds(nxt):
                continue
            kind = kinds.get(nxt, BLOCKED)
            if kind == WALKABLE:
                landing, move = nxt, "walk"
            elif kind == LEDGE and direction == "down":
                landing, move = (nxt[0], nxt[1] - 1), "jump"
                if not in_bounds(landing) or kinds.get(landing, BLOCKED) != WALKABLE:
                    continue
            else:
                continue
            if landing in previous:
                continue
            previous[landing] = (cell, direction, move)
            if landing in goals:
                path = []
                while previous[landing] is not None:
                    parent, d, m = previous[landing]
                    path.append((d, m, landing))
                    landing = parent
                return path[::-1]
            queue.append(landing)
    return None


def compress_path(path: List[Tuple[str, str, Cell]]) -> List[Tuple[str, int, str]]:
    runs: List[Tuple[str, int, str]] = []
    for direction, move, _ in path:
        if move == "walk" and runs and runs[-1][0] == direction and runs[-1][2] == "walk" and runs[-1][1] < MAX_STEPS_PER_MOVE:
            runs[-1] = (direction, runs[-1][1] + 1, "walk")
        else:
            runs.append((direction, 1, move))
    return runs


def approach_cells(kinds: Dict[Cell, str], target: Cell) -> Dict[Cell, str]:
    cells = {}
    for direction, cell in neighbours(target):
        if in_bounds(cell) and kinds.get(cell) == WALKABLE:
            facing = next(d for d, (dx, dy) in DIRECTIONS.items() if (cell[0] + dx, cell[1] + dy) == target)
            cells[cell] = facing
    return cells


def edge_cells(kinds: Dict[Cell, str], direction: str) -> List[Cell]:
    if direction == "up":
        cells = [(x, max(Y_RANGE)) for x in X_RANGE]
    elif direction == "down":
        cells = [(x, min(Y_RANGE)) for x in X_RANGE]
    elif direction == "left":
        cells = [(min(X_RANGE), y) for y in reversed(Y_RANGE)]
    else:
        cells = [(max(X_RANGE), y) for y in reversed(Y_RANGE)]
    return [cell for cell in cells if kinds.get(cell) == WALKABLE]


def is_on_edge(cell: Cell, direction: str) -> bool:
    return {
        "up": cell[1] == max(Y_RANGE),
        "down": cell[1] == min(Y_RANGE),
        "left": cell[0] == min(X_RANGE),
        "right": cell[0] == max(X_RANGE),
    }[direction]


def parse_cell(text: Optional[str]) -> Optional[Cell]:
    if not text:
        return None
    match = COORDINATE_PATTERN.search(text)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def verbalize_edges(kinds: Dict[Cell, str]) -> str:
    lines = []
    for direction in DIRECTIONS:
        cells = edge_cells(kinds, direction)
        listed = ", ".join(f"({x}, {y})" for x, y in cells) if cells else "none"
        lines.append(f"- {direction} edge: {listed}")
    return "\n".join(lines)
