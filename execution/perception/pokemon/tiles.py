from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from execution.artifact import StrategistArtifact, update
from execution.perception.matching import cell_key, split_quarters
from execution.perception.pokemon.grid import GRID_SKIP, cell_box, draw_box, split_grid
from execution.perception.pokemon.prompts import PLAYER_SIDES, TILE_IN_CELL_PROMPT, TILE_TAGS, fill_player_note
from utils import log_error, parse_key_value

TileKey = Tuple[int, int, str]

PLAYER_CELL = (0, 0)
QUARTER_OFFSETS = {
    "top_left": (0, 0),
    "top_right": (GRID_SKIP // 2, 0),
    "bottom_left": (0, GRID_SKIP // 2),
    "bottom_right": (GRID_SKIP // 2, GRID_SKIP // 2),
}


@dataclass
class Tile:
    image: Optional[np.ndarray]
    hash: Optional[str]
    tag: str
    description: Optional[str] = None


UNKNOWN_TILE = Tile(image=None, hash=None, tag="unknown", description=None)
PLAYER_TILE = Tile(image=None, hash=None, tag="player", description="the player")
NON_OBJECT_TAGS = ("ground", "ground decoration", "obstacle", "player", "unknown")
HALVES = {
    frozenset(("top_left", "top_right")): "top half",
    frozenset(("bottom_left", "bottom_right")): "bottom half",
    frozenset(("top_left", "bottom_left")): "left half",
    frozenset(("top_right", "bottom_right")): "right half",
}


def parse_tag(output: str) -> str:
    tag = (parse_key_value(output, "Category") or "unknown").strip().lower()
    return tag if tag in TILE_TAGS else "unknown"


def parse_description(output: str) -> Optional[str]:
    description = parse_key_value(output, "Description")
    if description is None or description.strip().lower() in ("none", ""):
        return None
    return description.strip().strip('"')


def cell_part(quarters) -> str:
    quarters = frozenset(quarters)
    if len(quarters) == 4:
        return ""
    if quarters in HALVES:
        return f" ({HALVES[quarters]})"
    names = [q.replace("_", "-") for q in ("top_left", "top_right", "bottom_left", "bottom_right") if q in quarters]
    return f" ({', '.join(names)} quarter{'s' if len(names) > 1 else ''})"


def verbalize_tiles(identified: Dict[TileKey, Tile]) -> str:
    by_cell: Dict[Tuple[int, int], Dict[str, Dict[str, Any]]] = {}
    for (x, y, quarter), tile in identified.items():
        if tile.tag in NON_OBJECT_TAGS:
            continue
        entry = by_cell.setdefault((x, y), {}).setdefault(tile.tag, {"quarters": set(), "descriptions": []})
        entry["quarters"].add(quarter)
        if tile.description and tile.description not in entry["descriptions"]:
            entry["descriptions"].append(tile.description)
    lines = ["You are at (0, 0). x increases to the right, y increases upward. Each (x, y) is one grid cell; "
             "ground and obstacles are not listed."]
    if not by_cell:
        lines.append("No objects, entrances or other notable things are visible.")
    for (x, y) in sorted(by_cell, key=lambda c: (-c[1], c[0])):
        for tag, entry in by_cell[(x, y)].items():
            details = f": {'; '.join(entry['descriptions'])}" if entry["descriptions"] else ""
            lines.append(f"({x}, {y}){cell_part(entry['quarters'])} {tag}{details}")
    return "\n".join(lines)


PLAYER_NEIGHBOURS = {
    (-1, 1): "diagonally up and to the left of",
    (0, 1): "directly above",
    (1, 1): "diagonally up and to the right of",
    (-1, 0): "directly to the left of",
    (1, 0): "directly to the right of",
    (-1, -1): "diagonally down and to the left of",
    (0, -1): "directly below",
    (1, -1): "diagonally down and to the right of",
}


def verbalize_cell(identified: Dict[TileKey, Tile], cell: Tuple[int, int]) -> str:
    groups: Dict[Tuple[str, Optional[str]], set] = {}
    for quarter in QUARTER_OFFSETS:
        tile = identified.get((cell[0], cell[1], quarter))
        if tile is not None:
            groups.setdefault((tile.tag, tile.description), set()).add(quarter)
    parts = []
    for (tag, description), quarters in groups.items():
        where = cell_part(quarters).strip().strip("()")
        parts.append(tag + (f" ({description})" if description else "") + (f" in the {where}" if where else ""))
    return "; ".join(parts) if parts else "nothing recognised"


def verbalize_neighbourhood(identified: Dict[TileKey, Tile]) -> str:
    return "\n".join(
        f"- ({x}, {y}), {position} the player: {verbalize_cell(identified, (x, y))}"
        for (x, y), position in PLAYER_NEIGHBOURS.items()
    )


class TileRecognizer(StrategistArtifact):
    # Not picklable: callers must call set_vlm_call() again after load().
    _TRANSIENT = ("vlm_call",)

    def __init__(
        self,
        name: str,
        game: str = "pokemon_red",
        parameters: Optional[dict] = None,
        max_new_tokens: int = 4000,
    ):
        self.vlm_call: Optional[Callable[..., Any]] = None
        self.name = name
        self.game = game
        self.parameters = parameters
        self.max_new_tokens = max_new_tokens
        self.tiles: List[Tile] = []
        self._by_hash: Dict[str, Tile] = {}

    def set_vlm_call(self, vlm_call: Callable[..., Any]) -> None:
        self.vlm_call = vlm_call

    def diff(self, old: "TileRecognizer") -> Dict[str, Any]:
        old_hashes = {tile.hash for tile in old.tiles}
        new_tiles = [tile for tile in self.tiles if tile.hash not in old_hashes]
        by_tag: Dict[str, int] = {}
        for tile in new_tiles:
            by_tag[tile.tag] = by_tag.get(tile.tag, 0) + 1
        return {"new_tiles": new_tiles, "new_tile_count_by_tag": by_tag}

    def _split(self, frame: np.ndarray) -> Dict[TileKey, np.ndarray]:
        pieces = {}
        for (x, y), cell in split_grid(frame).items():
            for quarter, piece in split_quarters(cell).items():
                pieces[(x, y, quarter)] = piece
        return pieces

    def identify_tiles(self, frame: np.ndarray) -> Dict[TileKey, Tile]:
        identified = {}
        for key, piece in self._split(frame).items():
            if key[:2] == PLAYER_CELL:
                identified[key] = PLAYER_TILE
            else:
                identified[key] = self._by_hash.get(cell_key(piece), UNKNOWN_TILE)
        return identified

    def record_tiles(self, frame: np.ndarray, episode_number: int) -> None:
        pieces = self._split(frame)
        new_groups: Dict[str, List[TileKey]] = {}
        for key, piece in pieces.items():
            if key[:2] == PLAYER_CELL:
                continue
            tile_hash = cell_key(piece)
            if tile_hash in self._by_hash:
                continue
            new_groups.setdefault(tile_hash, []).append(key)
        if not new_groups:
            return
        if self.vlm_call is None:
            log_error(f"Tile recognizer {self.name!r} has {len(new_groups)} new tiles to classify but no vlm_call. "
                      "Call set_vlm_call first.", self.parameters)

        cells = split_grid(frame)
        hashes, texts, images = [], [], []
        for tile_hash, members in new_groups.items():
            x, y, quarter = next((m for m in members if m[:2] not in PLAYER_SIDES), members[0])
            x0, y0, _, _ = cell_box(x, y)
            dx, dy = QUARTER_OFFSETS[quarter]
            left, top = x0 + dx, y0 + dy
            size = GRID_SKIP // 2
            hashes.append(tile_hash)
            texts.append(fill_player_note(TILE_IN_CELL_PROMPT.replace("[QUARTER]", quarter.replace("_", "-")), (x, y)))
            images.append([
                draw_box(frame, left, top, left + size - 1, top + size - 1),
                pieces[(x, y, quarter)],
                cells[(x, y)],
            ])

        outputs = self.vlm_call(texts=texts, images=images, max_new_tokens=self.max_new_tokens)
        self._add_tiles([
            Tile(image=np.array(pieces[new_groups[tile_hash][0]]), hash=tile_hash, tag=parse_tag(output),
                 description=parse_description(output))
            for tile_hash, output in zip(hashes, outputs)
        ], episode_number=episode_number)

    @update
    def _add_tiles(self, tiles: List[Tile]) -> None:
        for tile in tiles:
            self.tiles.append(tile)
            self._by_hash[tile.hash] = tile
