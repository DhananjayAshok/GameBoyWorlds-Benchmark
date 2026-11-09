"""
Overworld sprites (people) on the 16 px grid: cutting a sprite out of a screen, comparing two sprites by the shape
of their dark outline, and finding the squares a person stands on. Pure numpy, so it can be tested on synthetic frames.

A sprite is drawn ``SPRITE_PARAMS[game]["y_offset"]`` px above the top of its grid square. Two sprites are compared by
their black-pixel masks (so palette and background do not matter), with a left/right mirror allowed, because the game
draws a person facing left by flipping the sprite that faces right.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from execution.perception.pokemon.grid import GRID_SKIP, cell_box

Cell = Tuple[int, int]

#: ``auto``: the largest mask distance (out of 256 px) for an automatic same-look match. ``maybe``: the top of the
#: band that is left to a model; a distance above it is a different look. Measured on Red: the same picture and frame
#: differ by a median of 0, different pictures never come closer than 34.
SPRITE_PARAMS: Dict[str, Dict[str, Any]] = {
    "default": dict(y_offset=4, auto=12, maybe=33, min_quarters=3, scan_step=2, scan_range=8, min_black=24),
    "pokemon_red": dict(),
    "pokemon_crystal": dict(),      # not measured: re-check the thresholds before enabling there
}


def params(game: str) -> Dict[str, Any]:
    return {**SPRITE_PARAMS["default"], **SPRITE_PARAMS.get(game, {})}


def _screen(frame: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.squeeze(np.asarray(frame)))


def black_level(frame: np.ndarray) -> int:
    """The darkest grey on the screen: sprite outlines are drawn in it (0 on Red)."""
    return int(_screen(frame).min())


def sprite_window(frame: np.ndarray, cell: Cell, offset: Cell = (0, 0), y_offset: int = 4) -> Optional[np.ndarray]:
    """The 16x16 picture of the sprite standing on ``cell``, shifted by ``offset`` px; None if any part is off screen."""
    screen = _screen(frame)
    x0, y0, _, _ = cell_box(*cell)
    left, top = x0 + offset[0], y0 - y_offset + offset[1]
    if left < 0 or top < 0 or left + GRID_SKIP > screen.shape[1] or top + GRID_SKIP > screen.shape[0]:
        return None
    return screen[top:top + GRID_SKIP, left:left + GRID_SKIP].copy()


def black_mask(window: np.ndarray, level: int) -> np.ndarray:
    return np.asarray(window) <= level


def mask_distance(a: np.ndarray, b: np.ndarray, valid: Optional[np.ndarray] = None) -> Tuple[int, bool]:
    """``(distance, mirrored)``: differing black pixels out of 256, the better of as-is and left/right mirrored."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    plain, flipped = a ^ b, a[:, ::-1] ^ b
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        scale = 256.0 / max(1, int(valid.sum()))
        plain, flipped = plain[valid], flipped[valid]
    else:
        scale = 1.0
    d_plain, d_flip = int(round(plain.sum() * scale)), int(round(flipped.sum() * scale))
    return (d_flip, True) if d_flip < d_plain else (d_plain, False)


def stack_distances(keys: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Distance from ``mask`` to every key in ``keys`` (V, 16, 16), and whether the mirror won for each."""
    plain = (keys ^ mask).sum(axis=(1, 2))
    flipped = (keys ^ mask[:, ::-1]).sum(axis=(1, 2))
    return np.minimum(plain, flipped), flipped < plain


def tag_counts(identified: Dict[Tuple[int, int, str], Any], cell: Cell) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for quarter in ("top_left", "top_right", "bottom_left", "bottom_right"):
        tile = identified.get((cell[0], cell[1], quarter))
        if tile is not None:
            counts[tile.tag] = counts.get(tile.tag, 0) + 1
    return counts


def find_npc_cells(identified: Dict[Tuple[int, int, str], Any], min_quarters: int = 3) -> List[Cell]:
    """Squares other than the player's with at least ``min_quarters`` of 4 quarters tagged ``npc``."""
    cells = sorted({(x, y) for (x, y, _) in identified if (x, y) != (0, 0)})
    return [c for c in cells if tag_counts(identified, c).get("npc", 0) >= min_quarters]


def scan_cells(identified: Dict[Tuple[int, int, str], Any], exclude: Sequence[Cell], min_quarters: int = 2) -> List[Cell]:
    """Squares that might hold a person the tags did not fully catch: partly tagged ``npc``, or not recognised yet."""
    skip = set(exclude)
    out = []
    for cell in sorted({(x, y) for (x, y, _) in identified if (x, y) != (0, 0)}):
        if cell in skip:
            continue
        counts = tag_counts(identified, cell)
        if counts.get("npc", 0) >= min_quarters or counts.get("unknown", 0) >= min_quarters:
            out.append(cell)
    return out


def full_window_ok(identified: Dict[Tuple[int, int, str], Any], cell: Cell) -> bool:
    """Only a sprite whose four quarters are all the person (or three, with ground behind the fourth) may start a
    new look or view: a half-hidden sprite would store a bad picture."""
    counts = tag_counts(identified, cell)
    if counts.get("npc", 0) == 4:
        return True
    return counts.get("npc", 0) >= 3 and (counts.get("ground", 0) + counts.get("ground decoration", 0)) >= 1


def scan_for_views(frame: np.ndarray, keys: np.ndarray, cells: Sequence[Cell], level: int, scan_range: int = 8,
                   step: int = 2, auto: int = 12, y_offset: int = 4, min_black: int = 24
                   ) -> List[Tuple[Cell, Cell, int, int]]:
    """For each cell, the shifted window that best matches a known view: ``[(cell, offset, view index, distance)]``.
    Finds a person caught mid-step, or on a background the tile table does not know yet."""
    found: List[Tuple[Cell, Cell, int, int]] = []
    if len(keys) == 0:
        return found
    for cell in cells:
        best: Optional[Tuple[int, Cell, int]] = None
        for dy in range(-scan_range, scan_range + 1, step):
            for dx in range(-scan_range, scan_range + 1, step):
                window = sprite_window(frame, cell, (dx, dy), y_offset)
                if window is None:
                    continue
                mask = black_mask(window, level)
                if int(mask.sum()) < min_black:
                    continue
                distances, _ = stack_distances(keys, mask)
                index = int(distances.argmin())
                if best is None or int(distances[index]) < best[0]:
                    best = (int(distances[index]), (dx, dy), index)
        if best is not None and best[0] <= auto:
            found.append((cell, best[1], best[2], best[0]))
    return found
