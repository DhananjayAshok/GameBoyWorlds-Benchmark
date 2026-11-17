from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, Optional, Tuple

import numpy as np

from execution.perception.pokemon.grid import GRID_SKIP, LEFT_COL, TOP_ROW

if TYPE_CHECKING:
    from .model import Cell

PLAYER_COL, PLAYER_ROW = -LEFT_COL, TOP_ROW          # screen column/row of local (0, 0)

# A move is placed by pixel matching only if at most this fraction of pixels differ at the best shift.
PIXEL_THRESHOLD = 0.12
# A move is placed by edge matching only if at most this fraction of edge positions disagree.
STRUCTURE_THRESHOLD = 0.25
# Frames with fewer edge pixels than this are blank (a fade, a black transition frame).
MIN_EDGE_DENSITY = 0.02
# Step counts whose best shifts differ by less than this are treated as indistinguishable.
AMBIGUITY_MARGIN = 0.03
# How far, in pixels, a frame may be caught off the cell grid mid-scroll (a step scrolls 16 px).
PX_SLACK = 7
# Shifts that leave less than this fraction of the screen overlapping are not scored.
MIN_OVERLAP = 0.3
# Screen box of the player sprite (y0, y1, x0, x1), generous upward because the sprite can sit a few px high.
PLAYER_BOX = (PLAYER_ROW * GRID_SKIP - 6, (PLAYER_ROW + 1) * GRID_SKIP,
              PLAYER_COL * GRID_SKIP, (PLAYER_COL + 1) * GRID_SKIP)
# Screen-space unit of each direction, in pixels per cell: (+x right, +y DOWN).
SCREEN_UNIT: Dict[str, Cell] = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
# A screen that scrolled with no move pressed is searched this many cells each way.
SCRIPTED_MAX_STEPS = 4
# A cell whose pixels are all at or below this level is black.
BLACK_LEVEL = 40
# A screen is off the 16px grid only if its best alignment beats the next best by this much.
GRID_MARGIN = 0.02


def _screen(frame: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.squeeze(np.asarray(frame)))


def _uniqueness(a: np.ndarray, dx: int, dy: int, size: int) -> float:
    a = a[dy:, dx:]
    h, w = (a.shape[0] // size) * size, (a.shape[1] // size) * size
    blocks = a[:h, :w].reshape(h // size, size, w // size, size).swapaxes(1, 2).reshape(-1, size * size)
    return len({b.tobytes() for b in blocks}) / len(blocks)


def grid_offset(frame: np.ndarray) -> Optional[Cell]:
    a = _screen(frame)
    fine = sorted((_uniqueness(a, dx, dy, 8), (dx, dy)) for dy in range(0, 8, 2) for dx in range(0, 8, 2))
    if fine[1][0] - fine[0][0] < GRID_MARGIN:
        return None
    bx, by = fine[0][1]
    coarse = sorted((_uniqueness(a, dx, dy, GRID_SKIP), (dx, dy)) for dx in (bx, bx + 8) for dy in (by, by + 8))
    if coarse[1][0] - coarse[0][0] < GRID_MARGIN:
        return None
    return coarse[0][1]


def _still_lit(piece: np.ndarray, prior: np.ndarray) -> np.ndarray:
    new, prior, half = np.squeeze(piece), np.squeeze(prior), GRID_SKIP // 2
    lit = np.zeros(new.shape, dtype=bool)
    for qy in (0, half):
        for qx in (0, half):
            q = (slice(qy, qy + half), slice(qx, qx + half))
            lit[q] = new[q].max() > BLACK_LEVEL or prior[q].max() <= BLACK_LEVEL
    return lit


def grid_residual(shift: Cell) -> Cell:
    return tuple(-d if d <= GRID_SKIP // 2 else GRID_SKIP - d for d in shift)


def _local_cells(a: np.ndarray, shift: Cell = (0, 0)) -> Iterable[Tuple[Cell, np.ndarray]]:
    dx, dy = shift
    ci, cj = int(dx > GRID_SKIP // 2), int(dy > GRID_SKIP // 2)
    for row in range((a.shape[0] - dy) // GRID_SKIP):
        for col in range((a.shape[1] - dx) // GRID_SKIP):
            piece = a[dy + row * GRID_SKIP:dy + (row + 1) * GRID_SKIP, dx + col * GRID_SKIP:dx + (col + 1) * GRID_SKIP]
            yield (col + ci + LEFT_COL, TOP_ROW - (row + cj)), piece


def pixel_diff(before: np.ndarray, after: np.ndarray, sx: int, sy: int) -> Optional[float]:
    # after[y, x] is compared with before[y + sy, x + sx]: screen space, +x right, +y down
    a, b = _screen(before), _screen(after)
    h, w = a.shape
    if abs(sx) >= w or abs(sy) >= h:
        return None
    ny, nx = slice(max(0, -sy), h - max(0, sy)), slice(max(0, -sx), w - max(0, sx))
    oy, ox = slice(max(0, sy), h - max(0, -sy)), slice(max(0, sx), w - max(0, -sx))
    new, old = b[ny, nx], a[oy, ox]
    if new.size < MIN_OVERLAP * a.size:
        return None
    keep = np.ones(new.shape, dtype=bool)
    y0, y1, x0, x1 = PLAYER_BOX
    for oy0, ox0 in ((0, 0), (-sy, -sx)):            # the sprite now, and where it was before
        r0, r1 = y0 + oy0 - ny.start, y1 + oy0 - ny.start
        c0, c1 = x0 + ox0 - nx.start, x1 + ox0 - nx.start
        keep[max(0, r0):max(0, r1), max(0, c0):max(0, c1)] = False
    return float((new != old)[keep].mean())


def edges(frame: np.ndarray) -> np.ndarray:
    a = _screen(frame)
    e = np.zeros(a.shape, dtype=bool)
    e[:, :-1] |= a[:, 1:] != a[:, :-1]
    e[:-1, :] |= a[1:, :] != a[:-1, :]
    return e


def has_structure(frame: np.ndarray) -> bool:
    return edges(frame).mean() >= MIN_EDGE_DENSITY


def structure_diff(before_edges: np.ndarray, after_edges: np.ndarray, sx: int, sy: int) -> Optional[float]:
    a, b = before_edges, after_edges
    h, w = a.shape
    if abs(sx) >= w or abs(sy) >= h:
        return None
    ny, nx = slice(max(0, -sy), h - max(0, sy)), slice(max(0, -sx), w - max(0, sx))
    oy, ox = slice(max(0, sy), h - max(0, -sy)), slice(max(0, sx), w - max(0, -sx))
    new, old = b[ny, nx], a[oy, ox]
    if new.size < MIN_OVERLAP * a.size:
        return None
    keep = np.ones(new.shape, dtype=bool)
    y0, y1, x0, x1 = PLAYER_BOX
    for oy0, ox0 in ((0, 0), (-sy, -sx)):
        r0, r1 = y0 + oy0 - ny.start, y1 + oy0 - ny.start
        c0, c1 = x0 + ox0 - nx.start, x1 + ox0 - nx.start
        keep[max(0, r0 - 1):max(0, r1 + 1), max(0, c0 - 1):max(0, c1 + 1)] = False
    either = (new | old) & keep
    n = int(either.sum())
    if n == 0:
        return None
    return float(((new ^ old) & keep).sum() / n)


def register(before: np.ndarray, after: np.ndarray, direction: Optional[str], max_steps: int,
             reported: Optional[int] = None, residual: Cell = (0, 0)
             ) -> Optional[Tuple[Cell, Cell, bool]]:
    # pixel matching first; edge matching is the fallback that survives palette changes
    for structural in (False, True):
        placed = _register(before, after, direction, max_steps, reported, residual, structural=structural)
        if placed is not None:
            ux, uy = SCREEN_UNIT.get(direction, (0, 0))
            return (ux * placed[0], -uy * placed[0]), placed[1], structural
        if not structural and direction not in SCREEN_UNIT:
            scripted = _register_scripted(before, after, residual)
            if scripted is not None:
                return scripted[0], scripted[1], False
    return None


def _register_scripted(before: np.ndarray, after: np.ndarray, residual: Cell) -> Optional[Tuple[Cell, Cell]]:
    far, near = SCRIPTED_MAX_STEPS * GRID_SKIP + PX_SLACK, GRID_SKIP + PX_SLACK
    best: Dict[Cell, Tuple[float, Cell]] = {}
    for sx in range(-far, far + 1):
        for sy in range(-far, far + 1):
            if (sx, sy) == (0, 0) or min(abs(sx), abs(sy)) > near:
                continue
            kx, ky = int(round((sx + residual[0]) / GRID_SKIP)), int(round((sy + residual[1]) / GRID_SKIP))
            rx, ry = sx + residual[0] - GRID_SKIP * kx, sy + residual[1] - GRID_SKIP * ky
            if abs(rx) > PX_SLACK or abs(ry) > PX_SLACK:
                continue
            diff = pixel_diff(before, after, sx, sy)
            if diff is not None and ((kx, ky) not in best or diff < best[(kx, ky)][0]):
                best[(kx, ky)] = (diff, (rx, ry))
    if not best:
        return None
    k = min(best, key=lambda c: best[c][0])
    diff, new_residual = best[k]
    if diff > PIXEL_THRESHOLD or sum(d <= diff + AMBIGUITY_MARGIN for d, _ in best.values()) > 1:
        return None
    return (k[0], -k[1]), new_residual


def _register(before: np.ndarray, after: np.ndarray, direction: Optional[str], max_steps: int,
              reported: Optional[int], residual: Cell, structural: bool) -> Optional[Tuple[int, Cell]]:
    if direction in SCREEN_UNIT:
        ux, uy = SCREEN_UNIT[direction]
        # far shifts are where repeating textures alias, so look at most one cell past the reported count
        cap = max_steps if reported is None else min(max_steps, reported + 1)
        span = range(-PX_SLACK, cap * GRID_SKIP + PX_SLACK + 1)
    else:
        ux, uy, span = 0, 0, range(-PX_SLACK, PX_SLACK + 1)
    eb, ea = (edges(before), edges(after)) if structural else (None, None)
    threshold = STRUCTURE_THRESHOLD if structural else PIXEL_THRESHOLD
    best_by_k: Dict[int, Tuple[float, int]] = {}
    for s in span:
        if not (ux or uy or s == 0):
            continue
        diff = (structure_diff(eb, ea, ux * s, uy * s) if structural
                else pixel_diff(before, after, ux * s, uy * s))
        if diff is None:
            continue
        # s = 16 k + (residual_after - residual_before) along the axis, |residual| <= PX_SLACK
        along_before = residual[0] * ux + residual[1] * uy
        k = int(round((s + along_before) / GRID_SKIP))
        if k not in best_by_k or diff < best_by_k[k][0]:
            best_by_k[k] = (diff, s)
    if not best_by_k:
        return None
    k_best = min(best_by_k, key=lambda k: best_by_k[k][0])
    diff, s = best_by_k[k_best]
    if diff > threshold or k_best < 0:
        return None
    tied = [k for k, (d, _) in best_by_k.items() if d <= diff + AMBIGUITY_MARGIN and k >= 0]
    if len(tied) > 1:
        if reported is None or reported not in tied:
            return None
        k_best = reported
        s = best_by_k[k_best][1]
    along_after = s + (residual[0] * ux + residual[1] * uy) - GRID_SKIP * k_best
    new_residual = (residual[0], residual[1])
    if ux:
        new_residual = (along_after * ux, residual[1])
    elif uy:
        new_residual = (residual[0], along_after * uy)
    return k_best, new_residual
