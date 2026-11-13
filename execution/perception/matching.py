import hashlib
from itertools import combinations
from typing import Dict, Hashable, List, Optional, Tuple

import numpy as np

LEVELS = 4


def frame_key(frame: np.ndarray) -> Optional[str]:
    if frame is None or not hasattr(frame, "tobytes"):
        return None
    try:
        return hashlib.blake2b(np.ascontiguousarray(frame).tobytes(), digest_size=16).hexdigest()
    except Exception:
        return None
QUARTERS = ("top_left", "top_right", "bottom_left", "bottom_right")


def cell_key(cell: np.ndarray) -> Optional[str]:
    return frame_key(cell)


def quantize(cell: np.ndarray, levels: int = LEVELS) -> np.ndarray:
    return (cell.astype(np.uint16) * levels // 256).astype(np.uint8)


def mean_abs_diff(a: np.ndarray, b: np.ndarray, levels: int = LEVELS) -> float:
    qa = quantize(a, levels).astype(np.int16)
    qb = quantize(b, levels).astype(np.int16)
    return float(np.abs(qa - qb).mean() / (levels - 1))


def split_quarters(cell: np.ndarray) -> Dict[str, np.ndarray]:
    h, w = cell.shape[0] // 2, cell.shape[1] // 2
    return {
        "top_left": cell[:h, :w],
        "top_right": cell[:h, w:],
        "bottom_left": cell[h:, :w],
        "bottom_right": cell[h:, w:],
    }


def split_strips(cell: np.ndarray) -> Dict[str, np.ndarray]:
    h, w = cell.shape[0] // 2, cell.shape[1] // 2
    return {"top": cell[:h], "bottom": cell[h:], "left": cell[:, :w], "right": cell[:, w:]}


def is_repeated_tile(strip: np.ndarray) -> bool:
    h, w = strip.shape[:2]
    if w > h:
        first, second = strip[:, : w // 2], strip[:, w // 2 :]
    else:
        first, second = strip[: h // 2], strip[h // 2 :]
    return cell_key(first) == cell_key(second)


LEVELS_OF_MATCH =("tile", "row_strip", "column_strip", "cell")


def group_levels(
    cells: Dict[Hashable, np.ndarray],
) -> Dict[str, Dict[str, List[Tuple[Hashable, str]]]]:
    levels = {level: {} for level in LEVELS_OF_MATCH}
    for key, cell in cells.items():
        for part, piece in split_quarters(cell).items():
            levels["tile"].setdefault(cell_key(piece), []).append((key, part))
        for part, piece in split_strips(cell).items():
            level = "row_strip" if part in ("top", "bottom") else "column_strip"
            if is_repeated_tile(piece):
                continue
            levels[level].setdefault(cell_key(piece), []).append((key, part))
        levels["cell"].setdefault(cell_key(cell), []).append((key, "full"))
    return levels


def quarter_keys(cell: np.ndarray) -> Tuple[Optional[str], ...]:
    quarters = split_quarters(cell)
    return tuple(cell_key(quarters[q]) for q in QUARTERS)


def matching_quarters(a: Tuple[Optional[str], ...], b: Tuple[Optional[str], ...]) -> List[str]:
    return [q for q, ka, kb in zip(QUARTERS, a, b) if ka == kb]


def group_cells(
    cells: Dict[Hashable, np.ndarray],
) -> Tuple[List[List[Hashable]], Dict[int, Dict[int, List[str]]]]:
    by_keys = {}
    for key, cell in cells.items():
        by_keys.setdefault(quarter_keys(cell), []).append(key)
    signatures = list(by_keys)
    groups = [by_keys[s] for s in signatures]
    related = {i: {} for i in range(len(groups))}
    for i, j in combinations(range(len(groups)), 2):
        shared = matching_quarters(signatures[i], signatures[j])
        if len(shared) >= 2:
            related[i][j] = shared
            related[j][i] = shared
    return groups, related


def _distinct_per_row(values: np.ndarray) -> np.ndarray:
    ordered = np.sort(values, axis=1)
    return 1 + (np.diff(ordered, axis=1) != 0).sum(axis=1)


def shade_bank(images: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixels, per-image shade indices and shade counts of equally sized pieces, for shade_candidates."""
    pixels = np.stack([np.asarray(image).reshape(-1) for image in images]).astype(np.int64)
    shade_index = np.stack([np.unique(row, return_inverse=True)[1] for row in pixels]).astype(np.int64)
    return pixels, shade_index, _distinct_per_row(pixels)


def shade_candidates(bank: Tuple[np.ndarray, np.ndarray, np.ndarray], piece: np.ndarray,
                     min_shades: int = 2, ordered: bool = True) -> np.ndarray:
    """
    Indices of the bank pieces that ``piece`` is a recolouring of: the same picture in other shades, as when
    a Game Boy Color palette changes with the time of day. Shades may merge in either direction (night
    palettes map several day shades onto one), but each shade maps to one shade and their light-to-dark
    order is kept (``ordered``; some night palettes do reorder shades, so callers may drop it when other
    evidence says the screen was recoloured). Only the candidates that keep the most distinct shades are returned. Pieces with fewer than
    ``min_shades`` shades on either side match nothing: a single-shade piece is a recolouring of anything.
    """
    pixels, shade_index, n_shades = bank
    p = np.asarray(piece).reshape(-1).astype(np.int64)
    p_index = np.unique(p, return_inverse=True)[1].astype(np.int64)
    n_p = int(p_index.max()) + 1
    if n_p < min_shades or len(pixels) == 0:
        return np.zeros(0, dtype=np.int64)
    # piece = g(bank piece): each bank shade becomes one piece shade, in order
    forward = _distinct_per_row(shade_index * 256 + p[None, :]) == n_shades
    if ordered:
        forward &= np.all(np.diff(p[np.argsort(pixels, axis=1, kind="stable")], axis=1) >= 0, axis=1)
    # bank piece = g(piece): each piece shade becomes one bank shade, in order
    backward = _distinct_per_row(p_index[None, :] * 256 + pixels) == n_p
    if ordered:
        backward &= np.all(np.diff(pixels[:, np.argsort(p, kind="stable")], axis=1) >= 0, axis=1)
    ok = (n_shades >= min_shades) & (forward | backward)
    if not ok.any():
        return np.zeros(0, dtype=np.int64)
    kept = np.where(ok, np.minimum(n_shades, n_p), -1)
    return np.flatnonzero(kept == kept.max())
