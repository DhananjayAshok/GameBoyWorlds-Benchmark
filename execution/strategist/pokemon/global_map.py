"""
A global grid map assembled from the per-screen local maps.

The screen is a 10 x 9 grid of 16 px cells with the player at local (0, 0); x grows to the right
and y grows upward, the same convention as the rest of the navigation code. Every time the player
moves, the new screen is placed on a global grid at the player's new position and its cells are
merged in. Cells never placed are "unseen", which is what lets the strategist be told where it has
not been yet.

Placement does not trust the movement action's own step count: measured against RAM it overcounts
about one move in five (``judge_movement`` reads a few changed pixels as a step). Instead the step
count is read off the screen: the new frame is compared with the last placed frame at every
candidate shift along the move direction, and the shift whose cells match best wins. A move is
merged only when that match is clearly good. Otherwise the player is looked for in the cells
already mapped (re-localisation), and if not found a new map segment is started — a door warp, a
cut-scene teleport, or anything ambiguous never corrupts an existing map.

Only screen pixels are used. Nothing here reads RAM.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from execution.artifact import StrategistArtifact, update
from execution.perception.matching import cell_key, split_quarters
from execution.perception.pokemon.grid import GRID_SKIP, LEFT_COL, TOP_ROW
from execution.perception.pokemon.navigation import LEDGE_TAGS, WALKABLE_TAGS

Cell = Tuple[int, int]

N_COLS, N_ROWS = 10, 9
PLAYER_COL, PLAYER_ROW = -LEFT_COL, TOP_ROW          # screen column/row of local (0, 0)
QUARTERS = ("top_left", "top_right", "bottom_left", "bottom_right")
DIRECTIONS: Dict[str, Cell] = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}

WALKABLE, BLOCKED, LEDGE, UNKNOWN, OCCUPIED = "walkable", "blocked", "ledge", "unknown", "occupied"

#: A move is placed only if, at the best shift, at most this fraction of pixels differ...
PIXEL_THRESHOLD = 0.12
#: ...or, for the palette-proof fallback, at most this fraction of edge positions disagree.
STRUCTURE_THRESHOLD = 0.25
#: Frames with fewer edge pixels than this are blank (a fade, a black transition frame).
MIN_EDGE_DENSITY = 0.02
#: Step counts whose best shifts differ by less than this are treated as indistinguishable.
AMBIGUITY_MARGIN = 0.03
#: How far, in pixels, a frame may be caught off the cell grid mid-scroll (a step scrolls 16 px).
PX_SLACK = 7
#: Shifts that leave less than this fraction of the screen overlapping are not scored.
MIN_OVERLAP = 0.3
#: Screen box of the player sprite (y0, y1, x0, x1), generous upward for the Gen-1 4 px offset.
PLAYER_BOX = (PLAYER_ROW * GRID_SKIP - 6, (PLAYER_ROW + 1) * GRID_SKIP,
              PLAYER_COL * GRID_SKIP, (PLAYER_COL + 1) * GRID_SKIP)
#: Screen-space unit of each direction, in pixels per cell: (+x right, +y DOWN).
SCREEN_UNIT: Dict[str, Cell] = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
#: Re-localisation needs this many distinctive matching cells, and this fraction of the voters.
RELOCALISE_MIN_CELLS = 30
#: A cell picture votes for at most this many places it appears in a segment.
RELOCALISE_MAX_FANOUT = 25
RELOCALISE_MIN_FRACTION = 0.7
#: Consecutive unplaceable free-roam frames before a new segment is opened.
LOST_PATIENCE = 3
#: A segment younger than this many placements keeps checking whether it overlaps an older one.
MERGE_WINDOW = 40


def _screen(frame: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.squeeze(np.asarray(frame)))


def _cell(a: np.ndarray, col: int, row: int) -> np.ndarray:
    return a[row * GRID_SKIP:(row + 1) * GRID_SKIP, col * GRID_SKIP:(col + 1) * GRID_SKIP]


def pixel_diff(before: np.ndarray, after: np.ndarray, sx: int, sy: int) -> Optional[float]:
    """
    Fraction of overlapping pixels that differ once the camera has moved by (sx, sy) pixels in
    screen space (+x right, +y down), i.e. after[y, x] is compared with before[y + sy, x + sx].
    The player sprite is masked at both of its positions. None when too little overlaps.
    """
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
    """
    Where neighbouring pixels differ. A palette change — the flashes that open a wild battle, a
    fade — recolours the screen without moving any of these, so registering on edges survives it.
    """
    a = _screen(frame)
    e = np.zeros(a.shape, dtype=bool)
    e[:, :-1] |= a[:, 1:] != a[:, :-1]
    e[:-1, :] |= a[1:, :] != a[:-1, :]
    return e


def has_structure(frame: np.ndarray) -> bool:
    """False for blank, black or nearly uniform frames, which say nothing about position."""
    return edges(frame).mean() >= MIN_EDGE_DENSITY


def structure_diff(before_edges: np.ndarray, after_edges: np.ndarray, sx: int, sy: int) -> Optional[float]:
    """
    Like ``pixel_diff`` but on edge maps: of the overlapping positions where either frame has an
    edge, the fraction where only one does. Uniform areas carry no weight either way.
    """
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
             reported: Optional[int] = None, residual: Cell = (0, 0)) -> Optional[Tuple[int, Cell, bool]]:
    """
    Pixel matching first, which is the more accurate of the two on ordinary frames; edge matching
    as the fallback, which survives palette changes (battle-intro flashes, fades) that defeat it.
    """
    placed = _register(before, after, direction, max_steps, reported, residual, structural=False)
    if placed is not None:
        return placed[0], placed[1], False
    placed = _register(before, after, direction, max_steps, reported, residual, structural=True)
    return None if placed is None else (placed[0], placed[1], True)


def _register(before: np.ndarray, after: np.ndarray, direction: Optional[str], max_steps: int,
              reported: Optional[int], residual: Cell, structural: bool) -> Optional[Tuple[int, Cell]]:
    """
    How many cells the player moved along ``direction`` between two free-roam frames, read off the
    screen, and how far off the cell grid the new frame was caught.

    Frames are sometimes captured mid-scroll, a few pixels short of a whole step, so the search is
    over pixel shifts rather than whole cells. ``residual`` is how far off the grid ``before`` was,
    in screen pixels. Returns (cells moved along ``direction``, residual of ``after``), or None when
    no shift fits, or when two step counts fit equally well and the reported count does not single
    one out (a uniform texture, for instance).
    """
    if direction in SCREEN_UNIT:
        ux, uy = SCREEN_UNIT[direction]
        # the reported count overcounts but almost never undercounts by more than a ledge hop,
        # and far shifts are where repeating textures alias, so do not look beyond it
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


def classify(tags: Iterable[str]) -> str:
    tags = set(tags)
    if "npc" in tags:
        return OCCUPIED
    if tags <= WALKABLE_TAGS:
        return WALKABLE
    if tags <= WALKABLE_TAGS | LEDGE_TAGS:
        return LEDGE
    if "unknown" in tags and (tags - {"unknown"}) <= WALKABLE_TAGS | LEDGE_TAGS:
        return UNKNOWN
    return BLOCKED


@dataclass
class MapCell:
    kind: str
    tags: Tuple[str, ...]
    hashes: Tuple[Optional[str], ...]
    first_seen: int
    last_seen: int
    pixels: Optional[np.ndarray] = None


class GlobalMap(StrategistArtifact):
    """Segments of global grid, the player's place on them, and the bookkeeping to explain both."""

    def __init__(self) -> None:
        self.segments: List[Dict[Cell, MapCell]] = [{}]
        self.segment = 0
        self.pos: Cell = (0, 0)
        self.anchor: Optional[np.ndarray] = None
        self.residual: Cell = (0, 0)
        self.lost = 0
        self.faded = False
        self.merged_into: Dict[int, Tuple[int, Cell]] = {}
        self.pending: Optional[Tuple[str, int, Optional[int]]] = None
        self.links: List[Tuple[int, Cell, int, Cell]] = []
        self.step = 0
        self.visits: List[Counter] = [Counter()]
        self.trail: List[Tuple[int, int, Cell]] = []
        self.new_cells_by_episode: Dict[int, int] = {}
        self._new_this_episode = 0
        self.stats: Counter = Counter()
        self.exploring_until: Optional[int] = None

    # ------------------------------------------------------------------
    # Building the map
    # ------------------------------------------------------------------

    def observe(self, frame: np.ndarray, identified: Dict[Tuple[int, int, str], Any]) -> int:
        """Merge the current screen into the current segment at the player's position."""
        seg = self.segments[self.segment]
        a = _screen(frame)
        px, py = self.pos
        added = 0
        for row in range(N_ROWS):
            for col in range(N_COLS):
                x, y = col + LEFT_COL, TOP_ROW - row
                pixels = _cell(a, col, row).copy()
                quarters = split_quarters(_cell(a, col, row))
                hashes = tuple(cell_key(quarters[q]) for q in QUARTERS)
                if (x, y) == (0, 0):
                    tags, kind = ("player",), WALKABLE
                else:
                    tags = tuple(sorted({identified[(x, y, q)].tag for q in QUARTERS if (x, y, q) in identified}))
                    kind = classify(tags)
                g = (px + x, py + y)
                old = seg.get(g)
                if old is None:
                    seg[g] = MapCell(kind, tags, hashes, self.step, self.step, pixels)
                    added += 1
                    continue
                old.last_seen = self.step
                # an NPC standing on a cell, or a tile not yet recognised, never overwrites what the
                # cell was already known to be
                if kind in (OCCUPIED, UNKNOWN) and old.kind not in (OCCUPIED, UNKNOWN):
                    continue
                if (x, y) == (0, 0) and old.kind != OCCUPIED:
                    continue
                old.kind, old.tags, old.hashes, old.pixels = kind, tags, hashes, pixels
        self.visits[self.segment][self.pos] += 1
        self.trail.append((self.step, self.segment, self.pos))
        self._new_this_episode += added
        self.stats["cells_added"] += added
        return added

    def on_step(self, frame: np.ndarray, identified, free_roam: bool, direction: Optional[str] = None,
                max_steps: int = 0, reported: Optional[int] = None, was_free_roam: bool = True,
                faded: bool = False) -> str:
        """
        Account for one environment step. ``frame``/``identified`` describe the screen after it;
        ``direction``/``max_steps`` describe the movement attempted, if any. Returns what happened.
        """
        self.step += 1
        if faded:
            # the screen went blank during this step: a door, stairs or a battle ending. Whatever is
            # on screen now must be recognised from the map, not slid from the last frame.
            self.faded = True
        if not free_roam:
            # a battle, dialogue or menu: nothing to place now. Remember a move that was under way
            # so the next free-roam frame can be placed relative to the last good one.
            if was_free_roam and direction in DIRECTIONS:
                self.pending = (direction, max_steps, reported)
            self.stats["not_free_roam"] += 1
            return "not_free_roam"
        if not has_structure(frame):
            # a fade or a black transition frame: it says nothing about where the player is, so it
            # neither moves the player nor breaks the map. A move under way stays pending.
            if direction in DIRECTIONS:
                self.pending = (direction, max_steps, reported)
            self.stats["blank_frames"] += 1
            return "blank"
        if self.anchor is None:
            self.anchor, self.residual = _screen(frame), (0, 0)
            self.observe(frame, identified)
            return "start"
        if self.pending is not None and direction not in DIRECTIONS:
            direction, max_steps, reported = self.pending
        placed = None if self.faded else register(self.anchor, frame, direction, max_steps, reported,
                                                  self.residual)
        if placed is not None:
            k, self.residual, structural = placed
            if k:
                unit = DIRECTIONS[direction]
                self.pos = (self.pos[0] + unit[0] * k, self.pos[1] + unit[1] * k)
            self.pending, self.lost, self.anchor = None, 0, _screen(frame)
            self.stats["placed_by_edges" if structural else "placed"] += 1
            if self.residual == (0, 0) and not structural:
                self.observe(frame, identified)
                self._try_merge(frame)
            elif structural:
                # recoloured (a flash or fade): the position holds, the colours do not
                self.stats["recoloured_frames"] += 1
            else:
                # caught mid-scroll: cells straddle the grid; filled in from the next on-grid frame
                self.stats["off_grid_frames"] += 1
            return "moved" if k else "stayed"
        found = self.relocalise(frame)
        if found is not None:
            if found != (self.segment, self.pos):
                self.links.append((self.segment, self.pos, found[0], found[1]))
            self.segment, self.pos = found
            self.residual, self.pending, self.lost, self.anchor = (0, 0), None, 0, _screen(frame)
            self.faded = False
            self.stats["relocalised"] += 1
            self.observe(frame, identified)
            return "relocalised"
        # Not placed and not recognised. Often this is one bad frame — the first frame after a
        # battle, still fading in — so wait a few frames before deciding this is somewhere new.
        self.lost += 1
        if direction in DIRECTIONS and self.pending is None:
            self.pending = (direction, max_steps, reported)
        if self.lost < LOST_PATIENCE:
            self.stats["lost_frames"] += 1
            return "lost"
        self.segments.append({})
        self.visits.append(Counter())
        self.links.append((self.segment, self.pos, len(self.segments) - 1, (0, 0)))
        self.segment, self.pos = len(self.segments) - 1, (0, 0)
        self.residual, self.pending, self.lost, self.anchor = (0, 0), None, 0, _screen(frame)
        self.faded = False
        self.segment_born = self.stats["placed"]
        self.stats["new_segment"] += 1
        self.observe(frame, identified)
        return "new_segment"

    def _try_merge(self, frame: np.ndarray) -> None:
        """Loop closure: if this young segment turns out to overlap an older one, fold it in."""
        if self.segment == 0 or self.stats["placed"] - getattr(self, "segment_born", 0) > MERGE_WINDOW:
            return
        found = self.relocalise(frame, exclude=self.segment)
        if found is None:
            return
        target, target_pos = found
        dx, dy = target_pos[0] - self.pos[0], target_pos[1] - self.pos[1]
        src, dst = self.segments[self.segment], self.segments[target]
        for (x, y), c in src.items():
            g = (x + dx, y + dy)
            old = dst.get(g)
            if old is None or old.kind in (OCCUPIED, UNKNOWN) or c.last_seen >= old.last_seen:
                dst[g] = c
        for (x, y), v in self.visits[self.segment].items():
            self.visits[target][(x + dx, y + dy)] += v
        self.trail = [(t, target, (p[0] + dx, p[1] + dy)) if si == self.segment else (t, si, p)
                      for t, si, p in self.trail]
        self.merged_into[self.segment] = (target, (dx, dy))
        self.segments[self.segment] = {}
        self.visits[self.segment] = Counter()
        self.segment, self.pos = target, target_pos
        self.stats["merged"] += 1

    def relocalise(self, frame: np.ndarray, exclude: Optional[int] = None) -> Optional[Tuple[int, Cell]]:
        """
        Find the player in the cells already mapped by voting on exact cell matches. Blank cells
        (the black around an indoor room) match everywhere, so they do not vote, and the number of
        matches required scales with how much of the screen has content.
        """
        a = _screen(frame)
        local: Dict[Cell, Tuple[Optional[str], ...]] = {}
        for row in range(N_ROWS):
            for col in range(N_COLS):
                x, y = col + LEFT_COL, TOP_ROW - row
                piece = _cell(a, col, row)
                if (x, y) == (0, 0) or piece.min() == piece.max():
                    continue
                quarters = split_quarters(piece)
                local[(x, y)] = tuple(cell_key(quarters[q]) for q in QUARTERS)
        best = None
        for si, seg in enumerate(self.segments):
            if si == exclude or not seg:
                continue
            index: Dict[Tuple[Optional[str], ...], List[Cell]] = defaultdict(list)
            for g, c in seg.items():
                if c.kind != OCCUPIED:
                    index[c.hashes].append(g)
            votes: Counter = Counter()
            for (x, y), h in local.items():
                for g in index.get(h, ())[:RELOCALISE_MAX_FANOUT]:
                    votes[(g[0] - x, g[1] - y)] += 1
            if not votes:
                continue
            ranked = votes.most_common(2)
            offset, n = ranked[0]
            if len(ranked) > 1 and ranked[1][1] >= 0.8 * n:
                continue                       # two placements explain the screen equally well
            overlap = sum(1 for (x, y) in local if (offset[0] + x, offset[1] + y) in seg)
            # a small indoor room shows few content cells; ask for most of them rather than a fixed 30
            needed = min(RELOCALISE_MIN_CELLS, max(10, int(0.6 * len(local))))
            if n >= needed and n >= RELOCALISE_MIN_FRACTION * overlap:
                if best is None or n > best[0]:
                    best = (n, si, offset)
        return None if best is None else (best[1], best[2])

    def refresh(self, tile_lookup: Dict[str, Any]) -> int:
        """Re-classify cells whose tiles were unknown when seen, now that more tiles are known."""
        changed = 0
        for seg in self.segments:
            for c in seg.values():
                if c.kind != UNKNOWN:
                    continue
                tiles = [tile_lookup.get(h) for h in c.hashes]
                if any(t is None for t in tiles):
                    continue
                c.tags = tuple(sorted({t.tag for t in tiles}))
                c.kind = classify(c.tags)
                changed += 1
        return changed

    @update
    def end_episode(self, episode: int) -> int:
        """Close the episode's bookkeeping; decorated so the map is checkpointed every episode."""
        self.new_cells_by_episode[episode] = self._new_this_episode
        self._new_this_episode = 0
        return self.new_cells_by_episode[episode]

    def diff(self, old: "GlobalMap") -> Dict[str, Any]:
        return {"cells": self.n_cells - old.n_cells, "segments": len(self.segments) - len(old.segments)}

    # ------------------------------------------------------------------
    # Reading the map
    # ------------------------------------------------------------------

    @property
    def current(self) -> Dict[Cell, MapCell]:
        return self.segments[self.segment]

    @property
    def n_cells(self) -> int:
        return sum(len(s) for s in self.segments)

    @property
    def n_areas(self) -> int:
        return sum(1 for s in self.segments if s)

    def episodes_without_new_cells(self, episodes: Iterable[int]) -> int:
        """How many of the most recent episodes, newest first, added nothing."""
        count = 0
        for ep in sorted(episodes, reverse=True):
            if self.new_cells_by_episode.get(ep, 0) > 0:
                break
            count += 1
        return count

    def frontier(self) -> List[Cell]:
        """Walkable cells of the current segment with at least one neighbour never seen."""
        seg = self.current
        out = []
        for g, c in seg.items():
            if c.kind != WALKABLE:
                continue
            if any((g[0] + dx, g[1] + dy) not in seg for dx, dy in DIRECTIONS.values()):
                out.append(g)
        return out


# ----------------------------------------------------------------------------
# Feeding the map: every environment step, whoever takes it
# ----------------------------------------------------------------------------

def _movement(action, kwargs: Dict[str, Any], info: Dict[str, Any]) -> Tuple[Optional[str], int, Optional[int]]:
    """(direction, most cells it could have moved, the action's own reported count) for one step."""
    from gameboy_worlds.emulation.emulator import LowLevelActions
    from gameboy_worlds.interface.action import LowLevelAction
    from gameboy_worlds.interface.pokemon.actions import MoveStepsAction

    if action is MoveStepsAction or (isinstance(action, type) and issubclass(action, MoveStepsAction)):
        direction = kwargs.get("direction")
        steps = int(kwargs.get("steps") or 1)
        reported = (info.get("core", {}).get("action_return") or {}).get("n_steps_taken")
        # one extra cell allows for a ledge hop, which moves two cells on one press
        return direction, steps + 1, (int(reported) if reported is not None else None)
    if action is LowLevelAction or (isinstance(action, type) and issubclass(action, LowLevelAction)):
        arrows = {LowLevelActions.PRESS_ARROW_UP: "up", LowLevelActions.PRESS_ARROW_DOWN: "down",
                  LowLevelActions.PRESS_ARROW_LEFT: "left", LowLevelActions.PRESS_ARROW_RIGHT: "right"}
        direction = arrows.get(kwargs.get("low_level_action"))
        if direction is not None:
            return direction, 2, None
    return None, 0, None


class MappingEnvironment:
    """
    Wraps the environment the supervisor and its executors are handed, so that every step they
    take — moves, button presses, menus, battles — updates the global map. Behaviour is otherwise
    unchanged: every call is passed straight through, and a failure inside the map is logged and
    swallowed rather than allowed to disturb the run.
    """

    def __init__(self, env, global_map: GlobalMap, tile_recognizer, parameters: Optional[dict] = None) -> None:
        self._env = env
        self._map = global_map
        self._tiles = tile_recognizer
        self._parameters = parameters

    def step_high_level_action(self, action, **kwargs):
        from gameboy_worlds.emulation.pokemon import AgentState

        before = self._env.get_info()
        was_free = before.get("pokemon_core", {}).get("agent_state") == AgentState.FREE_ROAM
        result = self._env.step_high_level_action(action, **kwargs)
        try:
            info = result[4] or {}
            if not info.get("invalid_action"):
                frame = info["core"]["current_frame"]
                free = info.get("pokemon_core", {}).get("agent_state") == AgentState.FREE_ROAM
                direction, max_steps, reported = _movement(action, kwargs, info)
                passed = info["core"].get("transition_passed_frames")
                faded = passed is not None and any(not has_structure(f) for f in list(passed)[::2])
                identified = self._tiles.identify_tiles(frame) if free else {}
                self._map.on_step(frame, identified, free, direction, max_steps, reported, was_free, faded)
        except Exception as e:                                   # the map must never break a run
            from utils import log_warn
            log_warn(f"Global map update failed and was skipped: {type(e).__name__}: {e}", self._parameters)
        return result

    def step_str(self, input_str: str):
        action, kwargs = self._env.string_to_high_level_action(input_str)
        if action is None:
            return None, None, None, None, None
        return self.step_high_level_action(action, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._env, name)


# ----------------------------------------------------------------------------
# Rendering (debugging and the demo)
# ----------------------------------------------------------------------------

def render_segment(gmap: GlobalMap, segment: Optional[int] = None, trail: bool = True) -> Optional[np.ndarray]:
    """
    The segment as a picture assembled from the cells actually seen (BGR), unseen space hatched,
    frontier cells outlined green, the trail drawn in blue and the player boxed in red.
    """
    import cv2

    si = gmap.segment if segment is None else segment
    seg = gmap.segments[si]
    if not seg:
        return None
    xs = [g[0] for g in seg]; ys = [g[1] for g in seg]
    x0, x1, y0, y1 = min(xs) - 1, max(xs) + 1, min(ys) - 1, max(ys) + 1
    w, h = (x1 - x0 + 1) * GRID_SKIP, (y1 - y0 + 1) * GRID_SKIP
    canvas = np.full((h, w, 3), 60, np.uint8)
    canvas[::4, :, :] = 90                                       # hatch = never seen
    def box(g):
        return (g[0] - x0) * GRID_SKIP, (y1 - g[1]) * GRID_SKIP
    for g, c in seg.items():
        if c.pixels is None:
            continue
        px, py = box(g)
        canvas[py:py + GRID_SKIP, px:px + GRID_SKIP] = cv2.cvtColor(c.pixels.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if si == gmap.segment:
        for g in gmap.frontier():
            px, py = box(g)
            cv2.rectangle(canvas, (px, py), (px + GRID_SKIP - 1, py + GRID_SKIP - 1), (0, 200, 0), 1)
    if trail:
        pts = [p for _, s, p in gmap.trail if s == si]
        for a, b in zip(pts, pts[1:]):
            (ax, ay), (bx, by) = box(a), box(b)
            cv2.line(canvas, (ax + 8, ay + 8), (bx + 8, by + 8), (255, 120, 0), 1)
    if si == gmap.segment:
        px, py = box(gmap.pos)
        cv2.rectangle(canvas, (px, py), (px + GRID_SKIP - 1, py + GRID_SKIP - 1), (0, 0, 255), 2)
    return canvas


# ----------------------------------------------------------------------------
# What the strategist is told
# ----------------------------------------------------------------------------

LEGEND = ('@ you   . walkable   # blocked   = ledge (jump down only)   D door/stairs   " tall grass   '
          '~ water   n person   % not yet recognised   ? never seen')
SECTORS = ("east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east")
#: Episodes without a single new cell before exploration overrides the strategist's own task.
STALE_EPISODES = 3
#: Episodes in a row that did not complete before exploration overrides the strategist's own task.
FAILURE_EPISODES = 3
#: Once exploration takes over, it stays in charge for this many episodes, re-targeting each time.
EXPLORE_EPISODES = 3


def glyph(c: Optional[MapCell]) -> str:
    if c is None:
        return "?"
    tags = set(c.tags)
    if c.kind == OCCUPIED:
        return "n"
    if tags & {"entrance", "staircase", "elevator"}:
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
    """
    The nearest frontier cell in each compass direction, relative to the player (x right, y up),
    nearest first, plus how much frontier lies in each direction. A frontier cell on the current screen is kept only if the
    existing on-screen path search can reach it.
    """
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
    # one target per direction, so a strategist fixated on one side always sees the others
    out = sorted(nearest.values(), key=lambda t: t["steps"])
    return out[:limit], mass


def prompt_block(gmap: GlobalMap, recent_episodes: Iterable[int],
                 local_kinds: Optional[Dict[Cell, str]] = None,
                 failure_streak: int = 0, episode: Optional[int] = None,
                 failed_tasks: Iterable[str] = ()) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    The map section of the strategist's prompt, and — when the agent looks stuck — the frontier target
    that exploration should override the task with. Stuck means either ``STALE_EPISODES`` episodes in a
    row with nothing new mapped, or ``FAILURE_EPISODES`` episodes in a row that did not complete: an agent
    circling an obstacle still picks up a few squares each episode, so novelty alone misses it.
    """
    recent = list(recent_episodes)
    stale = gmap.episodes_without_new_cells(recent)
    targets, mass = frontier_targets(gmap, local_kinds)
    lines = [
        "MAP OF EVERYTHING YOU HAVE SEEN IN THIS AREA (north is up, one character per step):",
        ascii_window(gmap),
        "Legend: " + LEGEND,
        f"Squares mapped here: {len(gmap.current)} (in {gmap.n_areas} separate area(s) overall). "
        f"New squares found in the last episode: "
        f"{gmap.new_cells_by_episode.get(max(recent), 0) if recent else 0}; "
        f"episodes in a row with nothing new: {stale}.",
    ]
    if targets:
        lines.append("The nearest edge of the map you have NOT explored yet in each direction "
                     "(x = steps to the right, y = steps up, measured from you):")
        for t in targets:
            where = "on screen, reachable" if t["on_screen"] else "off screen"
            lines.append(f"- ({t['dx']}, {t['dy']}): {t['steps']} steps {t['sector']} ({where})")
        lines.append("Most unexplored edge lies: " +
                     ", ".join(f"{s} ({n} squares)" for s, n in mass.most_common(3)) + ".")
    else:
        lines.append("Every reachable edge of this area has been explored; the way on is a door, "
                     "stairs or a path you have not tried.")
    override = None
    triggered = stale >= STALE_EPISODES or failure_streak >= FAILURE_EPISODES
    # One episode of exploring is not enough to rule a direction out, and the strategist's own goals
    # pull it straight back to what was failing, so exploration stays in charge for a few episodes.
    active = (episode is not None and getattr(gmap, "exploring_until", None) is not None
              and episode <= gmap.exploring_until)
    if (triggered or active) and targets:
        if triggered and not active and episode is not None:
            gmap.exploring_until = episode + EXPLORE_EPISODES - 1
        heading = mass.most_common(1)[0][0]
        override = next(t for t in targets if t["sector"] == heading)   # every direction with mass has one
        if active and not triggered:
            left = gmap.exploring_until - episode + 1
            why = f"are exploring because your earlier plan kept failing ({left} episode(s) of exploring left)"
        elif stale >= STALE_EPISODES:
            why = f"have found nothing new for {stale} episodes in a row"
        else:
            why = f"have failed {failure_streak} episodes in a row"
        lines.append(
            f"You {why}, so what you have been trying is not working. This episode's task MUST be to "
            f"walk to the unexplored edge at ({override['dx']}, {override['dy']}), {override['steps']} steps "
            f"{override['sector']}, and keep going into the area beyond it.")
        failed = [t for t in dict.fromkeys(failed_tasks) if t][:3]
        if failed:
            lines.append("Do NOT set any of these recent tasks again — each of them failed: " +
                         "; ".join(f'"{t}"' for t in failed) + ".")
    return "\n".join(lines), override
