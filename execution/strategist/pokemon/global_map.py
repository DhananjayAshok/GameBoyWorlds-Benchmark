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

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from execution.artifact import StrategistArtifact, update
from execution.perception.matching import cell_key, split_quarters
from execution.perception.pokemon.grid import GRID_SKIP, LEFT_COL, TOP_ROW
from execution.perception.pokemon.navigation import LEDGE_TAGS, WALKABLE_TAGS
from execution.strategist.pokemon import prompts as P
from execution.strategist.pokemon.asking import ask, single_value

Cell = Tuple[int, int]

PLAYER_COL, PLAYER_ROW = -LEFT_COL, TOP_ROW          # screen column/row of local (0, 0)
QUARTERS = ("top_left", "top_right", "bottom_left", "bottom_right")
DIRECTIONS: Dict[str, Cell] = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}

WALKABLE, BLOCKED, LEDGE, UNKNOWN, OCCUPIED = "walkable", "blocked", "ledge", "unknown", "occupied"
#: What kind of place an area is, as the strategist judged when naming it.
OUTDOORS, BUILDING, CAVE = "outdoors", "building", "cave"
AREA_KINDS = (OUTDOORS, BUILDING, CAVE)

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
#: A screen that scrolled with no move pressed (a cutscene walking the player) is searched this many cells each way.
SCRIPTED_MAX_STEPS = 4
#: A cell this dark throughout is black. A frame that blacks out this many cells already mapped as something else
#: is a transition closing in (the wipe that opens a battle), not the map.
BLACK_LEVEL = 40
BLACKOUT_CELLS = 3
#: A wipe leaves the cells it has not reached yet as they were: at most this fraction of a cell's pixels changed...
WIPE_CELL_CHANGE = 0.2
#: ...on at least this many unchanged cells for every changed one. A new room changes nearly all of them.
WIPE_SAME_RATIO = 2
#: Re-localisation needs this many distinctive matching cells, and this fraction of the voters.
RELOCALISE_MIN_CELLS = 30
#: A cell picture votes for at most this many places it appears in a segment.
RELOCALISE_MAX_FANOUT = 25
RELOCALISE_MIN_FRACTION = 0.7
#: The best-voted placements, each checked cell by cell before one is accepted.
RELOCALISE_CANDIDATES = 5
#: Placements matching within this fraction of the best are a close call...
RELOCALISE_CLOSE_CALL = 0.8
#: ...settled only by a placement matching this many more distinctive cells (pictures seen at most
#: DISTINCTIVE_MAX_COPIES times in the segment) than every other close one.
DISTINCTIVE_LEAD = 2
DISTINCTIVE_MAX_COPIES = 2
#: Two ends of door links this close (in squares, either way) count as the same door.
DOOR_REACH = 1
#: Two differently named areas are only merged if the strategist judges them the same place; a verdict stands until
#: their overlap grows this many times larger.
MERGE_REJUDGE_GROWTH = 1.5
MERGE_JUDGE_ATTEMPTS = 2
#: Consecutive unplaceable free-roam frames before a new segment is opened.
LOST_PATIENCE = 3
#: A segment younger than this many placements keeps checking whether it overlaps an older one.
MERGE_WINDOW = 40
#: A screen is off the 16px grid only if its best alignment beats the next best by this much.
GRID_MARGIN = 0.02
#: Two segments are merged at the end of an episode when they share this many matching cells...
CONSOLIDATE_MIN_CELLS = 30
#: ...at least this fraction of their overlap matches...
CONSOLIDATE_MIN_AGREEMENT = 0.6
#: ...and their best offset has this many times the votes of any other...
CONSOLIDATE_MARGIN = 1.5
#: ...or, where repeating buildings or fences give a shifted offset nearly as many votes, it has at least this many
#: votes and its agreement beats that offset's by this much.
CONSOLIDATE_MIN_VOTES = 10
CONSOLIDATE_AGREEMENT_LEAD = 0.2
DOOR_TAGS = {"entrance", "staircase", "elevator"}
NEIGHBOURS_8 = tuple((ox, oy) for ox in (-1, 0, 1) for oy in (-1, 0, 1) if (ox, oy) != (0, 0))
#: A segment is only named once the player has been placed in it this many times.
MIN_NAME_OBSERVATIONS = 3
#: A segment is (re)named at the end of an episode once its map has grown by this many pixels since it was last
#: named or reviewed: about two screens.
RENAME_GROWTH_PIXELS = 2 * 90 * 16 * 16


def _screen(frame: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.squeeze(np.asarray(frame)))


def is_door(c: "MapCell") -> bool:
    tags = set(c.tags)
    return bool(tags & DOOR_TAGS) and "obstacle" not in tags


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
             reported: Optional[int] = None, residual: Cell = (0, 0)
             ) -> Optional[Tuple[Cell, Cell, bool]]:
    """
    Pixel matching first, which is the more accurate of the two on ordinary frames; edge matching
    as the fallback, which survives palette changes (battle-intro flashes, fades) that defeat it.
    """
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
class SegmentInfo:
    name: Optional[str] = None
    description: Optional[str] = None
    episode: Optional[int] = None
    step: Optional[int] = None
    aliases: List[Tuple[str, Optional[str]]] = field(default_factory=list)
    named_pixels: int = 0
    explored: Optional[int] = None
    kind: Optional[str] = None


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

    #: Set by MappingEnvironment to the tile recognizer's canonical_hash, so that a cell keeps its hashes
    #: when the palette changes with the time of day. Not saved; without it cells hash their raw pixels.
    _TRANSIENT = ("canonical_hash",)
    canonical_hash: Optional[Callable[[np.ndarray], Optional[str]]] = None

    def _cell_hashes(self, quarters: Dict[str, np.ndarray]) -> Tuple[Optional[str], ...]:
        hash_of = self.canonical_hash or cell_key
        return tuple(hash_of(quarters[q]) for q in QUARTERS)

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
        self.segment_born = 0
        self.warps: List[Tuple[int, int, Cell, int, Cell]] = []
        self.warp_moves: List[Optional[str]] = []
        self.fade_direction: Optional[str] = None
        self.routes: List[Dict[str, Any]] = []
        self.segment_info: Dict[int, SegmentInfo] = {0: SegmentInfo()}
        self.naming_queue: List[Dict[str, Any]] = []
        self.merge_verdicts: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.seen_entrances: Optional[set] = set()
        self.defeats: List[Dict[str, Any]] = []
        self.defeat_return: Optional[Dict[str, Any]] = None
        #: Cells (their four quarter hashes) the player was seen hopping down over. The tile labels
        #: can call a ledge an obstacle -- on Crystal they always do -- which would draw every ledge
        #: as a wall and hide that it is one-way. Read off the screen, never from RAM.
        self.ledge_patterns: set = set()

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        self.__dict__.setdefault("defeats", [])
        self.__dict__.setdefault("defeat_return", None)
        self.__dict__.setdefault("seen_entrances", None)
        self.__dict__.setdefault("exploring_until", None)
        self.__dict__.setdefault("segment_born", 0)
        self.__dict__.setdefault("warps", [])
        self.__dict__.setdefault("warp_moves", [None] * len(self.warps))
        self.__dict__.setdefault("fade_direction", None)
        self.__dict__.setdefault("routes", [])
        self.__dict__.setdefault("segment_info", {})
        self.__dict__.setdefault("naming_queue", [])
        self.__dict__.setdefault("merge_verdicts", {})
        self.__dict__.setdefault("ledge_patterns", set())

    def _classify(self, tags: Iterable[str], hashes: Tuple[Optional[str], ...]) -> str:
        kind = classify(tags)
        if kind == BLOCKED and hashes in self.ledge_patterns:
            return LEDGE
        return kind

    def _learn_ledges(self, start: Cell, end: Cell, direction: Optional[str]) -> int:
        """
        A move down that carried the player across a cell mapped as blocked went over a ledge: the
        only blocked-looking thing a player passes. Learn that cell's picture as a ledge and relabel
        every copy of it already mapped, so one hop reveals the whole ledge line. Only a move that crossed
        exactly one blocked cell counts, so a frame misplaced into a wall is unlikely to teach a ledge.
        """
        if direction != "down" or start[0] != end[0] or start[1] - end[1] < 2:
            return 0
        seg = self.segments[self.segment]
        path = [seg.get((start[0], y)) for y in range(start[1] - 1, end[1] - 1, -1)]
        blocked = [i for i, c in enumerate(path) if c is not None and c.kind == BLOCKED]
        # the hop lands on the cell after the ledge, so the ledge is never the last cell of the move
        if len(blocked) != 1 or blocked[0] == len(path) - 1:
            return 0
        pattern = path[blocked[0]].hashes
        if any(h is None for h in pattern) or pattern in self.ledge_patterns:
            return 0
        self.ledge_patterns.add(pattern)
        relabelled = 0
        for other in self.segments:
            for c in other.values():
                if c.kind == BLOCKED and c.hashes == pattern:
                    c.kind = LEDGE
                    relabelled += 1
        self.stats["ledges_learned"] += 1
        self.stats["ledge_cells_relabelled"] += relabelled
        return relabelled

    def names_differ(self, a: int, b: int) -> bool:
        def norm(name: Optional[str]) -> Optional[str]:
            return None if name is None else " ".join(re.sub(r"[^\w\s]", "", name.lower()).split())
        na, nb = norm(self.info(a).name), norm(self.info(b).name)
        return na is not None and nb is not None and na != nb

    def info(self, segment: int) -> "SegmentInfo":
        return self.segment_info.setdefault(segment, SegmentInfo())

    def set_name(self, segment: int, name: Optional[str], description: Optional[str], kind: str,
                 episode: int) -> bool:
        info = self.info(segment)
        info.named_pixels = self.filled_pixels(segment)
        name, description = name or info.name, description or info.description
        if (name, description, kind) == (info.name, info.description, info.kind):
            return False
        if info.name is not None and name != info.name:
            info.aliases.append((info.name, info.description))
        info.name, info.description, info.kind, info.episode, info.step = name, description, kind, episode, self.step
        return True

    def parent(self, segment: int) -> Optional[int]:
        segment = self.resolve(segment, (0, 0))[0]
        if self.info(segment).kind != BUILDING:
            return None
        exits, seen, queue = _exits(self), {segment}, [segment]
        while queue:
            for e in exits.get(queue.pop(0), []):
                to = self.resolve(e["to"], (0, 0))[0]
                if to in seen:
                    continue
                seen.add(to)
                kind = self.info(to).kind
                if kind == OUTDOORS:
                    return to
                if kind == BUILDING:
                    queue.append(to)
        return None

    def place_label(self, segment: Optional[int] = None) -> str:
        segment = self.resolve(self.segment if segment is None else segment, (0, 0))[0]
        parent = self.parent(segment)
        return self.area_label(segment) + (f" (inside {self.area_label(parent)})" if parent is not None else "")

    def prune_routes(self) -> None:
        self.routes = [r for r in self.routes
                       if self.segment not in {self.resolve(t, (0, 0))[0] for t in r["targets"]}]

    def filled_pixels(self, segment: int) -> int:
        return sum(c.pixels.size for c in self.segments[segment].values() if c.pixels is not None)

    def queue_grown(self) -> None:
        queued = {entry["segment"] for entry in self.naming_queue}
        for si, seg in enumerate(self.segments):
            if (seg and si not in queued and si not in self.merged_into
                    and self.filled_pixels(si) - self.info(si).named_pixels >= RENAME_GROWTH_PIXELS):
                self.naming_queue.append({"segment": si, "exit": None, "step": self.step, "marker": None,
                                          "reason": "grew"})
                queued.add(si)
        if (self.current and self.segment not in queued and self.info(self.segment).name is None
                and sum(self.visits[self.segment].values()) >= MIN_NAME_OBSERVATIONS):
            self.naming_queue.append({"segment": self.segment, "exit": None, "step": self.step, "marker": None,
                                      "reason": "here"})

    def mark_explored(self, episode: int) -> None:
        for si, seg in enumerate(self.segments):
            if not seg:
                continue
            info = self.info(si)
            done = any(c.kind == WALKABLE for c in seg.values()) and not self.frontier(si)
            if done and info.explored is None:
                info.explored = episode
                self.stats["segments_explored"] += 1
            elif not done:
                info.explored = None

    # ------------------------------------------------------------------
    # Building the map
    # ------------------------------------------------------------------

    def observe(self, frame: np.ndarray, identified: Dict[Tuple[int, int, str], Any],
                shift: Cell = (0, 0)) -> int:
        """Merge the current screen into the current segment at the player's position."""
        seg = self.segments[self.segment]
        a = _screen(frame)
        px, py = self.pos
        added = 0
        for (x, y), piece in _local_cells(a, shift):
            pixels = piece.copy()
            quarters = split_quarters(piece)
            hashes = self._cell_hashes(quarters)
            if (x, y) == (0, 0):
                tags, kind = ("player",), WALKABLE
            elif shift != (0, 0):
                tags, kind = ("unknown",), UNKNOWN
            else:
                tags = tuple(sorted({identified[(x, y, q)].tag for q in QUARTERS if (x, y, q) in identified}))
                kind = self._classify(tags, hashes)
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
            self.fade_direction = direction
        if not free_roam:
            # a battle, dialogue or menu: nothing to place now. Remember a move that was under way
            # so the next free-roam frame can be placed relative to the last good one.
            if was_free_roam and direction in DIRECTIONS:
                self.pending = (direction, max_steps, reported)
            self.fade_direction = None
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
            shift = self._grid_shift(frame)
            self.anchor, self.residual = _screen(frame), grid_residual(shift)
            self.observe(frame, identified, shift)
            return "start"
        if self.pending is not None and direction not in DIRECTIONS:
            direction, max_steps, reported = self.pending
        placed = None if self.faded else register(self.anchor, frame, direction, max_steps, reported,
                                                  self.residual)
        if placed is not None:
            delta, residual, structural = placed
            pos = (self.pos[0] + delta[0], self.pos[1] + delta[1])
            shift = tuple((-r) % GRID_SKIP for r in residual)
            if not structural and self._blacks_out(frame, self.segment, pos, shift):
                if direction in DIRECTIONS and self.pending is None:
                    self.pending = (direction, max_steps, reported)
                self.stats["blackout_frames"] += 1
                return "blank"
            if not structural:
                self._learn_ledges(self.pos, pos, direction)
            self.pos, self.residual = pos, residual
            self.pending, self.lost, self.anchor = None, 0, _screen(frame)
            self.stats["placed_by_edges" if structural else "placed"] += 1
            if structural:
                # recoloured (a flash or fade): the position holds, the colours do not
                self.stats["recoloured_frames"] += 1
            elif self.residual == (0, 0):
                self.observe(frame, identified)
                self._try_merge(frame)
            else:
                self.observe(frame, identified, shift)
                self.stats["off_grid_frames"] += 1
            return "moved" if delta != (0, 0) else "stayed"
        shift = self._grid_shift(frame)
        found = self.relocalise(frame, shift=shift, allowed=lambda si, q: not self.arrival_breaks_doors(si, q))
        if found is not None and self._blacks_out(frame, *found, shift):
            if direction in DIRECTIONS and self.pending is None:
                self.pending = (direction, max_steps, reported)
            self.stats["blackout_frames"] += 1
            return "blank"
        if found is not None:
            if found != (self.segment, self.pos):
                self._link(*found)
            self.segment, self.pos = found
            self.residual, self.pending, self.lost, self.anchor = grid_residual(shift), None, 0, _screen(frame)
            self.faded = False
            self.stats["relocalised"] += 1
            self.observe(frame, identified, shift)
            return "relocalised"
        if self._wipe_closing(frame):
            if direction in DIRECTIONS and self.pending is None:
                self.pending = (direction, max_steps, reported)
            self.stats["wipe_frames"] += 1
            return "blank"
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
        self.segment_info[len(self.segments) - 1] = SegmentInfo()
        self._link(len(self.segments) - 1, (0, 0))
        self.segment, self.pos = len(self.segments) - 1, (0, 0)
        self.residual, self.pending, self.lost, self.anchor = grid_residual(shift), None, 0, _screen(frame)
        self.faded = False
        self.segment_born = self.stats["placed"]
        self.stats["new_segment"] += 1
        self.observe(frame, identified, shift)
        return "new_segment"

    def _link(self, segment: int, pos: Cell) -> None:
        self.links.append((self.segment, self.pos, segment, pos))
        if not self.faded:
            return
        self.warps.append((self.step, self.segment, self.pos, segment, pos))
        self.warp_moves.append(self.fade_direction if self.fade_direction in DIRECTIONS else None)
        left = self.segment
        if (self.info(left).name is None and sum(self.visits[left].values()) >= MIN_NAME_OBSERVATIONS
                and all(entry["segment"] != left for entry in self.naming_queue)):
            self.naming_queue.append({"segment": left, "exit": self.pos, "step": self.step, "marker": None,
                                      "reason": "door"})

    def _blacks_out(self, frame: np.ndarray, segment: int, pos: Cell, shift: Cell = (0, 0)) -> bool:
        seg = self.segments[segment]
        n = 0
        for (x, y), piece in _local_cells(_screen(frame), shift):
            old = seg.get((pos[0] + x, pos[1] + y))
            if old is not None and old.pixels is not None:
                n += not _still_lit(piece, old.pixels).all()
        return n >= BLACKOUT_CELLS

    def _wipe_closing(self, frame: np.ndarray) -> bool:
        if self.faded:
            return False
        seg, shift = self.segments[self.segment], tuple((-r) % GRID_SKIP for r in self.residual)
        black = same = changed = 0
        for (x, y), piece in _local_cells(_screen(frame), shift):
            old = seg.get((self.pos[0] + x, self.pos[1] + y))
            if old is None or old.pixels is None or (x, y) == (0, 0):
                continue
            new, prior = piece.astype(int), np.squeeze(old.pixels).astype(int)
            lit = _still_lit(new, prior)
            if not lit.all():
                black += 1
            if not lit.any():
                continue
            if np.mean(np.abs(new - prior)[lit] > BLACK_LEVEL) < WIPE_CELL_CHANGE:
                same += 1
            else:
                changed += 1
        return black >= BLACKOUT_CELLS and same >= WIPE_SAME_RATIO * changed

    def _grid_shift(self, frame: np.ndarray) -> Cell:
        shift = grid_offset(frame) or (0, 0)
        if shift != (0, 0):
            self.stats["off_grid_unplaced"] += 1
        return shift

    def _try_merge(self, frame: np.ndarray) -> None:
        """Loop closure: if this young segment turns out to overlap an older one, fold it in."""
        if self.segment == 0 or self.stats["placed"] - self.segment_born > MERGE_WINDOW:
            return
        found = self.relocalise(frame, exclude=self.segment)
        if found is None:
            return
        target, target_pos = found
        offset = (target_pos[0] - self.pos[0], target_pos[1] - self.pos[1])
        if self.merge_breaks_doors(self.segment, target, offset):
            self.stats["merge_door_vetoes"] += 1
            return
        if self.names_differ(self.segment, target):
            return
        self._merge(self.segment, target, offset)

    def _merge(self, source: int, target: int, offset: Cell) -> None:
        dx, dy = offset
        src, dst = self.segments[source], self.segments[target]
        for (x, y), c in src.items():
            g = (x + dx, y + dy)
            old = dst.get(g)
            if old is None or old.kind in (OCCUPIED, UNKNOWN) or c.last_seen >= old.last_seen:
                dst[g] = c
        for (x, y), v in self.visits[source].items():
            self.visits[target][(x + dx, y + dy)] += v
        self.trail = [(t, target, (p[0] + dx, p[1] + dy)) if si == source else (t, si, p)
                      for t, si, p in self.trail]

        def moved(si: int, p: Cell) -> Tuple[int, Cell]:
            return (target, (p[0] + dx, p[1] + dy)) if si == source else (si, p)

        self.links = [(*moved(a, pa), *moved(b, pb)) for a, pa, b, pb in self.links]
        self.warps = [(t, *moved(a, pa), *moved(b, pb)) for t, a, pa, b, pb in self.warps]
        for entry in self.naming_queue:
            entry["segment"], entry["exit"] = moved(entry["segment"], entry["exit"])
        absorbed, kept = self.segment_info.pop(source, SegmentInfo()), self.info(target)
        if kept.name is None:
            absorbed.aliases += kept.aliases
            self.segment_info[target] = absorbed
        elif absorbed.name is not None:
            kept.aliases.append((absorbed.name, absorbed.description))
        self.merged_into[source] = (target, (dx, dy))
        self.segments[source] = {}
        self.visits[source] = Counter()
        if self.segment == source:
            self.segment, self.pos = target, (self.pos[0] + dx, self.pos[1] + dy)
        self.stats["merged"] += 1

    def overlap(self, source: int, target: int) -> Optional[Dict[str, Any]]:
        def content(seg):
            return {g: c for g, c in seg.items()
                    if c.kind != OCCUPIED and not (c.pixels is not None and c.pixels.min() == c.pixels.max())}
        src, dst = content(self.segments[source]), content(self.segments[target])
        index: Dict[Tuple[Optional[str], ...], List[Cell]] = defaultdict(list)
        for g, c in dst.items():
            index[c.hashes].append(g)
        votes: Counter = Counter()
        for g, c in src.items():
            places = index.get(c.hashes, ())
            if len(places) <= RELOCALISE_MAX_FANOUT:
                for p in places:
                    votes[(p[0] - g[0], p[1] - g[1])] += 1
        if not votes:
            return None
        def agreement(offset: Cell) -> Tuple[int, int]:
            shared = [(c, dst[(g[0] + offset[0], g[1] + offset[1])]) for g, c in src.items()
                      if (g[0] + offset[0], g[1] + offset[1]) in dst]
            return sum(1 for a, b in shared if a.hashes == b.hashes), len(shared)

        (dx, dy), _ = votes.most_common(1)[0]
        matches, shared = agreement((dx, dy))
        rivals = [(n, o) for o, n in votes.items() if max(abs(o[0] - dx), abs(o[1] - dy)) > 1]
        rival, rival_offset = max(rivals, default=(0, None))
        rival_matches, rival_shared = agreement(rival_offset) if rival_offset is not None else (0, 0)
        return {"offset": (dx, dy), "matches": matches, "shared": shared,
                "agreement": matches / shared if shared else 0.0,
                "votes": votes[(dx, dy)], "rival": rival,
                "rival_agreement": rival_matches / rival_shared if rival_shared else 0.0}

    def _judged_same(self, source: int, target: int, overlap: Dict[str, Any],
                     judge: Optional[Callable[[int, int, Dict[str, Any]], Optional[bool]]]) -> bool:
        if not self.names_differ(source, target):
            return True
        key = (min(source, target), max(source, target))
        verdict = self.merge_verdicts.get(key)
        if verdict is not None and overlap["matches"] < MERGE_REJUDGE_GROWTH * verdict["matches"]:
            return verdict["same"]
        if judge is None:
            return False
        same = judge(source, target, overlap)
        self.merge_verdicts[key] = {"same": bool(same), "matches": overlap["matches"], "step": self.step,
                                    "names": (self.info(source).name, self.info(target).name)}
        self.stats["merges_judged_same" if same else "merges_judged_different"] += 1
        return bool(same)

    def consolidate(self, judge: Optional[Callable[[int, int, Dict[str, Any]], Optional[bool]]] = None
                    ) -> List[Dict[str, Any]]:
        merged = []
        while True:
            live = [si for si, seg in enumerate(self.segments) if seg and si not in self.merged_into]
            best = None
            for i, target in enumerate(live):
                for source in live[i + 1:]:
                    o = self.overlap(source, target)
                    if (o is None or o["matches"] < CONSOLIDATE_MIN_CELLS
                            or o["agreement"] < CONSOLIDATE_MIN_AGREEMENT):
                        continue
                    if (o["votes"] < CONSOLIDATE_MARGIN * o["rival"]
                            and (o["votes"] < CONSOLIDATE_MIN_VOTES
                                 or o["agreement"] - o["rival_agreement"] < CONSOLIDATE_AGREEMENT_LEAD)):
                        continue
                    if best is not None and o["matches"] <= best["matches"]:
                        continue
                    if self.merge_breaks_doors(source, target, o["offset"]):
                        self.stats["merge_door_vetoes"] += 1
                        continue
                    if not self._judged_same(source, target, o, judge):
                        continue
                    best = dict(o, source=source, target=target)
            if best is None:
                return merged
            self._merge(best["source"], best["target"], best["offset"])
            self.stats["consolidated"] += 1
            merged.append(best)

    def relocalise(self, frame: np.ndarray, exclude: Optional[int] = None, shift: Cell = (0, 0),
                   allowed: Optional[Callable[[int, Cell], bool]] = None) -> Optional[Tuple[int, Cell]]:
        """
        Find the player in the cells already mapped by voting on exact cell matches. Blank cells
        (the black around an indoor room) match everywhere, so they do not vote, and the number of
        matches required scales with how much of the screen has content.
        """
        a = _screen(frame)
        local: Dict[Cell, Tuple[Optional[str], ...]] = {}
        for (x, y), piece in _local_cells(a, shift):
            if (x, y) == (0, 0) or piece.min() == piece.max():
                continue
            quarters = split_quarters(piece)
            local[(x, y)] = self._cell_hashes(quarters)
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
            ranked = sorted((self._screen_agreement(seg, local, o), o)
                            for o, _ in votes.most_common(RELOCALISE_CANDIDATES))[::-1]
            close = [(o, overlap) for (m, overlap), o in ranked if m >= RELOCALISE_CLOSE_CALL * ranked[0][0][0]]
            if len(close) > 1:
                distinctive = sorted(((self._distinctive_matches(seg, index, local, o), o, overlap)
                                      for o, overlap in close), reverse=True)
                if distinctive[0][0] - distinctive[1][0] < DISTINCTIVE_LEAD:
                    continue                  # two placements explain the screen equally well
                close = [(distinctive[0][1], distinctive[0][2])]
            offset, overlap = close[0]
            n = self._screen_agreement(seg, local, offset)[0]
            # a small indoor room shows few content cells; ask for most of them rather than a fixed 30
            needed = min(RELOCALISE_MIN_CELLS, max(10, int(0.6 * len(local))))
            if n >= needed and n >= RELOCALISE_MIN_FRACTION * overlap:
                if allowed is not None and not allowed(si, offset):
                    self.stats["relocalise_door_vetoes"] += 1
                    continue
                if best is None or n > best[0]:
                    best = (n, si, offset)
        return None if best is None else (best[1], best[2])

    @staticmethod
    def _screen_agreement(seg: Dict[Cell, MapCell], local: Dict[Cell, Tuple[Optional[str], ...]],
                          offset: Cell) -> Tuple[int, int]:
        placed = [(seg.get((offset[0] + x, offset[1] + y)), h) for (x, y), h in local.items()]
        placed = [(c, h) for c, h in placed if c is not None]
        return sum(1 for c, h in placed if c.kind != OCCUPIED and c.hashes == h), len(placed)

    @staticmethod
    def _distinctive_matches(seg: Dict[Cell, MapCell], index: Dict[Tuple[Optional[str], ...], List[Cell]],
                             local: Dict[Cell, Tuple[Optional[str], ...]], offset: Cell) -> int:
        return sum(1 for (x, y), h in local.items()
                   if len(index.get(h, ())) <= DISTINCTIVE_MAX_COPIES
                   and (c := seg.get((offset[0] + x, offset[1] + y))) is not None and c.hashes == h)

    @staticmethod
    def door_conflicts(warps: List[Tuple[int, int, Cell, int, Cell]], moves: List[Optional[str]]) -> int:
        def near(p: Cell, q: Cell) -> bool:
            return max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= DOOR_REACH

        conflicts, ends = 0, []
        for (_, a, pa, b, pb), move in zip(warps, moves):
            if a == b and move is not None:
                conflicts += 1
                continue
            if a == b and near(pa, pb):
                continue
            ends += [(a, pa, b, pb), (b, pb, a, pa)]
        for i, (s, p, other, q) in enumerate(ends):
            conflicts += any(s == s2 and near(p, p2) and (other != other2 or not near(q, q2))
                             for s2, p2, other2, q2 in ends[i + 1:])
        return conflicts

    def merge_breaks_doors(self, source: int, target: int, offset: Cell) -> bool:
        def moved(si: int, p: Cell) -> Tuple[int, Cell]:
            return (target, (p[0] + offset[0], p[1] + offset[1])) if si == source else (si, p)

        after = [(t, *moved(a, pa), *moved(b, pb)) for t, a, pa, b, pb in self.warps]
        return self.door_conflicts(after, self.warp_moves) > self.door_conflicts(self.warps, self.warp_moves)

    def arrival_breaks_doors(self, segment: int, pos: Cell) -> bool:
        if not self.faded:
            return False
        move = self.fade_direction if self.fade_direction in DIRECTIONS else None
        after = self.warps + [(self.step, self.segment, self.pos, segment, pos)]
        return (self.door_conflicts(after, self.warp_moves + [move])
                > self.door_conflicts(self.warps, self.warp_moves))

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
                c.kind = self._classify(c.tags, c.hashes)
                changed += 1
        return changed

    @update
    def end_episode(self, episode: int,
                    judge: Optional[Callable[[int, int, Dict[str, Any]], Optional[bool]]] = None) -> int:
        """Close the episode's bookkeeping; decorated so the map is checkpointed every episode."""
        self.consolidate(judge)
        self.mark_explored(episode)
        self.new_cells_by_episode[episode] = self._new_this_episode
        self._new_this_episode = 0
        return self.new_cells_by_episode[episode]

    def _door_at(self, segment: int, pos: Cell) -> Cell:
        seg = self.segments[segment]
        near = [(max(abs(ox), abs(oy)), abs(ox) + abs(oy), g) for ox, oy in NEIGHBOURS_8 + ((0, 0),)
                for g in [(pos[0] + ox, pos[1] + oy)] if g in seg and is_door(seg[g])]
        return min(near)[2] if near else pos

    def door_labels(self) -> Dict[Tuple[int, Cell], int]:
        """Every square a warp left from or arrived at, with the segment on the other side (latest warp wins)."""
        labels: Dict[Tuple[int, Cell], int] = {}
        for _, a, pa, b, pb in self.warps:
            labels[(a, self._door_at(a, pa))] = b
            labels[(b, self._door_at(b, pb))] = a
        return labels

    def area_label(self, segment: int) -> str:
        name = self.info(segment).name
        return name if name else f"area #{segment} (not named yet)"

    def door_list(self, segment: Optional[int] = None) -> List[Dict[str, Any]]:
        si = self.segment if segment is None else segment
        seg = self.segments[si]
        labels = {p: b for (s, p), b in self.door_labels().items() if s == si}
        order = {key: i for i, key in enumerate(labels)}
        cells = {g for g, c in seg.items() if is_door(c)} | set(labels)
        out, seen = [], set()
        for g in sorted(cells):
            if g in seen:
                continue
            group, stack = [], [g]
            while stack:
                h = stack.pop()
                if h in seen or h not in cells:
                    continue
                seen.add(h)
                group.append(h)
                stack += [(h[0] + ox, h[1] + oy) for ox, oy in NEIGHBOURS_8]
            known = sorted((order[h], labels[h]) for h in group if h in labels)
            out.append({"cells": sorted(group), "leads_to": known[-1][1] if known else None})
        return out

    def resolve(self, segment: int, pos: Cell) -> Tuple[int, Cell]:
        while segment in self.merged_into:
            segment, (dx, dy) = self.merged_into[segment]
            pos = (pos[0] + dx, pos[1] + dy)
        return segment, pos

    def diff(self, old: "GlobalMap") -> Dict[str, Any]:
        known: Dict[Tuple[int, Cell], MapCell] = {}
        for si, seg in enumerate(old.segments):
            for g, c in seg.items():
                known[self.resolve(si, g)] = c
        new_cells: Dict[int, List[Cell]] = defaultdict(list)
        reclassified: Dict[int, List[Cell]] = defaultdict(list)
        for si, seg in enumerate(self.segments):
            for g, c in seg.items():
                before = known.get((si, g))
                if before is None:
                    new_cells[si].append(g)
                elif before.kind == UNKNOWN and c.kind != UNKNOWN:
                    reclassified[si].append(g)
        return {
            "cells": self.n_cells - old.n_cells,
            "segments": len(self.segments) - len(old.segments),
            "new_cells": dict(new_cells),
            "reclassified": dict(reclassified),
            "opened": list(range(len(old.segments), len(self.segments))),
            "merged": {s: m for s, m in self.merged_into.items() if s not in old.merged_into},
            "steps": self.step - old.step,
            "position": ((old.segment, old.pos), (self.segment, self.pos)),
            "exploring": (old.exploring_until, self.exploring_until),
            "stats": {k: self.stats[k] - old.stats[k] for k in set(self.stats) | set(old.stats)
                      if self.stats[k] != old.stats[k]},
            "doors": self._new_doors(old),
        }

    def record_defeat(self) -> None:
        self.defeats.append({"step": self.step, "segment": self.segment, "pos": self.pos})
        self.stats["defeats"] += 1

    def defeat_route(self, defeat: Dict[str, Any]) -> str:
        segment, pos = self.resolve(defeat["segment"], defeat["pos"])
        where = f"{self.area_label(segment)}, at {pos} on that area's map"
        if segment == self.segment:
            dx, dy = pos[0] - self.pos[0], pos[1] - self.pos[1]
            return f"{where}: you are in that area now; it is at ({dx}, {dy}) from you (x = steps right, y = steps up)."
        return f"{where}.\n" + render_plan(self, [segment], plan_route(self, [segment]))

    def entrances(self) -> set:
        return {(si, g) for si, seg in enumerate(self.segments) if si not in self.merged_into
                for g, c in seg.items() if "entrance" in c.tags}

    def new_entrances(self) -> int:
        if self.seen_entrances is None:
            return 0
        return len(self.entrances() - {self.resolve(*key) for key in self.seen_entrances})

    def mark_entrances_seen(self) -> None:
        self.seen_entrances = self.entrances()

    def _new_doors(self, old: "GlobalMap") -> Dict[Tuple[int, Cell], int]:
        before = {self.resolve(*key): self.resolve(b, (0, 0))[0] for key, b in old.door_labels().items()}
        return {k: b for k, b in self.door_labels().items() if before.get(k) != b}

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

    def frontier(self, segment: Optional[int] = None) -> List[Cell]:
        """Walkable cells of the segment (the current one by default) with at least one neighbour never seen."""
        seg = self.current if segment is None else self.segments[segment]
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
        if tile_recognizer is not None and hasattr(tile_recognizer, "canonical_hash"):
            global_map.canonical_hash = tile_recognizer.canonical_hash
        self._parameters = parameters
        self.marker = None

    def step_high_level_action(self, action, **kwargs):
        from gameboy_worlds.emulation.pokemon import AgentState

        before = self._env.get_info()
        was_free = before.get("pokemon_core", {}).get("agent_state") == AgentState.FREE_ROAM
        result = self._env.step_high_level_action(action, **kwargs)
        try:
            info = result[4] or {}
            if (info.get("pokemon_core", {}).get("n_battles_lost", 0)
                    > before.get("pokemon_core", {}).get("n_battles_lost", 0)):
                self._map.record_defeat()
            if not info.get("invalid_action"):
                frame = info["core"]["current_frame"]
                free = info.get("pokemon_core", {}).get("agent_state") == AgentState.FREE_ROAM
                direction, max_steps, reported = _movement(action, kwargs, info)
                passed = info["core"].get("transition_passed_frames")
                faded = passed is not None and any(not has_structure(f) for f in passed)
                identified = self._tiles.identify_tiles(frame) if free else {}
                queued = len(self._map.naming_queue)
                self._map.on_step(frame, identified, free, direction, max_steps, reported, was_free, faded)
                for entry in self._map.naming_queue[queued:]:
                    entry["marker"] = self.marker() if self.marker is not None else None
        except Exception as e:                                   # the map must never break a run
            from utils import log_warn
            log_warn(f"Global map update failed and was skipped: {type(e).__name__}: {e}", self._parameters)
        return result

    @property
    def ledge_patterns(self) -> set:
        return self._map.ledge_patterns

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
    """
    The segment as a picture assembled from the cells actually seen (BGR), unseen space hatched,
    frontier cells outlined green, the trail drawn in blue and the player boxed in red.
    """
    import cv2

    si = gmap.segment if segment is None else segment
    seg = {g: c for g, c in gmap.segments[si].items() if upto_step is None or c.first_seen <= upto_step}
    bounds = bounds or segment_bounds(gmap, si)
    if not seg or bounds is None:
        return None
    x0, x1, y0, y1 = bounds
    w, h = (x1 - x0 + 1) * GRID_SKIP, (y1 - y0 + 1) * GRID_SKIP
    canvas = np.full((h, w, 3), 60, np.uint8)
    canvas[::4, :, :] = 90                                       # hatch = never seen
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


# ----------------------------------------------------------------------------
# What the strategist is told
# ----------------------------------------------------------------------------

LEGEND = ('@ you   . walkable   # blocked   = ledge (jump down only)   D door/stairs   " tall grass   '
          '~ water   n person   % not yet recognised   ? never seen')
SECTORS = ("east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east")
#: Episodes without a single new cell before the strategist is asked whether it is stuck.
STALE_EPISODES = 3
#: Episodes in a row that did not complete before the strategist is asked whether it is stuck.
FAILURE_EPISODES = 3
#: Once the strategist judges itself stuck, exploration stays in charge for this many episodes, re-targeting each time.
EXPLORE_EPISODES = 3
STUCK_ATTEMPTS = 2


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


NAME_ATTEMPTS = 2
MAX_NAMED_LINES = 30


def segment_ascii(gmap: GlobalMap, segment: int, mark: Optional[Cell] = None) -> str:
    seg, bounds = gmap.segments[segment], segment_bounds(gmap, segment)
    if bounds is None:
        return "(empty)"
    x0, x1, y0, y1 = bounds
    return "\n".join("".join("X" if (x, y) == mark else glyph(seg.get((x, y))) for x in range(x0, x1 + 1))
                     for y in range(y1, y0 - 1, -1))


UNCHANGED = "UNCHANGED"


def _parse_segment_name(review: bool):
    def parse(output: str) -> Tuple[Any, Optional[str]]:
        answer = []
        for key in ("Name", "Description"):
            value, problem = single_value(output, key)
            if problem is not None or not value:
                return None, problem or f"The `{key}:` line was empty."
            if value.strip().rstrip(".").upper() == UNCHANGED:
                if not review:
                    return None, f"This area has no {key.lower()} yet, so `{key}: {UNCHANGED}` is not possible."
                value = None
            answer.append(value)
        kind, problem = single_value(output, "Kind")
        kind = (kind or "").strip().rstrip(".").lower()
        if problem is not None or kind not in AREA_KINDS:
            return None, problem or f"`Kind:` must be one of {', '.join(AREA_KINDS)}, not {kind!r}."
        return (*answer, kind), None
    return parse


def name_segment(gmap: GlobalMap, segment: int, mark: Optional[Cell], left: bool, tasks: str, steps: str, vlm_call,
                 game: str, episode: int, parameters: Optional[dict] = None) -> Optional[Tuple[bool, str, str]]:
    import cv2
    from PIL import Image

    info = gmap.info(segment)
    review = info.name is not None
    named = [f"- {other.name}: {other.description}" for si, other in sorted(gmap.segment_info.items())
             if other.name is not None and si != segment and si not in gmap.merged_into]
    if left:
        situation, steps_title = "The player has just left an area", "What the player did before leaving the area"
        mark_image, mark_text = "where the player left the area", "where the player left"
    else:
        situation, steps_title = "The player has been exploring an area", "What the player did in the last episode"
        mark_image, mark_text = "where the player is now", "where the player is now"
    situation += " you have already named." if review else " that has no name yet."
    ask_text = (P.SEGMENT_NAME_REVIEW.replace("[NAME]", info.name).replace("[DESCRIPTION]", info.description or "")
                .replace("[BEFORE]", str(info.named_pixels // GRID_SKIP ** 2))
                .replace("[AFTER]", str(gmap.filled_pixels(segment) // GRID_SKIP ** 2))
                if review else P.SEGMENT_NAME_NEW)
    prompt = (P.SEGMENT_NAME_PROMPT.replace("[GAME]", game)
              .replace("[SITUATION]", situation)
              .replace("[MARK_IMAGE]", f"the magenta box is {mark_image}, and " if mark is not None else "and ")
              .replace("[MARK_TEXT]", f", with X {mark_text}" if mark is not None else "")
              .replace("[MAP]", segment_ascii(gmap, segment, mark))
              .replace("[LEGEND]", LEGEND)
              .replace("[TASKS]", tasks or "(none yet)")
              .replace("[STEPS_TITLE]", steps_title)
              .replace("[STEPS]", steps or "(nothing was recorded)")
              .replace("[NAMED]", "\n".join(named[-MAX_NAMED_LINES:]) or "(none yet)")
              .replace("[ASK]", ask_text)
              .replace("[NAME_FORMAT]", f'<a short name, or {UNCHANGED} to keep "{info.name}">' if review
                       else "<a short name>")
              .replace("[DESCRIPTION_FORMAT]", f"<one to three sentences, or {UNCHANGED} to keep the current one>"
                       if review else "<one to three sentences>"))
    canvas = render_segment(gmap, segment, highlight=[mark] if mark is not None else [], doors=True)
    images = [Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))] if canvas is not None else []
    answer = ask(vlm_call, prompt, _parse_segment_name(review), "Segment naming", NAME_ATTEMPTS, parameters,
                 images=images)
    if answer is None:
        return None
    changed = gmap.set_name(segment, answer[0], answer[1], answer[2], episode)
    gmap.stats["segments_renamed" if review and changed else "segments_named" if changed else "names_kept"] += 1
    return changed, info.name, info.description


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


def _door_text(gmap: GlobalMap, segment: int) -> str:
    leads = [gmap.area_label(d["leads_to"]) for d in gmap.door_list(segment) if d["leads_to"] is not None]
    unentered = sum(1 for d in gmap.door_list(segment) if d["leads_to"] is None)
    parts = [f"one to {name}" for name in leads] + ([f"{unentered} not entered yet"] if unentered else [])
    return ", ".join(parts) or "none known"


def _parse_merge_decision(output: str) -> Tuple[Any, Optional[str]]:
    decision, problem = single_value(output, "Decision")
    if problem is not None:
        return None, f"{problem} It must be SAME or DIFFERENT."
    decision = (decision or "").strip().strip(".").upper()
    if decision not in ("SAME", "DIFFERENT"):
        return None, f"The decision must be SAME or DIFFERENT, not {decision!r}."
    return decision == "SAME", None


def judge_merge(gmap: GlobalMap, source: int, target: int, overlap: Dict[str, Any], vlm_call, game: str,
                parameters: Optional[dict] = None) -> Optional[bool]:
    import cv2
    from PIL import Image

    a, b = gmap.info(target), gmap.info(source)
    prompt = (P.MERGE_JUDGE_PROMPT.replace("[GAME]", game)
              .replace("[NAME_A]", a.name or "").replace("[DESCRIPTION_A]", a.description or "")
              .replace("[DOORS_A]", _door_text(gmap, target))
              .replace("[NAME_B]", b.name or "").replace("[DESCRIPTION_B]", b.description or "")
              .replace("[DOORS_B]", _door_text(gmap, source))
              .replace("[SHARED]", str(overlap["shared"])).replace("[MATCHES]", str(overlap["matches"])))
    canvases = [render_segment(gmap, target, trail=False, doors=True),
                render_segment(gmap, source, trail=False, doors=True),
                render_overlay(gmap, source, target, overlap["offset"])]
    images = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in canvases if c is not None]
    return ask(vlm_call, prompt, _parse_merge_decision, "Merge judgement", MERGE_JUDGE_ATTEMPTS, parameters,
               images=images)


# ----------------------------------------------------------------------------
# Route planning
# ----------------------------------------------------------------------------

#: Cost of stepping onto a square of each kind; walls and squares never seen cannot be crossed.
STEP_COSTS = {WALKABLE: 1, UNKNOWN: 3, OCCUPIED: 5}
#: A door the player cannot reach over mapped squares is costed at this many times its straight-line distance.
UNMAPPED_PENALTY = 2
#: Squares the player walked through count as walkable whatever their tiles were recognised as. Positions are
#: recorded after each move, so a straight line between two of them is filled in, up to this long.
MAX_WALK_FILL = 8
#: A door's departure square that is missing or a wall moves to the nearest walkable square within this distance.
SNAP_RADIUS = 2
#: Routes the strategist looked up are kept, and shown at every dispatch, until it replans or arrives.
MAX_ACTIVE_ROUTES = 2
NAVIGATE_ATTEMPTS = 2
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


def _parse_choice(count: int):
    valid = f"numbers from 0 to {count - 1} separated by commas, or None"

    def parse(output: str) -> Tuple[Any, Optional[str]]:
        choice, problem = single_value(output, "Choice")
        if problem is not None:
            return None, f"{problem} It must hold {valid}."
        if choice is None:
            return [], None
        indices: List[int] = []
        for part in choice.split(","):
            part = part.strip().rstrip(".").strip()
            if not part.isdigit() or int(part) >= count:
                return None, f"{part!r} is not an area number. The choice must be {valid}."
            if int(part) not in indices:
                indices.append(int(part))
        return indices, None
    return parse


def navigate_map(gmap: GlobalMap, request: str, vlm_caller, game: str,
                 parameters: Optional[dict] = None) -> Optional[str]:
    live =[si for si, seg in enumerate(gmap.segments) if seg and si not in gmap.merged_into]
    plans = {si: plan_route(gmap, [si]) for si in live}
    entries = sorted(live, key=lambda si: (plans[si] is None, plans[si]["cost"] if plans[si] else 0, si))
    lines = []
    for index, si in enumerate(entries):
        plan, info = plans[si], gmap.info(si)
        note = ("no known route" if plan is None else "you are here" if not plan["hops"]
                else f"{len(plan['hops'])} door(s) away")
        lines.append(f"{index}. {gmap.area_label(si)}" + (f": {info.description}" if info.description else "")
                     + f" ({note})")
    prompt = (P.NAVIGATE_SELECT_PROMPT.replace("[GAME]", game)
              .replace("[REQUEST]", request)
              .replace("[CURRENT]", gmap.place_label())
              .replace("[PLACES]", "\n".join(lines)))
    indices = ask(vlm_caller("navigate_select"), prompt, _parse_choice(len(entries)), "Navigate selection",
                  NAVIGATE_ATTEMPTS, parameters)
    if indices is None:
        return None
    if not indices:
        return f"No area on your map fits \"{request}\"."
    chosen = [entries[i] for i in indices]
    plan = plan_route(gmap, chosen)
    if plan is not None and plan["hops"]:
        gmap.routes = (gmap.routes + [{"request": request, "targets": chosen, "step": gmap.step}])[-MAX_ACTIVE_ROUTES:]
        gmap.stats["routes_planned"] += 1
    return render_plan(gmap, chosen, plan)


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


def explore_elsewhere(gmap: GlobalMap) -> List[Dict[str, Any]]:
    groups: Dict[Any, Dict[str, Any]] = {}
    unreachable = []
    for si, seg in enumerate(gmap.segments):
        if not seg or si in gmap.merged_into or si == gmap.segment or not gmap.frontier(si):
            continue
        plan = plan_route(gmap, [si])
        if plan is None or not plan["hops"]:
            unreachable.append(si)
            continue
        first = plan["hops"][0]
        group = groups.setdefault((first["depart"], first["to"]), {"plan": plan, "target": si, "areas": []})
        group["areas"].append((si, plan))
        if plan["cost"] < group["plan"]["cost"]:
            group["plan"], group["target"] = plan, si
    out = sorted(groups.values(), key=lambda g: g["plan"]["cost"])
    for g in out:
        g["areas"].sort(key=lambda a: a[1]["cost"])
    if unreachable:
        out.append({"plan": None, "target": None, "areas": [(si, None) for si in unreachable]})
    return out


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


def explore_text(gmap: GlobalMap) -> str:
    here, lines = gmap.area_label(gmap.segment), []
    edge = len(gmap.frontier())
    if edge:
        lines.append(f"- {here}, where you are: {edge} unexplored edge square{'s' if edge != 1 else ''}.")
    unentered = sum(door["leads_to"] is None for door in gmap.door_list())
    if unentered:
        lines.append(f"- {here}, where you are: {unentered} door(s) or stairs not entered yet.")
    groups = explore_elsewhere(gmap)
    if groups:
        lines.append("Other areas with unexplored edges, grouped by the way out of here:")
        lines += elsewhere_lines(gmap, groups)
    return "\n".join(lines) or "(none)"


def exploring(gmap: GlobalMap, episode: Optional[int]) -> bool:
    return episode is not None and gmap.exploring_until is not None and episode <= gmap.exploring_until


def stuck_signs(gmap: GlobalMap, recent_episodes: Iterable[int], failure_streak: int) -> Optional[str]:
    stale = gmap.episodes_without_new_cells(list(recent_episodes))
    signs = []
    if stale >= STALE_EPISODES:
        signs.append(f"the last {stale} episodes found nothing new for your map")
    if failure_streak >= FAILURE_EPISODES:
        signs.append(f"the last {failure_streak} episodes did not complete their task")
    return " and ".join(signs) if signs else None


def _parse_stuck(output: str) -> Tuple[Any, Optional[str]]:
    decision, problem = single_value(output, "Stuck")
    if problem is not None:
        return None, f"{problem} It must be YES or NO."
    decision = (decision or "").strip().strip(".").upper()
    if decision not in ("YES", "NO"):
        return None, f"The answer must be YES or NO, not {decision!r}."
    return decision == "YES", None


def judge_stuck(gmap: GlobalMap, signs: str, episodes: str, frame: np.ndarray, vlm_call, game: str,
                parameters: Optional[dict] = None) -> Optional[bool]:
    here = gmap.info(gmap.segment)
    area = f'the area you named "{here.name}": {here.description}' if here.name else "an area you have not named yet"
    prompt = (P.STUCK_PROMPT.replace("[GAME]", game).replace("[SIGNS]", signs).replace("[EPISODES]", episodes)
              .replace("[AREA]", area).replace("[AREAS]", str(gmap.n_areas)))
    return ask(vlm_call, prompt, _parse_stuck, "Stuck check", STUCK_ATTEMPTS, parameters, images=[frame])


def prompt_block(gmap: GlobalMap, recent_episodes: Iterable[int],
                 local_kinds: Optional[Dict[Cell, str]] = None,
                 explore: bool = False, episode: Optional[int] = None,
                 failed_tasks: Iterable[str] = ()) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    The map section of the strategist's prompt, and — when the strategist judged itself stuck
    (``explore``) — the frontier target that exploration should override the task with.
    """
    recent = list(recent_episodes)
    stale = gmap.episodes_without_new_cells(recent)
    targets, mass = frontier_targets(gmap, local_kinds)
    here = gmap.info(gmap.segment)
    lines = [f"You are in the area you named {gmap.place_label()}: {here.description}"] if here.name else []
    if here.explored is not None:
        lines.append("You have fully explored this area: no edge of its map is left unexplored.")
    lines += [
        "MAP OF EVERYTHING YOU HAVE SEEN IN THIS AREA (north is up, one character per step):",
        ascii_window(gmap),
        "Legend: " + LEGEND,
        f"Squares mapped here: {len(gmap.current)} (in {gmap.n_areas} separate area(s) overall). "
        f"New squares found in the last episode: "
        f"{gmap.new_cells_by_episode.get(max(recent), 0) if recent else 0}; "
        f"episodes in a row with nothing new: {stale}.",
    ]
    lines += door_lines(gmap)
    lines += route_lines(gmap)
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
    triggered = explore and episode is not None
    # One episode of exploring is not enough to rule a direction out, and the strategist's own goals
    # pull it straight back to what was failing, so exploration stays in charge for a few episodes.
    active = exploring(gmap, episode)
    elsewhere = explore_elsewhere(gmap) if (triggered or active) and not targets else []
    if (triggered or active) and (targets or elsewhere):
        if triggered and not active:
            gmap.exploring_until = episode + EXPLORE_EPISODES - 1
        left = gmap.exploring_until - episode + 1
        stuck = (f"You judged that you are stuck, so what you have been trying is not working, and you are exploring "
                 f"({left} episode(s) of exploring left).")
        if targets:
            heading = mass.most_common(1)[0][0]
            override = next(t for t in targets if t["sector"] == heading)   # every direction with mass has one
            override = dict(override, kind="edge", area=gmap.segment)
            lines.append(
                f"{stuck} This episode's task MUST be to walk to the unexplored edge of {gmap.area_label(gmap.segment)} "
                f"at ({override['dx']}, {override['dy']}), {override['steps']} steps {override['sector']}, "
                f"and keep going into the area beyond it.")
        else:
            override = {"kind": "elsewhere", "areas": [si for g in elsewhere for si, _ in g["areas"]]}
            lines.append(
                f"{stuck} Nothing is left to explore in {gmap.area_label(gmap.segment)}. This episode's task MUST "
                f"be to head for one of the areas below that still has unexplored edges, and explore them. "
                f"Areas that share the way out of here are listed together:")
            lines += elsewhere_lines(gmap, elsewhere)
        failed = [t for t in dict.fromkeys(failed_tasks) if t][:3]
        if failed:
            lines.append("Do NOT set any of these recent tasks again — each of them failed: " +
                         "; ".join(f'"{t}"' for t in failed) + ".")
    return "\n".join(lines), override
