from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from execution.artifact import StrategistArtifact, update
from execution.perception.matching import cell_key, split_quarters
from execution.perception.pokemon.grid import GRID_SKIP
from execution.perception.pokemon.navigation import LEDGE_TAGS, WALKABLE_TAGS

from .registration import (BLACK_LEVEL, _local_cells, _screen, _still_lit, grid_offset, grid_residual, has_structure,
                           register)

Cell = Tuple[int, int]

QUARTERS = ("top_left", "top_right", "bottom_left", "bottom_right")
DIRECTIONS: Dict[str, Cell] = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}

WALKABLE, BLOCKED, LEDGE, UNKNOWN, OCCUPIED = "walkable", "blocked", "ledge", "unknown", "occupied"
# What kind of place an area is, as the strategist judged when naming it.
OUTDOORS, BUILDING, CAVE = "outdoors", "building", "cave"
AREA_KINDS = (OUTDOORS, BUILDING, CAVE)

BLACKOUT_CELLS = 3
# A cell counts as unchanged by a wipe if at most this fraction of its pixels changed.
WIPE_CELL_CHANGE = 0.2
# A wipe leaves at least this many unchanged cells for every changed one; a new room changes nearly all.
WIPE_SAME_RATIO = 2
# Re-localisation needs this many matching cells (fewer on a screen with little content).
RELOCALISE_MIN_CELLS = 30
# A cell picture votes for at most this many places it appears in a segment.
RELOCALISE_MAX_FANOUT = 25
# Re-localisation needs this fraction of the screen cells that land on mapped cells to match.
RELOCALISE_MIN_FRACTION = 0.7
# This many of the best-voted placements are checked cell by cell before one is accepted.
RELOCALISE_CANDIDATES = 5
# Placements matching within this fraction of the best are a close call.
RELOCALISE_CLOSE_CALL = 0.8
# A close call is settled only by a placement matching this many more distinctive cells than every other.
DISTINCTIVE_LEAD = 2
# A cell picture is distinctive if it appears at most this many times in the segment.
DISTINCTIVE_MAX_COPIES = 2
# Two ends of door links this close (in squares, either way) count as the same door.
DOOR_REACH = 1
# A verdict on merging two differently named areas stands until their overlap grows this many times larger.
MERGE_REJUDGE_GROWTH = 1.5
# Consecutive unplaceable free-roam frames before a new segment is opened.
LOST_PATIENCE = 3
# A segment younger than this many placements keeps checking whether it overlaps an older one.
MERGE_WINDOW = 40
# Two segments are merged at the end of an episode only if they share this many matching cells.
CONSOLIDATE_MIN_CELLS = 30
# Merging also needs at least this fraction of the overlap to match.
CONSOLIDATE_MIN_AGREEMENT = 0.6
# Merging also needs the best offset to have this many times the votes of any other.
CONSOLIDATE_MARGIN = 1.5
# Failing that margin (repeating scenery), the best offset needs at least this many votes.
CONSOLIDATE_MIN_VOTES = 10
# Failing that margin, the best offset's agreement must also beat the rival offset's by this much.
CONSOLIDATE_AGREEMENT_LEAD = 0.2
DOOR_TAGS = {"entrance", "staircase", "elevator"}
NEIGHBOURS_8 = tuple((ox, oy) for ox in (-1, 0, 1) for oy in (-1, 0, 1) if (ox, oy) != (0, 0))
# A segment is only named once the player has been placed in it this many times.
MIN_NAME_OBSERVATIONS = 3
# A segment is (re)named at the end of an episode once its map has grown by this many pixels: about two screens.
RENAME_GROWTH_PIXELS = 2 * 90 * 16 * 16


def is_door(c: "MapCell") -> bool:
    tags = set(c.tags)
    return bool(tags & DOOR_TAGS) and "obstacle" not in tags


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
        # a move down across exactly one cell mapped as blocked went over a ledge
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
        from .routes import _exits
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

    def clear_routes(self) -> None:
        self.routes = []

    def take_naming_queue(self) -> List[Dict[str, Any]]:
        queue, self.naming_queue = self.naming_queue, []
        return queue

    def count(self, stat: str, by: int = 1) -> None:
        self.stats[stat] += by

    def return_to_defeat(self, defeat: Dict[str, Any], episode: int) -> None:
        self.defeat_return = {"defeat": defeat, "episode": episode}

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

    def observe(self, frame: np.ndarray, identified: Dict[Tuple[int, int, str], Any],
                shift: Cell = (0, 0)) -> int:
        seg = self.segments[self.segment]
        a = _screen(frame)
        px, py = self.pos
        added = 0
        for (x, y), piece in _local_cells(a, shift):
            pixels = piece.copy()
            quarters = split_quarters(piece)
            hashes = tuple(cell_key(quarters[q]) for q in QUARTERS)
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
            # an NPC on a cell, or a tile not yet recognised, never overwrites what the cell was known to be
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
        self.step += 1
        if faded:
            # after a blank screen the position must be recognised from the map, not slid from the last frame
            self.faded = True
            self.fade_direction = direction
        if not free_roam:
            # remember a move under way so the next free-roam frame is placed relative to the last good one
            if was_free_roam and direction in DIRECTIONS:
                self.pending = (direction, max_steps, reported)
            self.fade_direction = None
            self.stats["not_free_roam"] += 1
            return "not_free_roam"
        if not has_structure(frame):
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
        # often one bad frame, still fading in: wait a few frames before deciding this is somewhere new
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
        a = _screen(frame)
        local: Dict[Cell, Tuple[Optional[str], ...]] = {}
        for (x, y), piece in _local_cells(a, shift):
            # blank cells match everywhere, so they do not vote
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
            ranked = sorted((self._screen_agreement(seg, local, o), o)
                            for o, _ in votes.most_common(RELOCALISE_CANDIDATES))[::-1]
            close = [(o, overlap) for (m, overlap), o in ranked if m >= RELOCALISE_CLOSE_CALL * ranked[0][0][0]]
            if len(close) > 1:
                distinctive = sorted(((self._distinctive_matches(seg, index, local, o), o, overlap)
                                      for o, overlap in close), reverse=True)
                if distinctive[0][0] - distinctive[1][0] < DISTINCTIVE_LEAD:
                    continue
                close = [(distinctive[0][1], distinctive[0][2])]
            offset, overlap = close[0]
            n = self._screen_agreement(seg, local, offset)[0]
            # a small room shows few content cells: ask for most of them rather than the fixed minimum
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
        from .routes import plan_route, render_plan
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
        count = 0
        for ep in sorted(episodes, reverse=True):
            if self.new_cells_by_episode.get(ep, 0) > 0:
                break
            count += 1
        return count

    def frontier(self, segment: Optional[int] = None) -> List[Cell]:
        seg = self.current if segment is None else self.segments[segment]
        out = []
        for g, c in seg.items():
            if c.kind != WALKABLE:
                continue
            if any((g[0] + dx, g[1] + dy) not in seg for dx, dy in DIRECTIONS.values()):
                out.append(g)
        return out
