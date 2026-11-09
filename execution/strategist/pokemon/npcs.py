"""
The people the player meets, as a sixth-and-a-half strategist artifact: every person seen on screen is cut out as a
sprite, followed from step to step, compared with everyone already in the registry and recorded with where and when.

Three levels, so a picture never sits on a person it might not belong to:

* a **View** is one picture of a sprite (one facing, one animation frame);
* a **Look** is the set of views of one sprite. A look can start with a single side view and grow when the same
  person is later seen turning. Several people can share a look (look-alikes);
* an **NPC** is one individual. It has no picture of its own, only a link to its look, a name that is unique in the
  registry, and its sightings and conversations.

Places are stored raw, as ``(segment, square)``, and translated through the global map's merge records every time
they are read, so a map merge never leaves a sighting pointing at a piece that no longer exists.

This module only watches: it reads the map and the tile table and never writes to either.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from execution.artifact import StrategistArtifact, update
from execution.perception.pokemon import sprites as S
from execution.strategist.pokemon.global_map import GlobalMap, StepEvent
from execution.strategist.pokemon.npc_names import draw_name
from utils import log_info, log_warn

Cell = Tuple[int, int]
FACINGS = ("down", "up", "left", "right")
KINDS = ("person", "trainer", "pokemon", "object", "unknown")

#: A frame is read for people only when the map has placed it.
PLACED_STATUS = {"start", "moved", "stayed", "relocalised", "new_segment"}
#: The same look within this many squares of where a person was last seen is that person (by code, no model).
NEAR = 4
#: A new person is compared with at most this many candidates by a model.
MAX_CANDIDATES = 8
#: A person standing still is noted again after this many map steps.
SIGHTING_EVERY = 10
#: A sprite caught mid-step or half hidden needs at least this many dark pixels to count as a person.
MIN_TAGGED_BLACK = 12


@dataclass
class View:
    pixels: np.ndarray                 # 16x16 greys, the sprite window as it was on screen
    key: np.ndarray                    # 16x16 bool, the black outline
    facing: Optional[str]              # which way it faces, once known
    mirrored_of: Optional[int]         # the view this is a left/right mirror of, if known
    first_seen: Tuple[int, int]        # (episode, map step)
    n_seen: int = 1


@dataclass
class Look:
    id: str
    views: List[View]
    kind_hint: Optional[str] = None    # "object" once a model says it is a statue, doll or machine
    links: List[Tuple[int, int, str]] = field(default_factory=list)   # (view, view, why)


@dataclass
class Sighting:
    episode: int
    map_step: int
    segment: int                       # the raw segment index at the time
    square: Cell                       # the raw square in that segment
    view: int
    offset: Cell = (0, 0)
    track: int = -1


@dataclass
class Conversation:
    id: str
    npc: Optional[str]
    sign: Optional[str]
    episode: int
    map_step_start: int
    map_step_end: Optional[int]
    segment: int
    square: Cell
    target: Optional[Cell]
    trigger: str
    crops: List[np.ndarray] = field(default_factory=list)
    text: str = ""
    summary: str = ""
    speaker_name: Optional[str] = None
    outcome: Optional[str] = None
    led_to: Optional[str] = None
    complete: bool = False


@dataclass
class NPC:
    id: str
    name: str
    look: str
    kind: str = "unknown"
    description: str = ""
    real_name: Optional[str] = None
    name_type: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    sightings: List[Sighting] = field(default_factory=list)
    conversations: List[str] = field(default_factory=list)
    needs: Optional[str] = None        # "card" | "judge" | None: model work still to do
    candidates: List[str] = field(default_factory=list)
    created: Tuple[int, int] = (0, 0)
    merged_into: Optional[str] = None


@dataclass
class Sign:
    id: str
    segment: int
    square: Cell
    conversations: List[str] = field(default_factory=list)
    text: str = ""


@dataclass
class Track:
    id: int
    npc: str
    segment: int
    square: Cell
    view: int
    last_step: int
    steps_seen: int = 1
    ambiguous: bool = False
    last_sight_step: int = -10_000
    last_sight_square: Optional[Cell] = None


def _manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class NPCRegistry(StrategistArtifact):
    """Every person the player has seen in one run. See the module docstring."""

    _TRANSIENT = ("_gmap", "_episode", "_tracks", "_last_people", "_stack", "_stack_ref", "_look_merges")

    def __init__(self, game: str, run_name: str) -> None:
        self.game = game
        self.run_name = run_name
        self.looks: Dict[str, Look] = {}
        self.npcs: Dict[str, NPC] = {}
        self.signs: Dict[str, Sign] = {}
        self.conversations: Dict[str, Conversation] = {}
        self.unattributed: List[str] = []
        self.look_aliases: Dict[str, str] = {}              # a merged-away look -> the look that took it in
        self.covisible: Set[FrozenSet[str]] = set()         # two people seen in one frame are never the same person
        self.next_ids: Counter = Counter()
        self.stats: Counter = Counter()
        self.version = 1
        self._reset_transient()

    def _reset_transient(self) -> None:
        self._gmap: Optional[GlobalMap] = None
        self._episode = 0
        self._tracks: Dict[int, Track] = {}
        self._last_people: Dict[Cell, str] = {}
        self._stack: Optional[np.ndarray] = None
        self._stack_ref: List[Tuple[str, int]] = []
        self._look_merges: Set[Tuple[str, str]] = set()

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        self._reset_transient()
        for name, default in (("look_aliases", dict), ("covisible", set), ("unattributed", list),
                              ("signs", dict), ("conversations", dict), ("stats", Counter),
                              ("next_ids", Counter), ("version", lambda: 1)):
            self.__dict__.setdefault(name, default())

    @property
    def _p(self) -> Dict[str, Any]:
        return S.params(self.game)

    # -- ids, names, lookups ----------------------------------------------------------------------

    def _next(self, prefix: str) -> str:
        self.next_ids[prefix] += 1
        return f"{prefix}_{self.next_ids[prefix]:03d}"

    def _taken_names(self) -> List[str]:
        return [x for n in self.npcs.values() for x in (n.name, n.real_name, *n.aliases) if x]

    def get(self, npc_id: str) -> NPC:
        """The NPC, following ``merged_into`` like ``GlobalMap.resolve``."""
        npc = self.npcs[npc_id]
        while npc.merged_into is not None:
            npc = self.npcs[npc.merged_into]
        return npc

    def live_npcs(self) -> List[NPC]:
        return [n for n in self.npcs.values() if n.merged_into is None]

    def look_of(self, npc: NPC) -> Look:
        look_id = npc.look
        while look_id in self.look_aliases:
            look_id = self.look_aliases[look_id]
        return self.looks[look_id]

    # -- places -----------------------------------------------------------------------------------

    def place(self, gmap: GlobalMap, sighting: Sighting) -> Tuple[int, Cell]:
        """Where a sighting is now: its raw place translated through the map's merges."""
        return gmap.resolve(sighting.segment, sighting.square)

    def last_seen(self, gmap: GlobalMap, npc: NPC) -> Optional[Tuple[int, Cell, Sighting]]:
        if not npc.sightings:
            return None
        newest = max(npc.sightings, key=lambda s: s.map_step)
        segment, square = self.place(gmap, newest)
        return segment, square, newest

    # -- the per-episode and per-step hooks -------------------------------------------------------

    def begin_episode(self, episode: int, gmap: GlobalMap) -> None:
        """Called by the strategist before the episode's first step: tracks never run across episodes."""
        self._gmap, self._episode = gmap, episode
        self._tracks, self._last_people = {}, {}

    def on_opening_screen(self, frame: np.ndarray, identified: Dict[Any, Any], free: bool) -> None:
        if free and self._gmap is not None:
            self._process(frame, identified, "start", 0)

    def on_env_step(self, ev: StepEvent) -> None:
        """A :class:`~execution.strategist.pokemon.global_map.MappingEnvironment` listener."""
        if self._gmap is None or not ev.free or ev.status not in PLACED_STATUS:
            return
        self._process(ev.frame, ev.identified, ev.status, ev.max_steps)

    # -- seeing people ----------------------------------------------------------------------------

    def _view_stack(self) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
        if self._stack is None:
            keys, ref = [], []
            for look in self.looks.values():
                for i, view in enumerate(look.views):
                    keys.append(view.key)
                    ref.append((look.id, i))
            self._stack = np.stack(keys) if keys else np.zeros((0, 16, 16), dtype=bool)
            self._stack_ref = ref
        return self._stack, self._stack_ref

    def _best_view(self, mask: np.ndarray) -> Optional[Tuple[int, str, int]]:
        keys, ref = self._view_stack()
        if len(keys) == 0:
            return None
        distances, _ = S.stack_distances(keys, mask)
        index = int(distances.argmin())
        return int(distances[index]), ref[index][0], ref[index][1]

    def _detect(self, frame: np.ndarray, identified: Dict[Any, Any]) -> List[Dict[str, Any]]:
        p = self._p
        level = S.black_level(frame)
        cells = S.find_npc_cells(identified, p["min_quarters"])
        found: List[Dict[str, Any]] = []
        for cell in cells:
            window = S.sprite_window(frame, cell, (0, 0), p["y_offset"])
            if window is None:
                continue
            mask = S.black_mask(window, level)
            if int(mask.sum()) < MIN_TAGGED_BLACK:
                continue
            found.append(dict(cell=cell, offset=(0, 0), window=window, mask=mask,
                              full=S.full_window_ok(identified, cell)))
        keys, _ = self._view_stack()
        if len(keys):
            for cell, offset, _, _ in S.scan_for_views(frame, keys, S.scan_cells(identified, cells), level,
                                                       p["scan_range"], p["scan_step"], p["auto"], p["y_offset"],
                                                       p["min_black"]):
                window = S.sprite_window(frame, cell, offset, p["y_offset"])
                if window is not None:
                    found.append(dict(cell=cell, offset=offset, window=window, mask=S.black_mask(window, level),
                                      full=False))
        return found

    def _process(self, frame: np.ndarray, identified: Dict[Any, Any], status: str, max_steps: int) -> None:
        gmap = self._gmap
        if status in ("relocalised", "new_segment"):
            self._tracks = {}
        if gmap.residual != (0, 0):                      # off the 16 px grid: tags are for the unshifted grid
            return
        pos, segment, step = gmap.pos, gmap.segment, gmap.step
        detections = self._detect(frame, identified)
        for det in detections:
            det["square"] = (pos[0] + det["cell"][0], pos[1] + det["cell"][1])
        live = {t.id: t for t in self._tracks.values() if t.segment == segment}
        allowance = max(1, min(max_steps, 4))

        # a detection close to another one could be mistaken for it: those tracks never link views
        crowded = {i for i, a in enumerate(detections) for j, b in enumerate(detections)
                   if i != j and _manhattan(a["square"], b["square"]) <= 2}

        pairs = []
        for i, det in enumerate(detections):
            for track in live.values():
                moved = _manhattan(det["square"], track.square)
                if moved <= allowance:
                    view = self.look_of(self.npcs[track.npc]).views[track.view]
                    pairs.append((moved, S.mask_distance(det["mask"], view.key)[0], i, track.id))
        pairs.sort()
        assigned: Dict[int, int] = {}
        taken: Set[int] = set()
        for _, _, i, tid in pairs:
            if i not in assigned and tid not in taken:
                assigned[i] = tid
                taken.add(tid)

        now: Dict[int, Track] = {}
        for i, tid in assigned.items():
            track = live[tid]
            self._update_track(track, detections[i], i in crowded, step)
            now[track.id] = track
        busy = {t.npc for t in now.values()}
        for i, det in enumerate(detections):
            if i in assigned:
                continue
            track = self._start_track(det, busy, segment, step)
            if track is not None:
                track.ambiguous = i in crowded
                now[track.id] = track
                busy.add(track.npc)
        self._tracks = now
        people = [t.npc for t in now.values()]
        self.covisible.update(frozenset(pair) for pair in _pairs(people))
        by_square = {d["square"]: d["cell"] for d in detections}
        self._last_people = {by_square[t.square]: t.npc for t in now.values() if t.square in by_square}
        self.stats["frames_with_people"] += bool(now)

    def _new_look(self, det: Dict[str, Any], step: int) -> Look:
        look = Look(id=self._next("look"), views=[View(det["window"].copy(), det["mask"].copy(), None, None,
                                                       (self._episode, step))])
        self.looks[look.id] = look
        self._stack = None
        self.stats["looks_created"] += 1
        return look

    def _new_npc(self, look: Look, needs: str, candidates: List[str], step: int) -> NPC:
        npc_id = self._next("npc")
        npc = NPC(id=npc_id, name=draw_name(self.game, self.run_name, npc_id, self._taken_names()), look=look.id,
                  needs=needs, candidates=candidates[:MAX_CANDIDATES], created=(self._episode, step))
        self.npcs[npc_id] = npc
        self.stats["npcs_created"] += 1
        return npc

    def _near(self, segment: int, square: Cell, exclude: Set[str], look_id: Optional[str] = None) -> List[NPC]:
        """People last seen within ``NEAR`` squares of ``square`` in the (resolved) segment, nearest first."""
        gmap = self._gmap
        found = []
        for npc in self.live_npcs():
            if npc.id in exclude or (look_id is not None and self.look_of(npc).id != look_id):
                continue
            seen = self.last_seen(gmap, npc)
            if seen is None or seen[0] != segment:
                continue
            distance = _manhattan(seen[1], square)
            if distance <= NEAR:
                found.append((distance, -seen[2].map_step, npc))
        return [npc for _, _, npc in sorted(found, key=lambda x: (x[0], x[1]))]

    def _start_track(self, det: Dict[str, Any], busy: Set[str], segment: int, step: int) -> Optional[Track]:
        """A person we were not following: who is it? Compared against every view of every look."""
        p, square = self._p, det["square"]
        best = self._best_view(det["mask"])
        npc: Optional[NPC] = None
        view_index = 0
        if best is not None and best[0] <= p["auto"]:
            look = self.looks[best[1]]
            if look.kind_hint == "object":
                self.stats["object_hits"] += 1
                return None
            same = [n for n in self.live_npcs() if self.look_of(n).id == look.id and n.id not in busy]
            near = self._near(segment, square, busy, look.id)
            if near:
                npc, view_index = near[0], best[2]
            elif det["full"] or int(det["mask"].sum()) >= p["min_black"]:
                # the same look, but nobody of that look was last seen here: a look-alike elsewhere is another
                # person until a model (or a conversation) says otherwise
                recent = sorted(same, key=lambda n: -max((s.map_step for s in n.sightings), default=-1))
                npc = self._new_npc(look, "judge" if same else "card", [n.id for n in recent], step)
                view_index = best[2]
            else:
                return None
        else:
            if not det["full"]:
                self.stats["ignored_partial"] += 1
                return None
            look = self._new_look(det, step)
            candidates = [n.id for n in self._near(segment, square, busy)]
            if best is not None and best[0] <= p["maybe"]:
                candidates += [n.id for n in self.live_npcs()
                               if self.look_of(n).id == best[1] and n.id not in busy and n.id not in candidates]
            npc = self._new_npc(look, "card", candidates, step)
        self.stats["tracks_started"] += 1
        self.next_ids["track"] += 1
        track = Track(id=self.next_ids["track"], npc=npc.id, segment=segment, square=square, view=view_index,
                      last_step=step)
        self._note_sighting(track, det, step)
        return track

    def _update_track(self, track: Track, det: Dict[str, Any], crowded: bool, step: int) -> None:
        p = self._p
        npc = self.get(track.npc)
        look = self.look_of(npc)
        current = look.views[track.view]
        distance, _ = S.mask_distance(det["mask"], current.key)
        track.square, track.last_step = det["square"], step
        track.steps_seen += 1
        track.ambiguous = track.ambiguous or crowded
        if distance <= p["auto"]:
            current.n_seen += 1
        elif det["full"] and not track.ambiguous:
            # the person changed picture while we watched: the same person turned or took a step
            nearest = sorted((S.mask_distance(det["mask"], v.key)[0], i) for i, v in enumerate(look.views))[0]
            if nearest[0] <= p["auto"]:
                track.view = nearest[1]
                look.views[nearest[1]].n_seen += 1
            else:
                look.views.append(View(det["window"].copy(), det["mask"].copy(), None, None,
                                       (self._episode, step)))
                look.links.append((track.view, len(look.views) - 1, "track"))
                track.view = len(look.views) - 1
                self._stack = None
                self.stats["views_added"] += 1
            other = self._best_view(det["mask"])
            if other is not None and other[1] != look.id and other[0] <= p["auto"]:
                self._look_merges.add((look.id, other[1]))
        self._note_sighting(track, det, step)

    def _note_sighting(self, track: Track, det: Dict[str, Any], step: int) -> None:
        if track.last_sight_square == det["square"] and step - track.last_sight_step < SIGHTING_EVERY:
            return
        self.npcs[track.npc].sightings.append(Sighting(self._episode, step, track.segment, det["square"], track.view,
                                                       det["offset"], track.id))
        track.last_sight_step, track.last_sight_square = step, det["square"]
        self.stats["sightings"] += 1

    # -- end of episode ---------------------------------------------------------------------------

    def merge_npcs(self, keep: NPC, drop: NPC) -> None:
        """Fold ``drop`` into ``keep``: they turned out to be one person."""
        keep.sightings += drop.sightings
        keep.conversations += drop.conversations
        if drop.name not in keep.aliases:
            keep.aliases.append(drop.name)
        keep.aliases += [a for a in drop.aliases if a not in keep.aliases]
        if drop.real_name and not keep.real_name:
            keep.real_name, keep.name_type = drop.real_name, drop.name_type
        for conversation in drop.conversations:
            if conversation in self.conversations:
                self.conversations[conversation].npc = keep.id
        self.covisible = {frozenset(keep.id if x == drop.id else x for x in pair) for pair in self.covisible}
        self.covisible = {pair for pair in self.covisible if len(pair) == 2}
        drop.sightings, drop.conversations, drop.merged_into, drop.needs = [], [], keep.id, None
        self.stats["npcs_merged"] += 1

    def merge_looks(self, keep_id: str, drop_id: str) -> None:
        """Two looks turned out to be one sprite: move the views over and re-point the people."""
        keep, drop = self.looks[keep_id], self.looks[drop_id]
        shift = len(keep.views)
        keep.views += drop.views
        keep.links += [(a + shift, b + shift, why) for a, b, why in drop.links]
        keep.links.append((0, shift, "track"))
        for npc in self.npcs.values():
            if self.look_aliases.get(npc.look, npc.look) == drop_id or npc.look == drop_id:
                npc.look = keep_id
                for s in npc.sightings:
                    s.view += shift
        self.look_aliases[drop_id] = keep_id
        del self.looks[drop_id]
        self._stack = None
        self.stats["looks_merged"] += 1

    def consolidate(self) -> None:
        """The code-only clean-up: look merges the tracks found, then people of one look that a map merge brought
        to the same place and that were never on screen together."""
        gmap = self._gmap
        for a, b in sorted(self._look_merges):
            a, b = self.look_aliases.get(a, a), self.look_aliases.get(b, b)
            if a != b and a in self.looks and b in self.looks:
                self.merge_looks(a, b)
        self._look_merges = set()
        people = sorted(self.live_npcs(), key=lambda n: n.created)
        for i, older in enumerate(people):
            if older.merged_into is not None:
                continue
            for newer in people[i + 1:]:
                if newer.merged_into is not None or self.look_of(newer).id != self.look_of(older).id:
                    continue
                if frozenset((older.id, newer.id)) in self.covisible:
                    continue
                a, b = self.last_seen(gmap, older), self.last_seen(gmap, newer)
                if a and b and a[0] == b[0] and _manhattan(a[1], b[1]) <= NEAR:
                    self.merge_npcs(older, newer)

    @update
    def end_episode(self, gmap: GlobalMap) -> None:
        """Close the episode's bookkeeping; decorated so the registry is saved every episode."""
        self._gmap = gmap
        self.consolidate()
        self._tracks, self._last_people = {}, {}

    # -- reading ----------------------------------------------------------------------------------

    def people_here(self, gmap: GlobalMap) -> List[Tuple[NPC, Cell]]:
        """People whose last sighting is in the player's current (resolved) segment, with that square."""
        found = []
        for npc in self.live_npcs():
            seen = self.last_seen(gmap, npc)
            if seen is not None and seen[0] == gmap.segment:
                found.append((npc, seen[1]))
        return found

    def diff(self, old: "NPCRegistry") -> Dict[str, Any]:
        new_npcs = [i for i in self.npcs if i not in old.npcs]
        return {
            "new_npcs": new_npcs,
            "new_looks": [i for i in self.looks if i not in old.looks],
            "new_views": {i: len(l.views) - len(old.looks[i].views) for i, l in self.looks.items()
                          if i in old.looks and len(l.views) != len(old.looks[i].views)},
            "new_conversations": [i for i in self.conversations if i not in old.conversations],
            "merged": {i: n.merged_into for i, n in self.npcs.items()
                       if n.merged_into and (i not in old.npcs or old.npcs[i].merged_into is None)},
            "renamed": [(i, old.npcs[i].name, n.name) for i, n in self.npcs.items()
                        if i in old.npcs and old.npcs[i].name != n.name],
            "kind_changed": [(i, old.npcs[i].kind, n.kind) for i, n in self.npcs.items()
                             if i in old.npcs and old.npcs[i].kind != n.kind],
            "new_signs": [i for i in self.signs if i not in old.signs],
        }


def _pairs(items: List[str]) -> List[Tuple[str, str]]:
    return [(a, b) for i, a in enumerate(items) for b in items[i + 1:] if a != b]
