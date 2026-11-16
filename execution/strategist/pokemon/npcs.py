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
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from execution.artifact import StrategistArtifact, update
from execution.perception.pokemon import sprites as S
from execution.perception.pokemon.grid import LEFT_COL, TOP_ROW, cell_box
from execution.strategist.pokemon.global_map import (DIRECTIONS, OCCUPIED, PX_SLACK, GlobalMap, StepEvent,
                                                     grid_offset, grid_residual)
from execution.strategist.pokemon import npc_model
from execution.strategist.pokemon.npc_map import LABEL_NOTE, WHERE_NOTES, maybe_lines
from execution.strategist.pokemon.npc_model import display_name
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
#: What a conversation can be with besides a person: a map square carrying one of these tags.
SIGN_TAGS = {"sign", "interactable object", "item"}
#: At most this many dialogue pictures of one conversation are read by the model.
MAX_OCR_CROPS = 12
OCR_PROMPT = "If there is no text in the image, just say NONE. Otherwise, perform OCR and state the text in this image:"
#: People not seen for more than this many episodes are left out of the strategist's list: they have likely moved.
STALE_EPISODES = 20
#: A person the tile table only half recognises (someone seated on a bench, partly behind a counter) is accepted when
#: at least ``WEAK_QUARTERS`` of the square's quarters read as a person, the outline is this heavy, and the same
#: square shows it in ``WEAK_FRAMES`` map steps.
WEAK_QUARTERS = 2
WEAK_BLACK = 60
WEAK_FRAMES = 2
#: On the map, a person not seen for more than this many episodes may have moved: drawn grey, listed as such.
STALE_MARK_EPISODES = 3
#: A square drawn like a sprite and tagged as a person where nobody was recognised (someone in tall grass, or cut off)
#: is shown to a model once it was seen like that in this many map steps; at most ``MAX_CONFIRM`` squares per episode.
CONFIRM_FRAMES = 2
MAX_CONFIRM = 6
#: The picture kept of such a square: the squares around it, this many in every direction.
CONFIRM_REACH = 2


@dataclass
class View:
    pixels: np.ndarray                 # 16x16 greys, the sprite window as it was on screen
    key: np.ndarray                    # 16x16 bool, the black outline
    facing: Optional[str]              # which way it faces, once known
    mirrored_of: Optional[int]         # the view this is a left/right mirror of, if known
    first_seen: Tuple[int, int]        # (episode, map step)
    n_seen: int = 1
    partial: bool = False              # a person seen only in part (in tall grass): never matched against others


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
    flipped: bool = False              # drawn as the mirror image of its view: the person turned left <-> right


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
    read: bool = False                 # the model has been asked to read the crops
    context: Optional[np.ndarray] = None   # the screen just before the conversation began
    summarised: bool = False           # the model has summarised the text (or gave up)
    name_type: Optional[str] = None    # "personal" | "class" for ``speaker_name``
    tries: int = 0
    how: str = "seen"                  # "seen" (on screen next to the player), "remembered" (the registry had them on
                                       # the square in front), "nearest" (they spoke first and were the only known
                                       # person near, a guess), "hidden" (a person never seen whole on the square in
                                       # front, found by this conversation) or "none"
    confidence: str = "high"           # "low" for a "nearest" guess, or a person who walks about attributed from memory


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
    context: Optional[np.ndarray] = None   # the screen (greys) on which this person was first seen
    context_cell: Optional[Cell] = None    # and the square the person stood on in it
    tries: int = 0                         # model answers that could not be used
    hidden: Optional[str] = None           # never seen whole when found: "conversation" (talked to on that square) or
                                           # "model" (a model saw someone in the picture of that square)


@dataclass
class Sign:
    id: str
    segment: int
    square: Cell
    conversations: List[str] = field(default_factory=list)
    text: str = ""


@dataclass
class MaybeSquare:
    """A square drawn like a sprite and tagged as a person where nobody was recognised: someone hidden (in tall grass)
    or cut off, not seen whole yet. Kept until a conversation or a model says who, or what, is there."""
    frames: int = 0                        # map steps it was seen like that
    last_step: int = -1
    crop: Optional[bytes] = None           # PNG of the squares around it, from the frame where it was nearest the player
    box: Cell = (0, 0)                     # the square's top-left pixel in the crop
    distance: int = 99                     # its distance from the player in that frame
    seen: Tuple[int, int] = (0, 0)         # (episode, map step) of that frame
    verdict: Optional[str] = None          # "person" | "pokemon" | "nothing", or "unclear" once a model gave up
    tries: int = 0


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
    last_sight_flipped: bool = False


@dataclass
class PersonMark:
    """One person on the map, for one area, as of one map step: derived from the registry on every read (like the
    global map's doors), never stored."""
    npc_id: str
    number: int                        # the number in the id: stable, so "P3" means the same person in every picture
    name: str                          # the display name (a real name once the person gave one)
    kind: str
    square: Cell                       # the (resolved) square last seen on
    facing: Optional[str]              # which way the person faced at that sighting, once a model said so
    last_seen: Tuple[int, int]         # (episode, map step)
    on_screen: bool
    talked: str                        # "yes" | "probably" (a guess at who spoke) | "no", in this area
    times_talked: int
    same_words: bool                   # talked to more than once here, the same words every time
    earlier: List[Cell]                # up to 3 earlier squares in this area, newest first
    spread: Optional[Tuple[int, int, int, int]]   # (min x, min y, max x, max y) of every square, for a walker
    stale: bool                        # may have moved: not seen for a while, or the square is on screen without them
    when: str                          # the freshness in words, for prompts


def frame_residual(frame: np.ndarray) -> Optional[Cell]:
    """How far off the cell grid the frame itself is drawn (as a residual), when its tiles show it clearly and it is
    within a scroll's reach; else None."""
    shift = grid_offset(frame)
    if shift is None:
        return None
    residual = grid_residual(shift)
    return residual if max(abs(residual[0]), abs(residual[1])) <= PX_SLACK else None


def _manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class NPCRegistry(StrategistArtifact):
    """Every person the player has seen in one run. See the module docstring."""

    _TRANSIENT = ("_gmap", "_episode", "_tracks", "_last_people", "_last_people_at", "_stack", "_stack_ref",
                  "_look_merges", "_weak", "_facing", "_open", "_frame", "_free_frame", "_live", "_a_step", "_resume")

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
        #: (raw segment, square) the tile table tagged as a person -> whether it was ever drawn like a sprite there
        #: (False: only ever like background, such as a statue or a bush)
        self.tagged_squares: Dict[Tuple[int, Cell], bool] = {}
        #: (raw segment, square) drawn like a sprite, tagged as a person, nobody recognised there (see MaybeSquare)
        self.maybe_squares: Dict[Tuple[int, Cell], MaybeSquare] = {}
        #: the conversation the last episode ended in the middle of: the next one may open in its text box
        self.interrupted: Optional[str] = None
        self.version = 1
        self._reset_transient()

    def _reset_transient(self) -> None:
        self._gmap: Optional[GlobalMap] = None
        self._episode = 0
        self._tracks: Dict[int, Track] = {}
        self._last_people: Dict[Cell, str] = {}                # map square -> person, on the latest free-roam frame
        self._last_people_at: Optional[Tuple[int, int]] = None  # (segment, map step) of that frame; None = not read
        self._stack: Optional[np.ndarray] = None
        self._stack_ref: List[Tuple[str, int]] = []
        self._look_merges: Set[Tuple[str, str]] = set()
        self._facing: Optional[str] = None                      # the way the player faces, once known
        self._open: Optional[str] = None                        # the conversation in progress
        self._weak: Dict[Tuple[int, Cell], Set[int]] = {}       # (segment, square) -> map steps it looked like a person
        self._frame: Optional[np.ndarray] = None                # the frame being read for people
        self._free_frame: Optional[np.ndarray] = None           # the last free-roam frame (a conversation's context)
        self._live: Optional[Callable[[str], Callable[..., Any]]] = None   # model caller by stage, for live summaries
        self._a_step: Optional[int] = None                      # map step of an A press that left the player in free roam
        self._resume: Optional[str] = None                      # ``interrupted``, for this episode's first step

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        self._reset_transient()
        for name, default in (("look_aliases", dict), ("covisible", set), ("unattributed", list),
                              ("signs", dict), ("conversations", dict), ("stats", Counter),
                              ("next_ids", Counter), ("version", lambda: 1), ("tagged_squares", dict),
                              ("maybe_squares", dict), ("interrupted", lambda: None)):
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

    def seen_whole(self, npc: NPC) -> bool:
        """False for a person only ever seen in part (in tall grass): not recognising them on screen says nothing."""
        return any(not v.partial for v in self.look_of(npc).views)

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

    def squares_in(self, gmap: GlobalMap, npc: NPC, segment: int) -> List[Cell]:
        """Every square this person has been seen on in the (resolved) segment, newest first."""
        out: List[Cell] = []
        for s in sorted(npc.sightings, key=lambda s: -s.map_step):
            seg, square = self.place(gmap, s)
            if seg == segment and square not in out:
                out.append(square)
        return out

    def _forget_screen(self) -> None:
        self._last_people, self._last_people_at = {}, None

    def on_screen(self, gmap: GlobalMap) -> Dict[Cell, str]:
        """The people detected on the latest free-roam frame, keyed by their square relative to the player now
        (empty when that frame was not read, or the player has since changed area)."""
        if self._last_people_at is None or self._last_people_at[0] != gmap.segment:
            return {}
        px, py = gmap.pos
        return {(x - px, y - py): npc for (x, y), npc in self._last_people.items()}

    def in_view(self, gmap: GlobalMap, square: Cell) -> bool:
        """Whether a square of the current (raw) segment was on the latest screen that was read for people."""
        if self._last_people_at is None or self._last_people_at[0] != gmap.segment:
            return False
        dx, dy = square[0] - gmap.pos[0], square[1] - gmap.pos[1]
        return LEFT_COL <= dx <= LEFT_COL + 9 and TOP_ROW - 8 <= dy <= TOP_ROW

    # -- the per-episode and per-step hooks -------------------------------------------------------

    def begin_episode(self, episode: int, gmap: GlobalMap) -> None:
        """Called by the strategist before the episode's first step: tracks never run across episodes."""
        self._gmap, self._episode = gmap, episode
        self._tracks = {}
        self._forget_screen()
        self._facing, self._open = None, None
        self._resume, self.interrupted = self.interrupted, None

    def on_opening_screen(self, frame: np.ndarray, identified: Dict[Any, Any], free: bool) -> None:
        """``free``: the screen is free roam and the map placed it (else it is not read for people)."""
        if free and self._gmap is not None:
            self._process(frame, identified, "start", 0)
            self._free_frame = frame

    def on_env_step(self, ev: StepEvent) -> None:
        """A :class:`~execution.strategist.pokemon.global_map.MappingEnvironment` listener."""
        if self._gmap is None:
            return
        try:                                                     # conversations first: dialogue steps are not free
            self._conversation_step(ev)
        except Exception as e:
            log_warn(f"NPC registry: conversation step skipped: {type(e).__name__}: {e}")
        if not ev.free:
            return
        if ev.status not in PLACED_STATUS:
            self._forget_screen()                                # not placed: who is on screen now is unknown
            return
        self._process(ev.frame, ev.identified, ev.status, ev.max_steps)
        self._free_frame = ev.frame

    # -- seeing people ----------------------------------------------------------------------------

    def _view_stack(self) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
        if self._stack is None:
            keys, ref = [], []
            for look in self.looks.values():
                for i, view in enumerate(look.views):
                    if not view.partial:                  # a picture of part of a person never matches anyone
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
                              full=S.full_window_ok(identified, cell) and S.complete_sprite(mask)))
        for cell in S.find_npc_cells(identified, WEAK_QUARTERS):
            if cell in cells:
                continue
            window = S.sprite_window(frame, cell, (0, 0), p["y_offset"])
            if window is None:
                continue
            mask = S.black_mask(window, level)
            if int(mask.sum()) >= WEAK_BLACK:
                found.append(dict(cell=cell, offset=(0, 0), window=window, mask=mask, full=False, weak=True))
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

    @staticmethod
    def _one_per_sprite(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Two detections whose windows overlap on screen are one sprite (a person straddling two squares, or found
        by both the tile tags and the scan): keep the whole, tag-backed one. Two people never overlap this much."""
        def origin(det):
            x0, y0, _, _ = cell_box(*det["cell"])
            return x0 + det["offset"][0], y0 + det["offset"][1]
        ranked = sorted(detections, key=lambda d: (bool(d.get("weak")), not d["full"], -int(d["mask"].sum())))
        kept: List[Dict[str, Any]] = []
        for det in ranked:
            ox, oy = origin(det)
            if all(abs(ox - origin(k)[0]) >= 12 or abs(oy - origin(k)[1]) >= 12 for k in kept):
                kept.append(det)
        return kept

    def _confirm_weak(self, detections: List[Dict[str, Any]], segment: int, step: int) -> List[Dict[str, Any]]:
        """Half-recognised people count only once the same square has shown one in ``WEAK_FRAMES`` map steps; then
        they may start a look like any other person (a seated person is whole, only the tile table is unsure)."""
        kept = [d for d in detections if not d.get("weak")]
        claimed = [d["square"] for d in kept]
        for det in sorted((d for d in detections if d.get("weak")), key=lambda d: -int(d["mask"].sum())):
            if any(_manhattan(det["square"], square) <= 1 for square in claimed):
                continue                                  # the same person, straddling two squares
            claimed.append(det["square"])
            seen = self._weak.setdefault((segment, det["square"]), set())
            seen.add(step)
            if len(seen) >= WEAK_FRAMES:
                det["full"] = True
                kept.append(det)
        return kept

    def _process(self, frame: np.ndarray, identified: Dict[Any, Any], status: str, max_steps: int) -> None:
        gmap = self._gmap
        self._frame = frame
        if status in ("relocalised", "new_segment"):
            self._tracks = {}
        if gmap.residual != (0, 0) and frame_residual(frame) != (0, 0):
            # off the 16 px grid: tags are for the unshifted grid. The map can keep a residual from a frame caught
            # mid-scroll after the screen is back on the grid, so the frame's own grid decides
            self._forget_screen()
            return
        pos, segment, step = gmap.pos, gmap.segment, gmap.step
        like = self._judge_tagged(frame, identified, segment, pos)
        detections = self._detect(frame, identified)
        for det in detections:
            det["square"] = (pos[0] + det["cell"][0], pos[1] + det["cell"][1])
        detections = self._confirm_weak(self._one_per_sprite(detections), segment, step)
        live ={t.id: t for t in self._tracks.values() if t.segment == segment}
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
        squares = {d["square"] for d in detections}
        self._last_people = {t.square: t.npc for t in now.values() if t.square in squares}
        self._last_people_at = (segment, step)
        self.stats["frames_with_people"] += bool(now)
        self._note_maybe(frame, like, {t.square for t in now.values()}, segment, pos, step)

    def _judge_tagged(self, frame: np.ndarray, identified: Dict[Any, Any], segment: int, pos: Cell) -> List[Cell]:
        """Note, for every square the tile table tags (even in part) as a person, whether it is drawn like a sprite;
        returns the screen cells that are."""
        like = []
        for cell in {(x, y) for (x, y, _), tile in identified.items() if tile.tag == "npc" and (x, y) != (0, 0)}:
            key = (segment, (pos[0] + cell[0], pos[1] + cell[1]))
            if S.sprite_like(frame, cell, self._p["y_offset"]):
                self.tagged_squares[key] = True
                like.append(cell)
            else:
                self.tagged_squares.setdefault(key, False)
        return like

    def _known_squares(self, segment: int) -> List[Cell]:
        """Every (resolved) square of the (resolved) segment a known person was seen on."""
        out = []
        for npc in self.live_npcs():
            for sighting in npc.sightings:
                seg, square = self.place(self._gmap, sighting)
                if seg == segment:
                    out.append(square)
        return out

    def _note_maybe(self, frame: np.ndarray, like: List[Cell], tracked: Set[Cell], segment: int, pos: Cell,
                    step: int) -> None:
        """Squares drawn like a sprite and tagged as a person where nobody is recognised (someone in tall grass, or cut
        off): count them and keep a small picture of each, from the frame where it was nearest the player, for the
        model check at the end of the episode."""
        import cv2

        if not like:
            return
        resolved_segment = self._gmap.resolve(segment, pos)[0]
        known = None
        screen = S._screen(frame)
        for cell in like:
            square = (pos[0] + cell[0], pos[1] + cell[1])
            if any(_manhattan(square, t) <= 1 for t in tracked):
                continue
            entry = self.maybe_squares.get((segment, square))
            if entry is not None and entry.verdict is not None:
                continue
            if known is None:
                known = self._known_squares(resolved_segment)
            if any(_manhattan(self._gmap.resolve(segment, square)[1], k) <= 1 for k in known):
                continue
            entry = self.maybe_squares.setdefault((segment, square), MaybeSquare())
            if entry.last_step != step:
                entry.frames, entry.last_step = entry.frames + 1, step
            distance = abs(cell[0]) + abs(cell[1])
            if entry.crop is None or distance < entry.distance:
                x0, y0, _, _ = cell_box(*cell)
                reach = CONFIRM_REACH * 16
                left, top = max(0, x0 - reach), max(0, y0 - reach)
                crop = screen[top:min(screen.shape[0], y0 + 16 + reach), left:min(screen.shape[1], x0 + 16 + reach)]
                entry.crop = cv2.imencode(".png", np.ascontiguousarray(crop, dtype=np.uint8))[1].tobytes()
                entry.box, entry.distance, entry.seen = (x0 - left, y0 - top), distance, (self._episode, step)

    def background_squares(self, gmap: GlobalMap) -> Set[Tuple[int, Cell]]:
        """(Resolved) squares the tile table calls a person that were only ever drawn like background, or where a
        model saw nobody."""
        return ({gmap.resolve(*key) for key, sprite in self.tagged_squares.items() if not sprite}
                | {gmap.resolve(*key) for key, m in self.maybe_squares.items() if m.verdict == "nothing"})

    def hidden_person(self, segment: int, square: Cell, why: str, when: Tuple[int, int],
                      window: Optional[np.ndarray] = None, level: int = 0, needs: Optional[str] = "card") -> NPC:
        """A person never seen whole, found on (raw) ``square``: talked to there, or seen there by a model. Their look
        is the part of the sprite that shows, marked partial so it never matches anyone; a whole sprite later seen on
        that square joins them as a person who stands still (``_static_at``)."""
        if window is None or window.shape != (16, 16):
            window = np.full((16, 16), 255, np.uint8)
        look = Look(id=self._next("look"), views=[View(np.array(window, copy=True), S.black_mask(window, level), None,
                                                       None, when, partial=True)])
        self.looks[look.id] = look
        npc = self._new_npc(look, needs, [], when[1])
        npc.created, npc.hidden = when, why
        npc.sightings.append(Sighting(when[0], when[1], segment, square, 0))
        self.stats[f"hidden_by_{why}"] += 1
        return npc

    def _hidden_at(self, offset: Cell) -> Optional[str]:
        """The person talked to on the square at ``offset`` from the player when nobody known is there: a square drawn
        like a sprite (or held by the map as a person) that no model said was empty. A new person, never seen whole."""
        g = self._gmap
        square = (g.pos[0] + offset[0], g.pos[1] + offset[1])
        segment, resolved = g.resolve(g.segment, square)
        entry = self.maybe_squares.get((g.segment, square))
        if entry is not None and entry.verdict == "nothing":
            return None
        if not (self.tagged_squares.get((g.segment, square)) or resolved in self.unconfirmed(g, segment)):
            return None
        if any(_manhattan(resolved, k) <= 1 for k in self._known_squares(segment)):
            return None
        window, level = None, 0
        if self._free_frame is not None:
            window = S.sprite_window(self._free_frame, offset, (0, 0), self._p["y_offset"])
            level = S.black_level(self._free_frame)
        npc = self.hidden_person(g.segment, square, "conversation", (self._episode, g.step), window, level)
        if self._free_frame is not None:
            npc.context, npc.context_cell = np.array(self._free_frame, copy=True), offset
        if entry is not None:
            entry.verdict, entry.crop = "person", None
        return npc.id

    def confirm_candidates(self, gmap: GlobalMap) -> List[Tuple[Tuple[int, Cell], MaybeSquare]]:
        """The squares to show a model: seen like a person in ``CONFIRM_FRAMES`` map steps, or held by the map as a
        person, nobody known on or next to them, not yet looked at; one per (resolved) square, most often seen first."""
        out, done = [], set()
        for key, m in sorted(self.maybe_squares.items(), key=lambda kv: -kv[1].frames):
            if m.verdict is not None or m.crop is None:
                continue
            segment, square = gmap.resolve(*key)
            if (segment, square) in done:
                continue
            if m.frames < CONFIRM_FRAMES and square not in self.unconfirmed(gmap, segment):
                continue
            if any(_manhattan(square, k) <= 1 for k in self._known_squares(segment)):
                continue
            done.add((segment, square))
            out.append((key, m))
        return out[:MAX_CONFIRM]

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

    def _static_at(self, segment: int, square: Cell, busy: Set[str]) -> Optional[NPC]:
        """The person last seen on exactly this square who has only ever been seen on that square (so, someone
        who stands still), and is not being followed right now."""
        gmap = self._gmap
        for npc in self.live_npcs():
            if npc.id in busy:
                continue
            seen = self.last_seen(gmap, npc)
            if seen is None or seen[0] != segment or seen[1] != square:
                continue
            if len({self.place(gmap, s) for s in npc.sightings}) == 1:
                return npc
        return None

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
            static = None if near else self._static_at(segment, square, busy)
            if near:
                npc, view_index = near[0], best[2]
            elif static is not None:
                # someone who never moves was last seen on exactly this square: they turned, even if the new picture
                # is a look-alike's (the Crystal officer facing left matched another officer's side view): one sprite
                npc, view_index = static, self._turned(static, det, step)
                if det["full"] and self.look_of(static).id != look.id:
                    self._look_merges.add((self.look_of(static).id, look.id))
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
                npc = self._static_at(segment, square, busy)
                if npc is None or not npc.hidden or not S.sprite_like(self._frame, det["cell"], p["y_offset"]):
                    self.stats["ignored_partial"] += 1
                    return None
                # someone already found but never seen whole (in tall grass), still there: a partial picture
                view_index = self._partial_view(npc, det, step)
                return self._begin_track(npc, det, segment, step, view_index)
            npc = self._static_at(segment, square, busy)
            if npc is not None:
                # a picture nobody has seen, on the square of someone who never moves: that person turned round
                view_index = self._turned(npc, det, step)
            else:
                look = self._new_look(det, step)
                candidates = [n.id for n in self._near(segment, square, busy)]
                if best is not None and best[0] <= p["maybe"]:
                    candidates += [n.id for n in self.live_npcs()
                                   if self.look_of(n).id == best[1] and n.id not in busy and n.id not in candidates]
                npc = self._new_npc(look, "card", candidates, step)
        return self._begin_track(npc, det, segment, step, view_index)

    def _begin_track(self, npc: NPC, det: Dict[str, Any], segment: int, step: int, view_index: int) -> Track:
        if npc.context is None and self._frame is not None:
            npc.context, npc.context_cell = np.array(self._frame, copy=True), det["cell"]
        self.stats["tracks_started"] += 1
        self.next_ids["track"] += 1
        track = Track(id=self.next_ids["track"], npc=npc.id, segment=segment, square=det["square"], view=view_index,
                      last_step=step)
        self._note_sighting(track, det, step)
        return track

    def _partial_view(self, npc: NPC, det: Dict[str, Any], step: int) -> int:
        """The partial view of a person never seen whole that this picture is (a new one if none is close). Partial
        views are never matched against anyone else (``_view_stack``), so they cannot join two people."""
        look = self.look_of(npc)
        partial = [(S.mask_distance(det["mask"], v.key)[0], i) for i, v in enumerate(look.views) if v.partial]
        if partial and min(partial)[0] <= self._p["auto"]:
            look.views[min(partial)[1]].n_seen += 1
            return min(partial)[1]
        look.views.append(View(det["window"].copy(), det["mask"].copy(), None, None, (self._episode, step),
                               partial=True))
        look.links.append((0, len(look.views) - 1, "partial"))
        self.stats["partial_views"] += 1
        return len(look.views) - 1

    def _turned(self, npc: NPC, det: Dict[str, Any], step: int) -> int:
        """A person who stands still, seen in a new picture on their square: add it as a view of their look (only a whole
        sprite is stored) and return the view to note the sighting with."""
        look = self.look_of(npc)
        last = self.last_seen(self._gmap, npc)[2]
        if not det["full"]:
            return last.view
        look.views.append(View(det["window"].copy(), det["mask"].copy(), None, None, (self._episode, step)))
        look.links.append((last.view, len(look.views) - 1, "same_square"))
        self._stack = None
        self.stats["views_added"] += 1
        self.stats["same_square"] += 1
        return len(look.views) - 1

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
        # The picture comparison allows a left/right mirror (the game draws "left" by flipping "right"), so a person
        # who turns that way is still the same picture: the turn is recorded here, per sighting, and a turn always
        # makes a new sighting.
        views = self.look_of(self.npcs[track.npc]).views
        flipped = bool(track.view < len(views) and S.mask_distance(det["mask"], views[track.view].key)[1])
        if (track.last_sight_square == det["square"] and step - track.last_sight_step < SIGHTING_EVERY
                and flipped == track.last_sight_flipped):
            return
        self.npcs[track.npc].sightings.append(Sighting(self._episode, step, track.segment, det["square"], track.view,
                                                       det["offset"], track.id, flipped))
        track.last_sight_step, track.last_sight_square, track.last_sight_flipped = step, det["square"], flipped
        self.stats["sightings"] += 1

    # -- conversations ----------------------------------------------------------------------------

    @staticmethod
    def _is_a_press(action: Any, kwargs: Dict[str, Any]) -> bool:
        from gameboy_worlds.emulation.emulator import LowLevelActions
        from gameboy_worlds.interface.action import LowLevelAction
        from gameboy_worlds.interface.pokemon.actions import InteractAction

        if isinstance(action, type) and issubclass(action, InteractAction):
            return True
        return (isinstance(action, type) and issubclass(action, LowLevelAction)
                and kwargs.get("low_level_action") == LowLevelActions.PRESS_BUTTON_A)

    def _conversation_step(self, ev: StepEvent) -> None:
        """Track which way the player faces, and open, fill and close conversations from the step stream."""
        from gameboy_worlds.emulation.pokemon import AgentState

        state = ev.info.get("pokemon_core", {}).get("agent_state") if ev.info else None
        if self._resume is not None:
            self._continue_conversation(ev)
        a_press = self._is_a_press(ev.action, ev.kwargs)
        # the text box can open a step after the A press that called it (seen on Crystal): that press still counts
        pressed = a_press or (self._a_step is not None and self._a_step == self._gmap.step - 1)
        self._a_step = self._gmap.step if a_press and ev.was_free and ev.free else None
        if ev.was_free and ev.free:
            if ev.direction is not None:
                self._facing = ev.direction
            elif ev.status in ("moved", "relocalised", "new_segment"):
                self._facing = None                              # moved some way we cannot read a direction from
        elif ev.status in ("relocalised", "new_segment"):
            self._facing = None
        # only text opens a conversation: a menu or a battle straight from free roam is not one (a trainer's
        # challenge starts with dialogue, so it still opens here and closes on the battle)
        if self._open is None and ev.was_free and state == AgentState.IN_DIALOGUE:
            self._open_conversation(ev, pressed)
        if self._open is not None:
            self._collect_crops(ev)
            if state == AgentState.FREE_ROAM:
                self._close_conversation("free_roam", True)
            elif state == AgentState.IN_BATTLE:
                self._close_conversation("battle", True)

    def _continue_conversation(self, ev: StepEvent) -> None:
        """The episode's first step: if the last episode ended in the middle of a conversation and this one opens in
        its text box (or a menu or battle it led to), it goes on: same speaker, more crops, read and summarised again
        once it ends."""
        conv = self.conversations.get(self._resume)
        self._resume = None
        g = self._gmap
        if (conv is None or self._open is not None or ev.was_free
                or g.resolve(g.segment, g.pos) != g.resolve(conv.segment, conv.square)):
            return
        conv.complete, conv.led_to, conv.map_step_end = False, None, None
        conv.text, conv.read, conv.summarised, conv.tries = "", False, False, 0
        self._open = conv.id
        self.stats["conversations_continued"] += 1

    def _sign_at(self, target: Cell) -> Optional[Cell]:
        g = self._gmap
        square = (g.pos[0] + target[0], g.pos[1] + target[1])
        cell = g.segments[g.segment].get(square)
        return square if cell is not None and set(cell.tags) & SIGN_TAGS else None

    def _open_conversation(self, ev: StepEvent, a_press: bool) -> None:
        g = self._gmap
        speaker: Optional[str] = None
        sign_square: Optional[Cell] = None
        on_screen = self.on_screen(g)
        neighbours = [n for c, n in on_screen.items() if abs(c[0]) + abs(c[1]) == 1]
        if a_press and self._facing is not None:
            target = DIRECTIONS[self._facing]
            speaker = on_screen.get(target)
            if speaker is None:
                sign_square = self._sign_at(target)
        elif len(set(neighbours)) == 1:                          # facing not known, or the person came to us
            speaker = neighbours[0]
        how, confidence = "seen", "high"
        if speaker is None and sign_square is None:              # not on this frame: use what the registry remembers
            if a_press and self._facing is not None:
                # only the square in front counts: an item or a sign beside a person is not that person
                speaker, how = self._remembered_at(DIRECTIONS[self._facing]), "remembered"
            elif not a_press:
                speaker, how, confidence = self._nearest_person(), "nearest", "low"
            if speaker is not None:
                # people move: memory of someone seen on several squares is weak, and gone once they roamed
                squares = self.squares_in(g, self.get(speaker), g.resolve(g.segment, g.pos)[0])
                if _spread(squares) > 1:
                    speaker = None
                elif len(squares) > 1:
                    confidence = "low"
            elif a_press and self._facing is not None:
                # nobody known on the square in front, but something like a person was: someone never seen whole
                speaker = self._hidden_at(DIRECTIONS[self._facing])
                if speaker is not None:
                    how, confidence = "hidden", "low"
        if speaker is None and sign_square is None:
            how = "none"
        conv = Conversation(
            id=self._next("conv"), npc=speaker, sign=None, episode=self._episode, map_step_start=g.step,
            map_step_end=None, segment=g.segment, square=g.pos, target=sign_square,
            trigger="a_press" if a_press else "npc_started", how=how, confidence=confidence)
        if speaker is not None:
            npc = self.get(speaker)
            conv.npc = npc.id
            npc.conversations.append(conv.id)
            seen = self.last_seen(g, npc)
            conv.target = seen[1] if seen else None
        elif sign_square is not None:
            sign = self._sign_for(g.segment, sign_square)
            conv.sign = sign.id
            sign.conversations.append(conv.id)
        else:
            self.unattributed.append(conv.id)
        if self._free_frame is not None:
            conv.context = np.array(self._free_frame, copy=True)
        self.conversations[conv.id] = conv
        self._open = conv.id
        self.stats["conversations"] += 1

    def _people_nearby(self) -> List[Tuple[int, Cell, NPC]]:
        """Every known person last seen in the player's area, as ``(distance, offset from the player, npc)``."""
        g = self._gmap
        segment, here = g.resolve(g.segment, g.pos)
        out = []
        for npc in self.live_npcs():
            if self.look_of(npc).kind_hint == "object":
                continue
            seen = self.last_seen(g, npc)
            if seen is not None and seen[0] == segment:
                offset = (seen[1][0] - here[0], seen[1][1] - here[1])
                out.append((abs(offset[0]) + abs(offset[1]), offset, npc))
        return out

    def _remembered_at(self, offset: Cell) -> Optional[str]:
        for _, where, npc in self._people_nearby():
            if where == offset:
                return npc.id
        return None

    def _nearest_person(self) -> Optional[str]:
        """Who spoke when someone started talking and nobody was seen next to the player on this frame: the one known
        person within 5 squares, a guess. Two or more within reach: nobody (we cannot tell which)."""
        people = [p for p in self._people_nearby() if 0 < p[0] <= 5]
        return people[0][2].id if len(people) == 1 else None

    def _sign_for(self, segment: int, square: Cell) -> Sign:
        g = self._gmap
        here = g.resolve(segment, square)
        for sign in self.signs.values():
            if g.resolve(sign.segment, sign.square) == here:
                return sign
        sign = Sign(id=self._next("sign"), segment=segment, square=square)
        self.signs[sign.id] = sign
        return sign

    def _collect_crops(self, ev: StepEvent) -> None:
        from execution.pokemon.executors import drop_prefix_crops

        conv = self.conversations[self._open]
        new = []
        for region in (ev.info.get("ocr") or {}).get("transition_ocr_regions") or []:
            stack = region.get("dialogue") if isinstance(region, dict) else None
            if stack is not None and len(stack):
                new.extend(np.asarray(c) for c in stack)
        if new:
            drop_prefix_crops(new, conv.crops)

    def _close_conversation(self, led_to: str, complete: bool) -> None:
        conv = self.conversations[self._open]
        conv.led_to, conv.complete, conv.map_step_end = led_to, complete, self._gmap.step
        self.interrupted = conv.id if led_to == "episode_end" else None
        if led_to == "battle" and conv.npc in self.npcs and self.get(conv.npc).kind == "unknown":
            self.get(conv.npc).kind = "trainer"
        if led_to == "battle" and conv.npc is None and conv.sign is None:
            # nobody known spoke and a battle followed straight away ("Wild X appeared!", or a trainer calling from
            # afar): the start of a battle, not a conversation, so it is never read, summarised or counted
            conv.trigger, conv.read, conv.summarised = "battle_start", True, True
            if conv.id in self.unattributed:
                self.unattributed.remove(conv.id)
            self.stats["battle_starts"] += 1
            self.stats["conversations"] -= 1
        self._open = None
        if conv.trigger != "battle_start":
            self._summarise_now(conv)

    def attach_model(self, vlm_caller: Optional[Callable[[str], Callable[..., Any]]]) -> None:
        """Let the registry call the model while a run goes on: each conversation is read and summarised the moment
        it ends. ``vlm_caller(stage)`` gives the caller for ``npc_ocr`` and ``npc_summary``. Not saved."""
        self._live = vlm_caller

    def _summarise_now(self, conv: Conversation) -> None:
        """Read (if needed) and summarise a conversation that just ended. Guarded: anything left undone is picked up
        by the end-of-episode steps."""
        if self._live is None or self._gmap is None:
            return
        try:
            if not conv.text and conv.crops and not conv.read:
                self.read_conversations(self._live("npc_ocr"), only=[conv])
            if conv.text.strip() and not conv.summarised:
                self.run_summaries(self._live("npc_summary"), self._gmap, only=[conv])
                if conv.npc in self.npcs:
                    self.apply_real_names()
        except Exception as e:
            log_warn(f"NPC registry: summarising {conv.id} on the spot failed ({type(e).__name__}: {e}); "
                     f"it is retried at the end of the episode")

    def finish_conversations(self) -> None:
        """An episode ends: a conversation still open is closed as interrupted (its text still counts)."""
        if self._open is not None and self._gmap is not None:
            self._close_conversation("episode_end", False)

    def read_conversations(self, vlm_call: Any, only: Optional[List[Conversation]] = None) -> None:
        """Ask the model to read the dialogue pictures of every conversation not read yet (one batched call)."""
        todo = [c for c in (only if only is not None else self.conversations.values())
                if c.crops and not c.text and not c.read]
        if not todo:
            return
        images = [[crop] for c in todo for crop in c.crops[:MAX_OCR_CROPS]]
        out = vlm_call(texts=[OCR_PROMPT] * len(images), images=images, max_new_tokens=4000)
        out = out if isinstance(out, list) else [out]
        at = 0
        for c in todo:
            n = len(c.crops[:MAX_OCR_CROPS])
            pieces = []
            for text in out[at:at + n]:
                text = text.rsplit("Response:", 1)[1] if "Response:" in text else text
                text = text.strip()
                for lead in ("The text in the image is:", "The text in this image is:"):
                    if text.startswith(lead):
                        text = text[len(lead):].strip()
                if text and text.lower() != "none":
                    pieces.append(text)
            at += n
            if not pieces and only is not None:
                continue                                         # nothing came back: try again at the episode end
            c.text, c.read = "\n".join(ordered_merge(pieces)), True
            self.stats["conversations_read"] += 1

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
    def end_episode(self, gmap: GlobalMap, vlm_caller: Optional[Callable[[str], Callable[..., Any]]] = None) -> None:
        """Close the episode's bookkeeping; decorated so the registry is saved every episode.

        ``vlm_caller(stage)`` gives the model caller for a stage (``npc_card``, ``npc_judge``, ``npc_confirm``,
        ``npc_summary``); without it only the code-only steps run. Each model step is guarded: a failure is logged and
        the run goes on.
        """
        self._gmap = gmap
        self.finish_conversations()
        self.consolidate()
        if vlm_caller is not None:
            self._model_steps(vlm_caller, gmap)
        self._tracks = {}
        self._forget_screen()

    def _model_steps(self, vlm_caller: Callable[[str], Callable[..., Any]], gmap: GlobalMap) -> None:
        for label, step in (("cards and judges", lambda: self.run_model(vlm_caller("npc_card"),
                                                                       vlm_caller("npc_judge"), gmap)),
                            ("hidden people", lambda: npc_model.run_confirm(self, vlm_caller("npc_confirm"), self.game,
                                                                            gmap, self._p["y_offset"])),
                            ("summaries", lambda: self.run_summaries(vlm_caller("npc_summary"), gmap)),
                            ("real names", self.apply_real_names)):
            try:
                step()
            except Exception as e:
                log_warn(f"NPC registry: {label} skipped: {type(e).__name__}: {e}")

    def run_model(self, vlm_card: Callable[..., Any], vlm_judge: Callable[..., Any], gmap: GlobalMap) -> None:
        npc_model.run_model(self, vlm_card, vlm_judge, self.game, gmap)

    def run_summaries(self, vlm_call: Callable[..., Any], gmap: GlobalMap,
                      only: Optional[List[Conversation]] = None) -> None:
        npc_model.run_summaries(self, vlm_call, self.game, gmap, only=only)

    @staticmethod
    def sighting_facing(look: Look, sighting: Sighting) -> Optional[str]:
        """Which way the person faced at a sighting: the view's facing (once a model said so), turned round if the
        sighting is the mirror image of the view and the view faces left or right."""
        facing = look.views[sighting.view].facing if sighting.view < len(look.views) else None
        if sighting.flipped and facing in ("left", "right"):
            return "left" if facing == "right" else "right"
        return facing

    def visits(self, npc: NPC) -> List[Tuple[int, Conversation, Optional[int]]]:
        """``(visit number, conversation, number of the earlier visit with the same words or None)`` for each time
        the player talked to ``npc``, oldest first."""
        talks = sorted((self.conversations[c] for c in npc.conversations if c in self.conversations),
                       key=lambda c: (c.episode, c.map_step_start))
        seen: Dict[str, int] = {}
        out = []
        for k, c in enumerate(talks, 1):
            words = npc_model._same_text(c.text)
            out.append((k, c, seen.get(words) if words else None))
            if words:
                seen.setdefault(words, k)
        return out

    def apply_real_names(self) -> None:
        npc_model.apply_real_names(self)

    # -- reading ----------------------------------------------------------------------------------

    def people_here(self, gmap: GlobalMap) -> List[Tuple[NPC, Cell]]:
        """People whose last sighting is in the player's current (resolved) segment, with that square."""
        found = []
        here = gmap.resolve(gmap.segment, gmap.pos)[0]
        for npc in self.live_npcs():
            seen = self.last_seen(gmap, npc)
            if seen is not None and seen[0] == here:
                found.append((npc, seen[1]))
        return found

    def when_seen(self, gmap: GlobalMap, npc: NPC, square: Cell) -> str:
        """How fresh the registry's place for a person is. People move, so a remembered place is memory, not truth:
        it carries its age, and a square on screen right now without the person says they may have moved."""
        seen = self.last_seen(gmap, npc)
        if npc.id in self.on_screen(gmap).values():
            return "on screen now"
        steps = gmap.step - seen[2].map_step
        if seen[2].episode == self._episode:
            when = f"last seen {steps} step(s) ago"
        else:
            when = f"last seen {self._episode - seen[2].episode} episode(s) ago"
        squares = self.squares_in(gmap, npc, seen[0])
        if len(squares) > 1:
            when += f", seen on {len(squares)} different squares (walks about)"
        if self.in_view(gmap, square):
            when += ("; not recognised on that square, which is on screen now: may have moved" if self.seen_whole(npc)
                     else "; only ever seen in part (hidden, for example in tall grass)")
        return when

    def people_block(self, gmap: GlobalMap, max_people: int = 8, max_chars: int = 1800, where: str = "here") -> str:
        """The block the strategist reads before writing a task: who has been seen in the area it is in, where they
        are from the player, and whether it has already talked to them (with what they said). Empty if nobody.
        ``where``: "here" (the screen now was placed), else a key of ``npc_map.WHERE_NOTES`` (told as remembered;
        "away" without positions)."""
        here = [(npc, square) for npc, square in self.people_here(gmap)
                if self._episode - self.last_seen(gmap, npc)[2].episode <= STALE_EPISODES
                and self.look_of(npc).kind_hint != "object"]
        maybe = [] if where == "away" else self.unconfirmed(gmap, gmap.resolve(gmap.segment, gmap.pos)[0])
        if not here and not maybe:
            return ""
        px, py = gmap.pos
        here.sort(key=lambda r: abs(r[1][0] - px) + abs(r[1][1] - py))
        lines = []
        for npc, square in here[:max_people]:
            dx, dy = square[0] - px, square[1] - py
            when = self.when_seen(gmap, npc, square)
            talks = [self.conversations[c] for c in npc.conversations
                     if c in self.conversations and (self.conversations[c].text or self.conversations[c].crops)
                     and gmap.resolve(self.conversations[c].segment, self.conversations[c].square)[0] == gmap.segment]
            if talks:
                last = talks[-1]
                heard = last.summary or last.text[:160].strip().replace("\n", " ")
                said = f": \"{heard}\"" if heard else ""
                status = f"talked to here (episode {last.episode}){said}"
            else:
                status = "not talked to yet"
            what = ", ".join(x for x in (npc.kind if npc.kind != "unknown" else "", npc.description[:70]) if x)
            at = "" if where == "away" else f" at ({dx}, {dy})"
            lines.append(f'- "{display_name(npc)}"{f" ({what})" if what else ""}{at}, {when}: {status}.')
        if len(here) > max_people:
            lines.append(f"...and {len(here) - max_people} more.")
        lines += maybe_lines(gmap, maybe)
        note = WHERE_NOTES[where] + "\n" if where in WHERE_NOTES else ""
        block = ("PEOPLE YOU HAVE SEEN IN THIS AREA (positions are steps from you: x right, y up). The quoted names are "
                 "labels for this list (unless one is a real name the person gave): the player does not know the labels, "
                 "so describe a person by where they stand, not by name, and never write a label into a task. Talking "
                 "to someone you have not talked to yet can give information or items.\n" + note
                 + "\n".join(lines))
        return "\n\n" + block[:max_chars]

    def talked_in(self, gmap: GlobalMap, npc: NPC) -> Set[int]:
        """The (resolved) segments where the player heard this person say something."""
        found = set()
        for conversation_id in npc.conversations:
            c = self.conversations.get(conversation_id)
            if c is not None and (c.text or c.crops):
                found.add(gmap.resolve(c.segment, c.square)[0])
        return found

    def _talks_in(self, gmap: GlobalMap, npc: NPC, segment: int) -> List[Conversation]:
        return [self.conversations[c] for c in npc.conversations
                if c in self.conversations and (self.conversations[c].text or self.conversations[c].crops)
                and gmap.resolve(self.conversations[c].segment, self.conversations[c].square)[0] == segment]

    def screen_labels(self, gmap: GlobalMap) -> Dict[Cell, str]:
        """What to say about each person on screen right now, keyed by the square relative to the player, for the
        screen text the strategist and the supervisor read: the quoted name, the kind, and whether the player has
        talked to them here. Someone who never moves, remembered on a square that is on screen but not recognised
        on this frame (a seated person the detector half misses), is added as remembered."""
        out: Dict[Cell, str] = {}
        people = dict(self.on_screen(gmap))
        remembered = set()
        segment = gmap.resolve(gmap.segment, gmap.pos)[0]
        for mark in self.map_people(gmap, segment):
            cell = (mark.square[0] - gmap.pos[0], mark.square[1] - gmap.pos[1])
            if (not mark.on_screen and self.in_view(gmap, mark.square) and not mark.earlier and cell not in people
                    and self._episode - mark.last_seen[0] <= STALE_MARK_EPISODES):
                people[cell] = mark.npc_id
                remembered.add(cell)
        for cell, npc_id in people.items():
            npc = self.get(npc_id)
            if self.look_of(npc).kind_hint == "object":
                continue
            talks = self._talks_in(gmap, npc, gmap.segment)
            if talks:
                heard = (talks[-1].summary or talks[-1].text)[:50].strip().replace("\n", " ")
                status = f'talked to here in episode {talks[-1].episode}' + (f': "{heard}"' if heard else "")
            else:
                status = "not talked to here"
                elsewhere = [seg for seg in self.talked_in(gmap, npc) if seg != gmap.segment]
                if elsewhere:
                    status += "; met before in another area"
            if cell in remembered:
                status = ("remembered here, not recognised on this frame: may have moved; " if self.seen_whole(npc)
                          else "seen only in part (hidden, for example in tall grass); ") + status
            kind = npc.kind if npc.kind != "unknown" else "person"
            name = (f'who gave the name "{npc.real_name}"' if npc.real_name
                    else f'labelled "{npc.name}" (the label is {LABEL_NOTE})')
            out[cell] = f'{kind} {name} ({status})'[:320]
        return out

    def map_people(self, gmap: GlobalMap, segment: int, upto_step: Optional[int] = None) -> List[PersonMark]:
        """The people layer of one area, the counterpart of ``GlobalMap.door_list``: everyone last seen in
        ``segment`` (as of map step ``upto_step``, or now), by number."""
        upto = float("inf") if upto_step is None else upto_step
        live = [n for n in self.live_npcs() if self.look_of(n).kind_hint != "object"]
        now = self._episode if upto_step is None else max(
            (s.episode for n in live for s in n.sightings if s.map_step <= upto), default=0)
        on_screen = set(self.on_screen(gmap).values()) if upto_step is None else set()
        marks = []
        for npc in sorted(live, key=lambda n: n.id):
            seen = sorted((s for s in npc.sightings if s.map_step <= upto), key=lambda s: -s.map_step)
            if not seen or self.place(gmap, seen[0])[0] != segment:
                continue
            squares: List[Cell] = []
            for sighting in seen:
                where = self.place(gmap, sighting)
                if where[0] == segment and where[1] not in squares:
                    squares.append(where[1])
            square, newest = squares[0], seen[0]
            views = self.look_of(npc).views
            talks = [c for c in self._talks_in(gmap, npc, segment) if c.map_step_start <= upto]
            sure = [c for c in talks if c.how in ("seen", "remembered") and getattr(c, "confidence", "high") == "high"]
            words = {npc_model._same_text(c.text) for c in talks}
            stale = now - newest.episode > STALE_MARK_EPISODES
            if upto_step is None:
                when = self.when_seen(gmap, npc, square)
                stale = stale or (npc.id not in on_screen and self.in_view(gmap, square) and self.seen_whole(npc))
            else:
                when = f"last seen in episode {newest.episode}"
            xs, ys = [q[0] for q in squares], [q[1] for q in squares]
            marks.append(PersonMark(
                npc_id=npc.id, number=int(npc.id.split("_")[1]), name=display_name(npc), kind=npc.kind,
                square=square, facing=self.sighting_facing(self.look_of(npc), newest),
                last_seen=(newest.episode, newest.map_step), on_screen=npc.id in on_screen,
                talked="yes" if sure else "probably" if talks else "no", times_talked=len(talks),
                same_words=len(talks) > 1 and len(words) == 1 and "" not in words, earlier=squares[1:4],
                spread=(min(xs), min(ys), max(xs), max(ys)) if len(squares) >= 3 else None, stale=stale, when=when))
        return marks

    def unconfirmed(self, gmap: GlobalMap, segment: int) -> List[Cell]:
        """Squares the map holds as a person (an ``occupied`` square the tile table tagged as a person) where no
        registry person was ever seen, and not only ever drawn like background: someone cut off by the screen edge
        or hidden, not yet seen whole."""
        background = {square for seg, square in self.background_squares(gmap) if seg == segment}
        known = []
        for npc in self.live_npcs():
            for sighting in npc.sightings:
                seg, square = self.place(gmap, sighting)
                if seg == segment:
                    known.append(square)
        out = []
        for square, cell in gmap.segments[segment].items():
            if (cell.kind == OCCUPIED and "npc" in cell.tags and square not in background
                    and all(_manhattan(square, k) > 1 for k in known)):
                out.append(square)
        return sorted(out)

    def digest_lines(self, episode: int, gmap: GlobalMap) -> str:
        """The conversations of one episode, to add to the digest the knowledge extraction reads."""
        lines = []
        for c in self.conversations.values():
            if c.episode != episode or not c.text.strip():
                continue
            who = (f'"{display_name(self.get(c.npc))}"' if c.npc in self.npcs else "a sign" if c.sign else "someone")
            lines.append(f'- {who} in {npc_model._area(self, gmap, gmap.resolve(c.segment, c.square)[0])}: '
                         f'"{c.text[:400].strip()}"')
        if not lines:
            return ""
        return ("\n\nConversations this episode (the quoted names are labels for a list, not names the game knows: "
                "to say who spoke, describe the person and where they stood):\n" + "\n".join(lines))

    def markers(self, gmap: GlobalMap, segment: int) -> List[Tuple[Cell, str, bool]]:
        """``(square, number, talked_to_here)`` for every person last seen in ``segment`` (see ``map_people``)."""
        return [(m.square, str(m.number), m.talked != "no") for m in self.map_people(gmap, segment)]

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


def ordered_merge(pieces: List[str]) -> List[str]:
    """Join the text of successive dialogue boxes into lines, in the order the boxes were on screen. A box that only
    repeats lines already there (the same box, or the box before it with its last line scrolled up) adds nothing new;
    the ▼ "more text" arrow is not text."""
    lines: List[str] = []
    for piece in pieces:
        new = [ln.replace("▼", "").replace("▶", "").strip() for ln in piece.splitlines()]
        new = [ln for ln in new if ln]
        for k in range(min(len(lines), len(new)), 0, -1):        # the box before, scrolled up by k lines
            if lines[-k:] == new[:k]:
                new = new[k:]
                break
        if new and not any(lines[i:i + len(new)] == new for i in range(len(lines) - len(new) + 1)):
            lines.extend(new)
    return lines


def _spread(squares: List[Cell]) -> int:
    """How far apart the squares a person was seen on are (0 for someone who never moved)."""
    return max((_manhattan(a, b) for a in squares for b in squares), default=0)


def _pairs(items: List[str]) -> List[Tuple[str, str]]:
    return [(a, b) for i, a in enumerate(items) for b in items[i + 1:] if a != b]
