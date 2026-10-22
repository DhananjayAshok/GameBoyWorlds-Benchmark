"""
The world as the strategist has mapped it: places, what they look like, and how they connect.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from execution.artifact import StrategistArtifact, update
from execution.report import EnvironmentStepRecord, SupervisorReport
from execution.strategist.pokemon import prompts as P
from execution.strategist.pokemon.asking import Parser, ask, single_value
from utils import log_error, log_info, log_warn
from utils.lm_inference import clean_value, parse_key_value, parse_yes_no

Place = Tuple[str, ...]
VLMCaller = Callable[[str], Callable[..., str]]

#: A pixel is black at or below this brightness, and a frame is black when at least BLACK_FRACTION of its
#: pixels are. Not "every pixel": the frame a door warp leaves behind is black except for a few grey lines
#: and sprites, which a strict test counts as a picture of a place.
BLACK_PIXEL_LEVEL = 10
BLACK_FRACTION = 0.95
BLACK_MERGE_GAP = 10
#: Positions in the frame list between the two context frames taken on each side of a black run. The
#: list holds 2 captures per button press, so 6 is about 3 presses: far enough apart to show the place,
#: close enough that the player is still in it.
CONTEXT_GAP = 6
#: Frames kept from one episode to the next, so a transition that straddles an episode boundary is whole.
TAIL_MAX_FRAMES = 120
#: Most no-button waits ``settle_screen`` makes while the screen is black.
MAX_SETTLE_STEPS = 8
LOCATION_ATTEMPTS = 3
MAX_IDENTITY_IMAGES = 5
PLACE_KINDS = ("major", "bridge", "internal", "unknown")


@dataclass
class VisualIdentity:
    common_tiles: Dict[str, int] = field(default_factory=dict)
    images: List[np.ndarray] = field(default_factory=list)

    def count_tiles(self, tile_hashes) -> None:
        for tile_hash in tile_hashes:
            if tile_hash is None:
                continue
            self.common_tiles[tile_hash] = self.common_tiles.get(tile_hash, 0) + 1

    def add_image(self, image: np.ndarray) -> None:
        self.images.append(image)

    def most_common(self, limit: int = 10) -> List[tuple]:
        return sorted(self.common_tiles.items(), key=lambda item: item[1], reverse=True)[:limit]


@dataclass
class InternalTransition:
    major_location: str
    node_source: str
    node_destination: str
    description: Optional[str] = None
    two_way: bool = False
    other_notes: Optional[str] = None


@dataclass
class LocationTransition:
    node_source_major: str
    node_destination_major: str
    node_source_internal: Optional[str] = None
    node_destination_internal: Optional[str] = None
    description: Optional[str] = None
    two_way: bool = False
    other_notes: Optional[str] = None


@dataclass
class InternalLocationGraph:
    transitions: List[InternalTransition] = field(default_factory=list)

    def add(self, transition: InternalTransition) -> InternalTransition:
        self.transitions.append(transition)
        return transition

    def out_of(self, node: str) -> List[InternalTransition]:
        return [t for t in self.transitions
                if t.node_source == node or (t.two_way and t.node_destination == node)]

    def neighbours(self, node: str) -> List[str]:
        out = []
        for transition in self.out_of(node):
            other = transition.node_destination if transition.node_source == node else transition.node_source
            if other not in out:
                out.append(other)
        return out


@dataclass
class MajorLocationGraph:
    transitions: List[LocationTransition] = field(default_factory=list)

    def add(self, transition: LocationTransition) -> LocationTransition:
        self.transitions.append(transition)
        return transition

    def out_of(self, major: str) -> List[LocationTransition]:
        return [t for t in self.transitions
                if t.node_source_major == major or (t.two_way and t.node_destination_major == major)]

    def neighbours(self, major: str) -> List[str]:
        out = []
        for transition in self.out_of(major):
            other = (transition.node_destination_major if transition.node_source_major == major
                     else transition.node_source_major)
            if other not in out:
                out.append(other)
        return out


@dataclass
class MajorLocation:
    name: str
    image_on_map: Optional[np.ndarray] = None
    visual_identity: VisualIdentity = field(default_factory=VisualIdentity)


@dataclass
class BridgeLocation:
    name: str
    accessible_through: List[str] = field(default_factory=list)
    visual_identity: VisualIdentity = field(default_factory=VisualIdentity)


@dataclass
class InternalLocation:
    name: str
    major_location: str
    visual_identity: VisualIdentity = field(default_factory=VisualIdentity)


@dataclass
class LocationStore(StrategistArtifact):
    majors: Dict[str, MajorLocation] = field(default_factory=dict)
    bridges: Dict[str, BridgeLocation] = field(default_factory=dict)
    internals: Dict[str, Dict[str, InternalLocation]] = field(default_factory=dict)
    graph: MajorLocationGraph = field(default_factory=MajorLocationGraph)
    internal_graphs: Dict[str, InternalLocationGraph] = field(default_factory=dict)
    current: Optional[Place] = None

    # -- reading ------------------------------------------------------------

    def node(self, name: str) -> Optional[Union[MajorLocation, BridgeLocation]]:
        return self.majors.get(name) or self.bridges.get(name)

    def location(self, place: Place) -> Optional[Union[MajorLocation, BridgeLocation, InternalLocation]]:
        return self.internal(*place) if len(place) == 2 else self.node(place[0])

    def describe_known(self) -> str:
        lines = []
        if self.majors:
            lines.append("Major: " + ", ".join(self.majors))
        if self.bridges:
            lines.append("Bridge: " + ", ".join(
                f"{name} (connects {', '.join(bridge.accessible_through)})" if bridge.accessible_through else name
                for name, bridge in self.bridges.items()))
        for major, rooms in self.internals.items():
            if rooms:
                lines.append(f"Internal, inside {major}: " + ", ".join(rooms))
        return "\n".join(lines) or "(none yet)"

    def connected(self, place: Place) -> List[str]:
        out = list(self.graph.neighbours(place[0]))
        for name in self.internal_graph(place[0]).neighbours(place[-1]):
            text = name if name == place[0] else f"{name} (inside {place[0]})"
            if text not in out:
                out.append(text)
        return out

    def internal(self, major_location: str, name: str) -> Optional[InternalLocation]:
        return self.internals.get(major_location, {}).get(name)

    def internal_graph(self, major_location: str) -> InternalLocationGraph:
        return self.internal_graphs.get(major_location) or InternalLocationGraph()

    def names(self) -> List[str]:
        return list(self.majors) + list(self.bridges)

    def identities(self) -> Dict[Tuple[str, ...], VisualIdentity]:
        out: Dict[Tuple[str, ...], VisualIdentity] = {}
        for name, location in {**self.majors, **self.bridges}.items():
            out[(name,)] = location.visual_identity
        for major, rooms in self.internals.items():
            for name, room in rooms.items():
                out[(major, name)] = room.visual_identity
        return out

    # -- updating -----------------------------------------------------------

    @update
    def add_major(self, name: str, image_on_map: Optional[np.ndarray] = None) -> MajorLocation:
        if self.node(name) is not None:
            log_error(f"Location {name!r} already exists.")
        self.majors[name] = MajorLocation(name=name, image_on_map=image_on_map)
        return self.majors[name]

    @update
    def add_bridge(self, name: str, accessible_through: Sequence[str] = ()) -> BridgeLocation:
        if self.node(name) is not None:
            log_error(f"Location {name!r} already exists.")
        self.bridges[name] = BridgeLocation(name=name, accessible_through=list(accessible_through))
        return self.bridges[name]

    @update
    def add_internal(self, name: str, major_location: str) -> InternalLocation:
        if self.internal(major_location, name) is not None:
            log_error(f"Internal location {name!r} already exists in {major_location!r}.")
        room = InternalLocation(name=name, major_location=major_location)
        self.internals.setdefault(major_location, {})[name] = room
        return room

    @update
    def add_bridge_access(self, name: str, major_location: str) -> None:
        bridge = self.bridges.get(name)
        if bridge is None:
            log_error(f"No bridge named {name!r}.")
        if major_location not in bridge.accessible_through:
            bridge.accessible_through.append(major_location)

    @update
    def set_map_image(self, name: str, image: np.ndarray) -> None:
        major = self.majors.get(name)
        if major is None:
            log_error(f"No major location named {name!r}.")
        major.image_on_map = image

    @update
    def set_current(self, place: Optional[Place]) -> None:
        self.current = place

    @update
    def add_transition(self, transition: LocationTransition) -> LocationTransition:
        return self.graph.add(transition)

    @update
    def add_internal_transition(self, transition: InternalTransition) -> InternalTransition:
        graph = self.internal_graphs.setdefault(transition.major_location, InternalLocationGraph())
        return graph.add(transition)

    @update
    def record_visual(self, name: str, major_location: Optional[str] = None,
                      tile_hashes: Iterable[Optional[str]] = (), image: Optional[np.ndarray] = None) -> None:
        location = self.internal(major_location, name) if major_location else self.node(name)
        if location is None:
            where = f" in {major_location!r}" if major_location else ""
            log_error(f"No location named {name!r}{where}.")
        location.visual_identity.count_tiles(tile_hashes)
        if image is not None:
            location.visual_identity.add_image(image)

    # -- diffing ------------------------------------------------------------

    def diff(self, old: "LocationStore") -> Dict[str, Any]:
        added_internals = {}
        for major, rooms in self.internals.items():
            old_rooms = old.internals.get(major, {})
            new_rooms = [name for name in rooms if name not in old_rooms]
            if new_rooms:
                added_internals[major] = new_rooms

        added_internal_transitions = {}
        for major, graph in self.internal_graphs.items():
            old_transitions = old.internal_graphs[major].transitions if major in old.internal_graphs else []
            new_transitions = [t for t in graph.transitions if t not in old_transitions]
            if new_transitions:
                added_internal_transitions[major] = new_transitions

        added_bridge_access = {}
        for name, bridge in self.bridges.items():
            before = old.bridges[name].accessible_through if name in old.bridges else []
            added = [major for major in bridge.accessible_through if major not in before]
            if added and name in old.bridges:
                added_bridge_access[name] = added

        map_images_set = []
        for name, major in self.majors.items():
            before = old.majors[name].image_on_map if name in old.majors else None
            if major.image_on_map is not None and (before is None or not np.array_equal(before, major.image_on_map)):
                map_images_set.append(name)

        visual_changed = {}
        old_identities = old.identities()
        for key, identity in self.identities().items():
            before = old_identities.get(key, VisualIdentity())
            images_added = len(identity.images) - len(before.images)
            tiles_counted = sum(identity.common_tiles.values()) - sum(before.common_tiles.values())
            if images_added or tiles_counted:
                visual_changed[key] = {"images_added": images_added, "tiles_counted": tiles_counted}

        return {
            "added_majors": [name for name in self.majors if name not in old.majors],
            "added_bridges": [name for name in self.bridges if name not in old.bridges],
            "added_internals": added_internals,
            "added_bridge_access": added_bridge_access,
            "map_images_set": map_images_set,
            "added_transitions": [t for t in self.graph.transitions if t not in old.graph.transitions],
            "added_internal_transitions": added_internal_transitions,
            "visual_changed": visual_changed,
            "current_changed": (old.current, self.current) if old.current != self.current else None,
        }


def place_text(place: Optional[Place]) -> str:
    if place is None:
        return "unknown"
    return f"{place[1]} (inside {place[0]})" if len(place) == 2 else place[0]


def _canonical(name: str, names: Iterable[str]) -> str:
    return next((known for known in names if known.strip().lower() == name.strip().lower()), name)


def _parse_place(store: LocationStore) -> Parser:
    def parse(output: str) -> Tuple[Any, Optional[str]]:
        kind = (parse_key_value(output, "Kind") or "").strip().strip("\"'").strip().lower()
        if kind not in PLACE_KINDS:
            return None, "The `Kind:` line must be Major, Bridge, Internal or Unknown."
        how = clean_value(parse_key_value(output, "How"))
        two_way = bool(parse_yes_no(output, "Two-way"))
        if kind == "unknown":
            return (kind, None, how, two_way), None
        name = clean_value(parse_key_value(output, "Name"))
        if name is None:
            return None, "The `Name:` line was missing or empty. Name the place, or write Kind: Unknown."
        if kind == "internal":
            parent = clean_value(parse_key_value(output, "Parent"))
            if parent is None:
                return None, "An Internal place needs the major location it is inside on the `Parent:` line."
            if _canonical(parent, store.bridges) in store.bridges:
                return None, f"{parent!r} is a bridge, not a major location. The parent must be a major location."
            parent = _canonical(parent, store.majors)
            name = _canonical(name, store.internals.get(parent, {}))
            if name.strip().lower() == parent.strip().lower():
                return None, "The room's name must differ from the major location it is inside."
            return (kind, (parent, name), how, two_way), None
        own, other = (store.majors, store.bridges) if kind == "major" else (store.bridges, store.majors)
        if _canonical(name, other) in other:
            other_kind = "bridge" if kind == "major" else "major location"
            return None, f"{name!r} is already known as a {other_kind}. Use that kind, or a different name."
        return (kind, (_canonical(name, own),), how, two_way), None
    return parse


def _parse_transition(output: str) -> Tuple[Any, Optional[str]]:
    verdict = parse_yes_no(output, "Transition")
    if verdict is None:
        return None, "The `Transition:` line must be Yes or No."
    return verdict, None


def _parse_verify(store: LocationStore) -> Parser:
    place_parser = _parse_place(store)

    def parse(output: str) -> Tuple[Any, Optional[str]]:
        same = (parse_key_value(output, "Same") or "").replace("*", "").strip().lower().rstrip(".")
        if same.startswith("yes"):
            return "same", None
        if same.startswith("unknown"):
            return "unknown", None
        if not same.startswith("no"):
            return None, "The `Same:` line must be Yes, No or Unknown."
        answer, problem = place_parser(output)
        if problem is not None:
            return None, problem
        kind, place, _, _ = answer
        if kind == "unknown":
            return "unknown", None
        return (kind, place), None
    return parse


def _enter(store: LocationStore, kind: str, place: Place, frame: np.ndarray, identified: Dict[Any, Any],
           episode_number: int) -> None:
    if kind == "major" and place[0] not in store.majors:
        store.add_major(place[0], episode_number=episode_number)
    if kind == "bridge" and place[0] not in store.bridges:
        store.add_bridge(place[0], episode_number=episode_number)
    if kind == "internal":
        if place[0] not in store.majors:
            store.add_major(place[0], episode_number=episode_number)
        if store.internal(*place) is None:
            store.add_internal(place[1], place[0], episode_number=episode_number)
    keep_image = len(store.location(place).visual_identity.images) < MAX_IDENTITY_IMAGES
    store.record_visual(place[-1], major_location=place[0] if len(place) == 2 else None,
                        tile_hashes=[tile.hash for tile in identified.values()],
                        image=frame if keep_image else None, episode_number=episode_number)
    store.set_current(place, episode_number=episode_number)


def locate(store: LocationStore, frame: np.ndarray, screen_tiles: str, identified: Dict[Any, Any],
           vlm_caller: VLMCaller, game: str, episode_number: int, parameters: Optional[dict] = None) -> Optional[Place]:
    prompt = (P.LOCATE_PROMPT.replace("[GAME]", game)
              .replace("[SCREEN_TILES]", screen_tiles or "(nothing was recognised)")
              .replace("[KNOWN]", store.describe_known()))
    answer = ask(vlm_caller("locate"), prompt, _parse_place(store), "Locate", LOCATION_ATTEMPTS, parameters,
                 images=[frame])
    if answer is None or answer[0] == "unknown":
        return None
    kind, place, _, _ = answer
    _enter(store, kind, place, frame, identified, episode_number)
    log_info(f"Located the player in {place_text(place)}.", parameters)
    return place


def verify_location(store: LocationStore, frame: np.ndarray, screen_tiles: str, identified: Dict[Any, Any],
                    vlm_caller: VLMCaller, game: str, episode_number: int, candidate: Optional[Place],
                    parameters: Optional[dict] = None) -> Optional[Place]:
    """Ask whether the screen still shows the place the store believes the player is in.

    Black-screen transitions cannot see a move that has no black screen (town to route), and a missed one
    leaves ``current`` wrong until the next detected move. This is the second line: one question per
    episode. To keep a naming slip from flipping the location, a different place is accepted at once only
    when the store already knows it; a new one has to be named twice in a row. A move found this way adds
    no edge to the map, since how the player got there is not known.

    :param candidate: The different place named by the previous check and not yet accepted, if any.
    :return: The candidate to carry to the next episode (``None`` once resolved).
    """
    current = store.current
    prompt = (P.LOCATE_VERIFY_PROMPT.replace("[GAME]", game)
              .replace("[CURRENT]", place_text(current))
              .replace("[SCREEN_TILES]", screen_tiles or "(nothing was recognised)")
              .replace("[KNOWN]", store.describe_known()))
    answer = ask(vlm_caller("locate_verify"), prompt, _parse_verify(store), "Location check",
                 LOCATION_ATTEMPTS, parameters, images=[frame])
    if answer is None or answer == "unknown":
        return candidate
    if answer == "same":
        return None
    kind, place = answer
    if place == current:
        return None
    if store.location(place) is not None or place == candidate:
        _enter(store, kind, place, frame, identified, episode_number)
        log_info(f"The location check moved the player from {place_text(current)} to {place_text(place)}.",
                 parameters)
        return None
    log_info(f"The location check thinks the player is in {place_text(place)}, not {place_text(current)}; "
             f"waiting for a second check.", parameters)
    return place


def episode_frames(report: SupervisorReport) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    for executor_report in report.executor_reports:
        for step in executor_report.steps:
            if not isinstance(step, EnvironmentStepRecord):
                continue
            if not frames:
                frames.append(step.frame_before)
            passed = [frame for state in step.transition_states for frame in state["core"]["passed_frames"]]
            frames.extend(passed or [step.frame_after])
    return frames


def is_black(frame: np.ndarray) -> bool:
    return float(np.mean(np.asarray(frame) <= BLACK_PIXEL_LEVEL)) >= BLACK_FRACTION


def black_runs(frames: Sequence[np.ndarray]) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    for index, frame in enumerate(frames):
        if not is_black(frame):
            continue
        if runs and index - runs[-1][1] <= BLACK_MERGE_GAP:
            runs[-1] = (runs[-1][0], index)
        else:
            runs.append((index, index))
    return runs


@dataclass
class TransitionCandidate:
    """One black run with the frames to show the model: indices into the frame list, oldest first."""
    run: Tuple[int, int]
    before: List[int]
    after: List[int]


@dataclass
class FrameTail:
    """The end of the frame stream so far, carried from episode to episode.

    ``frames[:resolved]`` have been examined. A black run with nothing yet after it is not resolved: it
    stays in the tail, and is finished when the next episode's frames arrive."""
    frames: List[np.ndarray] = field(default_factory=list)
    resolved: int = 0


def _first_not_black(frames: Sequence[np.ndarray], indices: Iterable[int]) -> Optional[int]:
    return next((i for i in indices if not is_black(frames[i])), None)


def find_transition_candidates(frames: Sequence[np.ndarray], resolved: int = 0, gap: int = CONTEXT_GAP
                               ) -> Tuple[List[TransitionCandidate], Optional[int], List[Tuple[int, int]]]:
    """Black runs in ``frames[resolved:]`` that can be put to the model.

    The model is shown the nearest screen before the run and one ``gap`` positions earlier, and the nearest
    screen after it and one ``gap`` positions later. A run with nothing after it yet is *pending* (its start
    is returned so the caller can keep it); a run with nothing before it cannot be judged and is returned
    in the third value, so it is reported rather than dropped silently.

    :return: ``(candidates, pending_from, skipped_runs)``.
    """
    candidates: List[TransitionCandidate] = []
    skipped: List[Tuple[int, int]] = []
    pending_from: Optional[int] = None
    for start, end in black_runs(frames):
        if start < resolved:
            continue
        after_first = _first_not_black(frames, range(end + 1, len(frames)))
        if after_first is None:
            pending_from = start
            break
        before_first = _first_not_black(frames, range(start - 1, -1, -1))
        if before_first is None:
            skipped.append((start, end))
            continue
        before_second = _first_not_black(frames, range(before_first - gap, -1, -1))
        after_second = _first_not_black(frames, range(after_first + gap, len(frames)))
        candidates.append(TransitionCandidate(
            (start, end),
            [i for i in (before_second, before_first) if i is not None],
            [i for i in (after_first, after_second) if i is not None]))
    return candidates, pending_from, skipped


def settle_screen(env, max_steps: int = MAX_SETTLE_STEPS) -> List[np.ndarray]:
    """Let the game run on, pressing nothing, while the screen is black.

    An episode can end on the middle of a door warp. Waiting here means the next task is dispatched from
    the new place rather than from a black screen, and the warp's frames are captured in this episode.

    :return: The frames captured while waiting.
    """
    emulator = env._emulator
    frames: List[np.ndarray] = []
    for _ in range(max_steps):
        if not is_black(env.get_info()["core"]["current_frame"]):
            break
        captured, _ = emulator.step(None)
        if captured is None:
            break
        frames.extend(captured)
    return frames


def _around(prompt: str, game: str, before: List[np.ndarray], after: List[np.ndarray]) -> str:
    return (prompt.replace("[GAME]", game)
            .replace("[N_BEFORE]", str(len(before)))
            .replace("[N_AFTER]", str(len(after))))


def _joins(source, destination, two_way: bool, wanted_source, wanted_destination) -> bool:
    return ((source, destination) == (wanted_source, wanted_destination)
            or (two_way and (destination, source) == (wanted_source, wanted_destination)))


def record_transition(store: LocationStore, previous: Optional[Place], new: Place, how: Optional[str],
                      two_way: bool, episode_number: int) -> Optional[Union[InternalTransition, LocationTransition]]:
    if previous is None or previous == new:
        return None
    if previous[0] == new[0] and previous[0] in store.majors:
        major, source, destination = previous[0], previous[-1], new[-1]
        if any(_joins(t.node_source, t.node_destination, t.two_way, source, destination)
               for t in store.internal_graph(major).transitions):
            return None
        return store.add_internal_transition(
            InternalTransition(major_location=major, node_source=source, node_destination=destination,
                               description=how, two_way=two_way), episode_number=episode_number)
    for bridge, other in ((previous[0], new[0]), (new[0], previous[0])):
        if bridge in store.bridges and other in store.majors and other not in store.bridges[bridge].accessible_through:
            store.add_bridge_access(bridge, other, episode_number=episode_number)
    source = (previous[0], previous[1] if len(previous) == 2 else None)
    destination = (new[0], new[1] if len(new) == 2 else None)
    if any(_joins((t.node_source_major, t.node_source_internal), (t.node_destination_major, t.node_destination_internal),
                  t.two_way, source, destination) for t in store.graph.transitions):
        return None
    return store.add_transition(
        LocationTransition(node_source_major=source[0], node_destination_major=destination[0],
                           node_source_internal=source[1], node_destination_internal=destination[1],
                           description=how, two_way=two_way), episode_number=episode_number)


def follow_transitions(store: LocationStore, new_frames: Sequence[np.ndarray], tail: FrameTail, tiles,
                       vlm_caller: VLMCaller, game: str, episode_number: int,
                       parameters: Optional[dict] = None) -> None:
    """Find the black screens in this episode's frames (and the end of the last episode's) and ask whether
    each was a move to a different place. ``tail`` is updated in place: a black run that has nothing after
    it yet is kept for the next episode instead of being dropped."""
    frames = tail.frames + list(new_frames)
    candidates, pending_from, skipped = find_transition_candidates(frames, tail.resolved)
    for start, end in skipped:
        log_warn(f"A black screen (frames {start}-{end}) had no earlier screen to compare with; "
                 f"no transition check was made.", parameters)
    if pending_from is not None:
        log_info(f"The episode ended on a black screen (frame {pending_from}); its transition check "
                 f"waits for the next frames.", parameters)
    resolved = pending_from if pending_from is not None else len(frames)
    drop = max(0, min(len(frames) - TAIL_MAX_FRAMES, resolved))
    tail.frames, tail.resolved = frames[drop:], resolved - drop

    for candidate in candidates:
        before = [frames[i] for i in candidate.before]
        after = [frames[i] for i in candidate.after]
        images = before + after
        moved = ask(vlm_caller("transition_check"), _around(P.TRANSITION_CHECK_PROMPT, game, before, after),
                    _parse_transition, "Transition check", LOCATION_ATTEMPTS, parameters, images=images)
        if not moved:
            continue
        previous = store.current
        connected = store.connected(previous) if previous is not None else []
        prompt = (_around(P.TRANSITION_NAME_PROMPT, game, before, after)
                  .replace("[PREVIOUS]", place_text(previous))
                  .replace("[CONNECTED]", ", ".join(connected) or "(none known yet)")
                  .replace("[KNOWN]", store.describe_known()))
        answer = ask(vlm_caller("transition_name"), prompt, _parse_place(store), "Transition naming",
                     LOCATION_ATTEMPTS, parameters, images=images)
        if answer is None or answer[0] == "unknown":
            store.set_current(None, episode_number=episode_number)
            log_info(f"The player left {place_text(previous)} for a place that could not be named.", parameters)
            continue
        kind, place, how, two_way = answer
        _enter(store, kind, place, after[-1], tiles.identify_tiles(after[-1]), episode_number)
        record_transition(store, previous, place, how, two_way, episode_number)
        log_info(f"The player moved from {place_text(previous)} to {place_text(place)}.", parameters)


# ----------------------------------------------------------------------------
# Path planning
# ----------------------------------------------------------------------------


EXPLORE_MAX_NEIGHBOURS = 1


@dataclass
class Hop:
    source: Place
    destination: Place
    how: Optional[str] = None
    notes: Optional[str] = None
    reversed: bool = False


def _node(major: str, name: Optional[str]) -> Place:
    return (major,) if name is None or name == major else (major, name)


def _edges(store: LocationStore) -> Dict[Place, List[Hop]]:
    edges: Dict[Place, List[Hop]] = {}

    def add(source: Place, destination: Place, how: Optional[str], notes: Optional[str], two_way: bool) -> None:
        edges.setdefault(source, []).append(Hop(source, destination, how, notes))
        if two_way:
            edges.setdefault(destination, []).append(Hop(destination, source, how, notes, reversed=True))

    for major, graph in store.internal_graphs.items():
        for t in graph.transitions:
            add(_node(major, t.node_source), _node(major, t.node_destination), t.description, t.other_notes,
                t.two_way)
    for t in store.graph.transitions:
        add(_node(t.node_source_major, t.node_source_internal),
            _node(t.node_destination_major, t.node_destination_internal), t.description, t.other_notes, t.two_way)
    return edges


def _search(store: LocationStore, source: Place) -> Tuple[Dict[Place, Optional[Hop]], Dict[Place, int]]:
    edges = _edges(store)
    parents: Dict[Place, Optional[Hop]] = {source: None}
    distances = {source: 0}
    queue = deque([source])
    while queue:
        place = queue.popleft()
        for hop in edges.get(place, []):
            if hop.destination not in parents:
                parents[hop.destination] = hop
                distances[hop.destination] = distances[place] + 1
                queue.append(hop.destination)
    return parents, distances


def known_places(store: LocationStore) -> List[Place]:
    places: List[Place] = [(name,) for name in store.names()]
    for major, rooms in store.internals.items():
        places += [(major, name) for name in rooms]
    for source, hops in _edges(store).items():
        for place in [source] + [hop.destination for hop in hops]:
            if place not in places:
                places.append(place)
    return places


def reachable(store: LocationStore, source: Optional[Place]) -> Optional[Dict[Place, int]]:
    if source is None:
        return None
    return _search(store, source)[1]


def route(store: LocationStore, source: Optional[Place], targets: Set[Place]) -> Optional[List[Hop]]:
    if source is None:
        return None
    parents, distances = _search(store, source)
    found = [place for place in distances if place in targets]
    if not found:
        return None
    hops: List[Hop] = []
    place = found[0]
    while parents[place] is not None:
        hops.append(parents[place])
        place = parents[place].source
    return hops[::-1]


def majors_on(hops: List[Hop]) -> List[str]:
    if not hops:
        return []
    majors: List[str] = []
    for place in [hops[0].source] + [hop.destination for hop in hops]:
        if not majors or majors[-1] != place[0]:
            majors.append(place[0])
    return majors


def render_route(hops: List[Hop]) -> str:
    if not hops:
        return "You are already there."
    lines = ["Passing through: " + " -> ".join(majors_on(hops))]
    for index, hop in enumerate(hops, start=1):
        line = f"{index}. {place_text(hop.source)} -> {place_text(hop.destination)}"
        if hop.how:
            line += f": {hop.how}" + (" (this was seen in the other direction)" if hop.reversed else "")
        if hop.notes:
            line += f" (notes: {hop.notes})"
        lines.append(line)
    return "\n".join(lines)


def explore(store: LocationStore) -> List[Dict[str, Any]]:
    outgoing: Dict[str, Set[str]] = {}
    neighbours: Dict[str, Set[str]] = {}
    for source, hops in _edges(store).items():
        for hop in hops:
            if hop.source[0] == hop.destination[0]:
                continue
            outgoing.setdefault(hop.source[0], set()).add(hop.destination[0])
            neighbours.setdefault(hop.source[0], set()).add(hop.destination[0])
            neighbours.setdefault(hop.destination[0], set()).add(hop.source[0])
    candidates = []
    for place in known_places(store):
        if len(place) != 1:
            continue
        name = place[0]
        if not outgoing.get(name):
            reason = "no known way out of it"
        elif len(neighbours.get(name, ())) <= EXPLORE_MAX_NEIGHBOURS:
            reason = f"connects to only {len(neighbours[name])} other known place"
        else:
            continue
        candidates.append({"place": place, "reason": reason, "route": route(store, store.current, {place})})
    return candidates


def _parse_selection(count: int) -> Parser:
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
                return None, f"{part!r} is not a place number. The choice must be {valid}."
            if int(part) not in indices:
                indices.append(int(part))
        return indices, None
    return parse


def navigate(store: LocationStore, request: str, vlm_caller: VLMCaller, game: str,
             parameters: Optional[dict] = None) -> Optional[str]:
    distances = reachable(store, store.current)
    if distances is None:
        return None
    entries = list(distances) + [place for place in known_places(store) if place not in distances]
    lines = []
    for index, place in enumerate(entries):
        if place not in distances:
            note = "no known route"
        elif distances[place] == 0:
            note = "you are here"
        else:
            note = f"{distances[place]} moves away"
        lines.append(f"{index}. {place_text(place)} ({note})")
    prompt = (P.NAVIGATE_SELECT_PROMPT.replace("[GAME]", game)
              .replace("[REQUEST]", request)
              .replace("[CURRENT]", place_text(store.current))
              .replace("[PLACES]", "\n".join(lines)))
    indices = ask(vlm_caller("navigate_select"), prompt, _parse_selection(len(entries)), "Navigate selection",
                  LOCATION_ATTEMPTS, parameters)
    if indices is None:
        log_warn(f"No usable place selection for {request!r}; giving no route.", parameters)
        return None
    if not indices:
        return f"No place on your map fits \"{request}\"."
    chosen = {entries[index] for index in indices}
    hops = route(store, store.current, chosen)
    if hops is None:
        names = ", ".join(place_text(place) for place in chosen)
        return f"{names} is on your map, but no route there from {place_text(store.current)} is known yet."
    return render_route(hops)
