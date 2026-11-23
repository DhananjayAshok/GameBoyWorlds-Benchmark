import json
import os
import threading
import time
from collections import deque
from typing import Optional

from utils import log_warn

PERIODIC_STEPS = 500
PERIODIC_SECONDS = 30
CLEARS_FILE = "clears.jsonl"
SUMMARY_FILE = "summary.json"
RATE_WINDOW_STEPS = 2000
_summary_lock = threading.Lock()


def _call(reader, default):
    try:
        value = reader()
    except Exception:
        return default
    return default if value is None else value


def map_key(location: dict) -> Optional[str]:
    if location.get("map_name"):
        return location["map_name"]
    if location.get("map_id") is not None:
        return f"map:{location['map_id']}"
    if location.get("map_group") is not None and location.get("map_number") is not None:
        return f"map:{location['map_group']}:{location['map_number']}"
    return None


class Progress:
    def __init__(self, *, directory: str, session: str, game: str, run: Optional[str],
                 memory, parameters: dict) -> None:
        self.directory = directory
        self.session = session
        self.game = game
        self.run = run
        self.memory = memory
        self._parameters = parameters
        self.maps: list = []
        self._seen_maps: set = set()
        self._reported_maps = 0
        self.tiles: dict = {}
        self._tile_count = 0
        self._reported_tiles = 0
        self._last_reading = None
        self._steps = 0
        self._new_tile_steps: Optional[int] = None
        self._new_tile_time = time.time()
        self._history: deque = deque()
        self._reported_events: set = set(self._events())
        self._reported_flags: set = set(self._flags())
        self._written_at = 0
        self._written_time = time.time()
        self._progress_steps: Optional[int] = None
        self._progress_seen: Optional[tuple] = None
        os.makedirs(directory, exist_ok=True)

    def _events(self) -> list:
        return _call(self.memory.events_raised, [])

    def _flags(self) -> list:
        reader = getattr(self.memory, "engine_flags_raised", None)
        return _call(reader, []) if reader else []

    def _badges(self) -> list:
        badges = _call(self.memory.badges, {})
        if "names" in badges:
            return list(badges["names"])
        return list(badges.get("johto", [])) + list(badges.get("kanto", []))

    def _map(self) -> Optional[str]:
        return map_key(_call(self.memory.location, {}))

    def lineage(self) -> dict:
        return {"maps": list(self.maps), "tiles": {name: sorted(cells) for name, cells in self.tiles.items()}}

    def restore(self, *, lineage: Optional[dict]) -> None:
        self.maps = list((lineage or {}).get("maps", []))
        self._seen_maps = set(self.maps)
        self._reported_maps = len(self.maps)
        self.tiles = {name: set(cells) for name, cells in (lineage or {}).get("tiles", {}).items()}
        self._tile_count = sum(len(cells) for cells in self.tiles.values())
        self._reported_tiles = self._tile_count
        self._reported_events = set(self._events())
        self._reported_flags = set(self._flags())
        self._last_reading = None
        self._new_tile_steps = None
        self._new_tile_time = time.time()
        self._history.clear()
        self._progress_steps = None
        self._progress_seen = None
        self.observe()

    def start_over(self) -> None:
        self.restore(lineage=None)

    def steps_without_progress(self, *, steps: int) -> int:
        maps, events, flags = set(self.maps), set(self._events()), set(self._flags())
        seen = self._progress_seen
        if seen is None or not (maps <= seen[0] and events <= seen[1] and flags <= seen[2]):
            self._progress_steps = steps
            seen = (set(), set(), set()) if seen is None else seen
            self._progress_seen = (seen[0] | maps, seen[1] | events, seen[2] | flags)
        return steps - self._progress_steps

    def observe(self, *, steps: Optional[int] = None) -> None:
        if steps is not None:
            self._steps = steps
            if self._new_tile_steps is None:
                self._new_tile_steps = steps
        location = _call(self.memory.location, {})
        name = map_key(location)
        x, y = location.get("x"), location.get("y")
        if not name or x is None or y is None:
            self._last_reading = None
            return
        settled = location.get("map_settled")
        if settled is False:
            self._last_reading = None
            return
        if settled is None:
            reading = (name, x, y)
            if reading != self._last_reading:
                self._last_reading = reading
                return
        if name not in self._seen_maps:
            self._seen_maps.add(name)
            self.maps.append(name)
        cells = self.tiles.setdefault(name, set())
        cell = x * 256 + y
        if cell not in cells:
            cells.add(cell)
            self._tile_count += 1
            self._new_tile_steps = self._steps
            self._new_tile_time = time.time()

    def due(self, *, steps: int) -> bool:
        if steps - self._written_at >= PERIODIC_STEPS:
            return True
        return steps != self._written_at and time.time() - self._written_time >= PERIODIC_SECONDS

    def write(self, *, kind: str, steps: int, subgoals: Optional[dict], success: Optional[float],
              save: Optional[str] = None) -> dict:
        self._written_at = steps
        self._written_time = time.time()
        events = self._events()
        flags = self._flags()
        party = _call(self.memory.party, [])
        record = {
            "t": round(time.time(), 1), "session": self.session, "run": self.run, "game": self.game,
            "kind": kind, "steps": steps, "save": save, "map": self._map(),
            "badges": self._badges(), "maps_reached": len(self.maps),
            "new_maps": self.maps[self._reported_maps:],
            "tiles": self._tile_count, "new_tiles": self._tile_count - self._reported_tiles,
            "steps_since_new_tile": max(0, steps - (self._new_tile_steps if self._new_tile_steps is not None else steps)),
            "seconds_since_new_tile": round(time.time() - self._new_tile_time, 1),
            "tiles_per_100_steps": self._rate(steps=steps),
            "events": len(events), "new_events": sorted(set(events) - self._reported_events),
            "new_flags": sorted(set(flags) - self._reported_flags),
            "party": [{"species": mon.get("species_name"), "level": mon.get("level")} for mon in party],
            "subgoals_completed": len((subgoals or {}).get("completed", [])),
            "subgoals_total": len((subgoals or {}).get("all", [])),
            "success": success,
        }
        self._reported_maps = len(self.maps)
        self._reported_tiles = self._tile_count
        self._reported_events = set(events)
        self._reported_flags = set(flags)
        self._append(f"{self.session}.jsonl", record)
        if kind == "cleared":
            self._append(CLEARS_FILE, record)
        self._summarize(record)
        return record

    def _rate(self, *, steps: int) -> Optional[float]:
        self._history.append((steps, self._tile_count))
        while len(self._history) > 1 and steps - self._history[1][0] >= RATE_WINDOW_STEPS:
            self._history.popleft()
        first_steps, first_tiles = self._history[0]
        if steps <= first_steps:
            return None
        return round(100.0 * (self._tile_count - first_tiles) / (steps - first_steps), 2)

    def _summarize(self, record: dict) -> None:
        keys = ("t", "run", "game", "kind", "steps", "map", "maps_reached", "tiles", "steps_since_new_tile",
                "seconds_since_new_tile", "tiles_per_100_steps", "badges")
        path = os.path.join(self.directory, SUMMARY_FILE)
        with _summary_lock:
            try:
                with open(path) as f:
                    summary = json.load(f)
            except (OSError, ValueError):
                summary = {}
            summary[self.session] = {key: record.get(key) for key in keys}
            try:
                with open(path + ".tmp", "w") as f:
                    json.dump(summary, f, indent=1)
                os.replace(path + ".tmp", path)
            except OSError as exc:
                log_warn(f"writing progress summary: {exc}", parameters=self._parameters)

    def _append(self, name: str, record: dict) -> None:
        try:
            with open(os.path.join(self.directory, name), "a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as exc:
            log_warn(f"writing progress for session {self.session}: {exc}", parameters=self._parameters)
