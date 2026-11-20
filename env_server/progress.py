import json
import os
import time
from typing import Optional

from utils import log_warn

PERIODIC_STEPS = 500
PERIODIC_SECONDS = 120
CLEARS_FILE = "clears.jsonl"


def _call(reader, default):
    try:
        value = reader()
    except Exception:
        return default
    return default if value is None else value


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
        self._reported_events: set = set(self._events())
        self._reported_flags: set = set(self._flags())
        self._written_at = 0
        self._written_time = time.time()
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
        return _call(self.memory.location, {}).get("map_name")

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
        self.observe()

    def start_over(self) -> None:
        self.restore(lineage=None)

    def observe(self) -> None:
        location = _call(self.memory.location, {})
        name = location.get("map_name")
        if not name:
            return
        if name not in self._seen_maps:
            self._seen_maps.add(name)
            self.maps.append(name)
        x, y = location.get("x"), location.get("y")
        if x is None or y is None:
            return
        cells = self.tiles.setdefault(name, set())
        cell = x * 256 + y
        if cell not in cells:
            cells.add(cell)
            self._tile_count += 1

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
        return record

    def _append(self, name: str, record: dict) -> None:
        try:
            with open(os.path.join(self.directory, name), "a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as exc:
            log_warn(f"writing progress for session {self.session}: {exc}", parameters=self._parameters)
