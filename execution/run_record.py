from __future__ import annotations

import json
import os
import time
from typing import List, Optional

from utils import log_warn

CHECKPOINTS_FILE = "checkpoints.jsonl"
TRAJECTORY_FILE = "trajectory.jsonl"
LAYERS = ("strategist", "supervisor", "executor")
KINDS = ("intention", "plan", "attempt", "outcome")
MAX_TEXT = 2000


class RunRecord:
    def __init__(self, directory: Optional[str], run: str, parameters: Optional[dict] = None) -> None:
        self.directory = directory
        self.run = run
        self._parameters = parameters
        if directory:
            os.makedirs(directory, exist_ok=True)

    def _append(self, name: str, record: dict) -> None:
        if not self.directory:
            return
        try:
            with open(os.path.join(self.directory, name), "a") as handle:
                handle.write(json.dumps(record) + "\n")
        except OSError as error:
            log_warn(f"Could not write {name} under {self.directory}: {error}", self._parameters)

    def checkpoint(self, *, checkpoint_id: int, episode: int, steps: Optional[int], server_save: str,
                   memory: List[str]) -> None:
        self._append(CHECKPOINTS_FILE, {"t": round(time.time(), 1), "run": self.run, "id": checkpoint_id,
                                        "episode": episode, "steps": steps, "server_save": server_save,
                                        "memory": memory})

    def event(self, *, episode: int, layer: str, kind: str, text: str, steps: Optional[int]) -> None:
        if layer not in LAYERS or kind not in KINDS:
            raise ValueError(f"a trajectory event is one of {KINDS} by one of {LAYERS}, got {kind!r} by {layer!r}")
        text = text.strip()
        if len(text) > MAX_TEXT:
            text = text[:MAX_TEXT] + " [...]"
        self._append(TRAJECTORY_FILE, {"t": round(time.time(), 1), "run": self.run, "episode": episode,
                                       "layer": layer, "kind": kind, "steps": steps, "text": text})
