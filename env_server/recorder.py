import json
import os
from typing import Optional

import numpy as np
from PIL import Image

from utils.log_handling import log_warn


def _jsonable(value, depth: int = 0):
    # Unlike codec.jsonable this summarises large arrays: frames are saved as PNG files beside the log.
    if depth > 8:
        return str(value)[:200]
    if isinstance(value, np.ndarray):
        return {"__shape__": list(value.shape), "dtype": str(value.dtype)} if value.size > 64 \
            else value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v, depth + 1) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)[:200]


class Recorder:
    """Host-side record of every step's full tracker report, whatever the client was sent."""

    def __init__(self, *, directory: str, session: str, task_id: Optional[str], game: Optional[str],
                 policy: str, parameters: dict) -> None:
        self._parameters = parameters
        self.directory = os.path.join(directory, policy, session)
        os.makedirs(os.path.join(self.directory, "frames"), exist_ok=True)
        self.n = 0
        with open(os.path.join(self.directory, "meta.json"), "w") as f:
            json.dump({"session": session, "task_id": task_id, "game": game, "policy": policy}, f, indent=1)

    def write(self, *, kind: str, action, info: Optional[dict], success, frame=None) -> None:
        try:
            if frame is not None:
                arr = np.asarray(frame)
                if arr.ndim == 3 and arr.shape[2] == 1:
                    arr = arr[:, :, 0]
                Image.fromarray(arr.astype(np.uint8)).save(
                    os.path.join(self.directory, "frames", f"{self.n:05d}.png"))
            with open(os.path.join(self.directory, "steps.jsonl"), "a") as f:
                f.write(json.dumps({"n": self.n, "kind": kind, "action": action, "success": success,
                                    "info": _jsonable(info)}) + "\n")
        except Exception as exc:
            log_warn(f"recorder: {exc}", parameters=self._parameters)
        self.n += 1

    def finish(self, *, success) -> None:
        try:
            path = os.path.join(self.directory, "meta.json")
            with open(path) as f:
                meta = json.load(f)
            meta.update({"success": success, "steps": self.n})
            with open(path, "w") as f:
                json.dump(meta, f, indent=1)
        except Exception as exc:
            log_warn(f"recorder finish: {exc}", parameters=self._parameters)
