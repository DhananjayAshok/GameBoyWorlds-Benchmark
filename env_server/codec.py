import base64
import enum
import io

import numpy as np
from PIL import Image

# Tracker fields served under obs.signals in every mode. Both are things the screen already shows;
# neither is grading state. Looked up by name because the group holding them is named per game.
OBS_SIGNAL_FIELDS = ("agent_state", "n_battles_lost")

# The only tracker groups a "scored" client sees: task success and subgoal progress.
SCORED_GROUPS = ("termination_truncation", "subgoals")

STATE_MODES = ("full", "scored")

FRAME_NDIM = 2

# The tracker holds objects whose attributes reach back into containers, so the walk is depth-bounded.
MAX_DEPTH = 12


def frame_png(frame: np.ndarray) -> str:
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def jsonable(value, depth: int = 0):
    """Total JSON encoder for the tracker report: converts everything, drops nothing."""
    if depth > MAX_DEPTH:
        return repr(value)
    if isinstance(value, (str, bool, int, float, type(None))):
        return value
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    # Before the container branches: a class must be named, not walked into.
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, np.ndarray):
        if value.dtype == np.uint8 and value.ndim >= FRAME_NDIM:
            if value.ndim == FRAME_NDIM or (value.ndim == 3 and value.shape[2] in (1, 3, 4)):
                return {"__png__": frame_png(value)}
            return [jsonable(item, depth + 1) for item in value]
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item, depth + 1) for item in value]
    return repr(value)


def state_for(*, report: dict, mode: str) -> dict:
    if mode == "full":
        return jsonable(report)
    if mode == "scored":
        return {group: jsonable(values) for group, values in report.items() if group in SCORED_GROUPS}
    raise ValueError(f"unknown state mode {mode!r}; one of {STATE_MODES}")


def obs_signals(*, report) -> dict:
    found: dict = {}
    if not isinstance(report, dict):
        return found
    for values in report.values():
        if not isinstance(values, dict):
            continue
        for field in OBS_SIGNAL_FIELDS:
            if field in values and field not in found:
                found[field] = jsonable(values[field])
    return found


def public_obs(*, frame: np.ndarray, goal: str, actions, steps: int, max_steps, report=None,
               action_success=None) -> dict:
    return {"frame": {"__png__": frame_png(frame)}, "goal": goal,
            "actions": dict(actions) if actions else {},
            "steps": int(steps), "max_steps": None if max_steps is None else int(max_steps),
            "action_success": jsonable(action_success),
            "signals": obs_signals(report=report)}


def _pngs(frames) -> list:
    return [] if frames is None else [{"__png__": frame_png(frame)} for frame in frames]


def transition(*, info) -> dict:
    core = info.get("core", {}) if isinstance(info, dict) else {}
    details = core.get("previous_action_details")
    states = details[2] if details else []
    return {"frames": _pngs(core.get("transition_passed_frames")),
            "frame_changed": bool(core.get("frame_changed")),
            "action_return": jsonable(details[4]) if details else None,
            "ocr": [{kind: _pngs(crops) for kind, crops in state.get("ocr", {}).get("ocr_regions", {}).items()}
                    for state in states]}


def audit_scored(*, payload) -> list:
    """Tracker groups in `payload` that a "scored" client must not have been given."""
    state = payload.get("state") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        return []
    return sorted(group for group in state if group not in SCORED_GROUPS)
