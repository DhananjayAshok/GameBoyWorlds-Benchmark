"""
What a client is allowed to see.

Two modes, chosen by the server's policy and nothing else:

    full    the emulator's state tracker report, whole. Every group, every key, nothing dropped.
    scored  task success and subgoal progress, and nothing else.

`gameboy_worlds.Environment.get_info()` returns the tracker report, which holds numpy arrays, enums and ragged
tuples, so `json.dumps` refuses it outright. `jsonable` is a total encoder: it converts rather than filters, so
"full" really is everything. Frame arrays go out as lossless PNG rather than nested integer lists, which is the
same bytes in a twentieth of the space.
"""
import base64
import enum
import io

import numpy as np
from PIL import Image

#: Served on every step, alongside the screen.
SERVED_STEP = ("action_success", "changed", "steps", "max_steps")

#: Tracker fields promoted into obs, under `obs.signals`, in every mode including `scored`.
#:
#: Both are things the screen already says: `agent_state` is whether the player is walking around or looking at a
#: dialogue box, a menu or a battle, and `n_battles_lost` counts blackouts, which play a distinctive animation. The
#: playthrough leg reads exactly these two out of RAM today, so promoting them is what lets a test-time run be
#: observation-only without being re-engineered first.
#:
#: Neither is grading state: the grader lives in `termination_truncation` and `subgoals`.
#:
#: Looked up by name across every group rather than at a fixed path, because the group holding them is named per
#: game -- `pokemon_core` here, `bomberman_max_core` elsewhere -- and a game whose tracker has neither key simply
#: gets an empty `signals`.
OBS_SIGNAL_FIELDS = ("agent_state", "n_battles_lost")

#: The two tracker groups a `scored` client sees. `termination_truncation` is where the benchmark's own pass/fail
#: lives (TestEnvironmentMixin puts success in `terminated`); `subgoals` is partial progress through the task.
SCORED_GROUPS = ("termination_truncation", "subgoals")

STATE_MODES = ("full", "scored")

#: A uint8 array with this many dimensions or more is a frame, or a stack of them.
FRAME_NDIM = 2

#: How deep `jsonable` will walk. The tracker holds action classes whose own attributes reach back into dicts and
#: lists, so an encoder that recurses on anything iterable must be bounded by construction rather than by trusting
#: the shape of what it is handed. Anything deeper is rendered as its repr.
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
    """Anything the tracker can hold, as JSON. Total: every branch returns, nothing is dropped.

    A uint8 array of two dimensions or more is a frame and goes out as PNG; a stack of them goes out as a list of
    PNGs. Every other array becomes nested lists. Enums become their name, numpy scalars their Python value, and
    anything else its repr.
    """
    if depth > MAX_DEPTH:
        return repr(value)
    if isinstance(value, (str, bool, int, float, type(None))):
        return value
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    # A class, not an instance: the tracker records the action's type in previous_action_details. Named before the
    # container branches, because a class is iterable-looking to duck typing and walking into one reaches the whole
    # module graph.
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


def full_state(*, report: dict) -> dict:
    """The tracker report, whole."""
    return jsonable(report)


def scored_state(*, report: dict) -> dict:
    """Task success and subgoal progress. Nothing else is read, so nothing else can leak."""
    return {group: jsonable(values) for group, values in report.items() if group in SCORED_GROUPS}


def state_for(*, report: dict, mode: str) -> dict:
    if mode == "full":
        return full_state(report=report)
    if mode == "scored":
        return scored_state(report=report)
    raise ValueError(f"unknown state mode {mode!r}; one of {STATE_MODES}")


def obs_signals(*, report) -> dict:
    """The `OBS_SIGNAL_FIELDS` present in `report`, by name, wherever they sit.

    First group to carry a name wins; a report with none of them yields `{}`.
    """
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
               action_success=None, changed=None) -> dict:
    return {"frame": {"__png__": frame_png(frame)}, "goal": goal,
            "actions": dict(actions) if actions else {},
            "steps": int(steps), "max_steps": None if max_steps is None else int(max_steps),
            "action_success": jsonable(action_success), "changed": jsonable(changed),
            "signals": obs_signals(report=report)}


def audit_scored(*, payload) -> list:
    """Tracker group names in `payload` that a `scored` client must not have been given.

    An allowlist check on the way out: it names what is present and unexpected rather than hunting for specific
    bad keys, so a change that widens the response fails `check.py` instead of leaking.
    """
    state = payload.get("state") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        return []
    return sorted(group for group in state if group not in SCORED_GROUPS)
