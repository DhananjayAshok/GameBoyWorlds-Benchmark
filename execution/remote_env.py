import io
import os
import sys
from typing import Any, Dict, Optional, Tuple, Type

import numpy as np
from PIL import Image

from execution.game_api import (ACTION_CLASSES, AgentState, BattleMenuAction, InteractAction, LowLevelAction,
                                LowLevelActions, MenuAction, MoveStepsAction, OpenMenuAction, PassDialogueAction,
                                PickAttackAction, WaitAction)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent_workspace"))
from envclient import Client  # noqa: E402

BUTTONS = {"a": LowLevelActions.PRESS_BUTTON_A, "b": LowLevelActions.PRESS_BUTTON_B,
           "start": LowLevelActions.PRESS_BUTTON_START, "up": LowLevelActions.PRESS_ARROW_UP,
           "down": LowLevelActions.PRESS_ARROW_DOWN, "left": LowLevelActions.PRESS_ARROW_LEFT,
           "right": LowLevelActions.PRESS_ARROW_RIGHT}
BUTTON_NAMES = {button: name for name, button in BUTTONS.items()}
DIRECTIONS = ("up", "down", "left", "right")
BATTLE_OPTIONS = ("fight", "pokemon", "bag", "run", "progress")
MENU_OPTIONS = ("up", "down", "left", "right", "confirm", "back")
LAST_ACTION_FIELDS = ("action_return", "transition_passed_frames", "previous_action_details")


def to_frame(png: bytes) -> np.ndarray:
    frame = np.array(Image.open(io.BytesIO(png)))
    return frame[:, :, None] if frame.ndim == 2 else frame


def to_frames(pngs) -> Optional[np.ndarray]:
    return np.stack([to_frame(png) for png in pngs]) if pngs else None


def parse_action(text: str) -> Tuple[Optional[Type], Optional[dict]]:
    text = text.lower().strip()
    if "(" not in text or ")" not in text:
        return None, None
    name = text.split("(")[0].strip()
    args = text.split("(")[1].split(")")[0].strip()
    if name == "button":
        button = BUTTONS.get(args)
        return (LowLevelAction, {"low_level_action": button}) if button is not None else (None, None)
    if name == "wait":
        return WaitAction, {}
    if name == "interact":
        return InteractAction, {}
    if name == "passdialogue":
        return PassDialogueAction, {}
    if name == "battlemenu":
        return (BattleMenuAction, {"option": args}) if args in BATTLE_OPTIONS else (None, None)
    if name == "pickattack":
        if not args.isnumeric() or not 1 <= int(args) <= 4:
            return None, None
        return PickAttackAction, {"option": int(args)}
    if name == "menu":
        return (MenuAction, {"menu_action": args}) if args in MENU_OPTIONS else (None, None)
    if name == "move":
        found = [d for d in DIRECTIONS if d in args]
        if len(found) != 1:
            return None, None
        steps = args.replace(found[0], "").strip()
        if not steps.isnumeric():
            return None, None
        return MoveStepsAction, {"direction": found[0], "steps": int(steps)}
    if name == "openmenu":
        return OpenMenuAction, {"option": args}
    return None, None


def format_action(action: Type, kwargs: Dict[str, Any]) -> str:
    name = action.__name__
    if name == LowLevelAction.__name__:
        return f"button({BUTTON_NAMES[kwargs['low_level_action']]})"
    if name == MoveStepsAction.__name__:
        return f"move({kwargs['direction']} {int(kwargs['steps'])})"
    if name == BattleMenuAction.__name__:
        return f"battlemenu({kwargs['option']})"
    if name == PickAttackAction.__name__:
        return f"pickattack({int(kwargs['option'])})"
    if name == MenuAction.__name__:
        return f"menu({kwargs['menu_action']})"
    if name == OpenMenuAction.__name__:
        return f"openmenu({kwargs['option']})"
    no_args = {WaitAction.__name__: "wait()", InteractAction.__name__: "interact()",
               PassDialogueAction.__name__: "passdialogue()"}
    if name in no_args:
        return no_args[name]
    raise ValueError(f"{name} has no action string on the environment server")


class RemoteEmulator:
    def __init__(self, env: "RemoteEnvironment") -> None:
        self._env = env
        self.session_path = None

    @property
    def step_count(self) -> int:
        return self._env.steps

    def check_if_done(self) -> bool:
        return self._env.done


class RemoteEnvironment:
    def __init__(self, session, owned: bool = False) -> None:
        self._session = session
        self._owned = owned
        self._emulator = RemoteEmulator(self)
        self.described = session.describe()
        self.game = self.described["game"]
        self._named_saves = bool(self.described.get("named_saves"))
        self._absorb(session.observe(), frame_changed=False)

    @classmethod
    def attach(cls, session_id: Optional[str] = None) -> "RemoteEnvironment":
        session_id = session_id or os.environ.get("ENVCTL_SESSION")
        if not session_id:
            raise RuntimeError("no session to attach to: pass one or set $ENVCTL_SESSION")
        return cls(Client.from_config().attach(session_id))

    @classmethod
    def open(cls, game: str, run: Optional[str] = None) -> "RemoteEnvironment":
        session = Client.from_config().open(task_id=f"gb.{game}.playthrough", run=run)
        try:
            session.reset()
            return cls(session, owned=True)
        except Exception:
            session.close()
            raise

    def _absorb(self, out: dict, *, frame_changed: bool) -> None:
        obs = out["obs"]
        signals = obs.get("signals") or {}
        self.steps = int(obs.get("steps") or 0)
        self.done = bool(out.get("done"))
        self.success = out.get("success")
        self._actions = obs.get("actions") or {}
        last_action = {key: value for key, value in getattr(self, "_info", {}).get("core", {}).items()
                       if key in LAST_ACTION_FIELDS}
        self._info = {"core": {**last_action, "current_frame": to_frame(obs["frame"]),
                               "frame_changed": frame_changed},
                      "pokemon_core": {"agent_state": AgentState[signals["agent_state"]],
                                       "n_battles_lost": int(signals.get("n_battles_lost") or 0)}}

    def get_info(self) -> Dict[str, Dict[str, Any]]:
        return {group: dict(values) if isinstance(values, dict) else values for group, values in self._info.items()}

    def get_observation(self) -> np.ndarray:
        return self._info["core"]["current_frame"]

    def get_action_strings(self, return_all: bool = False) -> Dict[Type, str]:
        unknown = sorted(set(self._actions) - set(ACTION_CLASSES))
        if unknown:
            raise RuntimeError(f"the server offers actions this harness cannot take: {unknown}")
        return {ACTION_CLASSES[name]: text for name, text in self._actions.items()}

    def string_to_high_level_action(self, input_str: str) -> Tuple[Optional[Type], Optional[dict]]:
        return parse_action(input_str)

    def step_str(self, input_str: str):
        action, kwargs = parse_action(input_str)
        if action is None:
            return None, None, None, None, None
        return self.step_high_level_action(action, **kwargs)

    def step_high_level_action(self, action: Type, **kwargs):
        out = self._session.step(format_action(action, kwargs))
        if not out.get("valid"):
            self._absorb(out, frame_changed=False)
            self._info["invalid_action"] = True
            return self.get_observation(), 0.0, False, False, self.get_info()
        transition = out["obs"].get("transition") or {}
        self._absorb(out, frame_changed=bool(transition.get("frame_changed")))
        action_return = transition.get("action_return")
        states = [{"core": {}, "ocr": {"ocr_regions": {kind: to_frames(crops) for kind, crops in ocr.items()
                                                       if crops}}}
                  for ocr in (transition.get("ocr") or [{}])]
        if action_return is not None:
            states[-1]["core"]["action_return"] = action_return
        self._info["core"].update({
            "action_return": action_return,
            "transition_passed_frames": to_frames(transition.get("frames")),
            "previous_action_details": (action, kwargs, states, out["obs"].get("action_success"), action_return)})
        self._info["ocr"] = {"transition_ocr_regions": [state["ocr"]["ocr_regions"] for state in states]}
        info = self.get_info()
        terminated = self.done and self.success == 1.0
        return self.get_observation(), 0.0, terminated, self.done and not terminated, info

    def save_custom_state(self, name: str) -> None:
        if self._named_saves:
            self._session.save(name)
        else:
            self._session.checkpoint(name)

    def load_custom_state(self, name: str) -> None:
        if self._named_saves:
            self._absorb(self._session.load(name), frame_changed=True)
            return
        label = self._session.describe().get("label")
        if label != name:
            raise RuntimeError(f"the server resumed from checkpoint {label!r}, not {name!r}: the game and this "
                               "harness's saved memory are out of step")

    def reset(self):
        if not self._owned:
            raise RuntimeError("a remote session is opened and reset by the server's operator, not the harness")
        self._absorb(self._session.reset(), frame_changed=True)
        return self.get_observation(), self.get_info()

    def close(self) -> None:
        if self._owned:
            self._session.close()
            self._owned = False
