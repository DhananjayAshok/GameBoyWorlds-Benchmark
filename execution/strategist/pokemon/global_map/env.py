from __future__ import annotations

import traceback
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from .registration import has_structure

if TYPE_CHECKING:
    from .model import GlobalMap


def _movement(action, kwargs: Dict[str, Any], info: Dict[str, Any]) -> Tuple[Optional[str], int, Optional[int]]:
    from gameboy_worlds.emulation.emulator import LowLevelActions
    from gameboy_worlds.interface.action import LowLevelAction
    from gameboy_worlds.interface.pokemon.actions import MoveStepsAction

    if action is MoveStepsAction or (isinstance(action, type) and issubclass(action, MoveStepsAction)):
        direction = kwargs.get("direction")
        steps = int(kwargs.get("steps") or 1)
        reported = (info.get("core", {}).get("action_return") or {}).get("n_steps_taken")
        # one extra cell allows for a ledge hop, which moves two cells on one press
        return direction, steps + 1, (int(reported) if reported is not None else None)
    if action is LowLevelAction or (isinstance(action, type) and issubclass(action, LowLevelAction)):
        arrows = {LowLevelActions.PRESS_ARROW_UP: "up", LowLevelActions.PRESS_ARROW_DOWN: "down",
                  LowLevelActions.PRESS_ARROW_LEFT: "left", LowLevelActions.PRESS_ARROW_RIGHT: "right"}
        direction = arrows.get(kwargs.get("low_level_action"))
        if direction is not None:
            return direction, 2, None
    return None, 0, None


class MappingEnvironment:
    """Wraps the environment so every step taken through it updates the global map; a map failure ends the run."""

    def __init__(self, env, global_map: GlobalMap, tile_recognizer, parameters: Optional[dict] = None) -> None:
        self._env = env
        self._map = global_map
        self._tiles = tile_recognizer
        self._parameters = parameters
        self.marker = None

    def step_high_level_action(self, action, **kwargs):
        from gameboy_worlds.emulation.pokemon import AgentState

        before = self._env.get_info()
        was_free = before.get("pokemon_core", {}).get("agent_state") == AgentState.FREE_ROAM
        result = self._env.step_high_level_action(action, **kwargs)
        try:
            info = result[4] or {}
            if (info.get("pokemon_core", {}).get("n_battles_lost", 0)
                    > before.get("pokemon_core", {}).get("n_battles_lost", 0)):
                self._map.record_defeat()
            core = info.get("core", {})
            if not info.get("invalid_action") and core.get("current_frame") is not None:
                frame = core["current_frame"]
                free = info.get("pokemon_core", {}).get("agent_state") == AgentState.FREE_ROAM
                direction, max_steps, reported = _movement(action, kwargs, info)
                passed = core.get("transition_passed_frames")
                faded = passed is not None and any(not has_structure(f) for f in passed)
                identified = self._tiles.identify_tiles(frame) if free else {}
                queued = len(self._map.naming_queue)
                self._map.on_step(frame, identified, free, direction, max_steps, reported, was_free, faded)
                for entry in self._map.naming_queue[queued:]:
                    entry["marker"] = self.marker() if self.marker is not None else None
        except Exception as e:
            from utils import log_error
            # log_error raises
            name = getattr(action, "__name__", repr(action))
            log_error(f"Global map update failed on {name}({kwargs}): {type(e).__name__}: {e}\n"
                      f"was_free={was_free}, segment={self._map.segment}, pos={self._map.pos}, "
                      f"cells={self._map.n_cells}\n{traceback.format_exc()}", self._parameters)
        return result

    @property
    def ledge_patterns(self) -> set:
        return self._map.ledge_patterns

    def step_str(self, input_str: str):
        action, kwargs = self._env.string_to_high_level_action(input_str)
        if action is None:
            return None, None, None, None, None
        return self.step_high_level_action(action, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._env, name)
