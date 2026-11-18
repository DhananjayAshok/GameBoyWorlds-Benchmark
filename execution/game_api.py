from enum import Enum
from typing import Dict, Tuple

import numpy as np

try:
    from gameboy_worlds.emulation.emulator import LowLevelActions
    from gameboy_worlds.emulation.parser import StateParser
    from gameboy_worlds.emulation.pokemon import AgentState
    from gameboy_worlds.interface import Environment, HighLevelAction
    from gameboy_worlds.interface.action import LowLevelAction, WaitAction
    from gameboy_worlds.interface.pokemon.actions import (HARD_MAX_STEPS, BattleMenuAction, InteractAction,
                                                          MenuAction, MoveStepsAction, OpenMenuAction,
                                                          PassDialogueAction, PickAttackAction)

    STAND_IN = False
except ImportError:
    STAND_IN = True

    HARD_MAX_STEPS = 20

    class AgentState(Enum):
        FREE_ROAM = 0
        IN_DIALOGUE = 1
        IN_MENU = 2
        IN_BATTLE = 3

    class LowLevelActions(Enum):
        PRESS_ARROW_DOWN = "down"
        PRESS_ARROW_LEFT = "left"
        PRESS_ARROW_RIGHT = "right"
        PRESS_ARROW_UP = "up"
        PRESS_BUTTON_A = "a"
        PRESS_BUTTON_B = "b"
        PRESS_BUTTON_START = "start"

    class Environment:
        pass

    class HighLevelAction:
        pass

    class LowLevelAction(HighLevelAction):
        pass

    class WaitAction(HighLevelAction):
        pass

    class MoveStepsAction(HighLevelAction):
        pass

    class InteractAction(HighLevelAction):
        pass

    class PassDialogueAction(HighLevelAction):
        pass

    class BattleMenuAction(HighLevelAction):
        pass

    class PickAttackAction(HighLevelAction):
        pass

    class MenuAction(HighLevelAction):
        pass

    class OpenMenuAction(HighLevelAction):
        pass

    class StateParser:
        @staticmethod
        def capture_box(current_frame: np.ndarray, start_x: int, start_y: int, width: int,
                        height: int) -> np.ndarray:
            end_x, end_y = start_x + width, start_y + height
            start_x, start_y = max(0, start_x), max(0, start_y)
            end_x, end_y = min(current_frame.shape[1], end_x), min(current_frame.shape[0], end_y)
            return current_frame[start_y:end_y, start_x:end_x, :]

        @staticmethod
        def capture_grid_cells(current_frame: np.ndarray, *, quadrant: str = None, grid_skip: int = 16,
                               x_offset=0, y_offset=-2) -> Dict[Tuple[int, int], np.ndarray]:
            if quadrant is not None and quadrant.lower() not in ("tl", "tr", "bl", "br"):
                raise ValueError(f"Invalid quadrant: {quadrant}. Must be one of 'TL', 'TR', 'BL', 'BR'")
            x_iter = ([-x_offset] if x_offset != 0 else []) + list(range(0, current_frame.shape[1], grid_skip))
            y_iter = ([-y_offset] if y_offset != 0 else []) + list(range(0, current_frame.shape[0], grid_skip))
            cells = {}
            for x in x_iter:
                for y in y_iter:
                    x_cell = (x_iter.index(x) - len(x_iter) // 2) + 1 * (x_offset == 0)
                    y_cell = -(y_iter.index(y) - len(y_iter) // 2) + 1 * (y_offset == 0)
                    if quadrant is not None:
                        q = quadrant.lower()
                        if ((q == "tl" and (x_cell > 0 or y_cell < 0)) or (q == "tr" and (x_cell < 0 or y_cell < 0))
                                or (q == "bl" and (x_cell > 0 or y_cell > 0))
                                or (q == "br" and (x_cell < 0 or y_cell > 0))):
                            continue
                    cells[(x_cell, y_cell)] = StateParser.capture_box(current_frame, x + x_offset, y + y_offset,
                                                                      grid_skip, grid_skip)
            return cells

        @staticmethod
        def reform_image(grid_cells: Dict[Tuple[int, int], np.ndarray]) -> np.ndarray:
            if len(grid_cells) == 1:
                return list(grid_cells.values())[0]
            xs = sorted({coord[0] for coord in grid_cells})
            ys = sorted({coord[1] for coord in grid_cells})
            rows = [np.concatenate([grid_cells[(x, y)] for x in xs], axis=1) for y in ys]
            if len(rows) == 1:
                return rows[0]
            back_offset = 2 if rows[-1].shape != rows[-2].shape else 1
            return np.concatenate([rows[i] for i in range(len(rows) - back_offset, -1, -1)], axis=0)


ACTION_CLASSES = {cls.__name__: cls for cls in (LowLevelAction, WaitAction, MoveStepsAction, InteractAction,
                                                PassDialogueAction, BattleMenuAction, PickAttackAction,
                                                MenuAction, OpenMenuAction)}
