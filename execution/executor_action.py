from utils import object_detection
from typing import Tuple, List, Dict, Any, Optional, Callable
import numpy as np
from abc import ABC, abstractmethod
from gameboy_worlds.emulation.parser import StateParser


def _get_quadrants(
    grid_cells: Dict[Tuple[int, int], np.ndarray],
) -> Dict[str, Dict]:
    coords = grid_cells.keys()
    xs = sorted(set(c[0] for c in coords))
    ys = sorted(set(c[1] for c in coords))
    mid_x = xs[len(xs) // 2]
    mid_y = ys[len(ys) // 2]
    lower_x = [x for x in xs if x < mid_x]
    higher_x = [x for x in xs if x >= mid_x]
    lower_y = [y for y in ys if y < mid_y]
    higher_y = [y for y in ys if y >= mid_y]
    raw = {
        "tl": {(x, y): grid_cells[(x, y)] for x in lower_x for y in higher_y},
        "tr": {(x, y): grid_cells[(x, y)] for x in higher_x for y in higher_y},
        "bl": {(x, y): grid_cells[(x, y)] for x in lower_x for y in lower_y},
        "br": {(x, y): grid_cells[(x, y)] for x in higher_x for y in lower_y},
    }
    return {
        q: {"cells": cells, "screen": StateParser.reform_image(cells)}
        for q, cells in raw.items()
    }


def coord_to_string(coord: Tuple[int, int]) -> str:
    x, y = coord
    parts = []
    if x > 0:
        parts.append(f"{x} steps to right from you")
    elif x < 0:
        parts.append(f"{-x} steps to left from you")
    if y > 0:
        parts.append(f"{y} steps up from you")
    elif y < 0:
        parts.append(f"{-y} steps down from you")
    return "(" + ", ".join(parts) + ")"


def coords_to_string(coords: List[Tuple[int, int]]) -> str:
    return "[" + ", ".join(coord_to_string(c) for c in coords) + "]"


class ExecutorAction(ABC):
    """Passive action an Executor calls to read the game state; never calls ``emulator.step``."""

    def __init__(self, vlm_call: Callable[..., Any], parameters: Dict[str, Any]) -> None:
        self._vlm_call = vlm_call
        self._parameters = parameters

    @abstractmethod
    def _execute(self, info: Dict[str, Dict[str, Any]], **kwargs) -> Tuple[Dict[str, Any], int]:
        raise NotImplementedError

    def is_valid(self, info: Dict[str, Dict[str, Any]], **kwargs) -> Tuple[bool, Optional[str]]:
        return True, None

    @classmethod
    @abstractmethod
    def verbalize(cls) -> str:
        # The invocation syntax must not contain ``:``; use ``=`` for arguments.
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def string_to_kwargs(cls, action_str: str) -> Dict[str, Any]:
        raise NotImplementedError

    def execute(self, info: Dict[str, Dict[str, Any]], **kwargs) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
        valid, reason = self.is_valid(info, **kwargs)
        if not valid:
            return {"invalid": reason or "invalid call"}, None
        return self._execute(info, **kwargs)


class LocateAction(ExecutorAction):
    """Success codes: -1 not found, 0 one definitive, 1 many definitive, 2 one potential, 3 many potential."""

    pre_described_options: Dict[str, str] = {}
    """Maps known target names to VLM-ready descriptions. Subclasses override to add entries."""

    def coord_to_string(self, coord: Tuple[int, int]) -> str:
        return coord_to_string(coord)

    def coords_to_string(self, coords: List[Tuple[int, int]]) -> str:
        return coords_to_string(coords)

    def is_valid(self, info: Dict[str, Dict[str, Any]], target: str = None,
                 **kwargs) -> Tuple[bool, Optional[str]]:
        if target is None or (isinstance(target, str) and target.strip()):
            return True, None
        return False, f"target must be a non-empty string, got {target!r}"

    def check_for_target(self, description: str, screens: List[np.ndarray]) -> List[bool]:
        return object_detection(description=description, images=screens)

    def get_centroid(
        self, cells: Dict[Tuple[int, int], np.ndarray]
    ) -> Tuple[float, float]:
        xs = [coord[0] for coord in cells.keys()]
        ys = [coord[1] for coord in cells.keys()]
        return ((min(xs) + max(xs)) // 2, (min(ys) + max(ys)) // 2)

    def get_cells_found(
        self,
        grid_cells: Dict[Tuple[int, int], np.ndarray],
        description: str,
    ) -> Tuple[bool, List[Tuple[int, int]], List[Tuple[int, int]]]:
        quadrant_keys = ["tl", "tr", "bl", "br"]
        if len(grid_cells) == 1:
            screen = list(grid_cells.values())[0]
            if self.check_for_target(description, [screen])[0]:
                return True, list(grid_cells.keys()), list(grid_cells.keys())
            return False, [], []

        quadrants = _get_quadrants(grid_cells)
        screens = [quadrants[q]["screen"] for q in quadrant_keys]
        quadrant_founds = self.check_for_target(description, screens)

        if not any(quadrant_founds):
            return False, [], []

        potential_cells = []
        quadrant_definites = []
        for i, quadrant in enumerate(quadrant_keys):
            if not quadrant_founds[i]:
                continue
            cells = quadrants[quadrant]["cells"]
            if len(cells) < 4:
                potential_cells.append(self.get_centroid(cells))
                cell_keys = list(cells.keys())
                cell_screens = [cells[k] for k in cell_keys]
                for j, found in enumerate(self.check_for_target(description, cell_screens)):
                    if found:
                        quadrant_definites.append(cell_keys[j])
            else:
                found_in_quadrant, quadrant_potentials, recursive_definites = self.get_cells_found(
                    cells, description
                )
                quadrant_definites.extend(recursive_definites)
                if found_in_quadrant:
                    if quadrant_potentials:
                        potential_cells.extend(quadrant_potentials)
                    else:
                        potential_cells.append(self.get_centroid(cells))

        return True, potential_cells, quadrant_definites

    def do_location(
        self, info: Dict[str, Any], description: str
    ) -> Tuple[Dict[str, Any], int]:
        frame = info["core"]["current_frame"]
        grid_cells = StateParser.capture_grid_cells(frame)
        found, potential_cells, definitive_cells = self.get_cells_found(grid_cells, description)
        ret_dict = {
            "found": found,
            "potential_cells": potential_cells,
            "definitive_cells": definitive_cells,
            "potential_cells_str": self.coords_to_string(potential_cells),
            "definitive_cells_str": self.coords_to_string(definitive_cells),
        }
        if not found:
            action_success = -1
        elif not definitive_cells:
            if len(potential_cells) == 1:
                action_success = 2
            elif len(potential_cells) > 1:
                action_success = 3
            else:
                action_success = -1
        else:
            action_success = 0 if len(definitive_cells) == 1 else 1
        return ret_dict, action_success

    def _execute(self, info: Dict[str, Dict[str, Any]], target: str, **kwargs) -> Tuple[Dict[str, Any], int]:
        key = target.lower().strip()
        description = self.pre_described_options.get(key, target)
        return self.do_location(info, description)
