from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from execution.game_api import (AgentState, BattleMenuAction, InteractAction, MenuAction, MoveStepsAction,
                                OpenMenuAction, PassDialogueAction, PickAttackAction)

from execution.executors import PolicyExecutor
from execution.perception.pokemon.navigation import (DIRECTIONS, WALKABLE, approach_cells,
                                                     compress_path, edge_cells, in_bounds,
                                                     parse_cell, shortest_path, verbalize_edges)
from execution.pokemon.executor_actions import CheckInteractionAction
from execution.report import EnvironmentStepRecord
from utils import log_error, ocr, parse_int, parse_key_value, parse_yes_no


DIALOGUE_BACKGROUND = 248
CURSOR_MARGIN = 12


def is_prefix_crop(earlier: np.ndarray, later: np.ndarray) -> bool:
    if earlier.shape != later.shape:
        return False
    a, b = earlier[:, :-CURSOR_MARGIN], later[:, :-CURSOR_MARGIN]
    differ = a != b
    return bool(np.all(a[differ] >= DIALOGUE_BACKGROUND))


def captured_dialogue_crops(records: List[EnvironmentStepRecord]) -> List[Any]:
    crops = []
    for record in records:
        for state in record.transition_states:
            step_crops = state.get("ocr", {}).get("ocr_regions", {}).get("dialogue")
            if step_crops is None or len(step_crops) == 0:
                continue
            crops.extend(step_crops)
    return crops


def drop_prefix_crops(crops: List[Any], kept: Optional[List[Any]] = None) -> List[Any]:
    kept = [] if kept is None else kept
    for crop in crops:
        if kept and is_prefix_crop(kept[-1], crop):
            kept[-1] = crop
        else:
            kept.append(crop)
    return kept


class InteractionExecutor(PolicyExecutor):
    CARDINAL_DIRECTIONS = {cell: name for name, cell in CheckInteractionAction.cardinals.items()}

    def __init__(self, env, task, max_steps: int = 3, identified=None, **kwargs):
        self.identified = identified
        self.check_result: Optional[dict] = None
        self.dialogue_text: Optional[str] = None
        self.dialogue_crops: List[Any] = []
        self.captured_crops: List[Any] = []
        super().__init__(env, task, max_steps, **kwargs)

    def _fail(self, reason: str, notes: Optional[str] = None) -> int:
        self.report.termination_reason = reason
        self.report.notes = notes
        return -1

    def _execute(self) -> int:
        self._last_terminated = False
        self._last_truncated = False
        info = self._get_state()
        if info["pokemon_core"]["agent_state"] != AgentState.FREE_ROAM:
            return self._fail("not_free_roam")

        check = CheckInteractionAction(self._vlm_call, self._parameters)
        result, code = check.execute(info, target=self._task, identified=self.identified)
        self.check_result = result
        if code is None:
            return self._fail("invalid_check", result["invalid"])
        if code == -1:
            return self._fail("target_not_found")
        if code == 1:
            return self._fail("target_only_diagonal",
                              f"Target found at {result['found_cells_str']}, which cannot be faced")

        direction = result["direction"]
        records = []
        self._begin_scripted(f"face {direction} and interact with {self._task}")
        turn = self._take_action(MoveStepsAction, direction=direction, steps=1)
        records.append(turn)
        if self._last_terminated or self._last_truncated:
            return self._finish(records)
        action_return = turn.transition_states[-1]["core"].get("action_return") or {}
        if action_return.get("n_steps_taken"):
            return self._fail("target_cell_walkable",
                              f"Pressed {direction} to face the target at {result['found_cells_str']}, "
                              "but the player walked onto that cell instead, so nothing is there")

        interact = self._take_action(InteractAction)
        records.append(interact)
        if interact.action_success != 1:
            return self._fail("interaction_did_nothing")

        in_dialogue = self._get_state()["pokemon_core"]["agent_state"] == AgentState.IN_DIALOGUE
        if in_dialogue and not (self._last_terminated or self._last_truncated):
            records.append(self._take_action(PassDialogueAction))
        return self._finish(records)

    def _finish(self, records: List[EnvironmentStepRecord]) -> int:
        self.dialogue_text = self._read_dialogue(records)
        self.report.notes = self.dialogue_text
        final_state = self._get_state()["pokemon_core"]["agent_state"]
        if self._last_terminated:
            self.report.termination_reason = "terminated"
            return 1
        if self._last_truncated:
            self.report.termination_reason = "truncated"
            return 2
        if final_state == AgentState.IN_DIALOGUE:
            self.report.termination_reason = "dialogue_not_cleared"
        elif final_state == AgentState.FREE_ROAM:
            self.report.termination_reason = "dialogue_finished"
        else:
            self.report.termination_reason = f"dialogue_ended_in_{final_state.name.lower()}"
        return 0

    def _read_dialogue(self, records: List[EnvironmentStepRecord]) -> str:
        self.captured_crops.extend(captured_dialogue_crops(records))
        frames = drop_prefix_crops(self.captured_crops)
        self.dialogue_crops = frames
        if not frames:
            return ""
        return "\n".join(ocr(frames, self._vlm, do_merge=True, parameters=self._parameters))


MOVE_OK = 0
MOVE_TERMINATED = 1
MOVE_TRUNCATED = 2
MOVE_NOT_FREE_ROAM = -1
MOVE_TARGET_NOT_FOUND = -2
MOVE_TARGET_AMBIGUOUS = -3
MOVE_BAD_LOCATION = -4
MOVE_NO_PATH = -5
MOVE_BLOCKED = -6
MOVE_INTERRUPTED = -7
MOVE_OUT_OF_STEPS = -8
LOCATION_FAIL_CODES = (MOVE_TARGET_NOT_FOUND, MOVE_TARGET_AMBIGUOUS, MOVE_BAD_LOCATION)

GRID_DESCRIPTION = """The screen is a grid of cells. The player is at (0, 0). x increases to the right, from -4 at the left edge to 5 at the right edge. y increases upward, from 4 at the top edge to -4 at the bottom edge."""


class GridMoveExecutor(PolicyExecutor):
    LOCATE_TAG = "locate"
    MAX_NEW_TOKENS = 4000

    def __init__(self, env, task, max_steps: int = 10, screen_tiles: Optional[str] = None,
                 cell_kinds: Optional[Dict[Tuple[int, int], str]] = None, **kwargs):
        self.screen_tiles = screen_tiles
        self.cell_kinds = cell_kinds
        self.target_cell: Optional[Tuple[int, int]] = None
        self.plan: Optional[List[Tuple[str, int, str]]] = None
        self._n_moves = 0
        super().__init__(env, task, max_steps, screen_tiles=screen_tiles, cell_kinds=cell_kinds, **kwargs)

    def _fail(self, code: int, reason: str, notes: Optional[str] = None) -> int:
        self.report.termination_reason = reason
        self.report.notes = notes
        return code

    def _locate(self, prompt: str) -> str:
        frame = self._get_state()["core"]["current_frame"]
        return self._vlm_call(self.LOCATE_TAG, texts=prompt, images=[frame], max_new_tokens=self.MAX_NEW_TOKENS)

    def _check_inputs(self) -> Optional[int]:
        if self.screen_tiles is None or self.cell_kinds is None:
            log_error(f"{self.__class__.__name__} needs screen_tiles and cell_kinds from the supervisor.",
                      self._parameters)
        if self._get_state()["pokemon_core"]["agent_state"] != AgentState.FREE_ROAM:
            return self._fail(MOVE_NOT_FREE_ROAM, "not_free_roam")
        return None

    def _step(self, direction: str, steps: int) -> Optional[int]:
        remaining = steps
        attempts = 0
        while remaining > 0:
            if attempts >= steps + 2:
                return self._fail(MOVE_BLOCKED, "move_blocked", f"Could not finish moving {direction} {steps}")
            if self._n_moves >= self._max_steps:
                return self._fail(MOVE_OUT_OF_STEPS, "out_of_steps")
            attempts += 1
            self._n_moves += 1
            record = self._take_action(MoveStepsAction, direction=direction, steps=remaining)
            if self._last_terminated:
                return self._fail(MOVE_TERMINATED, "terminated")
            if self._last_truncated:
                return self._fail(MOVE_TRUNCATED, "truncated")
            action_return = (record.transition_states[-1]["core"].get("action_return") or {}) if record.transition_states else {}
            taken = action_return.get("n_steps_taken", 0)
            if record.action_success == 2:
                return self._fail(MOVE_INTERRUPTED, "move_interrupted",
                                  f"Left free roam after {taken} steps {direction}: "
                                  f"{self._get_state()['pokemon_core']['agent_state'].name}")
            remaining -= taken
            if taken == 0 and not action_return.get("rotated"):
                return self._fail(MOVE_BLOCKED, "move_blocked", f"Blocked moving {direction} with {remaining} steps left")
            if record.action_success in (-1, 1) and remaining > 0 and taken > 0:
                return self._fail(MOVE_BLOCKED, "move_blocked", f"Stopped moving {direction} with {remaining} steps left")
        return None

    def _walk(self, goals: Dict[Tuple[int, int], Optional[str]]) -> Optional[int]:
        path = shortest_path(self.cell_kinds, (0, 0), goals)
        if path is None:
            return self._fail(MOVE_NO_PATH, "no_path", f"No walkable path from (0, 0) to any of {sorted(goals)}")
        self.plan = compress_path(path)
        end = path[-1][2] if path else (0, 0)
        facing = goals[end]
        description = ", ".join(f"{m} {d} {n}" for d, n, m in self.plan) or "stay"
        self._begin_scripted(f"walk to {end} ({description})" + (f" and face {facing}" if facing else ""))
        for direction, n_steps, move in self.plan:
            code = self._step(direction, n_steps)
            if code is not None:
                return code
        if facing is not None:
            code = self._step(facing, 1)
            if code is not None and code != MOVE_BLOCKED:
                return code
        return None


class OnScreenMoveExecutor(GridMoveExecutor):
    LOCATE_TAG = "locate_target"
    LOCATE_PROMPT = """You are helping a player move to something on the screen in a Game Boy Pokemon game.

The player wants to move to: "[TARGET]"

The image is the current game screen. [GRID_DESCRIPTION] Here is what has been recognised on the screen:
[SCREEN_TILES]

Find the exact cell of the target.
- If exactly one thing on the screen matches the target, give its coordinates. If it covers several cells, give the one closest to the player.
- If more than one thing on the screen could be the target, answer Ambiguous and list every candidate.
- If nothing on the screen matches the target, answer Not Found.

Respond in exactly this format:
Reasoning: <what matches the target and where>
Location: <(x, y), Ambiguous, or Not Found>
Candidates: <if Ambiguous, each candidate with its coordinates and what sets it apart from the others; otherwise None>
[STOP]"""

    def _execute(self) -> int:
        code = self._check_inputs()
        if code is not None:
            return code
        prompt = (self.LOCATE_PROMPT.replace("[TARGET]", self._task.strip())
                  .replace("[GRID_DESCRIPTION]", GRID_DESCRIPTION)
                  .replace("[SCREEN_TILES]", self.screen_tiles))
        output = self._locate(prompt)
        location = (parse_key_value(output, "Location") or "").strip()
        lowered = location.lower()
        if "ambiguous" in lowered:
            return self._fail(MOVE_TARGET_AMBIGUOUS, "target_ambiguous", parse_key_value(output, "Candidates"))
        if "not found" in lowered:
            return self._fail(MOVE_TARGET_NOT_FOUND, "target_not_found", parse_key_value(output, "Reasoning"))
        target = parse_cell(location)
        if target is None or not in_bounds(target):
            return self._fail(MOVE_BAD_LOCATION, "bad_location", f"Unusable location {location!r}")
        self.target_cell = target
        if target == (0, 0):
            return MOVE_OK
        if self.cell_kinds.get(target) == WALKABLE:
            goals = {target: None}
        else:
            goals = approach_cells(self.cell_kinds, target)
            if not goals:
                return self._fail(MOVE_NO_PATH, "no_path", f"Nothing walkable next to the target at {target}")
        code = self._walk(goals)
        if code is not None:
            return code
        self.report.termination_reason = "arrived"
        self.report.notes = f"Moved to the target at {target}"
        return MOVE_OK


class OffScreenMoveExecutor(GridMoveExecutor):
    LOCATE_TAG = "locate_edge"
    LOCATE_PROMPT = """You are helping a player walk off the edge of the screen in a Game Boy Pokemon game.

The player wants to: "[TARGET]"

The image is the current game screen. [GRID_DESCRIPTION] Here is what has been recognised on the screen:
[SCREEN_TILES]

These cells on each edge of the screen can be walked on:
[EDGE_CELLS]

Choose the edge cell the player should walk to so that they can then keep walking off the screen as described. Pick a cell from the list above on the edge in the direction the player wants to go. If no listed cell fits the description, answer Not Found.

Respond in exactly this format:
Reasoning: <which edge and which cell, and why>
Direction: <up, down, left or right>
Location: <(x, y) or Not Found>
[STOP]"""

    def _execute(self) -> int:
        code = self._check_inputs()
        if code is not None:
            return code
        prompt = (self.LOCATE_PROMPT.replace("[TARGET]", self._task.strip())
                  .replace("[GRID_DESCRIPTION]", GRID_DESCRIPTION)
                  .replace("[SCREEN_TILES]", self.screen_tiles)
                  .replace("[EDGE_CELLS]", verbalize_edges(self.cell_kinds)))
        output = self._locate(prompt)
        location = (parse_key_value(output, "Location") or "").strip()
        direction = (parse_key_value(output, "Direction") or "").strip().lower()
        direction_aliases = {"north": "up", "south": "down", "west": "left", "east": "right"}
        direction = direction_aliases.get(direction, direction)
        if "not found" in location.lower():
            return self._fail(MOVE_TARGET_NOT_FOUND, "edge_not_found", parse_key_value(output, "Reasoning"))
        target = parse_cell(location)
        if direction not in DIRECTIONS:
            return self._fail(MOVE_BAD_LOCATION, "bad_direction", f"Unusable direction {direction!r}")
        cells = edge_cells(self.cell_kinds, direction)
        if target is not None and target not in cells and cells:
            target = min(cells, key=lambda c: (c[0] - target[0]) ** 2 + (c[1] - target[1]) ** 2)
        if target is None or target not in cells:
            return self._fail(MOVE_BAD_LOCATION, "bad_location",
                              f"{location!r} is not a walkable cell on the {direction} edge")
        self.target_cell = target
        code = self._walk({target: None})
        if code is not None:
            return code
        code = self._step(direction, 1)
        if code is not None:
            return code
        self.report.termination_reason = "left_screen"
        self.report.notes = f"Walked to {target} and stepped off the screen {direction}"
        return MOVE_OK


BATTLE_ENDED = 0
BATTLE_TERMINATED = 1
BATTLE_TRUNCATED = 2
BATTLE_LEFT_TO_DIALOGUE = 3
BATTLE_LEFT_TO_MENU = 4
BATTLE_NOT_IN_BATTLE = -1
BATTLE_UNREADABLE = -2
BATTLE_ACTION_FAILED = -3
BATTLE_OUT_OF_STEPS = -4

FREE_ROAM_CHECK_PRESSES = 5
AUTO_PROGRESS_PRESSES = 4
BATTLE_OPTIONS = ("fight", "bag", "pokemon", "run", "progress")
LEFT_BATTLE_CODES = {
    AgentState.FREE_ROAM: BATTLE_ENDED,
    AgentState.IN_DIALOGUE: BATTLE_LEFT_TO_DIALOGUE,
    AgentState.IN_MENU: BATTLE_LEFT_TO_MENU,
}


class CombatExecutor(PolicyExecutor):
    MAX_NEW_TOKENS = 4000
    MAX_INVALID = 3
    STEP_TAG = "battle_step"

    STEP_PROMPT = """You are playing a Game Boy Pokemon game and are in a battle.

[INSTRUCTIONS]
The image is the current battle screen.
[HISTORY]
Choose what to do next:
- fight: attack, and say which of the four moves to use (1 is the top one)
- progress: press through battle text, or continue when there is no choice to make
- run: try to escape. This never works against another trainer
- bag: open the bag to use an item. This hands over to the menu system, so only choose it if an item is needed
- pokemon: open your team to switch. This hands over to the menu system, so only choose it if switching is needed

Respond in exactly this format:
Reasoning: <what the screen shows and what to do about it>
Action: <one of fight, progress, run, bag, pokemon>
Move: <1, 2, 3 or 4 if the action is fight, otherwise None>
[STOP]"""

    MOVE_TAG = "battle_move"
    MOVE_ATTEMPTS = 3
    MAX_PREVIOUS_CHARS = 1500

    MOVE_PROMPT = """You are playing a Game Boy Pokemon game and chose to fight in a battle.

[INSTRUCTIONS]
The image is the current battle screen, showing your moves.
[HISTORY]
Your previous reply was:
[PREVIOUS]

That reply could not be used: it did not give the number of a move. Choose which of the four moves to use, where 1 is the top one.

Respond in exactly this format:
Reasoning: <which move to use and why>
Move: <1, 2, 3 or 4>
[STOP]"""

    def __init__(self, env, task, max_steps: int = 60, **kwargs):
        self.history: List[str] = []
        self.dialogue_crops: List[Any] = []
        self.dialogue_text: List[str] = []
        super().__init__(env, task, max_steps, **kwargs)

    def _fail(self, code: int, reason: str, notes: Optional[str] = None) -> int:
        self.report.termination_reason = reason
        self.report.notes = notes
        return code

    def _agent_state(self):
        return self._get_state()["pokemon_core"]["agent_state"]

    def _read_new_text(self, record: EnvironmentStepRecord) -> List[str]:
        before = len(self.dialogue_crops)
        last_before = self.dialogue_crops[-1] if self.dialogue_crops else None
        drop_prefix_crops(captured_dialogue_crops([record]), self.dialogue_crops)
        start = before
        if last_before is not None and self.dialogue_crops[before - 1] is not last_before:
            start = before - 1
        fresh = self.dialogue_crops[start:]
        if not fresh:
            return []
        texts = ocr(fresh, self._vlm, do_merge=True, parameters=self._parameters)
        self.dialogue_text.extend(texts)
        return texts

    def _history_block(self) -> str:
        if not self.history:
            return ""
        return "\nWhat has happened in this battle so far:\n" + "\n".join(self.history) + "\n"

    def _instructions_block(self) -> str:
        task = (self._task or "").strip()
        if not task or task.lower() in ("win the current battle", "none"):
            return "Win this battle.\n"
        return f"Your instructions for this battle: \"{task}\"\n"

    def _choose_move(self, output: str) -> Optional[int]:
        move = parse_int(output, "Move", lo=1, hi=4)
        for _ in range(self.MOVE_ATTEMPTS):
            if move is not None:
                return move
            self._record_invalid(output, reason="no move number")
            frame = self._get_state()["core"]["current_frame"]
            prompt = (self.MOVE_PROMPT.replace("[INSTRUCTIONS]", self._instructions_block())
                      .replace("[HISTORY]", self._history_block())
                      .replace("[PREVIOUS]", output[:self.MAX_PREVIOUS_CHARS]))
            output = self._vlm_call(self.MOVE_TAG, texts=prompt, images=[frame],
                                    max_new_tokens=self.MAX_NEW_TOKENS)
            move = parse_int(output, "Move", lo=1, hi=4)
        if move is None:
            self._record_invalid(output, reason="no move number")
        return move

    def _record_outcome(self, state) -> int:
        code = LEFT_BATTLE_CODES.get(state, BATTLE_ENDED)
        reason = {
            BATTLE_ENDED: "battle_ended",
            BATTLE_LEFT_TO_DIALOGUE: "left_battle_to_dialogue",
            BATTLE_LEFT_TO_MENU: "left_battle_to_menu",
        }[code]
        self.report.termination_reason = reason
        self.report.notes = "\n".join(self.dialogue_text)
        return code

    def _execute(self) -> int:
        self._last_terminated = False
        self._last_truncated = False
        if self._agent_state() != AgentState.IN_BATTLE:
            return self._fail(BATTLE_NOT_IN_BATTLE, "not_in_battle",
                              f"The player is {self._agent_state().name.lower()}, not in a battle.")

        consecutive_invalid = 0
        consecutive_failures = 0
        for step in range(self._max_steps):
            state = self._agent_state()
            if state == AgentState.FREE_ROAM:
                texts = []
                for _ in range(FREE_ROAM_CHECK_PRESSES):
                    record = self._take_action_string("button(b)")
                    if self._last_terminated:
                        return self._fail(BATTLE_TERMINATED, "terminated")
                    if self._last_truncated:
                        return self._fail(BATTLE_TRUNCATED, "truncated")
                    texts.extend(self._read_new_text(record))
                state = self._agent_state()
                entry = f"- step {step + 1}: the screen did not look like a battle, pressed B {FREE_ROAM_CHECK_PRESSES} times"
                if texts:
                    entry += "\n  text on screen: " + " / ".join(texts)
                self.history.append(entry)
                if state != AgentState.IN_BATTLE:
                    return self._record_outcome(state)
                continue
            if state != AgentState.IN_BATTLE:
                return self._record_outcome(state)

            frame = self._get_state()["core"]["current_frame"]
            prompt = (self.STEP_PROMPT.replace("[INSTRUCTIONS]", self._instructions_block())
                      .replace("[HISTORY]", self._history_block()))
            output = self._vlm_call(self.STEP_TAG, texts=prompt, images=[frame],
                                    max_new_tokens=self.MAX_NEW_TOKENS)
            option = (parse_key_value(output, "Action") or "").strip().lower()
            option = next((name for name in BATTLE_OPTIONS if name in option), None)
            if option is None:
                self._record_invalid(output)
                consecutive_invalid += 1
                if consecutive_invalid >= self.MAX_INVALID:
                    return self._fail(BATTLE_UNREADABLE, "action_unreadable", output)
                continue
            consecutive_invalid = 0

            record = self._take_action(BattleMenuAction, option=option)
            if self._last_terminated:
                return self._fail(BATTLE_TERMINATED, "terminated")
            if self._last_truncated:
                return self._fail(BATTLE_TRUNCATED, "truncated")
            entry = f"- step {step + 1}: chose {option}"
            if record.action_success in (1, 2):
                entry += " (could not escape)"
            elif record.action_success == -1:
                entry += " (the game did not accept it)"
                consecutive_failures += 1
            else:
                consecutive_failures = 0

            if option == "fight" and record.action_success == 0:
                move = self._choose_move(output)
                if move is None:
                    entry += ", but no move number was given"
                else:
                    attack = self._take_action(PickAttackAction, option=move)
                    entry += f", used move {move}"
                    if attack.action_success == 1:
                        entry += " (no PP left)"
                    elif attack.action_success == -1:
                        entry += " (the game did not accept it)"
                    if self._last_terminated:
                        return self._fail(BATTLE_TERMINATED, "terminated")
                    if self._last_truncated:
                        return self._fail(BATTLE_TRUNCATED, "truncated")
                    record = attack

            texts = self._read_new_text(record)
            if option == "progress" and texts:
                auto = 0
                while auto < AUTO_PROGRESS_PRESSES and self._agent_state() == AgentState.IN_BATTLE:
                    extra = self._take_action(BattleMenuAction, option="progress")
                    if self._last_terminated:
                        return self._fail(BATTLE_TERMINATED, "terminated")
                    if self._last_truncated:
                        return self._fail(BATTLE_TRUNCATED, "truncated")
                    if extra.action_success == -1:
                        break
                    auto += 1
                    more = self._read_new_text(extra)
                    texts.extend(more)
                    if not more:
                        break
                if auto:
                    entry += f", then pressed progress {auto} more time{'s' if auto > 1 else ''}"
            if texts:
                entry += "\n  text on screen: " + " / ".join(texts)
            self.history.append(entry)
            if consecutive_failures >= self.MAX_INVALID:
                return self._fail(BATTLE_ACTION_FAILED, "actions_not_accepted", self._history_block())

        return self._fail(BATTLE_OUT_OF_STEPS, "out_of_steps", self._history_block())


MENU_OK = 0
MENU_TERMINATED = 1
MENU_TRUNCATED = 2
MENU_NOT_IN_MENU = -1
MENU_OPEN_FAILED = -2
MENU_UNREADABLE = -3
MENU_OUT_OF_STEPS = -4

MENU_PRESSES = ("up", "down", "left", "right", "confirm", "back")
MENU_STATES = (AgentState.IN_MENU, AgentState.IN_BATTLE, AgentState.IN_DIALOGUE)


class MenuExecutor(PolicyExecutor):
    MAX_NEW_TOKENS = 4000
    MAX_INVALID = 3
    OPEN_TAG = "menu_open"
    STEP_TAG = "menu_step"

    OPEN_PROMPT = """You are playing a Game Boy Pokemon game and are about to open the main menu with the Start button.

Objective: "[TASK]"

The image is the current game screen.

Which entry of the main menu should be opened first?
- pokedex: the Pokedex
- pokemon: your Pokemon team, for checking, reordering, healing or using their moves outside battle
- bag: your items
- trainer: your trainer card

Respond in exactly this format:
Reasoning: <which entry the objective needs and why>
Entry: <one of pokedex, pokemon, bag, trainer>
[STOP]"""

    STEP_PROMPT = """You are playing a Game Boy Pokemon game and are using the game's menus.

Objective: "[TASK]"

The image is the current screen.
[HISTORY]
First decide whether the objective has already been achieved on this screen. If it has not, choose the single button press that best moves towards it.

The presses you can make:
- up, down, left, right: move the cursor
- confirm: press A to choose what the cursor is on
- back: press B to go back, or to close the current menu

Respond in exactly this format:
Reasoning: <what is on the screen, and what should be pressed next>
Complete: <yes or no>
Press: <one of up, down, left, right, confirm, back>
[STOP]"""

    def __init__(self, env, task, max_steps: int = 15, is_start_menu: bool = False, **kwargs):
        self.is_start_menu = is_start_menu
        self.presses: List[str] = []
        super().__init__(env, task, max_steps, is_start_menu=is_start_menu, **kwargs)

    def _fail(self, code: int, reason: str, notes: Optional[str] = None) -> int:
        self.report.termination_reason = reason
        self.report.notes = notes
        return code

    def _agent_state(self):
        return self._get_state()["pokemon_core"]["agent_state"]

    def _open_main_menu(self) -> Optional[int]:
        frame = self._get_state()["core"]["current_frame"]
        output = self._vlm_call(self.OPEN_TAG, texts=self.OPEN_PROMPT.replace("[TASK]", self._task.strip()),
                                images=[frame], max_new_tokens=self.MAX_NEW_TOKENS)
        entry = (parse_key_value(output, "Entry") or "").strip().lower()
        entry = next((option for option in OpenMenuAction.options if option in entry), None)
        if entry is None:
            return self._fail(MENU_OPEN_FAILED, "menu_entry_unreadable", output)
        record = self._take_action(OpenMenuAction, option=entry)
        if self._last_terminated:
            return self._fail(MENU_TERMINATED, "terminated")
        if self._last_truncated:
            return self._fail(MENU_TRUNCATED, "truncated")
        if record.action_success != 0 or self._agent_state() not in MENU_STATES:
            return self._fail(MENU_OPEN_FAILED, "menu_did_not_open",
                              f"Tried to open {entry}, success={record.action_success}, state={self._agent_state().name}")
        return None

    def _history_block(self) -> str:
        if not self.presses:
            return ""
        return f"\nPresses made so far, in order: {', '.join(self.presses)}.\n"

    def _execute(self) -> int:
        self._last_terminated = False
        self._last_truncated = False
        state = self._agent_state()
        if state not in MENU_STATES:
            if not self.is_start_menu:
                return self._fail(MENU_NOT_IN_MENU, "not_in_menu",
                                  f"The objective needs a menu that is already open, but the player is {state.name.lower()}.")
            if state != AgentState.FREE_ROAM:
                return self._fail(MENU_NOT_IN_MENU, "not_free_roam",
                                  f"Cannot open the main menu while {state.name.lower()}.")
            self._begin_scripted(f"open the main menu for: {self._task}")
            code = self._open_main_menu()
            if code is not None:
                return code

        consecutive_invalid = 0
        for _ in range(self._max_steps):
            frame = self._get_state()["core"]["current_frame"]
            prompt = (self.STEP_PROMPT.replace("[TASK]", self._task.strip())
                      .replace("[HISTORY]", self._history_block()))
            output = self._vlm_call(self.STEP_TAG, texts=prompt, images=[frame],
                                    max_new_tokens=self.MAX_NEW_TOKENS)
            if parse_yes_no(output, "Complete"):
                self.report.termination_reason = "objective_complete"
                self.report.notes = f"Presses: {', '.join(self.presses) or 'none'}"
                return MENU_OK
            press = (parse_key_value(output, "Press") or "").strip().lower()
            press = next((option for option in MENU_PRESSES if option in press), None)
            if press is None:
                self._record_invalid(output)
                consecutive_invalid += 1
                if consecutive_invalid >= self.MAX_INVALID:
                    return self._fail(MENU_UNREADABLE, "press_unreadable", output)
                continue
            consecutive_invalid = 0
            self.presses.append(press)
            self._take_action(MenuAction, menu_action=press)
            if self._last_terminated:
                return self._fail(MENU_TERMINATED, "terminated")
            if self._last_truncated:
                return self._fail(MENU_TRUNCATED, "truncated")
        return self._fail(MENU_OUT_OF_STEPS, "out_of_steps", f"Presses: {', '.join(self.presses) or 'none'}")


DIALOGUE_CLEARED = 0
DIALOGUE_TERMINATED = 1
DIALOGUE_TRUNCATED = 2
DIALOGUE_LEFT_TO_MENU = 3
DIALOGUE_LEFT_TO_BATTLE = 4
DIALOGUE_NOT_IN_DIALOGUE = -1
DIALOGUE_STUCK = -2

LEFT_DIALOGUE_CODES = {
    AgentState.FREE_ROAM: DIALOGUE_CLEARED,
    AgentState.IN_MENU: DIALOGUE_LEFT_TO_MENU,
    AgentState.IN_BATTLE: DIALOGUE_LEFT_TO_BATTLE,
}


class DialogueExecutor(PolicyExecutor):
    def __init__(self, env, task, max_steps: int = 10, **kwargs):
        self.dialogue_crops: List[Any] = []
        self.dialogue_text: str = ""
        super().__init__(env, task, max_steps, **kwargs)

    def _fail(self, code: int, reason: str, notes: Optional[str] = None) -> int:
        self.report.termination_reason = reason
        self.report.notes = notes
        return code

    def _agent_state(self):
        return self._get_state()["pokemon_core"]["agent_state"]

    def _finish(self, code: int, reason: str) -> int:
        if self.dialogue_crops:
            self.dialogue_text = "\n".join(ocr(self.dialogue_crops, self._vlm, do_merge=True,
                                               parameters=self._parameters))
        self.report.termination_reason = reason
        self.report.notes = self.dialogue_text
        return code

    def _execute(self) -> int:
        self._last_terminated = False
        self._last_truncated = False
        state = self._agent_state()
        if state != AgentState.IN_DIALOGUE:
            return self._fail(DIALOGUE_NOT_IN_DIALOGUE, "not_in_dialogue",
                              f"The player is {state.name.lower()}, not in dialogue.")

        self._begin_scripted(f"pass the dialogue: {self._task}")
        for _ in range(self._max_steps):
            record = self._take_action(PassDialogueAction)
            drop_prefix_crops(captured_dialogue_crops([record]), self.dialogue_crops)
            if self._last_terminated:
                return self._finish(DIALOGUE_TERMINATED, "terminated")
            if self._last_truncated:
                return self._finish(DIALOGUE_TRUNCATED, "truncated")
            state = self._agent_state()
            if state != AgentState.IN_DIALOGUE:
                code = LEFT_DIALOGUE_CODES.get(state, DIALOGUE_CLEARED)
                return self._finish(code, {
                    DIALOGUE_CLEARED: "dialogue_cleared",
                    DIALOGUE_LEFT_TO_MENU: "left_dialogue_to_menu",
                    DIALOGUE_LEFT_TO_BATTLE: "left_dialogue_to_battle",
                }[code])
            if record.action_success == -1:
                return self._finish(DIALOGUE_STUCK, "dialogue_stuck")
        return self._finish(DIALOGUE_STUCK, "out_of_steps")


FREE_OK = 0
FREE_TERMINATED = 1
FREE_TRUNCATED = 2
FREE_UNREADABLE = -1
FREE_OUT_OF_STEPS = -2


class FreeExecutor(PolicyExecutor):
    MAX_NEW_TOKENS = 4000
    MAX_INVALID = 3
    STEP_TAG = "free_step"

    STEP_PROMPT = """You are playing a Game Boy Pokemon game, pressing one button at a time.

What you have been asked to do: "[TASK]"

Stop when: [TERMINATION]

The image is the current screen.
[HISTORY]
First decide, from this screen, whether the condition to stop at has been met. If it has not, choose the single button press that best moves towards it.

The buttons are: a, b, start, up, down, left, right.

Respond in exactly this format:
Reasoning: <what is on the screen, and whether the stopping condition is met>
Complete: <yes or no>
Press: <one of a, b, start, up, down, left, right>
[STOP]"""

    DEFAULT_TERMINATION = "what you were asked to do has visibly happened on screen"

    def __init__(self, env, task, max_steps: int = 15, termination: Optional[str] = None, **kwargs):
        self.presses: List[str] = []
        self.dialogue_crops: List[Any] = []
        self.dialogue_text: str = ""
        self.termination = (termination or "").strip() or self.DEFAULT_TERMINATION
        super().__init__(env, task, max_steps, termination=self.termination, **kwargs)

    def _fail(self, code: int, reason: str, notes: Optional[str] = None) -> int:
        self.report.termination_reason = reason
        self.report.notes = notes
        return code

    def _history_block(self) -> str:
        if not self.presses:
            return ""
        return f"\nButtons pressed so far, in order: {', '.join(self.presses)}.\n"

    def _finish(self, code: int, reason: str) -> int:
        if self.dialogue_crops:
            self.dialogue_text = "\n".join(ocr(self.dialogue_crops, self._vlm, do_merge=True,
                                               parameters=self._parameters))
        self.report.termination_reason = reason
        self.report.notes = "\n".join(filter(None, [f"Presses: {', '.join(self.presses) or 'none'}",
                                                    self.dialogue_text]))
        return code

    def _execute(self) -> int:
        self._last_terminated = False
        self._last_truncated = False
        consecutive_invalid = 0
        for _ in range(self._max_steps):
            frame = self._get_state()["core"]["current_frame"]
            prompt = (self.STEP_PROMPT.replace("[TASK]", self._task.strip())
                      .replace("[TERMINATION]", self.termination)
                      .replace("[HISTORY]", self._history_block()))
            output = self._vlm_call(self.STEP_TAG, texts=prompt, images=[frame],
                                    max_new_tokens=self.MAX_NEW_TOKENS)
            if parse_yes_no(output, "Complete"):
                return self._finish(FREE_OK, "termination_met")
            answer = (parse_key_value(output, "Press") or "").strip().lower().strip(".")
            press = answer.split()[0] if answer else ""
            record = self._take_action_string(f"button({press})") if press.isalpha() else None
            if record is None:
                self._record_invalid(output)
                consecutive_invalid += 1
                if consecutive_invalid >= self.MAX_INVALID:
                    return self._finish(FREE_UNREADABLE, "press_unreadable")
                continue
            consecutive_invalid = 0
            self.presses.append(press)
            drop_prefix_crops(captured_dialogue_crops([record]), self.dialogue_crops)
            if self._last_terminated:
                return self._finish(FREE_TERMINATED, "terminated")
            if self._last_truncated:
                return self._finish(FREE_TRUNCATED, "truncated")
        return self._finish(FREE_OUT_OF_STEPS, "out_of_steps")
