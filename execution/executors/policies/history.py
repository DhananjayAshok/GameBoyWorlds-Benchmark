from __future__ import annotations

from typing import Callable, List, Protocol

from execution.game_api import LowLevelAction

from execution.report import EnvironmentStepRecord, StepRecord, action_name

DEFAULT_HISTORY_K = 5


class HistoryPolicy(Protocol):
    """The memory half of an executor."""

    name: str

    def reset(self) -> None:
        pass

    def observe(self, steps: List[StepRecord]) -> None:
        pass

    def render(self) -> str:
        pass


class NoHistoryPolicy:
    name = "none"

    def __init__(self, **kwargs) -> None:
        pass

    def reset(self) -> None:
        pass

    def observe(self, steps: List[StepRecord]) -> None:
        pass

    def render(self) -> str:
        return ""


def _action_line(record: StepRecord) -> str:
    action_str = action_name(record)
    changed = record.frame_changed
    if issubclass(record.action_class, LowLevelAction):
        # LowLevelActions report success=0 by convention, not as a failure signal.
        return f"  {action_str}{'' if changed else ' [no change]'}"
    status = ("ok" if record.action_success == 1
              else "failed" if record.action_success == 0 else "unknown")
    return f"  {action_str}  [{status}{'' if changed else ', no change'}]"


STUCK_HINT = """
If you have been trying to execute the same action repeatedly 
(specifically A or B) and especially if you get the [no change] message on your recent actions, consider that you may be stuck in a loop, and should try something else. 
Look at the screen deeply and use the visual cues to guide your decision making. 
If you are trying to interact with something, you likely have the incorrect orientation and need to slightly adjust your positioning
"""


class ActionHistoryPolicy:
    name = "actions"

    def __init__(self, history_k: int = DEFAULT_HISTORY_K, **kwargs) -> None:
        self._history_k = history_k
        self._records: List[StepRecord] = []

    def reset(self) -> None:
        self._records = []

    def observe(self, steps: List[StepRecord]) -> None:
        self._records.extend(steps)

    def render(self) -> str:
        if not self._records:
            return ""
        recent = self._records[-self._history_k:]
        lines = ["Recent actions (oldest first):"]
        lines += [_action_line(record) for record in recent]
        stuck = STUCK_HINT if any(isinstance(r, EnvironmentStepRecord)
                                  and not r.frame_changed for r in recent) else ""
        return "\n".join(lines) + stuck + "\n\n"


class VisualHistoryPolicy:
    name = "visual"

    DIFF_PROMPT = """
You are watching someone play the GameBoy game [GAME].

Image 1 is the screen BEFORE they pressed [ACTION]. Image 2 is the screen AFTER.

In one short sentence, say what changed between the two screens as a result of that action. If nothing changed, say exactly: nothing changed.
[STOP]"""

    def __init__(self, call: Callable = None, game: str = "",
                 history_k: int = DEFAULT_HISTORY_K, **kwargs) -> None:
        self._call = call
        self._game = game
        self._history_k = history_k
        self._entries: List[tuple] = []

    def reset(self) -> None:
        self._entries = []

    def observe(self, steps: List[StepRecord]) -> None:
        if not steps:
            return

        pending: List[list] = []
        prompts, images = [], []
        for record in steps:
            if (self._call is None or record.frame_before is None
                    or record.frame_after is None):
                continue
            action_str = action_name(record)
            pending.append([action_str, None])
            prompts.append(self.DIFF_PROMPT
                           .replace("[GAME]", self._game)
                           .replace("[ACTION]", action_str))
            images.append([record.frame_before, record.frame_after])

        if not pending:
            return

        responses = iter(self._call("frame_diff", texts=prompts, images=images,
                                    max_new_tokens=1000) if prompts else [])
        for entry in pending:
            if entry[1] is None:
                entry[1] = _first_sentence(next(responses, ""))
            self._entries.append((entry[0], entry[1]))

    def render(self) -> str:
        if not self._entries:
            return ""
        lines = ["What your recent actions did (oldest first):"]
        for action_str, change in self._entries[-self._history_k:]:
            lines.append(f"  {action_str} -> {change}")
        stuck = STUCK_HINT if any("nothing changed" in change.lower()
                                  for _, change in self._entries[-self._history_k:]) else ""
        return "\n".join(lines) + stuck + "\n\n"


def _first_sentence(response: str) -> str:
    text = (response or "").strip()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return "(no description)"


AVAILABLE_HISTORY_POLICIES = {
    policy.name: policy
    for policy in (NoHistoryPolicy, ActionHistoryPolicy, VisualHistoryPolicy)
}
""" History policies by registry name — the second half of an arm's identity. """
