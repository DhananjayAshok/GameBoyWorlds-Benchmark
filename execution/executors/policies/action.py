from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

from utils import parse_action_line, parse_int, parse_key_value


@dataclass(frozen=True)
class Decision:
    actions: List[str]
    reasoning: Optional[str] = None


class ActionPolicy(Protocol):
    """The action-selection half of an executor."""

    # Must be in ``execution.report.ACTION_TAGS``.
    tag: str
    name: str
    # ``None`` means the executor's default budget.
    max_new_tokens: Optional[int]
    done_check_reasoning_label: str

    def instruction(self) -> str:
        pass

    def response_format(self) -> str:
        pass

    def action_format(self) -> str:
        pass

    def parse(self, response: str) -> Optional[Decision]:
        pass

    def parse_error(self) -> str:
        pass

    def unknown_action_error(self, action: str) -> str:
        pass


def _one_action_format() -> str:
    return "Action: <one environment action>"


def _split_top_level(body: str) -> List[str]:
    parts, current, depth = [], "", 0
    for char in body:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    parts.append(current)
    return [part.strip() for part in parts if part.strip()]


class SingleActionPolicy:
    tag = "action"
    name = "single"
    max_new_tokens = None
    done_check_reasoning_label = "The reasoning given for that action was:"

    def instruction(self) -> str:
        return "Reason about the best next action, then respond in exactly this format:"

    def response_format(self) -> str:
        return "Reasoning: <your reasoning>\n[ACTION_FORMAT]"

    def action_format(self) -> str:
        return _one_action_format()

    def parse(self, response: str) -> Optional[Decision]:
        action = parse_action_line(response)
        if action is None:
            return None
        return Decision(actions=[action],
                        reasoning=(parse_key_value(response, "Reasoning") or "").strip()
                        or None)

    def parse_error(self) -> str:
        return ("Your previous response could not be parsed. "
                "You must end your response with:\n"
                "  Action: <action>\n"
                "  [STOP]")

    def unknown_action_error(self, action: str) -> str:
        return (f"You tried to do '{action}' but that is not a recognised action. DO NOT "
                f"use '{action}' in your response. Choose exactly one from the listed "
                "actions.")


class ScoredActionPolicy:
    tag = "score"
    name = "scored"
    # Truncation is silent (the parser maxes over whatever lines arrived), so keep this generous.
    max_new_tokens = 700
    done_check_reasoning_label = "The justification given for scoring that action highest was:"

    def instruction(self) -> str:
        return ("Score each available action on how useful it would be RIGHT NOW for "
                "making progress toward the task (1=useless, 5=very useful), and justify "
                "each score in one sentence.\n\nRespond with one line per action in "
                "exactly this format:")

    def response_format(self) -> str:
        return "<action>: <one-sentence reason for the score>: <score>\n..."

    def action_format(self) -> str:
        return _one_action_format()

    def parse_score_line(self, line: str) -> Optional[Tuple[str, str, int]]:
        stripped = line.strip()
        first, last = stripped.find(":"), stripped.rfind(":")
        if first == -1:
            return None
        action = stripped[:first].strip()
        reasoning = stripped[first + 1:last].strip() if last > first else ""
        # Synthetic keyed line so parse_int never sees digits in the justification.
        score = parse_int(f"score: {stripped[last + 1:]}", "score", 1, 5)
        if not action or score is None:
            return None
        return action, reasoning, score

    def parse(self, response: str) -> Optional[Decision]:
        best_score, best_action, best_reasoning = -1, None, None
        for line in response.splitlines():
            parsed = self.parse_score_line(line)
            if parsed is None:
                continue
            action, reasoning, score = parsed
            if score > best_score:
                best_score, best_action, best_reasoning = score, action, reasoning or None
        if best_action is None:
            return None
        return Decision(actions=[best_action], reasoning=best_reasoning)

    def parse_error(self) -> str:
        return ("Could not determine a valid action from scoring. Score every action "
                "again, one per line, in exactly this format:\n"
                "  <action>: <one-sentence reason for the score>: <score 1-5>")

    def unknown_action_error(self, action: str) -> str:
        return (f"Highest-scored action '{action}' is not a recognised action string. "
                "Re-score using exact action strings from the list.")


class SequenceActionPolicy:
    tag = "action"
    name = "sequence"
    max_new_tokens = None
    done_check_reasoning_label = ("The reasoning given for the planned sequence this "
                                  "action came from was:")

    def instruction(self) -> str:
        return ("Plan a short sequence of actions (1-5) to make progress on the task. "
                "Respond in exactly this format:")

    def response_format(self) -> str:
        return "Reasoning: <your reasoning>\n[ACTION_FORMAT]"

    def action_format(self) -> str:
        return "Action: <ACTION1, ACTION2, ...>"

    def parse(self, response: str) -> Optional[Decision]:
        for line in response.splitlines():
            stripped = line.strip()
            if not stripped.lower().startswith("action:"):
                continue
            actions = _split_top_level(stripped[len("action:"):].strip())
            if not actions:
                return None
            return Decision(actions=actions,
                            reasoning=(parse_key_value(response, "Reasoning") or "").strip()
                            or None)
        return None

    def parse_error(self) -> str:
        return ("Your previous response could not be parsed. "
                "You must end your response with:\n"
                "  Action: ACTION1, ACTION2, ...\n"
                "  [STOP]")

    def unknown_action_error(self, action: str) -> str:
        return (f"'{action}' in the planned sequence is not a recognised action. "
                "Re-plan with valid actions.")


AVAILABLE_ACTION_POLICIES = {
    policy.name: policy
    for policy in (SingleActionPolicy, ScoredActionPolicy, SequenceActionPolicy)
}
""" Action policies by registry name — the first half of an arm's identity. """
