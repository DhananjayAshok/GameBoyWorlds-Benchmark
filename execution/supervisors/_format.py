from __future__ import annotations

from typing import List, Optional

from execution.report import (ACTION_TAGS, DONE_CHECK_TAG, EnvironmentStepRecord,
                              action_name, says_complete)
from utils import parse_key_value


def action_trace(report) -> str:
    lines, n_actions = [], 0
    for entry in report.vlm_call_log:
        if entry.tag == DONE_CHECK_TAG:
            verdict = "yes" if says_complete(entry.response) else "no"
            reason = parse_key_value(entry.response, "Reasoning")
            lines.append(f"            ↳ finished? {verdict}"
                         + (f" — {reason}" if reason else ""))
            continue
        if entry.tag not in ACTION_TAGS:
            continue
        reason = parse_key_value(entry.response, "Reasoning")
        for i, step in enumerate(entry.steps or [None]):
            if isinstance(step, EnvironmentStepRecord):
                name = action_name(step)
            elif step is None:
                name = "(no action)"
            else:
                name = "INVALID"
            n_actions += 1
            if i == 0:
                lines.append(f"         {n_actions}. {name} — {reason or '(no reasoning given)'}")
            else:
                lines.append(f"         {n_actions}. {name} — (same planned sequence)")
    return "\n".join(lines) if lines else "         (no actions taken)"


def attempt_history_line(report, env_steps: list, summaries: List[str],
                         hint: Optional[str], verdict: str,
                         regression: Optional[str] = None) -> str:
    return (
        f"{report.termination_reason} after {len(env_steps)} step(s)\n"
        f"       hint given: {hint or '(none — the step text was the only instruction)'}\n"
        f"       what the player pressed, and why they said they pressed it:\n"
        f"{action_trace(report)}\n"
        f"       what visibly happened: "
        f"{' '.join(summaries) if summaries else '(nothing summarised)'}\n"
        + (f"       REGRESSION: {regression}\n" if regression else "")
        + f"       verdict: {verdict}"
    )
