from __future__ import annotations

from typing import List, Optional, Tuple

from execution.report import EnvironmentStepRecord, ExecutorReport, parse_completion
from execution.supervisors._format import action_trace, attempt_history_line
from execution.supervisors.base import Supervisor
from execution.supervisors.checker import summarise_trajectory_segments
from execution.supervisors.prompts import (JUDGE_CONSOLIDATE_PROMPT, JUDGE_SLICE_PROMPT,
                                           RESUME_HINT_PROMPT)
from utils import log_info, parse_key_value


class RevisingSupervisor(Supervisor):
    """Run the task in short legs, critiquing and re-hinting after each failure."""

    def __init__(self, *args, max_leg_steps: int = 5, max_attempts_per_target: int = 3,
                 final_attempt_multiplier: int = 3, max_frames_per_slice: int = 8,
                 max_history_attempts: Optional[int] = None, verbose: bool = False,
                 **kwargs) -> None:
        self.max_leg_steps = max_leg_steps
        self.max_attempts_per_target = max_attempts_per_target
        self.final_attempt_multiplier = final_attempt_multiplier
        self._max_frames_per_slice = max_frames_per_slice
        self.max_history_attempts = (max_attempts_per_target if max_history_attempts is None
                                     else max_history_attempts)
        self.verbose = verbose
        self.step_log: List[dict] = []
        self.last_diagnosis: Optional[str] = None
        super().__init__(*args, **kwargs)

    @property
    def max_final_attempts(self) -> int:
        floor = self.max_attempts_per_target * self.final_attempt_multiplier
        return max(floor, self._max_steps)

    def _run_config(self) -> dict:
        return {**super()._run_config(),
                "max_leg_steps": self.max_leg_steps,
                "max_attempts_per_target": self.max_attempts_per_target,
                "max_final_attempts": self.max_final_attempts,
                "max_frames_per_slice": self._max_frames_per_slice,
                "max_history_attempts": self.max_history_attempts}

    def _resolve_targets(self) -> List[Optional[str]]:
        # A ``None`` target means the benchmark task itself.
        return [None]

    def _knowledge(self) -> str:
        return ""

    def _diagnose_failure(self, index: int, targets: List[Optional[str]],
                          history: List[str], judgement: str,
                          regression: Optional[str]) -> Optional[List[str]]:
        return None

    def _check_regression(self, summaries: List[str]) -> Optional[str]:
        return None

    def _on_target_cleared(self, target: str) -> None:
        pass

    def _on_targets_replaced(self, targets: List[Optional[str]]) -> None:
        pass

    def _say(self, message: str) -> None:
        if self.verbose:
            log_info(message, self._parameters)

    def _current_frame(self):
        return self._env.get_info()["core"]["current_frame"]

    def _segment_summaries(self, env_steps: list, target: str) -> List[str]:
        return summarise_trajectory_segments(
            env_steps, JUDGE_SLICE_PROMPT, self._game, target,
            self._vlm_caller("judge_slice"),
            self._max_new_tokens, self._max_frames_per_slice,
        )

    def _run_leg(self, leg_task: str, hint: Optional[str], self_terminate: bool,
                 budget: int) -> ExecutorReport:
        return self.call_executor(
            leg_task,
            hint=hint,
            allow_self_termination=self_terminate,
            max_steps=min(self.max_leg_steps, budget),
        )

    @staticmethod
    def _budget_spent(report: ExecutorReport) -> int:
        # Floored at 1 so the budget strictly decreases.
        return max(1, len(report.steps))

    def _leg_spec(self, target: Optional[str], is_last: bool,
                  hint: Optional[str]) -> Tuple[str, Optional[str], bool]:
        if target is None:
            return self._task, hint, False
        if is_last:
            return self._task, hint or target, False
        return target, hint, True

    def judge_step(self, env_steps: list, target: str, stop_reason: str) -> Tuple:
        if not env_steps:
            return False, "No environment steps were taken in this attempt.", []

        summaries = self._segment_summaries(env_steps, target)
        prompt = (
            JUDGE_CONSOLIDATE_PROMPT
            .replace("[GAME]", self._game)
            .replace("[TASK]", target)
            .replace("[SEGMENT_SUMMARIES]", "\n".join(summaries))
            .replace("[STOP_REASON]", stop_reason or "unknown")
        )
        output = self._vlm_call("judge", texts=prompt, images=[self._current_frame()])
        verdict = parse_completion(output)
        reasoning = (parse_key_value(output, "Reasoning") or "").strip()
        if verdict is None:
            self._log_truncated_verdict()
        return verdict is True, reasoning, summaries

    def _log_truncated_verdict(self) -> None:
        from utils import log_warn
        log_warn(f"[supervisor] completion check returned no 'Complete:' line within "
                 f"{self._max_new_tokens} tokens; treating as not complete. "
                 f"Raise --supervisor_max_new_tokens if this repeats.", self._parameters)

    def write_resume_hint(self, summaries: List[str], target: str, judgement: str,
                          previous_hint: str = "", report=None,
                          regression: Optional[str] = None) -> Optional[str]:
        prior_block = (f'Previous hint, which did not work (do not simply repeat it):\n'
                       f'"{previous_hint}"\n\n' if previous_hint else "")
        trace_block = (f"\n\nWhat they pressed, and the reason they gave for each:\n"
                       f"{action_trace(report)}" if report is not None else "")
        prompt = (
            RESUME_HINT_PROMPT
            .replace("[GAME]", self._game)
            .replace("[TASK]", target)
            .replace("[SEGMENT_SUMMARIES]",
                     ("\n".join(summaries) or "  (no actions taken)") + trace_block)
            .replace("[JUDGEMENT]", judgement or "the screen does not show the step complete")
            .replace("[REGRESSION_BLOCK]",
                     f"**{regression} Say what to do about that first — the player cannot "
                     f"finish this step from a state they have gone backwards into.**\n\n"
                     if regression else "")
            .replace("[INSIGHTS]", self._knowledge() or "(nothing recorded)")
            .replace("[PRIOR_HINT_BLOCK]", prior_block)
        )
        output = self._vlm_call("hint", texts=prompt, images=[self._current_frame()])
        self.last_diagnosis = (parse_key_value(output, "Diagnosis") or "").strip() or None
        hint = (parse_key_value(output, "Hint") or "").strip()
        if self.last_diagnosis:
            self._say(f"      diagnosis: {self.last_diagnosis}")
        return hint or None

    def _handle_failed_attempt(self, *, index: int, targets: List[Optional[str]],
                               history: List[str], attempt: dict, report,
                               env_steps: list, summaries: List[str], verdict: str,
                               hint_target: str, hint_reason: str, hint: Optional[str],
                               attempts_at_target: int, attempt_cap: int) -> Tuple:
        regression = self._check_regression(summaries)
        if regression:
            attempt["regression"] = regression

        history.append(attempt_history_line(
            report, env_steps, summaries, attempt["hint"], verdict, regression))
        if self.max_history_attempts > 0:
            del history[:-self.max_history_attempts]
        else:
            history.clear()

        replacement = self._diagnose_failure(index, targets, history, verdict, regression)
        if replacement:
            return replacement, hint, False

        if attempts_at_target >= attempt_cap:
            return None, hint, True

        hint = self.write_resume_hint(summaries, hint_target, hint_reason, hint or "",
                                      report=report, regression=regression) or hint
        if self.last_diagnosis:
            attempt["diagnosis"] = self.last_diagnosis
        return None, hint, False

    def _evaluate(self) -> dict:
        self.step_log = []
        budget = self._max_steps
        targets = self._resolve_targets()

        index = 0
        while index < len(targets):
            if budget <= 0:
                self._say(f"  step budget exhausted at target {index + 1}/{len(targets)}")
                break

            target = targets[index]
            is_last = index == len(targets) - 1
            record = {"step": target, "attempts": [], "replans": [], "cleared": False}
            # These three are reset on a replan; ``record`` is not, it audits the whole slot.
            hint: Optional[str] = None
            history: List[str] = []
            attempts_at_target = 0

            while budget > 0:
                leg_task, leg_hint, self_terminate = self._leg_spec(target, is_last, hint)
                attempts_at_target += 1

                self._say(f"\n  --- target {index + 1}/{len(targets)}"
                          f" attempt {attempts_at_target}"
                          f"{' (final)' if is_last else ''}, {budget} steps left")
                if target is not None:
                    self._say(f"      {target}")
                if leg_hint and leg_hint != target:
                    self._say(f"      hint: {leg_hint}")

                report = self._run_leg(leg_task, leg_hint, self_terminate, budget)
                env_steps = [s for s in report.steps
                             if isinstance(s, EnvironmentStepRecord)]
                budget -= self._budget_spent(report)
                attempt = {"termination_reason": report.termination_reason,
                           "n_steps": len(env_steps), "hint": leg_hint}
                self._say(f"      -> {report.termination_reason} after "
                          f"{len(env_steps)} step(s)")

                # `max_invalid` is NOT here: it is re-hintable, not terminal.
                if report.termination_reason in ("terminated", "truncated"):
                    attempt["cleared"] = report.termination_reason == "terminated"
                    record["attempts"].append(attempt)
                    record["cleared"] = attempt["cleared"]
                    index = len(targets)          # leave the outer loop too
                    break

                if is_last:
                    # Not judged: only the environment terminating clears the last target.
                    attempt["cleared"] = False
                    record["attempts"].append(attempt)
                    summaries = (self._segment_summaries(env_steps, self._task)
                                 if env_steps else [])
                    verdict = "the environment did not signal the task complete"
                    hint_target, hint_reason = self._task, verdict
                    attempt_cap = self.max_final_attempts
                else:
                    complete, reasoning, summaries = self.judge_step(
                        env_steps, target, report.termination_reason
                    )
                    attempt["cleared"] = complete
                    attempt["judgement"] = reasoning
                    record["attempts"].append(attempt)
                    self._say(f"      judge: {'COMPLETE' if complete else 'NOT COMPLETE'} "
                              f"— {reasoning}")
                    if complete:
                        record["cleared"] = True
                        self._on_target_cleared(target)
                        break
                    verdict = f"judged incomplete — {reasoning}"
                    hint_target, hint_reason = target, reasoning
                    attempt_cap = self.max_attempts_per_target

                replacement, hint, exhausted = self._handle_failed_attempt(
                    index=index, targets=targets, history=history, attempt=attempt,
                    report=report, env_steps=env_steps, summaries=summaries,
                    verdict=verdict, hint_target=hint_target, hint_reason=hint_reason,
                    hint=hint, attempts_at_target=attempts_at_target,
                    attempt_cap=attempt_cap,
                )

                if replacement:
                    record["replans"].append(replacement)
                    targets[index:] = replacement
                    self._on_targets_replaced(targets)
                    target = targets[index]
                    is_last = index == len(targets) - 1
                    record["step"] = target
                    hint = None
                    history = []
                    attempts_at_target = 0
                    continue

                if exhausted:
                    self._say(f"      target not cleared after {attempts_at_target} "
                              f"attempts; {'giving up' if is_last else 'moving on'}.")
                    break

            self.step_log.append(record)
            index += 1

        return self._extras()

    def _extras(self) -> dict:
        return {"step_log": self.step_log}

    def process_executor_return(self, report: ExecutorReport) -> ExecutorReport:
        return report
