from __future__ import annotations

from typing import List, Optional, Type

from gameboy_worlds.interface import HighLevelAction

from execution.executors.base import MAX_CONSECUTIVE_INVALID, Executor
from execution.executors.policies import (AVAILABLE_ACTION_POLICIES,
                                          AVAILABLE_HISTORY_POLICIES)
from execution.report import EnvironmentStepRecord, StepRecord


class PolicyExecutor(Executor):
    """An executor assembled from an action policy and a history policy."""

    ACTION_POLICY = "single"
    HISTORY_POLICY = "none"

    STEP_PROMPT = """
Task: [TASK][HINT_BLOCK]

You are playing a GameBoy game. The current screen is shown in the image.

[ERROR_BLOCK]Available environment actions:
[ACTION_LIST]

[CONTEXT_SECTION][INSTRUCTION]
[RESPONSE_FORMAT]
[STOP]"""

    def __init__(self, env, task, max_steps,
                 action_policy=None, history_policy=None,
                 history_k: int = 5, **kwargs):
        action_policy = action_policy or self.ACTION_POLICY
        history_policy = history_policy or self.HISTORY_POLICY
        if isinstance(action_policy, str):
            action_policy = AVAILABLE_ACTION_POLICIES[action_policy]()
        if isinstance(history_policy, str):
            history_policy = AVAILABLE_HISTORY_POLICIES[history_policy](
                # `self._vlm_call` records, so the policy's auxiliary calls are logged too.
                call=self._vlm_call, game=kwargs.get("game", ""), history_k=history_k,
            )
        self._action_policy = action_policy
        self._history_policy = history_policy
        # Set before super().__init__, which runs the whole episode.
        super().__init__(env, task, max_steps, **kwargs)

    @property
    def DONE_CHECK_REASONING_LABEL(self) -> str:  # noqa: N802 - shadows a base constant
        return self._action_policy.done_check_reasoning_label

    def _run_config(self, extra_kwargs):
        return {**super()._run_config(extra_kwargs),
                "action_policy": self._action_policy.name,
                "history_policy": self._history_policy.name}

    def _take_action(self, action_class: Type[HighLevelAction], **kwargs) -> EnvironmentStepRecord:
        before_info = self._env.get_info()
        frame_before = before_info["core"]["current_frame"]
        obs, reward, terminated, truncated, info = self._env.step_high_level_action(
            action_class, **kwargs
        )
        if "previous_action_details" in info.get("core", {}):
            _, _, transition_states, action_success, _ = info["core"]["previous_action_details"]
        else:
            transition_states, action_success = [], -1

        record = EnvironmentStepRecord(
            frame_before=frame_before,
            frame_after=info["core"]["current_frame"],
            action_class=action_class,
            kwargs=kwargs,
            transition_states=transition_states,
            action_success=action_success,
            reward=reward,
            frame_changed=info["core"]["frame_changed"],
        )
        self._record_step(record)
        self._last_terminated = terminated
        self._last_truncated = truncated
        self._last_frame_changed = record.frame_changed
        return record

    @staticmethod
    def _error_block(error_message: Optional[str]) -> str:
        return f"[ERROR] {error_message}\n\n" if error_message is not None else ""

    def _action_list_block(self) -> str:
        return "\n".join(f"  {s}" for s in self._get_action_strings().values())

    def _build_prompt(self, error_message: Optional[str]) -> str:
        return (
            self.STEP_PROMPT
            .replace("[TASK]", self._task)
            .replace("[HINT_BLOCK]", self._hint_block())
            .replace("[ERROR_BLOCK]", self._error_block(error_message))
            .replace("[ACTION_LIST]", self._action_list_block())
            .replace("[CONTEXT_SECTION]", self._history_policy.render())
            .replace("[INSTRUCTION]", self._action_policy.instruction())
            .replace("[RESPONSE_FORMAT]", self._action_policy.response_format())
            # Last: the response format may itself contain the placeholder.
            .replace("[ACTION_FORMAT]", self._action_policy.action_format())
        )

    def _execute(self) -> int:
        self._last_terminated = False
        self._last_truncated = False
        self._history_policy.reset()

        error_message: Optional[str] = None
        n_env_steps = 0
        consecutive_invalid = 0

        while n_env_steps < self._max_steps:
            frame = self._get_state()["core"]["current_frame"]
            prompt = self._build_prompt(error_message)

            call_kwargs = {"texts": prompt, "images": [frame]}
            if self._action_policy.max_new_tokens is not None:
                call_kwargs["max_new_tokens"] = self._action_policy.max_new_tokens
            response = self._vlm_call(self._action_policy.tag, **call_kwargs)

            decision = self._action_policy.parse(response)
            if decision is None:
                self._record_invalid(str(response))
                error_message = self._action_policy.parse_error()
                n_env_steps += 1
                consecutive_invalid += 1
                if consecutive_invalid >= MAX_CONSECUTIVE_INVALID:
                    self.report.termination_reason = "max_invalid"
                    return -1
                continue

            self._last_reasoning = decision.reasoning or self._last_reasoning

            outcome, n_env_steps, consecutive_invalid, error_message = self._run_decision(
                decision, n_env_steps, consecutive_invalid)
            if outcome is not None:
                return outcome

        self.report.termination_reason = "max_steps"
        return 0

    def _run_decision(self, decision, n_env_steps: int, consecutive_invalid: int):
        records: List[EnvironmentStepRecord] = []
        steps: List[StepRecord] = []
        error_message: Optional[str] = None
        aborted = False

        for action_str in decision.actions:
            if n_env_steps >= self._max_steps:
                break

            action_class, action_kwargs = self._env.string_to_high_level_action(action_str)
            if action_class is None:
                self._record_invalid(f"Unrecognised action string: {action_str!r}",
                                     reason="unrecognised action")
                error_message = self._action_policy.unknown_action_error(action_str)
                n_env_steps += 1
                consecutive_invalid += 1
                aborted = True
                break

            record = self._take_action(action_class, **(action_kwargs or {}))
            records.append(record)
            steps.append(record)
            n_env_steps += 1
            consecutive_invalid = 0

            if self._last_terminated:
                self._finish_decision(steps)
                self.report.termination_reason = "terminated"
                return 1, n_env_steps, consecutive_invalid, error_message
            if self._last_truncated:
                self._finish_decision(steps)
                self.report.termination_reason = "truncated"
                return 2, n_env_steps, consecutive_invalid, error_message

            # Low-level actions report success=0 by convention, not as a failure.
            if (len(decision.actions) > 1
                    and not _is_low_level(action_class)
                    and record.action_success == 0):
                error_message = (f"Action '{action_str}' failed (blocked or invalid). "
                                 "Re-plan.")
                aborted = True
                break

        self._finish_decision(steps)

        if consecutive_invalid >= MAX_CONSECUTIVE_INVALID:
            self.report.termination_reason = "max_invalid"
            return -1, n_env_steps, consecutive_invalid, error_message

        # Checked only at a decision boundary; an aborted decision is never checked.
        if records and not aborted:
            outcome = self._maybe_self_terminate(records[-1], n_env_steps)
            if outcome is not None:
                return outcome, n_env_steps, consecutive_invalid, error_message

        return None, n_env_steps, consecutive_invalid, error_message

    def _finish_decision(self, steps: List[StepRecord]) -> None:
        if steps:
            self._history_policy.observe(steps)


def _is_low_level(action_class) -> bool:
    from gameboy_worlds.interface.action import LowLevelAction
    return issubclass(action_class, LowLevelAction)


def make_executor_class(action_name: str, history_name: str) -> type:
    name = f"{action_name}_{history_name}"
    return type(name, (PolicyExecutor,), {
        "ACTION_POLICY": action_name,
        "HISTORY_POLICY": history_name,
        "__doc__": f"Executor arm: {action_name} action policy, {history_name} history.",
    })
