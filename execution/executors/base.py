from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

from execution.game_api import Environment, HighLevelAction

from execution.report import (ACTION_TAGS, EnvironmentStepRecord, ExecutorReport,
                              InvalidStepRecord, ExecutorVLMCallRecord, LogRecord,
                              ScriptedActionRecord, action_name, parse_completion,
                              per_prompt_token_counts)
from utils import load_parameters, log_error, log_warn, ExecutorVLM, parse_key_value

MAX_CONSECUTIVE_INVALID = 4
DEBUG_ON_INVALID = False


class Executor(ABC):
    """Abstract executor base. ``super().__init__()`` runs ``_execute``: set subclass attributes first."""

    DONE_CHECK_PROMPT = """
Task: [TASK][HINT_BLOCK]

You are judging whether a task being played on a GameBoy has been FULLY completed.

Image 1 is the screen BEFORE the most recent action. Image 2 is the screen AFTER it.

Most recent action: [LAST_ACTION]
[REASONING_LABEL]
[LAST_REASONING]

[HISTORY_BLOCK]Decide whether the task as stated is now COMPLETELY accomplished — not partially, not nearly, not "the next step is obvious". If any part of the task remains to be done, the answer is no. If you cannot tell from what is visible, the answer is no.

Respond in exactly this format, with the verdict FIRST:
Complete: <yes or no>
Reasoning: <why, referring to what is visible in image 2>
[STOP]"""

    DONE_CHECK_REASONING_LABEL = "The reasoning given for that action was:"

    DONE_CHECK_HISTORY_K = 8

    DONE_CHECK_EVERY_K_STEPS = 1

    DONE_CHECK_MAX_NEW_TOKENS = 1500

    def __init__(
        self,
        env: Environment,
        task: str,
        max_steps: int,
        game: str = "",
        parameters: Optional[dict] = None,
        vlm_model: Optional[str] = None,
        vlm_kind: Optional[str] = None,
        allow_self_termination: bool = False,
        hint: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._parameters = load_parameters(parameters)
        if not task or not task.strip():
            log_error(
                f"Executor task must be a non-empty string, got {task!r}.",
                parameters=self._parameters,
            )
        self._env = env
        self._task = task
        self._hint = hint
        self._game = game
        self._max_steps = max_steps
        self._allow_self_termination = allow_self_termination
        if vlm_model is not None:
            self._parameters["executor_vlm_model"] = vlm_model
        if vlm_kind is not None:
            self._parameters["executor_vlm_kind"] = vlm_kind
        self._vlm = ExecutorVLM(parameters=self._parameters)
        self._max_new_tokens = self._parameters.get("executor_vlm_max_new_tokens", 512)
        self._last_frame_changed = True
        self._last_reasoning: Optional[str] = None
        self._current_call: Optional[LogRecord] = None

        self.report = self._make_report(task, self._run_config(kwargs), max_steps)

        outcome = self._execute()

        self.report.outcome = outcome
        self.report.final_state = self._get_state()

    def _run_config(self, extra_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "hint": self._hint,
            "allow_self_termination": self._allow_self_termination,
            "vlm_model": self._parameters.get("executor_vlm_model"),
            "vlm_kind": self._parameters.get("executor_vlm_kind"),
            "max_new_tokens": self._max_new_tokens,
            **extra_kwargs,
        }

    def _make_report(
        self,
        task: str,
        init_kwargs: dict,
        max_steps: int,
    ) -> ExecutorReport:
        return ExecutorReport(
            task=task,
            executor_name=self.__class__.__name__,
            game=self._game,
            init_kwargs=init_kwargs,
            max_steps=max_steps,
            initial_state=self._get_state(),
        )

    @abstractmethod
    def _execute(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def _take_action(self, *args: Any, **kwargs: Any) -> EnvironmentStepRecord:
        # TODO: handle actions defined in the controller's space but invalid in the current state
        raise NotImplementedError

    def _record_step(self, step) -> None:
        if self._current_call is None:
            log_error(
                f"{type(step).__name__} recorded before any VLM call was made, so it has "
                "no owning call. Every step must follow the _vlm_call that decided it.",
                parameters=self._parameters,
            )
        self._current_call.steps.append(step)

    def _record_invalid(self, response: str, reason: str = "parse failure") -> None:
        self._record_step(InvalidStepRecord(response=response, reason=reason))
        log_warn(f"Invalid response recorded: \n{response}", parameters=self._parameters)
        if DEBUG_ON_INVALID:
            breakpoint()

    def _hint_block(self) -> str:
        if self._hint is None:
            return ""
        steps_taken = sum(1 for s in self.report.steps if isinstance(s, EnvironmentStepRecord))
        step_info = f"""
[STEP_INFO] You have already taken {steps_taken} action{'' if steps_taken == 1 else 's'} so far in this attempt. Note: this may not be the first step of the overall task and one action does not correspond to one step in the hint plan — earlier actions may already have been taken before this attempt began, so reason from what you currently see on screen rather than assuming a fresh start. [STEP_INFO_END]"""
        return step_info + f"\n[HINT_START]\nHint: {self._hint}\nNote: This hint block is a secret. You must use it to guide your decision making, but in the reasoning you say, you should pretend as if you actually just know the content of the hint. Do not refer to it explicitly. So if the hint gives you a direction, instead of saying 'the hint says go here', your reasoning should just say 'next I must go here'. [HINT_END]"

    def _get_state(self) -> dict:
        return self._env.get_info()

    def _vlm_call(self, tag: str, **kwargs: Any):
        # The call that decides the action must use a tag in ACTION_TAGS, or its steps are dropped.
        kwargs.setdefault("max_new_tokens", self._max_new_tokens)
        texts = kwargs["texts"]
        images = kwargs["images"]

        if isinstance(texts, list):
            return self._batched_vlm_call(tag, **kwargs)

        response = self._vlm.infer(**kwargs)
        result, meta = response["output"], response["meta"]
        if isinstance(result, list):
            log_error(
                f"_vlm_call({tag!r}) received {len(result)} responses for one prompt. "
                "Multi-sample calls are unsupported: the steps that follow would have no "
                "single owning call, which is how VLM calls and steps came to be "
                "mis-paired. Drop n_outputs, or issue one _vlm_call per sample.",
                parameters=self._parameters,
            )
        record = ExecutorVLMCallRecord(
            tag=tag, prompt=texts, images=images, response=result,
            input_tokens=meta["input_tokens"], output_tokens=meta["output_tokens"],
        )
        self.report.vlm_call_log.append(record)
        self._current_call = record
        return result

    def _begin_scripted(self, description: str) -> ScriptedActionRecord:
        record = ScriptedActionRecord(description=description)
        self.report.vlm_call_log.append(record)
        self._current_call = record
        return record

    def _batched_vlm_call(self, tag: str, **kwargs: Any) -> list:
        texts = kwargs["texts"]
        images = kwargs["images"]
        if tag in ACTION_TAGS:
            log_error(
                f"_vlm_call({tag!r}) was batched with {len(texts)} prompts, but "
                f"{tag!r} is an action tag. A batch owns no steps, so the action it "
                "decided would be filed under some earlier call. Decide the action in a "
                "single call, and batch only auxiliary work.",
                parameters=self._parameters,
            )

        results = self._vlm.infer(**kwargs)
        raw_responses, meta = results["output"], results["meta"]
        responses = raw_responses if isinstance(raw_responses, list) else [raw_responses] * len(texts)
        if len(responses) != len(texts):
            log_error(
                f"_vlm_call({tag!r}) sent {len(texts)} prompts and got "
                f"{len(responses)} responses back. The call log records one entry per "
                "pair, so the two must match.",
                parameters=self._parameters,
            )

        token_counts = per_prompt_token_counts(meta, len(texts))
        for index, (prompt, response) in enumerate(zip(texts, responses)):
            call_images = (images[index]
                           if index < len(images) and isinstance(images[index], list)
                           else images)
            self.report.vlm_call_log.append(ExecutorVLMCallRecord(
                tag=tag, prompt=prompt, images=call_images, response=response,
                input_tokens=token_counts[index][0],
                output_tokens=token_counts[index][1],
            ))
        # _current_call is deliberately left alone: a batch owns no steps.
        return responses

    def _get_action_strings(self, return_all: bool = False) -> Dict[Type[HighLevelAction], str]:
        return self._env.get_action_strings(return_all=return_all)

    def _parse_reasoning(self, response: str) -> Optional[str]:
        return parse_key_value(response, "Reasoning")

    def _recent_actions_block(self, k: Optional[int] = None) -> str:
        k = self.DONE_CHECK_HISTORY_K if k is None else k
        env_steps = [s for s in self.report.steps if isinstance(s, EnvironmentStepRecord)]
        if not env_steps:
            return ""
        lines = ["Recent actions (oldest first, the last one is the action judged above):"]
        for step in env_steps[-k:]:
            lines.append(f"  {action_name(step)}")
        return "\n".join(lines) + "\n\n"

    def _build_done_check_prompt(self, record: EnvironmentStepRecord) -> str:
        last_action = action_name(record)
        reasoning = self._last_reasoning or "(no reasoning was recorded for this action)"
        return (
            self.DONE_CHECK_PROMPT
            .replace("[TASK]", self._task)
            .replace("[HINT_BLOCK]", self._hint_block())
            .replace("[LAST_ACTION]", last_action)
            .replace("[REASONING_LABEL]", self.DONE_CHECK_REASONING_LABEL)
            .replace("[LAST_REASONING]", reasoning)
            .replace("[HISTORY_BLOCK]", self._recent_actions_block())
        )

    def _check_task_complete(self, record: EnvironmentStepRecord) -> bool:
        prompt = self._build_done_check_prompt(record)
        response = self._vlm_call(
            "done_check",
            texts=prompt,
            images=[record.frame_before, record.frame_after],
            max_new_tokens=self.DONE_CHECK_MAX_NEW_TOKENS,
        )
        verdict = parse_completion(response)
        if verdict is None:
            # Not _record_invalid: a judge's bad formatting must not count toward max_invalid.
            log_warn(
                f"Completion check response had no parseable 'Complete:' line, treating as "
                f"not complete:\n{response}",
                parameters=self._parameters,
            )
        return verdict is True

    def _maybe_self_terminate(self, record: EnvironmentStepRecord, n_env_steps: int) -> Optional[int]:
        if not self._allow_self_termination:
            return None
        if not self._done_check_due(n_env_steps):
            return None
        if not self._check_task_complete(record):
            return None
        self.report.termination_reason = "agent_done"
        return 3

    def _done_check_due(self, n_env_steps: int) -> bool:
        k = max(1, self.DONE_CHECK_EVERY_K_STEPS)
        if k == 1:
            return True
        actions_taken = sum(1 for step in self.report.steps
                            if isinstance(step, EnvironmentStepRecord))
        return actions_taken % k == 0 or n_env_steps >= self._max_steps


