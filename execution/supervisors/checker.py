from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple, Type

from gameboy_worlds.interface import Environment

from execution.executors import Executor
from execution.report import EnvironmentStepRecord, ExecutorReport, action_name
from execution.supervisors.base import Supervisor
from execution.supervisors.prompts import (
    CRITIQUE_CONSOLIDATE_PROMPT,
    CRITIQUE_SLICE_PROMPT,
    DESCRIBE_CONSOLIDATE_PROMPT,
    DESCRIBE_SLICE_PROMPT,
    JUDGE_BINARY_PROMPT,
    JUDGE_GOAL_CONDITION_NOTE,
)
from utils import parse_int, parse_key_value, parse_yes_no, sum_meta, zero_meta, VLM


class AttemptCheckerSupervisor(Supervisor):
    """Run the executor on a fixed task, then judge completion with a two-stage VLM pipeline."""

    def __init__(
        self,
        task: str,
        executor_class: Type[Executor],
        env: Environment,
        game: str,
        max_steps: int,
        evaluation_lookback: int = 8,
        hint: Optional[str] = None,
        goal_condition: Optional[str] = None,
        allow_self_termination: bool = False,
        supervisor_vlm_model: Optional[str] = None,
        supervisor_vlm_kind: Optional[str] = None,
        max_new_tokens: int = 4800,
        parameters: Optional[dict] = None,
        **executor_kwargs: Any,
    ) -> None:
        self._evaluation_lookback = evaluation_lookback
        self._hint = hint
        # Judge-side only; never forwarded to the executor.
        self._goal_condition = goal_condition
        self._allow_self_termination = allow_self_termination
        super().__init__(
            task=task,
            executor_class=executor_class,
            env=env,
            game=game,
            max_steps=max_steps,
            supervisor_vlm_model=supervisor_vlm_model,
            supervisor_vlm_kind=supervisor_vlm_kind,
            max_new_tokens=max_new_tokens,
            parameters=parameters,
            **executor_kwargs,
        )

    def _evaluate(self) -> dict:
        return self.call_executor(self._task, hint=self._hint,
                                  allow_self_termination=self._allow_self_termination)

    _DESCRIBE_SLICE_SIZE = 10

    def _describe_trajectory(self, env_steps: list) -> str:
        all_frames = [s.frame_after for s in env_steps]
        total = len(all_frames)
        slice_size = self._DESCRIBE_SLICE_SIZE

        if total <= slice_size:
            output = self._vlm_call(
                "describe_slice",
                texts=DESCRIBE_SLICE_PROMPT
                    .replace("[GAME]", self._game)
                    .replace("[START_IDX]", "1")
                    .replace("[END_IDX]", str(total))
                    .replace("[TOTAL]", str(total)),
                images=all_frames,
            )
            return parse_key_value(output, "Description") or output.strip()

        windows = window_trajectory(
            env_steps, DESCRIBE_SLICE_PROMPT, game=self._game,
            call=self._vlm_caller("describe_slice"),
            max_new_tokens=self._max_new_tokens, slice_size=slice_size,
        )

        segment_descriptions = []
        for (start, end), output in windows:
            desc = parse_key_value(output, "Description") or output.strip()
            segment_descriptions.append(f"Frames {start + 1}-{end}: {desc}")

        consolidate_prompt = (
            DESCRIBE_CONSOLIDATE_PROMPT
            .replace("[GAME]", self._game)
            .replace("[SEGMENT_DESCRIPTIONS]", "\n".join(segment_descriptions))
        )
        output = self._vlm_call(
            "describe_consolidate",
            texts=consolidate_prompt,
        )
        return parse_key_value(output, "Description") or output.strip()

    def process_executor_return(self, report: ExecutorReport) -> dict:
        self.last_report = report
        env_steps = [s for s in report.steps if isinstance(s, EnvironmentStepRecord)]
        k = min(self._evaluation_lookback, len(env_steps))

        run_meta = {
            "termination_reason": report.termination_reason,
            "n_env_steps": len(env_steps),
            "max_steps": report.max_steps,
        }

        if k == 0:
            return {
                "success": False,
                "safe_success_point": None,
                "description": "",
                "reasoning": "No environment steps were taken.",
                **run_meta,
            }

        description = self._describe_trajectory(env_steps)

        final_frames = [s.frame_after for s in env_steps[-k:]]
        goal_note = (
            JUDGE_GOAL_CONDITION_NOTE.replace("[GOAL_CONDITION]", self._goal_condition)
            if self._goal_condition else ""
        )
        judge_prompt = (
            JUDGE_BINARY_PROMPT
            .replace("[TASK]", self._task)
            .replace("[DESCRIPTION]", description)
            .replace("[GOAL_CONDITION_NOTE]", goal_note)
        )
        judge_output = self._vlm_call(
            "judge",
            texts=judge_prompt,
            images=final_frames,
        )

        reasoning = parse_key_value(judge_output, "Reasoning") or ""
        safe_frame = parse_int(judge_output, "Safe success point")
        safe_success_point = _frame_to_call_cutoff(report.vlm_call_log, safe_frame)

        success = parse_yes_no(judge_output, "Success") is True
        return {"success": success, "safe_success_point": safe_success_point,
                "description": description, "reasoning": reasoning, **run_meta}


def _frame_to_call_cutoff(
    vlm_call_log: list,
    safe_frame: Optional[int],
) -> Optional[int]:
    # ``safe_frame`` is a 1-based env-frame number; the result is a call-log slice index, not a frame.
    if safe_frame is None:
        return None
    env_frames = 0
    for call_idx, entry in enumerate(vlm_call_log):
        env_frames += len(entry.env_steps)
        if env_frames >= safe_frame:
            return call_idx + 1  # keep through the call that produced this frame
    return len(vlm_call_log)


def window_trajectory(
    env_steps: list,
    slice_prompt: str,
    *,
    game: str,
    call: Callable[..., Any],
    max_new_tokens: int,
    task: str = "",
    slice_size: int = 8,
) -> List[tuple]:
    frames = [s.frame_after for s in env_steps]
    if not frames:
        return []

    total = len(env_steps)
    wants_actions = "[ACTION_SEQUENCE]" in slice_prompt
    action_lines_all = [f"  {i + 1}. {action_name(step)}"
                        for i, step in enumerate(env_steps)] if wants_actions else []

    segment_ranges = []
    segment_prompts = []
    segment_images = []
    for start in range(0, total, slice_size):
        end = min(start + slice_size, total)
        slice_actions = "\n".join(action_lines_all[start:end]) or "  (no actions taken)"
        prompt = (
            slice_prompt
            .replace("[GAME]", game)
            .replace("[TASK]", task)
            .replace("[START_IDX]", str(start + 1))
            .replace("[END_IDX]", str(end))
            .replace("[TOTAL]", str(total))
            .replace("[ACTION_SEQUENCE]", slice_actions)
        )
        segment_ranges.append((start, end))
        segment_prompts.append(prompt)
        segment_images.append(frames[start:end])

    outputs = call(texts=segment_prompts, images=segment_images,
                   max_new_tokens=max_new_tokens)
    return list(zip(segment_ranges, outputs))


def summarise_trajectory_segments(
    env_steps: list,
    slice_prompt: str,
    game: str,
    task: str,
    call: Callable[..., Any],
    max_new_tokens: int,
    max_frames_per_slice: int = 8,
) -> List[str]:
    windows = window_trajectory(
        env_steps, slice_prompt, game=game, task=task, call=call,
        max_new_tokens=max_new_tokens, slice_size=max_frames_per_slice,
    )

    segment_summaries = []
    for (start, end), output in windows:
        for line in output.splitlines():
            if line.strip().lower().startswith("segment summary:"):
                summary = line.strip()[len("segment summary:"):].strip()
                segment_summaries.append(f"Steps {start + 1}-{end}: {summary}")
                break
        else:
            segment_summaries.append(f"Steps {start + 1}-{end}: {output.strip()}")
    return segment_summaries


def derive_critique_hint(
    env_steps: list,
    task: str,
    game: str,
    vlm: VLM,
    max_new_tokens: int,
    previous_hint: str = "",
    max_frames_per_slice: int = 8,
) -> Tuple[str, Optional[int], Optional[int]]:
    if not env_steps:
        empty = zero_meta()
        return previous_hint, empty["input_tokens"], empty["output_tokens"]

    metas = []

    def call(**kwargs: Any) -> Any:
        result = vlm.infer(**kwargs)
        metas.append(result["meta"])
        return result["output"]

    segment_summaries = summarise_trajectory_segments(
        env_steps, CRITIQUE_SLICE_PROMPT, game, task,
        call, max_new_tokens, max_frames_per_slice
    )
    if not segment_summaries:
        total = sum_meta(*metas)
        return previous_hint, total["input_tokens"], total["output_tokens"]

    prior_block = (
        f'Previous hint (refine or build on this):\n"{previous_hint}"\n\n'
        if previous_hint else ""
    )
    consolidate_prompt = (
        CRITIQUE_CONSOLIDATE_PROMPT
        .replace("[GAME]", game)
        .replace("[TASK]", task)
        .replace("[SEGMENT_SUMMARIES]", "\n".join(segment_summaries))
        .replace("[PRIOR_HINT_BLOCK]", prior_block)
    )
    output = call(texts=consolidate_prompt, max_new_tokens=max_new_tokens)
    total = sum_meta(*metas)
    return (parse_key_value(output, "hint") or output.strip(),
            total["input_tokens"], total["output_tokens"])


