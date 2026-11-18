from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union

import matplotlib.pyplot as plt
import numpy as np

from execution.game_api import HighLevelAction

from python_scripts import paths
from utils import load_parameters, log_info, parse_yes_no, sum_optional


@dataclass
class EnvironmentStepRecord:
    frame_before: np.ndarray
    frame_after: np.ndarray
    action_class: Type[HighLevelAction]
    kwargs: Dict[str, Any]
    transition_states: List[Dict[str, Any]]
    action_success: int
    frame_changed: bool
    reward: float = 0.0


@dataclass
class InvalidStepRecord:
    response: str
    reason: str


StepRecord = Union[EnvironmentStepRecord, InvalidStepRecord]


def per_prompt_token_counts(
    meta: Dict[str, Any], n_prompts: int
) -> List[tuple]:
    def spread(value: Any) -> List[Optional[int]]:
        if isinstance(value, list) and len(value) == n_prompts:
            return list(value)
        total = sum_optional(value) if isinstance(value, list) else value
        rest = None if total is None else 0
        return [total] + [rest] * (n_prompts - 1)

    inputs = spread(meta.get("input_tokens"))
    outputs = spread(meta.get("output_tokens"))
    return list(zip(inputs, outputs))


# Image 1 is the current screen, so the picked prediction is ``images[choice - 1]``.
WORLD_MODEL_FIRST_CHOICE = 2


@dataclass
class WorldModelDecision:
    """Similarity is cosine between unit-norm encoder embeddings; pixel error is mean absolute grey level, 0-255."""

    candidates: List[str]
    action_indices: List[int]
    choice: Optional[int] = None
    predicted_similarity: Optional[List[float]] = None
    copy_similarity: Optional[float] = None
    predicted_pixel_error: Optional[float] = None
    copy_pixel_error: Optional[float] = None
    reconstruction_pixel_error: Optional[float] = None
    frame_changed: Optional[bool] = None

    @property
    def chosen(self) -> Optional[int]:
        return None if self.choice is None else self.choice - WORLD_MODEL_FIRST_CHOICE

    @property
    def chosen_label(self) -> Optional[str]:
        return None if self.chosen is None else self.candidates[self.chosen]

    @property
    def scored(self) -> bool:
        return self.predicted_similarity is not None

    @property
    def chosen_similarity(self) -> Optional[float]:
        return self.predicted_similarity[self.chosen] if self.scored else None

    @property
    def chosen_was_best_match(self) -> Optional[bool]:
        if not self.scored:
            return None
        chosen = self.chosen_similarity
        return all(s < chosen for i, s in enumerate(self.predicted_similarity)
                   if i != self.chosen)

    @property
    def action_spread(self) -> Optional[float]:
        if not self.scored:
            return None
        return max(self.predicted_similarity) - min(self.predicted_similarity)

    def image_caption(self, image_index: int) -> str:
        if image_index == 0:
            return "current screen"
        candidate = image_index - 1
        caption = f"predicted after {self.candidates[candidate]}"
        return caption + " (chosen)" if candidate == self.chosen else caption

    def describe(self) -> str:
        if self.choice is None:
            return "world model: no valid choice"
        text = f"world model: chose image {self.choice} = {self.chosen_label}"
        if not self.scored:
            return text
        best = max(range(len(self.candidates)), key=lambda i: self.predicted_similarity[i])
        return (
            f"{text} · similarity to outcome {self.chosen_similarity:.3f} "
            f"(nothing-changes {self.copy_similarity:.3f}) · closest prediction: "
            f"{self.candidates[best]} · spread across actions {self.action_spread:.3f} · "
            f"pixel error {self.predicted_pixel_error:.1f} (nothing-changes "
            f"{self.copy_pixel_error:.1f}, decoder floor {self.reconstruction_pixel_error:.1f})"
            f"{'' if self.frame_changed else ' · screen unchanged'}"
        )


def summarize_world_model(decisions: List[WorldModelDecision]) -> Optional[str]:
    scored = [d for d in decisions if d.scored]
    if not scored:
        return None
    lines = [f"World model: {len(scored)} scored step(s) of {len(decisions)} decision(s)"]
    for label, group in (("all steps", scored),
                         ("screen changed", [d for d in scored if d.frame_changed])):
        if not group:
            lines.append(f"  {label}: none")
            continue
        chance = float(np.mean([1 / len(d.candidates) for d in group]))
        lines.append(
            f"  {label} (n={len(group)}): "
            f"similarity to outcome {np.mean([d.chosen_similarity for d in group]):.3f} "
            f"vs nothing-changes {np.mean([d.copy_similarity for d in group]):.3f}; "
            f"beat nothing-changes "
            f"{np.mean([d.chosen_similarity > d.copy_similarity for d in group]):.0%}; "
            f"chosen action's prediction was the closest "
            f"{np.mean([d.chosen_was_best_match for d in group]):.0%} (chance {chance:.0%}); "
            f"spread across actions {np.mean([d.action_spread for d in group]):.3f}; "
            f"pixel error {np.mean([d.predicted_pixel_error for d in group]):.1f} "
            f"vs nothing-changes {np.mean([d.copy_pixel_error for d in group]):.1f} "
            f"(decoder floor {np.mean([d.reconstruction_pixel_error for d in group]):.1f})"
        )
    return "\n".join(lines)


@dataclass
class ExecutorVLMCallRecord:
    tag: str
    images: List[np.ndarray]
    prompt: str
    response: str
    steps: List[StepRecord] = field(default_factory=list)
    # Token counts are ``None`` when the backend did not report them, which is not zero.
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    world_model: Optional[WorldModelDecision] = None

    @property
    def env_steps(self) -> List[EnvironmentStepRecord]:
        return [s for s in self.steps if isinstance(s, EnvironmentStepRecord)]


@dataclass
class ScriptedActionRecord:
    description: str
    steps: List[StepRecord] = field(default_factory=list)
    tag: str = "scripted"
    images: List[np.ndarray] = field(default_factory=list)
    prompt: str = ""
    response: str = ""
    input_tokens: Optional[int] = 0
    output_tokens: Optional[int] = 0

    @property
    def env_steps(self) -> List[EnvironmentStepRecord]:
        return [s for s in self.steps if isinstance(s, EnvironmentStepRecord)]


LogRecord = Union[ExecutorVLMCallRecord, ScriptedActionRecord]


@dataclass
class ExecutorReport:
    """Complete record of a single executor run."""

    task: str
    executor_name: str
    game: str
    init_kwargs: Dict[str, Any]
    max_steps: int
    initial_state: Dict[str, Any]
    vlm_call_log: List[LogRecord] = field(default_factory=list)
    final_state: Optional[Dict[str, Any]] = None
    outcome: Optional[int] = None
    notes: Optional[str] = None
    termination_reason: Optional[str] = None
    """
    Why execution ended.  Set by the executor's ``_execute`` method.

    Expected values:

    - ``"max_steps"``  — the environment-step budget was exhausted.
    - ``"terminated"`` — the environment signalled a terminal state.
    - ``"truncated"``  — the environment signalled truncation.
    - ``"agent_done"`` — the post-step completion check said the task is finished
      (only possible under ``allow_self_termination``).
    - ``"max_invalid"``— too many consecutive unparseable action responses.
    - ``None``         — execution has not yet completed.
    """

    @property
    def steps(self) -> List[StepRecord]:
        return [step for call in self.vlm_call_log for step in call.steps]

    @property
    def invalid_steps(self) -> List[str]:
        return [s.response for s in self.steps if isinstance(s, InvalidStepRecord)]

    @property
    def total_input_tokens(self) -> Optional[int]:
        return sum_optional([call.input_tokens for call in self.vlm_call_log])

    @property
    def total_output_tokens(self) -> Optional[int]:
        return sum_optional([call.output_tokens for call in self.vlm_call_log])

    @property
    def world_model_decisions(self) -> List[WorldModelDecision]:
        # A ScriptedActionRecord has no ``world_model`` field, hence getattr.
        decisions = (getattr(call, "world_model", None) for call in self.vlm_call_log)
        return [decision for decision in decisions if decision is not None]

    def __str__(self) -> str:
        lines: List[str] = []

        if not self.vlm_call_log:
            lines.append("  (no VLM calls recorded)")
            return "\n".join(lines)

        for call_idx, entry in enumerate(self.vlm_call_log):
            display_idx = call_idx + 1
            tag_label = f"[{entry.tag.upper()}]"
            lines.append(f"\n  ┌─ {tag_label} (call {display_idx})" + "─" * max(0, 48 - len(tag_label)))
            lines.append("")
            if isinstance(entry, ScriptedActionRecord):
                lines.append(f"  │ Scripted: {entry.description}")
            else:
                lines.append("  | Prompt:")
                lines.append(_indent(entry.prompt, "  │   "))
                lines.append("  │ VLM output:")
                lines.append(_indent(entry.response, "  │   "))

            for step in entry.steps:
                if isinstance(step, InvalidStepRecord):
                    lines.append(f"  │ → INVALID  ({step.reason})")
                else:
                    lines.append(f"  │ → {_step_summary(step)}")
            decision = getattr(entry, "world_model", None)
            if decision is not None:
                lines.append(f"  │ → {decision.describe()}")

            if entry.tag in ACTION_TAGS and not entry.steps:
                lines.append("  │ → (no step recorded)")
            elif entry.tag == DONE_CHECK_TAG:
                verdict = "yes" if says_complete(entry.response) else "no"
                ended = (call_idx == len(self.vlm_call_log) - 1
                         and self.termination_reason == "agent_done")
                lines.append(f"  │ → COMPLETE: {verdict}"
                             + ("  → SELF-TERMINATED  (agent_done)" if ended else ""))

            lines.append("  └" + "─" * 57)

        world_model_summary = summarize_world_model(self.world_model_decisions)
        if world_model_summary:
            lines.append(_indent(world_model_summary, "  "))

        return "\n".join(lines)

    def _save_images(self) -> None:
        parameters = load_parameters()
        img_save_path = paths.executor_frames_dir(
            parameters,
            game=self.game,
            executor=self.executor_name,
            model=self.init_kwargs.get("vlm_model"),
            task=self.task.lower(),
        )
        if os.path.exists(img_save_path):
            shutil.rmtree(img_save_path)
        os.makedirs(img_save_path)

        for call_idx, entry in enumerate(self.vlm_call_log):
            display_idx = call_idx + 1
            for i, image in enumerate(entry.images):
                img_path = os.path.join(img_save_path, f"{display_idx}_{i}.png")
                plt.imshow(image)
                plt.savefig(img_path)
                plt.clf()

    def show(self) -> str:
        text = str(self)
        self._save_images()
        log_info(text)
        return text


def _indent(text: str, prefix: str = "      ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def action_name(record) -> str:
    try:
        return record.action_class.get_action_name(**record.kwargs)
    except Exception:
        return record.action_class.__name__


def _step_summary(step: StepRecord) -> str:
    if isinstance(step, EnvironmentStepRecord):
        return f"ENV   {step.action_class.__name__}({step.kwargs})"
    return f"INVALID  ({step.reason})"


ACTION_TAGS = {"action", "score"}

DONE_CHECK_TAG = "done_check"


def parse_completion(response: str) -> Optional[bool]:
    return parse_yes_no(response, "Complete")


def says_complete(response: str) -> bool:
    return parse_completion(response) is True


@dataclass
class SupervisorVLMCallRecord:
    # TODO: add a SupervisorInvalidStepRecord; an unparseable verdict currently reads as "no".
    stage: str
    images: List[np.ndarray]
    prompt: str
    response: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


@dataclass
class SupervisorReport:
    """Complete record of one supervisor run."""

    task: str
    supervisor_name: str
    game: str
    init_kwargs: Dict[str, Any] = field(default_factory=dict)
    event_log: List[Union[SupervisorVLMCallRecord, ExecutorReport]] = field(default_factory=list)
    narrative: List[str] = field(default_factory=list)
    digest: Optional[str] = None

    @property
    def supervisor_calls(self) -> List[SupervisorVLMCallRecord]:
        return [e for e in self.event_log if isinstance(e, SupervisorVLMCallRecord)]

    @property
    def executor_reports(self) -> List[ExecutorReport]:
        return [e for e in self.event_log if isinstance(e, ExecutorReport)]

    @property
    def n_invalid(self) -> int:
        return sum(len(report.invalid_steps) for report in self.executor_reports)

    @property
    def supervisor_input_tokens(self) -> Optional[int]:
        return sum_optional([call.input_tokens for call in self.supervisor_calls])

    @property
    def supervisor_output_tokens(self) -> Optional[int]:
        return sum_optional([call.output_tokens for call in self.supervisor_calls])

    @property
    def executor_input_tokens(self) -> Optional[int]:
        return sum_optional([report.total_input_tokens for report in self.executor_reports])

    @property
    def executor_output_tokens(self) -> Optional[int]:
        return sum_optional([report.total_output_tokens for report in self.executor_reports])

    @property
    def total_input_tokens(self) -> Optional[int]:
        return sum_optional([self.supervisor_input_tokens, self.executor_input_tokens])

    @property
    def total_output_tokens(self) -> Optional[int]:
        return sum_optional([self.supervisor_output_tokens, self.executor_output_tokens])

    @property
    def world_model_decisions(self) -> List[WorldModelDecision]:
        return [d for report in self.executor_reports for d in report.world_model_decisions]

    def __str__(self) -> str:
        if not self.event_log:
            return "  (no supervisor events recorded)"
        lines: List[str] = []
        leg = 0
        for event in self.event_log:
            if isinstance(event, SupervisorVLMCallRecord):
                lines.append(f"===== SUPERVISOR [{event.stage}] =====")
                lines.append(_indent(f"Prompt:\n{event.prompt}"))
                lines.append(_indent(f"Response:\n{event.response}"))
            else:
                leg += 1
                hint = event.init_kwargs.get("hint") if event.init_kwargs else None
                header = f"===== EXECUTOR LEG {leg}: {event.task!r} ====="
                lines.append(header)
                if hint:
                    lines.append(_indent(f"hint: {hint}"))
                lines.append(str(event))
        legs_with_decisions = [r for r in self.executor_reports if r.world_model_decisions]
        if len(legs_with_decisions) > 1:
            lines.append("===== WORLD MODEL (whole episode) =====")
            lines.append(_indent(summarize_world_model(self.world_model_decisions) or "", "  "))
        return "\n".join(lines)
