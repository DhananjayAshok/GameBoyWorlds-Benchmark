from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Type

from execution.game_api import Environment

from execution.executors import Executor
from execution.report import (ExecutorReport, SupervisorReport, SupervisorVLMCallRecord,
                              per_prompt_token_counts)
from utils import VLM, load_parameters, log_error


class Supervisor(ABC):
    """Abstract base class for supervisor agents, which wrap one or more executor runs."""

    def __init__(
        self,
        task: str,
        executor_class: Type[Executor],
        env: Environment,
        game: str,
        max_steps: int,
        supervisor_vlm_model: Optional[str] = None,
        supervisor_vlm_kind: Optional[str] = None,
        max_new_tokens: int = 4800,
        parameters: Optional[dict] = None,
        **executor_kwargs: Any,
    ) -> None:
        self._task = task
        self._executor_class = executor_class
        self._env = env
        self._game = game
        self._max_steps = max_steps
        self._supervisor_vlm_model = supervisor_vlm_model
        self._supervisor_vlm_kind = supervisor_vlm_kind
        self._max_new_tokens = max_new_tokens
        self._vlm_instance: Optional[VLM] = None
        self._parameters = load_parameters(parameters)
        self._executor_kwargs = executor_kwargs
        # Built here so a partial record exists even if ``evaluate`` raises.
        self.report = SupervisorReport(
            task=task,
            supervisor_name=self.__class__.__name__,
            game=game,
            init_kwargs=self._run_config(),
        )
        self._evaluated = False

    def _run_config(self) -> dict:
        # Called from ``__init__``: subclasses must set their attributes before ``super().__init__()``.
        return {
            "supervisor_vlm_model": self._supervisor_vlm_model,
            "supervisor_vlm_kind": self._supervisor_vlm_kind,
            "max_new_tokens": self._max_new_tokens,
            "max_steps": self._max_steps,
            "executor_kwargs": dict(self._executor_kwargs),
        }

    @property
    def _vlm(self) -> VLM:
        if self._vlm_instance is None:
            if not self._supervisor_vlm_model:
                log_error(
                    f"{self.__class__.__name__} tried to make a VLM call but no "
                    f"supervisor_vlm_model was given. Pass one, or use DummySupervisor if "
                    f"the arm is meant to do no reasoning.",
                    self._parameters,
                )
            self._vlm_instance = VLM(self._supervisor_vlm_model, self._supervisor_vlm_kind)
        return self._vlm_instance

    def _vlm_call(self, stage: str, **kwargs: Any) -> Any:
        result, records = self._vlm_infer(stage, **kwargs)
        self.report.event_log.extend(records)
        return result

    def _vlm_infer(self, stage: str, **kwargs: Any) -> tuple:
        # Safe off the main thread: records are returned, not filed, so the caller keeps the log ordered.
        kwargs.setdefault("max_new_tokens", self._max_new_tokens)
        inferred = self._vlm.infer(**kwargs)
        result, meta = inferred["output"], inferred["meta"]
        texts = kwargs.get("texts")
        images = kwargs.get("images") or []
        records = []
        if isinstance(texts, list):
            responses = result if isinstance(result, list) else [result] * len(texts)
            token_counts = per_prompt_token_counts(meta, len(texts))
            for index, (prompt, response) in enumerate(zip(texts, responses)):
                call_images = images[index] if index < len(images) and isinstance(images[index], list) else images
                records.append(SupervisorVLMCallRecord(
                    stage=stage, images=call_images, prompt=prompt, response=response,
                    input_tokens=token_counts[index][0],
                    output_tokens=token_counts[index][1],
                ))
        else:
            records.append(SupervisorVLMCallRecord(
                stage=stage, images=images, prompt=texts, response=result,
                input_tokens=meta["input_tokens"], output_tokens=meta["output_tokens"],
            ))
        return result, records

    def _vlm_caller(self, stage: str):
        def call(**kwargs: Any) -> Any:
            return self._vlm_call(stage, **kwargs)
        return call

    def evaluate(self) -> dict:
        if self._evaluated:
            log_error(
                f"{self.__class__.__name__}.evaluate() called twice on one instance. The "
                "report accumulates, so this would merge two episodes into one record. "
                "Construct a new supervisor per episode.",
                self._parameters,
            )
        self._evaluated = True
        extras = self._evaluate() or {}
        return {"report": self.report, **extras}

    @abstractmethod
    def _evaluate(self) -> Optional[dict]:
        raise NotImplementedError

    def call_executor(self, task: str, *, hint: Optional[str] = None,
                      allow_self_termination: bool = False,
                      max_steps: Optional[int] = None) -> Any:
        run_kwargs = dict(self._executor_kwargs)
        run_kwargs["hint"] = hint
        run_kwargs["allow_self_termination"] = allow_self_termination
        executor = self._executor_class(
            env=self._env,
            task=task,
            game=self._game,
            max_steps=self._max_steps if max_steps is None else max_steps,
            parameters=self._parameters,
            **run_kwargs,
        )
        self.report.event_log.append(executor.report)
        return self.process_executor_return(executor.report)

    @abstractmethod
    def process_executor_return(self, report: ExecutorReport) -> Any:
        raise NotImplementedError



