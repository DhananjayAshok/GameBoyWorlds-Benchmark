from __future__ import annotations

import os
import weakref
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from cleanrl_utils.port_gameboy_worlds import (
    FRAME_STACK,
    CNNEmbedder,
    WorldModel,
    load_action_space,
)

from execution.executors.base import MAX_CONSECUTIVE_INVALID
from execution.executors.executor import PolicyExecutor
from execution.report import (WORLD_MODEL_FIRST_CHOICE, EnvironmentStepRecord,
                              WorldModelDecision)
from python_scripts import paths
from utils import depathify, log_error, parse_int

WORLD_MODEL_EXECUTOR = "world_model"

# Keyed on both files' paths, mtimes and the device, so a retrained checkpoint is reloaded.
_LOADED_MODELS: Dict[tuple, Tuple[WorldModel, dict]] = {}

# Keyed on the env object, not ``id(env)``: ids are reused after garbage collection.
_FRAME_STACKS: "weakref.WeakKeyDictionary[Any, deque]" = weakref.WeakKeyDictionary()


class WorldModelExecutor(PolicyExecutor):
    """Choose an action by showing the VLM what the world model predicts each one does."""

    ACTION_POLICY = "single"
    HISTORY_POLICY = "none"

    WORLD_MODEL_RUN_NAME: Optional[str] = None

    WORLD_MODEL_GAME: Optional[str] = None

    DONE_CHECK_REASONING_LABEL = (
        "The reason given for choosing that predicted outcome was:"
    )

    STEP_PROMPT = """
Task: [TASK][HINT_BLOCK]

You are playing a GameBoy game.

[ERROR_BLOCK]Image 1 is the CURRENT screen.

The remaining images are predictions from a learned world model of what the screen would
look like after each available action. They are reconstructions, so they are blurry and
imperfect — judge them on the overall change they show, not on fine detail.

[CHOICE_LIST]

Pick the single predicted screen that most directly advances the task. Respond in exactly
this format:
Reasoning: <your reasoning>
Choice: <the image number you pick>
[STOP]"""

    def __init__(self, env, task, max_steps,
                 observation_embedder_run_name: Optional[str] = None,
                 **kwargs):
        # Must all be set before super().__init__(), which runs the whole episode.
        self._world_model_run_name = self.WORLD_MODEL_RUN_NAME
        self._observation_embedder_run_name = (
            observation_embedder_run_name or self._world_model_run_name
        )
        self._game_for_paths = self.WORLD_MODEL_GAME or kwargs.get("game", "")
        self._frames: deque = _FRAME_STACKS.setdefault(env, deque(maxlen=FRAME_STACK))
        self._load_world_model(kwargs.get("parameters"))
        super().__init__(env, task, max_steps, **kwargs)

    def _load_world_model(self, parameters) -> None:
        if not self._world_model_run_name:
            log_error(
                "WorldModelExecutor has no checkpoint. Build the class with "
                "make_world_model_executor_class(run_name) — run_benchmark.py does this "
                "from --world_model_run_name.",
                parameters=parameters,
            )
        wm_dir = paths.world_model_dir(parameters, game=self._game_for_paths,
                                       run_name=self._world_model_run_name)
        emb_dir = paths.observation_embedder_dir(
            parameters, game=self._game_for_paths,
            run_name=self._observation_embedder_run_name,
        )
        wm_file = paths.require(os.path.join(wm_dir, paths.WORLD_MODEL_FILENAME),
                                "world_model", parameters, self._game_for_paths)
        emb_file = paths.require(os.path.join(emb_dir, paths.OBSERVATION_ENCODER_FILENAME),
                                 "observation_encoder", parameters, self._game_for_paths)
        wm_stat, emb_stat = os.stat(wm_file), os.stat(emb_file)
        # Retraining overwrites the directory, so the mtime tells two runs under one name apart.
        self._checkpoints = {
            "world_model_checkpoint": wm_file,
            "world_model_checkpoint_mtime": _mtime(wm_stat),
            "observation_encoder_checkpoint": emb_file,
            "observation_encoder_checkpoint_mtime": _mtime(emb_stat),
        }

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        files = (wm_file, emb_file, str(self._device))
        key = (*files, wm_stat.st_mtime_ns, emb_stat.st_mtime_ns)
        if key not in _LOADED_MODELS:
            for stale in [k for k in _LOADED_MODELS if k[:len(files)] == files]:
                del _LOADED_MODELS[stale]
            # normalized_observations must match how the world model was trained.
            embedder = CNNEmbedder(seed=1, normalized_observations=True).to(self._device)
            # Not CNNEmbedder.load(): it has no map_location, so GPU checkpoints fail on CPU.
            embedder.load_state_dict(torch.load(emb_file, map_location=self._device))
            world_model = WorldModel(embedder=embedder, load_path=wm_dir).to(self._device)
            world_model.eval()
            spec = world_model.action_space_spec or load_action_space(wm_dir)
            _LOADED_MODELS[key] = (world_model, spec)
        self._world_model, self._action_space_spec = _LOADED_MODELS[key]

    @property
    def _controller(self):
        return self._env._controller

    def _candidate_actions(self) -> List[Tuple[int, type, Dict[str, Any], str]]:
        candidates = []
        for entry in self._action_space_spec["actions"]:
            space_action = (entry["sub_index"], entry["sub_action"])
            action_class, kwargs = self._controller._space_action_to_high_level_action(
                space_action
            )
            if action_class.__name__ != entry["action_class"]:
                log_error(
                    f"World model action {entry['index']} was trained as "
                    f"{entry['action_class']} (on {self._action_space_spec.get('game')!r}, "
                    f"{self._action_space_spec.get('controller_variant')!r} controller), but "
                    f"this environment's controller maps it to {action_class.__name__}. The "
                    "checkpoint cannot drive this controller.",
                    parameters=self._parameters,
                )
            candidates.append((entry["index"], action_class, kwargs,
                               self._action_label(action_class, kwargs)))
        return candidates

    @staticmethod
    def _action_label(action_class: type, kwargs: Dict[str, Any]) -> str:
        if not kwargs:
            return action_class.__name__
        parts = [str(getattr(value, "name", None) or value) for value in kwargs.values()]
        return f"{action_class.__name__}({', '.join(parts)})"

    @staticmethod
    def _to_grey(frame: np.ndarray) -> np.ndarray:
        return np.asarray(frame).reshape(144, 160)

    def _observe(self, frame: np.ndarray) -> None:
        grey = self._to_grey(frame)
        if not self._frames:
            for _ in range(FRAME_STACK):
                self._frames.append(grey)
        else:
            self._frames.append(grey)

    def _seed_frames(self) -> None:
        current = self._get_state()["core"]["current_frame"]
        if not self._frames or not np.array_equal(self._frames[-1], self._to_grey(current)):
            self._frames.clear()
            self._observe(current)

    def _current_latents(self) -> torch.Tensor:
        stack = np.stack(list(self._frames)).astype(np.float32)
        with torch.no_grad():
            latents = self._world_model.embedder.embed(stack)
        return latents.reshape(-1)

    def _predict(self, candidates) -> Tuple[torch.Tensor, List[np.ndarray]]:
        latents = self._current_latents()
        embedder = self._world_model.embedder
        actions = torch.tensor([[float(index)] for index, _, _, _ in candidates],
                               device=latents.device)
        x = torch.cat([latents.expand(len(candidates), -1), actions], dim=1)
        with torch.no_grad():
            embeddings = self._world_model(x)
            pixels = (
                embedder.denormalize_reconstruction(embedder.decode(embeddings))
                .cpu()
                .numpy()
                .clip(0, 255)
            )
        frames = [p.reshape(144, 160, 1).astype(np.uint8) for p in pixels]
        return embeddings, frames

    def _score(self, decision: WorldModelDecision, predicted_embeddings: torch.Tensor,
               predicted_frames: List[np.ndarray], record: EnvironmentStepRecord) -> None:
        embedder = self._world_model.embedder
        before = self._to_grey(record.frame_before).astype(np.float32)
        after = self._to_grey(record.frame_after).astype(np.float32)
        with torch.no_grad():
            actual = embedder.embed(after)
            before_embedding = embedder.embed(before)[0]
            reconstruction = (
                embedder.denormalize_reconstruction(embedder.decode(actual))
                .cpu()
                .numpy()
                .clip(0, 255)
                .reshape(144, 160)
            )
        chosen_frame = predicted_frames[decision.chosen].reshape(144, 160).astype(np.float32)

        decision.predicted_similarity = (predicted_embeddings @ actual[0]).tolist()
        decision.copy_similarity = float(before_embedding @ actual[0])
        decision.predicted_pixel_error = float(np.abs(chosen_frame - after).mean())
        decision.copy_pixel_error = float(np.abs(before - after).mean())
        decision.reconstruction_pixel_error = float(np.abs(reconstruction - after).mean())
        decision.frame_changed = bool(record.frame_changed)

    def _choice_list_block(self, candidates) -> str:
        lines = ["Predicted outcomes:"]
        for offset, (_, _, _, label) in enumerate(candidates):
            lines.append(f"  Image {offset + WORLD_MODEL_FIRST_CHOICE}: after taking {label}")
        return "\n".join(lines)

    def _build_wm_prompt(self, candidates, error_message: Optional[str]) -> str:
        return (
            self.STEP_PROMPT
            .replace("[TASK]", self._task)
            .replace("[HINT_BLOCK]", self._hint_block())
            .replace("[ERROR_BLOCK]", self._error_block(error_message))
            .replace("[CHOICE_LIST]", self._choice_list_block(candidates))
        )

    def _execute(self) -> int:
        self._last_terminated = False
        self._last_truncated = False
        self._history_policy.reset()
        # Not cleared: the stack is shared with earlier legs on this environment.
        self._seed_frames()

        error_message: Optional[str] = None
        n_env_steps = 0
        consecutive_invalid = 0
        last_choice = WORLD_MODEL_FIRST_CHOICE - 1

        while n_env_steps < self._max_steps:
            frame = self._get_state()["core"]["current_frame"]

            candidates = self._candidate_actions()
            predicted_embeddings, predicted_frames = self._predict(candidates)
            prompt = self._build_wm_prompt(candidates, error_message)

            response = self._vlm_call(
                "action", texts=prompt, images=[frame] + predicted_frames
            )
            decision = WorldModelDecision(
                candidates=[label for _, _, _, label in candidates],
                action_indices=[index for index, _, _, _ in candidates],
            )
            self._current_call.world_model = decision

            last_choice = WORLD_MODEL_FIRST_CHOICE + len(candidates) - 1
            choice = parse_int(str(response), "Choice", WORLD_MODEL_FIRST_CHOICE, last_choice)
            if choice is None:
                self._record_invalid(str(response), reason="no valid choice")
                error_message = (
                    "Your previous response could not be parsed. You must end your "
                    f"response with:\n  Choice: <an image number between "
                    f"{WORLD_MODEL_FIRST_CHOICE} and {last_choice}>"
                )
                n_env_steps += 1
                consecutive_invalid += 1
                if consecutive_invalid >= MAX_CONSECUTIVE_INVALID:
                    self.report.termination_reason = "max_invalid"
                    return -1
                continue

            consecutive_invalid = 0
            error_message = None
            self._last_reasoning = self._parse_reasoning(str(response)) or self._last_reasoning
            decision.choice = choice

            _, action_class, kwargs, _ = candidates[decision.chosen]
            record = self._take_action(action_class, **kwargs)
            # Pushed even when the screen did not change: training stacks did the same.
            self._observe(record.frame_after)
            self._score(decision, predicted_embeddings, predicted_frames, record)
            n_env_steps += 1

            # Outcome codes and termination reasons must match PolicyExecutor's loop.
            if self._last_terminated:
                self._finish_decision([record])
                self.report.termination_reason = "terminated"
                return 1
            if self._last_truncated:
                self._finish_decision([record])
                self.report.termination_reason = "truncated"
                return 2

            self._finish_decision([record])
            outcome = self._maybe_self_terminate(record, n_env_steps)
            if outcome is not None:
                return outcome

        self.report.termination_reason = "max_steps"
        return 0

    def _run_config(self, extra_kwargs):
        return {**super()._run_config(extra_kwargs),
                "world_model_run_name": self._world_model_run_name,
                "world_model_game": self._game_for_paths,
                "observation_embedder_run_name": self._observation_embedder_run_name,
                **self._checkpoints}


def world_model_executor_name(run_name: str, source_game: Optional[str] = None) -> str:
    if source_game:
        return f"{WORLD_MODEL_EXECUTOR}_{depathify(source_game)}_{depathify(run_name)}"
    return f"{WORLD_MODEL_EXECUTOR}_{depathify(run_name)}"


def make_world_model_executor_class(run_name: str, source_game: Optional[str] = None) -> type:
    source = f" trained on {source_game}" if source_game else ""
    return type(world_model_executor_name(run_name, source_game), (WorldModelExecutor,), {
        "WORLD_MODEL_RUN_NAME": run_name,
        "WORLD_MODEL_GAME": source_game,
        "__doc__": f"World-model arm playing with the '{run_name}' checkpoint{source}.",
    })


def _mtime(stat: os.stat_result) -> str:
    return datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
