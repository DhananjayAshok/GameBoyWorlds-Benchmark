# Must stay cheap to import: only utils.fundamental, utils.log_handling and utils.parameter_handling.

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

from utils.fundamental import depathify
from utils.log_handling import log_error
from utils.parameter_handling import load_parameters


PRODUCERS = {
    "grouped": "scripts/rl/create_all_traj.sh",
    "curiosity": "scripts/vlm/infer_tasks.sh (via scripts/pipeline/curiosity_all_tasks.sh)",
    "tasks": "scripts/vlm/propose_zeroshot.sh",
    "attempts": "scripts/vlm/attempt_tasks.sh",
    "benchmark": "scripts/benchmark.sh (via scripts/pipeline/serve_and_benchmark.sh)",
    "info": "scripts/vlm/build_info.sh",
    "insights": "scripts/vlm/build_info.sh (stage A)",
    "world_model": "scripts/core_rl/train_world_model.sh",
    "observation_encoder": "scripts/core_rl/train_observation_embedder.sh",
}

GROUPED_FILENAME = "grouped_global_high_reward_trajectories.pkl"
INFO_DIR_NAME = "info_docs"
INFO_DOC_FILENAME = "info.json"
INSIGHTS_FILENAME = "insights.jsonl"
TASKS_FILENAME = "zeroshot_tasks.jsonl"
CURIOSITY_ANNOTATION_STEM = "trajectory_annotation"
ALL_TRAJECTORIES_FILENAME = "all_trajectories.csv"
SUCCESS_TRAJECTORIES_STEM = "success_trajectories"
WORLD_MODEL_FILENAME = "world_model.pt"
OBSERVATION_ENCODER_FILENAME = "observation_encoder.pt"

REPORT_FILENAME = "report.pkl.gz"

# Must match execution.registry.AVAILABLE_SUPERVISORS.
BENCHMARK_SUPERVISORS = (
    "dummy",
    "revision",
    "subgoal",
    "info_subgoal_retrieval",
    "info_subgoal_parametric",
)

# "zeroshot" is accepted as an alias for "attempt" wherever a source is taken.
SOURCES = ("attempt", "curiosity")


def model_save_name(model_name: str) -> str:
    # Must fold the same way as model_save_name in scripts/core/utils.sh.
    return model_name.split("/")[-1].lower()


def finetuned_model_name(*, model_name: str, game: str, run_name: str, mode: str) -> str:
    # Must match the served-model name built in serve_and_benchmark.sh.
    return f"{model_save_name(model_name)}-{game}-{run_name}-{mode}"


def normalise_source(source: str, parameters=None) -> str:
    if source == "zeroshot":
        return "attempt"
    if source in SOURCES:
        return source
    log_error(f"Unknown source '{source}'. Choose from {list(SOURCES)} (or 'zeroshot').",
              parameters)


def source_label(artifact_dir: str) -> str:
    parts = os.path.normpath(artifact_dir).split(os.sep)
    if len(parts) >= 3 and parts[-3] == 'curiosity':
        return 'curiosity'
    stem = parts[-2] if len(parts) >= 2 else os.path.basename(artifact_dir)
    if stem.endswith('_attempts'):
        stem = stem[: -len('_attempts')]
        stem = stem.rsplit('_', 1)[0]
    if stem.startswith('zeroshot_tasks'):
        return 'zeroshot'
    return stem or 'unknown'


def require(path: str, kind: str, parameters=None, game: Optional[str] = None) -> str:
    if not os.path.exists(path):
        log_error(
            f"Required artifact not found: {path}\n"
            f"  This is produced by: {PRODUCERS.get(kind, kind)}\n"
            + (f"  Run that stage for game '{game}' before this command." if game else ""),
            parameters,
        )
    return path


def _roots(parameters) -> tuple[dict, str, str]:
    parameters = load_parameters(parameters)
    return parameters, parameters["storage_dir"], parameters["results_dir"]


def grouped_dir(parameters=None, *, game: str, run_name: str) -> str:
    _, storage, _ = _roots(parameters)
    return os.path.join(storage, "grouped_trajectories", game, run_name)


def grouped_file(parameters=None, *, game: str, run_name: str, init_state: str) -> str:
    return os.path.join(grouped_dir(parameters, game=game, run_name=run_name),
                        init_state, GROUPED_FILENAME)


def init_states(parameters=None, *, game: str, run_name: str) -> list[str]:
    parameters, _, _ = _roots(parameters)
    root = require(grouped_dir(parameters, game=game, run_name=run_name),
                   "grouped", parameters, game)
    states = sorted(
        d for d in os.listdir(root)
        if os.path.exists(grouped_file(parameters, game=game, run_name=run_name, init_state=d))
    )
    if not states:
        log_error(f"No grouped trajectory files under {root}.\n"
                  f"  Produced by: {PRODUCERS['grouped']}", parameters)
    return states


def world_model_dir(parameters=None, *, game: str, run_name: str) -> str:
    _, storage, _ = _roots(parameters)
    return os.path.join(storage, "curiosity_buffers", "world_model", game, run_name)


def world_model_file(parameters=None, *, game: str, run_name: str) -> str:
    return os.path.join(world_model_dir(parameters, game=game, run_name=run_name),
                        WORLD_MODEL_FILENAME)


def observation_embedder_dir(parameters=None, *, game: str, run_name: str) -> str:
    _, storage, _ = _roots(parameters)
    return os.path.join(storage, "observation_embedders", game, run_name)


def observation_embedder_file(parameters=None, *, game: str, run_name: str) -> str:
    return os.path.join(observation_embedder_dir(parameters, game=game, run_name=run_name),
                        OBSERVATION_ENCODER_FILENAME)


def proposed_tasks_dir(parameters=None, *, game: str, model_name: str) -> str:
    _, storage, _ = _roots(parameters)
    return os.path.join(storage, "proposed_tasks", game, model_save_name(model_name))


def curiosity_annotation(parameters=None, *, game: str, model_name: str, run_name: str) -> str:
    return os.path.join(curiosity_dir(parameters, game=game, model_name=model_name,
                                      run_name=run_name),
                        f"{CURIOSITY_ANNOTATION_STEM}.json")


def curiosity_dir(parameters=None, *, game: str, model_name: str, run_name: str) -> str:
    return os.path.join(proposed_tasks_dir(parameters, game=game, model_name=model_name),
                        "curiosity", run_name)


def zeroshot_dir(parameters=None, *, game: str, model_name: str) -> str:
    return os.path.join(proposed_tasks_dir(parameters, game=game, model_name=model_name),
                        "zeroshot")


def tasks_file(parameters=None, *, game: str, model_name: str) -> str:
    return os.path.join(zeroshot_dir(parameters, game=game, model_name=model_name),
                        TASKS_FILENAME)


def attempts_dir_from_stem(tasks_path: str, *, executor: str, controller_variant: str) -> str:
    return os.path.splitext(tasks_path)[0] + f"_{executor}_{controller_variant}_attempts"


def attempts_dir(parameters=None, *, game: str, model_name: str, executor: str,
                 controller_variant: str) -> str:
    return attempts_dir_from_stem(
        tasks_file(parameters, game=game, model_name=model_name), executor=executor,
        controller_variant=controller_variant)


def all_trajectories_csv(parameters=None, *, game: str, model_name: str, executor: str, controller_variant: str) -> str:
    return os.path.join(attempts_dir(parameters, game=game, model_name=model_name,
                                     executor=executor, controller_variant=controller_variant), ALL_TRAJECTORIES_FILENAME)


def success_trajectories_json(parameters=None, *, game: str, model_name: str,
                              executor: str, controller_variant: str) -> str:
    return f"{success_trajectories_stem(parameters, game=game, model_name=model_name, executor=executor, controller_variant=controller_variant)}.json"


def success_trajectories_pkl(parameters=None, *, game: str, model_name: str,
                             executor: str, controller_variant: str) -> str:
    return f"{success_trajectories_stem(parameters, game=game, model_name=model_name, executor=executor, controller_variant=controller_variant)}.pkl"


def success_trajectories_stem(parameters=None, *, game: str, model_name: str,
                              executor: str, controller_variant: str) -> str:
    return os.path.join(attempts_dir(parameters, game=game, model_name=model_name,
                                     executor=executor, controller_variant=controller_variant), SUCCESS_TRAJECTORIES_STEM)


def info_source_stem(parameters=None, *, game: str, model_name: str, run_name: str,
                     executor: str, controller_variant: str, source: str) -> str:
    parameters, _, _ = _roots(parameters)
    if normalise_source(source, parameters) == "curiosity":
        return os.path.join(curiosity_dir(parameters, game=game, model_name=model_name,
                                          run_name=run_name), CURIOSITY_ANNOTATION_STEM)
    return success_trajectories_stem(parameters, game=game, model_name=model_name,
                                     executor=executor, controller_variant=controller_variant)


def info_dir_from_stem(trajectory_path: str) -> str:
    return os.path.join(os.path.dirname(trajectory_path), INFO_DIR_NAME)


def source_info_dir(parameters=None, *, game: str, model_name: str, run_name: str,
                    executor: str, controller_variant: str, source: str = "attempt") -> str:
    return info_dir_from_stem(
        info_source_stem(parameters, game=game, model_name=model_name, run_name=run_name,
                         executor=executor, controller_variant=controller_variant, source=source))


def info_dir(parameters=None, *, game: str, model_name: str, run_name: str,
             executor: str, controller_variant: str) -> str:
    return source_info_dir(parameters, game=game, model_name=model_name, run_name=run_name,
                           executor=executor, controller_variant=controller_variant, source="attempt")


def curiosity_info_dir(parameters=None, *, game: str, model_name: str, run_name: str,
                       executor: str, controller_variant: str) -> str:
    return source_info_dir(parameters, game=game, model_name=model_name, run_name=run_name,
                           executor=executor, controller_variant=controller_variant, source="curiosity")


def info_doc(parameters=None, *, game: str, model_name: str, run_name: str, executor: str, controller_variant: str,
             source: str = "attempt") -> str:
    return os.path.join(source_info_dir(parameters, game=game, model_name=model_name,
                                        run_name=run_name, executor=executor, controller_variant=controller_variant, source=source),
                        INFO_DOC_FILENAME)


def insights_jsonl(parameters=None, *, game: str, model_name: str, run_name: str,
                   executor: str, controller_variant: str, source: str = "attempt") -> str:
    return os.path.join(source_info_dir(parameters, game=game, model_name=model_name,
                                        run_name=run_name, executor=executor, controller_variant=controller_variant, source=source),
                        INSIGHTS_FILENAME)


def parametric_doc(parameters=None, *, game: str, model_name: str) -> str:
    _, storage, _ = _roots(parameters)
    return os.path.join(storage, "parametric_docs", game, model_save_name(model_name),
                        INFO_DOC_FILENAME)


def tile_recognizer_file(parameters=None, *, game: str, name: str) -> str:
    parameters, storage, _ = _roots(parameters)
    segment = depathify(name)
    if not segment:
        log_error(f"Tile recognizer name {name!r} has no usable characters for a file name.", parameters)
    return os.path.join(storage, "playthrough_artifacts", game, "perception", "tile_recognizer", segment + ".pkl")


# The on-disk segment is deliberately "strategist", not "playthrough": existing trees are filed under it.
PLAYTHROUGH_SEGMENT = "strategist"


def playthrough_dir(parameters=None, *, game: str, name: str) -> str:
    parameters, storage, _ = _roots(parameters)
    segment = depathify(name)
    if not segment:
        log_error(f"Playthrough name {name!r} has no usable characters for a directory name.", parameters)
    return os.path.join(storage, "playthrough_artifacts", game, PLAYTHROUGH_SEGMENT, segment)


def playthrough_sessions_root(*, game: str) -> str:
    return os.path.join(sessions_dir(game=game), PLAYTHROUGH_SEGMENT)


def playthrough_session_rel(path: str, *, game: str, parameters=None) -> str:
    # The trailing separator matters: without it a sibling like strategist_old/ would pass the check.
    root = playthrough_sessions_root(game=game) + os.sep
    if not path.startswith(root):
        log_error(f"Session directory {path!r} is not under {root!r}, so it cannot be "
                  f"recorded relative to it.", parameters)
    return path[len(root):]


def playthrough_session_abs(value: str, *, game: str) -> str:
    return os.path.join(playthrough_sessions_root(game=game), value)


def playthrough_video(value: str, *, game: str) -> Optional[str]:
    path = os.path.join(playthrough_session_abs(value, game=game), "videos", "0.mp4")
    return path if os.path.exists(path) else None


def benchmark_dir(parameters=None, *, game: str) -> str:
    _, _, results = _roots(parameters)
    return os.path.join(results, "benchmark", game)


def _extra_part(extra_name: Optional[str]) -> str:
    if extra_name is None or extra_name == "" or extra_name == "none":
        return ""
    return f"_{depathify(str(extra_name))}"


def benchmark_stem(*, supervisor: str, executor: str, controller_variant: str, model: str,
                   extra_name: Optional[str] = None,
                   n_tasks: Optional[int] = None) -> str:
    extra = _extra_part(extra_name)
    base = f"{supervisor}_{executor}_{controller_variant}_{model}{extra}"
    return f"{base}_first{n_tasks}" if n_tasks is not None else base


def benchmark_session_name(*, supervisor: str, executor: str, controller_variant: str,
                           model: str, extra_name: Optional[str] = None) -> str:
    return (f"benchmark_{supervisor}_{executor}_{controller_variant}_{model}"
            f"{_extra_part(extra_name)}")


def benchmark_csv(parameters=None, *, game: str, supervisor: str, executor: str,
                  controller_variant: str, model: str, extra_name: Optional[str] = None,
                  n_tasks: Optional[int] = None, create: bool = False) -> str:
    # model is already a save-name here, not a full model name.
    directory = benchmark_dir(parameters, game=game)
    if create:
        os.makedirs(directory, exist_ok=True)
    stem = benchmark_stem(supervisor=supervisor, executor=executor,
                          controller_variant=controller_variant, model=model,
                          extra_name=extra_name, n_tasks=n_tasks)
    return os.path.join(directory, f"{stem}.csv")


def benchmark_series_csv(parameters=None, *, game: str) -> str:
    import pandas as pd
    parameters, _, _ = _roots(parameters)
    tests_dir = os.path.join(parameters["project_root"], "GameBoyWorlds", "benchmark", "tests")
    require(tests_dir, "benchmark", parameters, game)
    for filename in sorted(os.listdir(tests_dir)):
        if not filename.endswith(".csv"):
            continue
        path = os.path.join(tests_dir, filename)
        if game in set(pd.read_csv(path)["game"].unique()):
            return path
    log_error(f"No series CSV under {tests_dir} contains game '{game}'.", parameters)


TRAIN_FLAG_COLUMN = "can_train_from_init_state"


def train_games(parameters=None, *, game: str) -> list[str]:
    import pandas as pd
    parameters, _, _ = _roots(parameters)
    frame = pd.read_csv(benchmark_series_csv(parameters, game=game))
    if TRAIN_FLAG_COLUMN not in frame.columns:
        log_error(
            f"Series CSV for '{game}' has no {TRAIN_FLAG_COLUMN!r} column, so its training "
            "titles cannot be determined.", parameters)
    flag = frame[TRAIN_FLAG_COLUMN]
    # pandas hands back bool or str depending on whether any cell is blank.
    truthy = flag.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))
    return sorted(frame.loc[truthy, "game"].unique())


def gameboy_worlds_storage() -> str:
    # GameBoyWorlds' own storage_dir, a different tree from this project's storage_dir.
    from gameboy_worlds.utils import load_parameters as gbw_load_parameters
    return gbw_load_parameters()["storage_dir"]


def sessions_dir(*, game: str) -> str:
    return os.path.join(gameboy_worlds_storage(), "sessions", game)


def model_checkpoint_dir(parameters=None, *, game: str, run_name: str, mode: str,
                         model_name: str) -> str:
    _, storage, _ = _roots(parameters)
    return os.path.join(storage, "models", f"{game}-{run_name}-{mode}",
                        model_save_name(model_name))


def debug_dir(parameters=None, *, game: str, stage: str, sub: tuple = (),
              output_dir: Optional[str] = None) -> str:
    _, _, results = _roots(parameters)
    root = output_dir or os.path.join(results, "debug", game)
    path = os.path.join(root, stage, *sub)
    os.makedirs(path, exist_ok=True)
    return path


def debug_frames_dir(parameters=None, *, game: str, stage: str, sub: tuple = ()) -> str:
    _, storage, _ = _roots(parameters)
    path = os.path.join(storage, "tmp", "debug_frames", game, stage, *sub)
    os.makedirs(path, exist_ok=True)
    return path


def executor_frames_dir(parameters=None, *, game: str, executor: str,
                        model: Optional[str], task: str) -> str:
    from utils.fundamental import depathify
    _, storage, _ = _roots(parameters)
    task_str = depathify(task)
    if not task_str:
        log_error(
            f"Cannot derive an image directory from task {task!r}: it contains no word "
            "characters.", parameters)
    stem = f"{executor}_{model_save_name(model) if model else 'unknown_model'}"
    return os.path.join(storage, "tmp", "executor_frames", game, stem, task_str)


def api_image_cache_dir(parameters=None, *, unique_id: str) -> str:
    parameters, storage, _ = _roots(parameters)
    return os.path.join(storage, "tmp", "api_image_cache", unique_id)


def trajectory_render_dir(parameters=None, *, name: str, group: Optional[int] = None) -> str:
    parameters, storage, _ = _roots(parameters)
    path = os.path.join(storage, "tmp", "trajectories", name)
    return path if group is None else os.path.join(path, f"group_{group}")


@dataclass
class Paths:
    parameters: Optional[dict] = None
    game: Optional[str] = None
    run_name: str = "my_run"
    executor: str = "single_actions"
    controller_variant: str = "low_level"
    extra_name: Optional[str] = None
    model_name: Optional[str] = None
    output_dir: Optional[str] = None
    mode: str = "both"
    storage_dir: str = field(init=False)
    results_dir: str = field(init=False)

    def __post_init__(self):
        self.parameters = load_parameters(self.parameters)
        self.storage_dir = self.parameters["storage_dir"]
        self.results_dir = self.parameters["results_dir"]

    @property
    def _model(self) -> str:
        if self.model_name is None:
            log_error("model_name is required for this command but was not provided.",
                      self.parameters)
        return self.model_name

    @property
    def model_save_name(self) -> str:
        return model_save_name(self._model)

    @property
    def finetuned_model_name(self) -> str:
        return finetuned_model_name(model_name=self._model, game=self.game,
                                    run_name=self.run_name, mode=self.mode)

    def require(self, path: str, kind: str) -> str:
        return require(path, kind, self.parameters, self.game)

    def grouped_dir(self) -> str:
        return grouped_dir(self.parameters, game=self.game, run_name=self.run_name)

    def grouped_file(self, init_state: str) -> str:
        return grouped_file(self.parameters, game=self.game, run_name=self.run_name,
                            init_state=init_state)

    def init_states(self) -> list[str]:
        return init_states(self.parameters, game=self.game, run_name=self.run_name)

    def proposed_tasks_dir(self) -> str:
        return proposed_tasks_dir(self.parameters, game=self.game, model_name=self._model)

    def curiosity_annotation(self) -> str:
        return curiosity_annotation(self.parameters, game=self.game, model_name=self._model,
                                    run_name=self.run_name)

    def zeroshot_dir(self) -> str:
        return zeroshot_dir(self.parameters, game=self.game, model_name=self._model)

    def tasks_file(self) -> str:
        return tasks_file(self.parameters, game=self.game, model_name=self._model)

    def attempts_dir(self) -> str:
        return attempts_dir(self.parameters, game=self.game, model_name=self._model,
                            executor=self.executor, controller_variant=self.controller_variant)

    def all_trajectories_csv(self) -> str:
        return all_trajectories_csv(self.parameters, game=self.game, model_name=self._model,
                                    executor=self.executor, controller_variant=self.controller_variant)

    def success_trajectories_json(self) -> str:
        return success_trajectories_json(self.parameters, game=self.game,
                                         model_name=self._model, executor=self.executor,
                                         controller_variant=self.controller_variant)

    def success_trajectories_pkl(self) -> str:
        return success_trajectories_pkl(self.parameters, game=self.game,
                                        model_name=self._model, executor=self.executor,
                                        controller_variant=self.controller_variant)

    def _info_kwargs(self) -> dict[str, Any]:
        return dict(game=self.game, model_name=self._model, run_name=self.run_name,
                    executor=self.executor, controller_variant=self.controller_variant)

    def info_source_stem(self, source: str) -> str:
        return info_source_stem(self.parameters, source=source, **self._info_kwargs())

    def info_dir(self) -> str:
        return info_dir(self.parameters, **self._info_kwargs())

    def curiosity_info_dir(self) -> str:
        return curiosity_info_dir(self.parameters, **self._info_kwargs())

    def source_info_dir(self, source: str) -> str:
        return source_info_dir(self.parameters, source=source, **self._info_kwargs())

    def info_doc(self, source: str = "attempt") -> str:
        return info_doc(self.parameters, source=source, **self._info_kwargs())

    def insights_jsonl(self, source: str = "attempt") -> str:
        return insights_jsonl(self.parameters, source=source, **self._info_kwargs())

    def parametric_doc(self) -> str:
        return parametric_doc(self.parameters, game=self.game, model_name=self._model)

    def tile_recognizer_file(self, name: str) -> str:
        return tile_recognizer_file(self.parameters, game=self.game, name=name)

    def playthrough_dir(self, name: str) -> str:
        return playthrough_dir(self.parameters, game=self.game, name=name)

    def benchmark_csv(self, bench_game: str, model: str, supervisor: str,
                      n_tasks: Optional[int] = None, create: bool = False) -> str:
        return benchmark_csv(self.parameters, game=bench_game, supervisor=supervisor,
                             executor=self.executor,
                             controller_variant=self.controller_variant,
                             model=model, extra_name=self.extra_name, n_tasks=n_tasks,
                             create=create)

    def benchmark_series_csv(self) -> str:
        return benchmark_series_csv(self.parameters, game=self.game)

    def gameboy_worlds_storage(self) -> str:
        return gameboy_worlds_storage()

    def sessions_dir(self, game: Optional[str] = None) -> str:
        return sessions_dir(game=game or self.game)

    def model_checkpoint_dir(self) -> str:
        return model_checkpoint_dir(self.parameters, game=self.game, run_name=self.run_name,
                                    mode=self.mode, model_name=self._model)

    def debug_dir(self, stage: str, *sub: str) -> str:
        return debug_dir(self.parameters, game=self.game, stage=stage, sub=sub,
                         output_dir=self.output_dir)

    def debug_frames_dir(self, stage: str, *sub: str) -> str:
        return debug_frames_dir(self.parameters, game=self.game, stage=stage, sub=sub)
