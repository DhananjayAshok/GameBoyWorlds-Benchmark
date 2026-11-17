from __future__ import annotations

import gzip
import os
import pickle
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd
from tqdm import tqdm

from gameboy_worlds import get_test_environment

from execution.info_doc import load_document
from python_scripts import paths
from python_scripts.paths import REPORT_FILENAME
from utils import depathify, log_error, log_info, log_warn


# Order matters: arm-specific columns are appended after these, SESSION_COLUMN last.
COMMON_COLUMNS = [
    "game",
    "task",
    "success",
    "n_steps",
    "n_invalid",
    "supervisor_input_tokens",
    "supervisor_output_tokens",
    "executor_input_tokens",
    "executor_output_tokens",
    "subgoals_reached",
    "all_subgoals",
    "report",
]

SESSION_COLUMN = "session_dirs"


def select_tasks(tasks: pd.DataFrame, n_tasks: Optional[int]) -> pd.DataFrame:
    if n_tasks is None:
        return tasks
    if not (1 <= n_tasks <= len(tasks)):
        raise ValueError(f"n_tasks must be between 1 and {len(tasks)}, got {n_tasks}")
    return tasks.head(n_tasks).reset_index(drop=True)


def results_path(parameters: dict, game: str, *, supervisor: str, executor: str,
                 controller_variant: str, model: str, extra_name: Optional[str],
                 n_tasks: Optional[int]) -> str:
    return paths.benchmark_csv(parameters, game=game, supervisor=supervisor,
                               executor=executor, controller_variant=controller_variant,
                               model=model, extra_name=extra_name, n_tasks=n_tasks,
                               create=True)


def load_checkpoint(save_path: str, regenerate: bool, columns: list,
                    parameters: dict) -> tuple:
    if regenerate or not os.path.exists(save_path):
        return [], 0

    existing = pd.read_csv(save_path)
    if len(existing.columns) != len(columns):
        log_error(
            f"Cannot resume {save_path}: it has {len(existing.columns)} columns but this "
            f"run writes {len(columns)}. It was written by an older version. Pass "
            f"--regenerate to overwrite it, or move it aside to keep it.",
            parameters,
        )
    log_info(f"Resuming from checkpoint: {len(existing)} tasks already completed "
             f"in {save_path}")
    return existing.values.tolist(), len(existing)


def load_documents(info_docs: str, parameters: dict) -> list:
    documents = []
    for path in [p.strip() for p in info_docs.split(",") if p.strip()]:
        if not os.path.exists(path):
            log_error(f"info document not found at {path}. Produced by: "
                      "scripts/vlm/build_info.sh", parameters)
        document = load_document(path, parameters=parameters)
        label = document.provenance.label or "(no provenance recorded)"
        documents.append(document)
        log_info(f"Loaded info document '{label}' — {len(document.task_entries)} task / "
                 f"{len(document.image_entries)} image entries ({path})")
    if not documents:
        log_error("No usable --info_docs given.", parameters)
    return documents


@dataclass
class PlayResult:
    report: Any
    extras: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        legs = self.report.executor_reports if self.report is not None else []
        return bool(legs) and legs[-1].termination_reason == "terminated"


@dataclass
class EpisodeOutcome:
    success: bool = False
    n_steps: int = 0
    n_invalid: int = 0
    # None means the backend did not report usage, which is distinct from 0.
    supervisor_input_tokens: Optional[int] = None
    supervisor_output_tokens: Optional[int] = None
    executor_input_tokens: Optional[int] = None
    executor_output_tokens: Optional[int] = None
    subgoals_reached: list = field(default_factory=list)
    subgoals_all: Any = None
    error: bool = True
    report_str: Optional[str] = None
    session_dirs: list = field(default_factory=list)
    extras: dict = field(default_factory=dict)


def session_path_of(environment) -> Optional[str]:
    emulator = getattr(environment, "_emulator", None)
    return getattr(emulator, "session_path", None) if emulator is not None else None


def save_report(session_path: str, *, supervisor: str, row, executor_name: str,
                controller_variant: str, model: str, extra_name: Optional[str] = None,
                info_docs: Optional[list] = None, report=None) -> Optional[str]:
    payload = {
        "supervisor": supervisor,
        "game": row["game"],
        "task": row["task"],
        "init_state": row.get("init_state"),
        "executor": executor_name,
        "controller_variant": controller_variant,
        "model": model,
        "extra_name": extra_name,
        "info_docs": info_docs,
        "report": report,
    }
    path = os.path.join(session_path, REPORT_FILENAME)
    try:
        with gzip.open(path, "wb", compresslevel=6) as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return path
    except Exception as error:  # noqa: BLE001 - archiving must never fail a run
        log_warn(f"Could not write report to {path}: {error}")
        return None


def run_episode(row, play: Callable[[Any], PlayResult], *, supervisor: str,
                controller_variant: str, executor_name: str, model: str,
                extra_name: Optional[str] = None, info_docs: Optional[list] = None,
                **emulator_kwargs) -> EpisodeOutcome:
    outcome = EpisodeOutcome()
    mission = row["task"]
    # depathify: a task containing "/" must not splice a directory separator into the session path.
    task_str = depathify(mission).lower()
    emulator_kwargs = dict(emulator_kwargs)
    emulator_kwargs["session_name"] += f"/{task_str}/"
    emulator_kwargs["wait_ticks"] = 20

    try:
        environment = get_test_environment(
            row=row, controller_variant=controller_variant, **emulator_kwargs
        )
        # Captured before close(): the emulator removes an empty session directory on close.
        session_path = session_path_of(environment)

        result = play(environment)

        last_state = environment.get_info()
        outcome.subgoals_all = last_state["subgoals"]["all"]
        for subgoal in last_state["subgoals"]["completed"]:
            if subgoal not in outcome.subgoals_reached:
                outcome.subgoals_reached.append(subgoal)

        outcome.n_steps += last_state["core"]["steps"]
        outcome.n_invalid += result.report.n_invalid if result.report is not None else 0
        outcome.report_str = str(result.report) if result.report is not None else None
        if result.report is not None:
            outcome.supervisor_input_tokens = result.report.supervisor_input_tokens
            outcome.supervisor_output_tokens = result.report.supervisor_output_tokens
            outcome.executor_input_tokens = result.report.executor_input_tokens
            outcome.executor_output_tokens = result.report.executor_output_tokens
        outcome.extras = result.extras

        if session_path is not None:
            save_report(session_path, supervisor=supervisor, row=row,
                        executor_name=executor_name,
                        controller_variant=controller_variant,
                        model=model, extra_name=extra_name, info_docs=info_docs,
                        report=result.report)
            outcome.session_dirs.append(session_path)

        environment.close()
        outcome.error = False
        outcome.success = result.success

    except Exception as error:  # noqa: BLE001 - one bad task must not end the sweep
        outcome.error = True
        log_warn(f"Error during execution of task '{mission}': {error}\n"
                 f"{traceback.format_exc()}")

    return outcome


def common_row(row, outcome: EpisodeOutcome) -> list:
    return [
        row["game"],
        row["task"],
        outcome.success,
        outcome.n_steps,
        outcome.n_invalid,
        outcome.supervisor_input_tokens,
        outcome.supervisor_output_tokens,
        outcome.executor_input_tokens,
        outcome.executor_output_tokens,
        outcome.subgoals_reached,
        outcome.subgoals_all,
        outcome.report_str,
    ]


def run_sweep(tasks: pd.DataFrame, *, columns: list, save_path: str, results: list,
              n_completed: int,
              run_one: Callable[[Any], EpisodeOutcome],
              build_row: Callable[[Any, EpisodeOutcome], list],
              on_episode: Optional[Callable[[Any, EpisodeOutcome], None]] = None) -> list:
    for i, row in tqdm(tasks.iterrows(), total=len(tasks)):
        if i < n_completed:
            continue

        outcome = run_one(row)
        if outcome.error:
            log_error(f"Error occurred during execution of task '{row['task']}' "
                      "- exiting loop")
        if on_episode is not None:
            on_episode(row, outcome)

        results.append(build_row(row, outcome))
        pd.DataFrame(results, columns=columns).to_csv(save_path, index=False)
        log_info(f"Saved benchmark results to {save_path}")
    return results
