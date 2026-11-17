import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import click
import numpy as np
from tqdm import tqdm

from gameboy_worlds import get_environment
from execution.registry import AVAILABLE_EXECUTORS
from execution.report import EnvironmentStepRecord
from execution.supervisors import AttemptCheckerSupervisor, derive_critique_hint
from python_scripts import paths
from utils import log_info, log_warn, log_error, sum_optional, VLM, HuggingFaceModel


def _reconstruct_trajectory(env_steps: list, init_state: str) -> tuple:
    if not env_steps:
        return None
    # observations[i] is the frame before action i; observations[-1] is the final frame
    observations = np.array(
        [env_steps[0].frame_before] + [s.frame_after for s in env_steps]
    )
    actions = [
        None for s in env_steps
    ]  # executors produce only high-level actions
    high_level_actions = [(s.action_class, s.kwargs) for s in env_steps]
    rewards = [s.reward for s in env_steps]
    return (observations, actions, high_level_actions, rewards, init_state)


def _attempt_task(
    group_idx: str,
    init_state: str,
    task_str: str,
    game: str,
    model_name: str,
    vlm_kind: str,
    executor_class,
    max_steps: int,
    lookback: int,
    controller_variant: str,
    max_attempts: int,
    checker_max_new_tokens: int,
    max_new_tokens: int,
    critique_vlm: VLM,
    parameters: dict,
    verbose: bool,
) -> tuple[dict, tuple | None]:
    env = get_environment(
        game=game,
        controller_variant=controller_variant,
        environment_variant="default",
        init_state=init_state,
        max_steps=max_steps,
        headless=True,
        save_video=False,
    )

    hint = ""
    result = None
    trajectory = None
    critique_input_tokens = 0
    critique_output_tokens = 0

    try:
        for attempt in range(max_attempts):
            env.reset()

            if verbose:
                log_info(f"  Attempt {attempt + 1}/{max_attempts}"
                         + (f" | hint: {hint}" if hint else ""), parameters)

            supervisor = AttemptCheckerSupervisor(
                task=task_str,
                executor_class=executor_class,
                env=env,
                game=game,
                max_steps=max_steps,
                evaluation_lookback=lookback,
                allow_self_termination=True,
                supervisor_vlm_model=model_name,
                supervisor_vlm_kind=vlm_kind,
                max_new_tokens=checker_max_new_tokens,
                parameters=parameters,
                vlm_model=model_name,
                vlm_kind=vlm_kind,
                hint=hint or None,
            )
            result = supervisor.evaluate()
            executor_reports = result["report"].executor_reports
            env_steps = [
                s for report in executor_reports for s in report.steps
                if isinstance(s, EnvironmentStepRecord)
            ]
            trajectory = _reconstruct_trajectory(env_steps, init_state)

            if verbose:
                log_info(f"  Result: {'success' if result['success'] else 'failure'} "
                         f"| {result.get('description', '')}", parameters)

            if result["success"]:
                break

            if attempt < max_attempts - 1:
                hint, input_tokens, output_tokens = derive_critique_hint(
                    env_steps, task_str, game, critique_vlm, max_new_tokens, hint
                )
                # sum_optional, not +=: None must propagate, not count as zero.
                critique_input_tokens = sum_optional(
                    [critique_input_tokens, input_tokens])
                critique_output_tokens = sum_optional(
                    [critique_output_tokens, output_tokens])
                if verbose:
                    log_info(f"  Derived hint: {hint}", parameters)
    finally:
        env.close()

    result_record = {
        "init_state": init_state,
        "task_string": task_str,
        "success": result["success"],
        "description": result.get("description", ""),
        "reasoning": result.get("reasoning", ""),
        "n_tries": attempt + 1,
        "final_hint": hint,
        # "agent_done" is the executor's own claim; success above is the checker's independent verdict.
        "termination_reason": result.get("termination_reason"),
        "n_env_steps": result.get("n_env_steps"),
        "max_steps": result.get("max_steps"),
        "critique_input_tokens": critique_input_tokens,
        "critique_output_tokens": critique_output_tokens,
    }
    return result_record, trajectory


@click.command(name="attempt_tasks")
@click.option(
    "--tasks_path",
    required=True,
    help="Path to the JSONL produced by propose_tasks_zeroshot.",
)
@click.option(
    "--executor",
    "executor_name",
    default="single_actions",
    show_default=True,
    type=click.Choice(list(AVAILABLE_EXECUTORS.keys())),
    help="Executor class to use for task attempts.",
)
@click.option(
    "--max_steps",
    default=50,
    show_default=True,
    help="Env-step budget per task attempt.",
)
@click.option(
    "--lookback",
    default=8,
    show_default=True,
    help="Number of final frames passed to the checker VLM.",
)
@click.option(
    "--controller_variant",
    default="low_level",
    show_default=True,
    help="Controller variant passed to get_environment.",
)
@click.option(
    "--max_attempts",
    default=5,
    show_default=True,
    help="Maximum number of attempts per task. On each failure a hint is derived and passed to the next attempt.",
)
@click.option(
    "--checker_max_new_tokens",
    default=2000,
    show_default=True,
    help="Token budget for each checker VLM call.",
)
@click.option(
    "--max_concurrency",
    default=16,
    show_default=True,
    help="Max concurrent (init_state, task) attempt pipelines. Forced to 1 for --verbose or a huggingface vlm_kind.",
)
@click.pass_obj
def attempt_tasks_cmd(
    obj, tasks_path, executor_name, max_steps, lookback, controller_variant, max_attempts, checker_max_new_tokens, max_concurrency
):
    """Attempt each proposed task with a VLM executor and check for success."""
    parameters = obj["parameters"]
    game = obj["game"]
    model_name = obj["model_name"]
    vlm_kind = obj["vlm_kind"]
    overwrite = obj["overwrite"]
    verbose = obj["verbose"]

    executor_class = AVAILABLE_EXECUTORS[executor_name]
    max_new_tokens = obj["max_new_tokens"]
    critique_vlm = VLM(model_name, vlm_kind)

    if not os.path.exists(tasks_path):
        log_error(f"tasks_path '{tasks_path}' does not exist.", parameters)

    out_dir = paths.attempts_dir_from_stem(tasks_path, executor=executor_name,
                                           controller_variant=controller_variant)
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, paths.ALL_TRAJECTORIES_FILENAME)
    json_path = os.path.join(out_dir, f"{paths.SUCCESS_TRAJECTORIES_STEM}.json")
    pkl_path = os.path.join(out_dir, f"{paths.SUCCESS_TRAJECTORIES_STEM}.pkl")
    checkpoint_json = os.path.join(out_dir, "checkpoint.json")
    checkpoint_pkl = os.path.join(out_dir, "checkpoint.pkl")

    if os.path.exists(csv_path) and not overwrite:
        log_info(f"Skipping attempts — output already exists at {csv_path}. Use --overwrite to rerun.")
        return

    if os.path.exists(checkpoint_json) and not overwrite:
        with open(checkpoint_json, "r") as f:
            results = json.load(f)
        trajectories = {}
        if os.path.exists(checkpoint_pkl):
            with open(checkpoint_pkl, "rb") as f:
                trajectories = pickle.load(f)
        # A group is done only when both its result and its trajectory are on disk: the two writes are not atomic.
        unpaired = [group_idx for group_idx in results if group_idx not in trajectories]
        for group_idx in unpaired:
            del results[group_idx]
        if unpaired:
            log_warn(
                f"checkpoint mismatch: {len(unpaired)} group_idx(s) had a saved result but "
                f"no saved trajectory ({unpaired[:5]}{' ...' if len(unpaired) > 5 else ''}) "
                "— rerunning them."
            )
        log_info(
            f"Resuming from checkpoint — {len(results)} group_idxs already done.",
            parameters,
        )
    else:
        results = {}
        trajectories = {}

    with open(tasks_path, "r") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    jobs = []
    for line_number, record in enumerate(lines):
        init_state = record["init_state"]
        tasks = record.get("tasks", [])
        for task_index, task_str in enumerate(tasks):
            group_idx = f"{line_number}_{task_index}"
            if group_idx in results and not overwrite:
                continue
            jobs.append((group_idx, init_state, task_str))

    # HuggingFaceModel is not safe for concurrent generate() calls.
    effective_workers = (
        1 if (verbose or isinstance(critique_vlm._vlm, HuggingFaceModel)) else max_concurrency
    )

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        future_to_group = {
            executor.submit(
                _attempt_task,
                group_idx,
                init_state,
                task_str,
                game,
                model_name,
                vlm_kind,
                executor_class,
                max_steps,
                lookback,
                controller_variant,
                max_attempts,
                checker_max_new_tokens,
                max_new_tokens,
                critique_vlm,
                parameters,
                verbose,
            ): group_idx
            for group_idx, init_state, task_str in jobs
        }

        for future in tqdm(as_completed(future_to_group), total=len(future_to_group), desc="Attempting tasks"):
            group_idx = future_to_group[future]
            result_record, trajectory = future.result()
            results[group_idx] = result_record
            trajectories[group_idx] = trajectory

            with open(checkpoint_json, "w") as f:
                json.dump(results, f, indent=2)
            with open(checkpoint_pkl, "wb") as f:
                pickle.dump(trajectories, f)

    # Written first and unconditionally: checkpoint.json, the only other copy of these fields, is deleted below.
    pd.DataFrame([
        {
            "group_idx": group_idx,
            "init_state": res["init_state"],
            "task_string": res["task_string"],
            "success": res["success"],
            "n_tries": res["n_tries"],
            "final_hint": res["final_hint"],
            "judge_description": res["description"],
            "judge_reasoning": res["reasoning"],
            "termination_reason": res["termination_reason"],
            "n_env_steps": res["n_env_steps"],
            "max_steps": res["max_steps"],
            "critique_input_tokens": res["critique_input_tokens"],
            "critique_output_tokens": res["critique_output_tokens"],
        }
        for group_idx, res in results.items()
    ]).to_csv(csv_path, index=False)

    success_trajectories = {gid: traj for gid, traj in trajectories.items() if results.get(gid, {}).get("success")}
    success_json = {gid: res["task_string"] for gid, res in results.items() if res.get("success")}

    # The success files are not written, so build_info fails loudly on a missing input.
    if not success_json:
        log_error(
            f"attempt_tasks: 0 of {len(results)} attempted tasks succeeded, so there are no "
            f"trajectories to distil an info document from. Attempt records were still written to "
            f"{csv_path} — inspect them before rerunning. Rerunning this stage needs "
            "--overwrite, since that CSV is its skip-if-exists marker.",
            parameters,
        )

    with open(pkl_path, "wb") as f:
        pickle.dump(success_trajectories, f)
    with open(json_path, "w") as f:
        json.dump(success_json, f, indent=2)

    log_info(f"Saved results       -> {csv_path}")
    log_info(f"Saved success tasks -> {json_path}")
    log_info(f"Saved trajectories  -> {pkl_path}")

    if os.path.exists(checkpoint_json):
        os.remove(checkpoint_json)
    if os.path.exists(checkpoint_pkl):
        os.remove(checkpoint_pkl)
