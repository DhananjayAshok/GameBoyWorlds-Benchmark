import json
import os
import pickle
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import click
import pandas as pd
from tqdm import tqdm

from gameboy_worlds import get_environment
from execution.registry import AVAILABLE_EXECUTORS
from execution.report import EnvironmentStepRecord
from execution.supervisors import AttemptCheckerSupervisor, derive_critique_hint
from utils import log_info, log_warn, log_error, VLM, HuggingFaceModel


def _format_guidance(guidance_dict: dict) -> str:
    lines = []
    if guidance_dict.get("summary"):
        lines.append(f"Summary: {guidance_dict['summary']}")
    steps = guidance_dict.get("steps", [])
    if steps:
        lines.append("Steps:")
        for i, step in enumerate(steps, 1):
            lines.append(f"  {i}. {step}")
    return "\n".join(lines)


def _episode_seed(base_seed: int, group_idx: str, attempt: int) -> int:
    # Injective given task_index < 100 and attempt < 100. Never hash(), which is salted per process.
    group_idx = str(group_idx)
    if "_" in group_idx:
        head, _, tail = group_idx.partition("_")
        major, minor = int(head), int(tail)
    else:
        major, minor = int(group_idx), 0
    return base_seed + major * 10000 + minor * 100 + attempt


def _report_env_steps(report) -> list:
    return [
        step
        for executor_report in report.executor_reports
        for step in executor_report.steps
        if isinstance(step, EnvironmentStepRecord)
    ]


def _report_vlm_calls(report) -> list:
    # The flattening order defines the call_idx that clean_practice and create_dataset index with.
    return [
        call
        for executor_report in report.executor_reports
        for call in executor_report.vlm_call_log
    ]


def _practice_episode(
    group_idx: str,
    attempt: int,
    record: dict,
    guidance_str: str,
    game: str,
    model_name: str,
    vlm_kind: str,
    executor_class,
    max_steps: int,
    n_random_actions: int,
    lookback: int,
    controller_variant: str,
    checker_max_new_tokens: int,
    max_new_tokens: int,
    critique_vlm: VLM,
    base_seed: int,
    parameters: dict,
    verbose: bool,
):
    task_str = record["task"]
    init_state = record["init_state"]
    goal_condition = record.get("goal_condition", "") or None

    try:
        env = get_environment(
            game=game,
            controller_variant=controller_variant,
            environment_variant="default",
            init_state=init_state,
            max_steps=max_steps + n_random_actions + 50,  # extra buffer for random actions and potential overshooting
            headless=True,
            save_video=False,
        )
    except Exception:
        log_info(f"[{group_idx}_{attempt}] env creation failed:\n{traceback.format_exc()}", parameters)
        return None

    try:
        if verbose:
            log_info(f"Group [{group_idx}] attempt {attempt} task: {task_str}", parameters)
            if guidance_str:
                log_info(f"  Guidance: {guidance_str}", parameters)

        seed = _episode_seed(base_seed, group_idx, attempt)
        env.reset(seed=seed)
        random_actions = [env.action_space.sample() for _ in range(n_random_actions)]

        for action in random_actions:
            env.step(action)

        def _make_supervisor(hint):
            return AttemptCheckerSupervisor(
                task=task_str,
                executor_class=executor_class,
                env=env,
                game=game,
                max_steps=max_steps,
                evaluation_lookback=lookback,
                allow_self_termination=False,
                hint=hint,
                goal_condition=goal_condition,
                supervisor_vlm_model=model_name,
                supervisor_vlm_kind=vlm_kind,
                max_new_tokens=checker_max_new_tokens,
                parameters=parameters,
                vlm_model=model_name,
                vlm_kind=vlm_kind,
            )

        supervisor = _make_supervisor(guidance_str or None)

        result = supervisor.evaluate()

        failed = not result["success"]
        derived_hint = ""
        if failed:
            env_steps = _report_env_steps(result["report"])
            derived_hint, _, _ = derive_critique_hint(
                env_steps, task_str, game, critique_vlm, max_new_tokens
            )
            hint_str = f"{guidance_str}\nSpecific hint: {derived_hint}" if guidance_str else f"Specific hint: {derived_hint}"
            # A fresh supervisor: evaluate() refuses to run twice on one instance.
            supervisor = _make_supervisor(hint_str)
            supervisor._env.reset()
            for action in random_actions:
                supervisor._env.step(action)
            result = supervisor.evaluate()

        if verbose:
            log_info(f"  Attempt {attempt}: {'success' if result['success'] else 'failure'} "
                     f"| {result.get('description', '')}", parameters)
            log_info(supervisor.last_report.show(), parameters)
            breakpoint()

        row = {
            "group_idx": group_idx,
            "attempt": attempt,
            "task_string": task_str,
            "success": result["success"],
            "safe_success_point": result.get("safe_success_point"),
            "seed": seed,
            "used_retry": bool(failed),
            "derived_hint": derived_hint,
            "guidance": guidance_str or "",
            "goal_condition": goal_condition or "",
            "judge_description": result.get("description", ""),
            "judge_reasoning": result.get("reasoning", ""),
            "termination_reason": result.get("termination_reason"),
            "n_env_steps": result.get("n_env_steps"),
        }
        return row, _report_vlm_calls(result["report"])
    except Exception:
        log_info(f"[{group_idx}_{attempt}] episode failed:\n{traceback.format_exc()}", parameters)
        return None
    finally:
        env.close()


@click.command(name="practice_tasks")
@click.option(
    "--guidance_path",
    required=True,
    help="Path to trajectory_guidance.json produced by infer_guidance.",
)
@click.option(
    "--n_attempts",
    default=3,
    show_default=True,
    help="Number of independent attempts per task.",
)
@click.option(
    "--max_total_practice_runs",
    default=4000,
    show_default=True,
    help="Cap on total practice episodes (tasks x n_attempts). If tasks x n_attempts "
    "exceeds this, n_attempts is truncated to the largest value that keeps the total "
    "under this cap (min 1 attempt per task). Sized so a run finishes in ~5 days.",
)
@click.option(
    "--n_random_actions",
    default=5,
    show_default=True,
    help="Random low-level env steps taken before each attempt to perturb start state.",
)
@click.option(
    "--max_steps",
    default=50,
    show_default=True,
    help="Env-step budget per attempt.",
)
@click.option(
    "--lookback",
    default=8,
    show_default=True,
    help="Number of final frames passed to the checker VLM.",
)
@click.option(
    "--executor",
    "executor_name",
    default="single_actions",
    show_default=True,
    type=click.Choice(list(AVAILABLE_EXECUTORS.keys())),
    help="Executor class to use.",
)
@click.option(
    "--controller_variant",
    default="low_level",
    show_default=True,
    help="Controller variant passed to get_environment.",
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
    help="Max concurrent practice episodes. Forced to 1 for --verbose or a huggingface vlm_kind.",
)
@click.pass_obj
def practice_tasks_cmd(
    obj,
    guidance_path,
    n_attempts,
    max_total_practice_runs,
    n_random_actions,
    max_steps,
    lookback,
    executor_name,
    controller_variant,
    checker_max_new_tokens,
    max_concurrency,
):
    """Run repeated supervised practice attempts on inferred tasks."""
    parameters = obj["parameters"]
    game = obj["game"]
    model_name = obj["model_name"]
    vlm_kind = obj["vlm_kind"]
    overwrite = obj["overwrite"]
    verbose = obj["verbose"]
    base_seed = parameters["random_seed"]
    max_new_tokens = obj["max_new_tokens"]

    executor_class = AVAILABLE_EXECUTORS[executor_name]
    vlm = VLM(model_name, vlm_kind)

    if not os.path.exists(guidance_path):
        log_error(f"guidance_path '{guidance_path}' does not exist.", parameters)

    out_dir = os.path.join(os.path.dirname(guidance_path), f"practice_{executor_name}")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "results.csv")
    checkpoint_path = os.path.join(out_dir, "checkpoint.json")

    if os.path.exists(csv_path) and not overwrite:
        log_info(f"Skipping practice — output already exists at {csv_path}. Use --overwrite to rerun.")
        return

    if os.path.exists(checkpoint_path) and not overwrite:
        with open(checkpoint_path, "r") as f:
            rows = json.load(f)
        done = {(r["group_idx"], r["attempt"]) for r in rows}
    else:
        rows = []
        done = set()

    with open(guidance_path, "r") as f:
        guidance_data = {str(k): v for k, v in json.load(f).items()}

    guidance_strs = {g: _format_guidance(rec.get("guidance", {})) for g, rec in guidance_data.items()}

    n_tasks = len(guidance_data)
    n_attempts_requested = n_attempts
    if n_tasks > 0 and n_tasks * n_attempts > max_total_practice_runs:
        revised_n_attempts = max(1, max_total_practice_runs // n_tasks)
        log_warn(
            f"Total practice episodes ({n_tasks} tasks x {n_attempts} attempts = "
            f"{n_tasks * n_attempts}) exceeds max_total_practice_runs "
            f"({max_total_practice_runs}). Truncating n_attempts {n_attempts} -> "
            f"{revised_n_attempts} (new total {n_tasks * revised_n_attempts}).",
            parameters,
        )
        n_attempts = revised_n_attempts

    all_jobs = [(g, a) for g in guidance_data for a in range(n_attempts)]
    jobs = [j for j in all_jobs if j not in done]
    log_info(f"{len(done)}/{len(all_jobs)} episodes already done — running {len(jobs)}.", parameters)

    # HuggingFaceModel is not safe for concurrent generate() calls.
    effective_workers = 1 if (verbose or isinstance(vlm._vlm, HuggingFaceModel)) else max_concurrency

    config = {
        "guidance_path": os.path.abspath(guidance_path),
        "game": game,
        "model_name": model_name,
        "vlm_kind": vlm_kind,
        "executor": executor_name,
        "controller_variant": controller_variant,
        "n_tasks": n_tasks,
        "n_attempts_requested": n_attempts_requested,
        "n_attempts_effective": n_attempts,
        "max_total_practice_runs": max_total_practice_runs,
        "n_random_actions": n_random_actions,
        "max_steps": max_steps,
        "lookback": lookback,
        "checker_max_new_tokens": checker_max_new_tokens,
        "max_new_tokens": max_new_tokens,
        "base_seed": base_seed,
        "max_concurrency": max_concurrency,
        "effective_workers": effective_workers,
        "n_episodes_planned": len(all_jobs),
        "n_episodes_already_done": len(done),
    }
    config_tmp = os.path.join(out_dir, "config.json.tmp")
    with open(config_tmp, "w") as f:
        json.dump(config, f, indent=2)
    os.replace(config_tmp, os.path.join(out_dir, "config.json"))

    with ThreadPoolExecutor(max_workers=effective_workers) as pool:
        future_to_job = {
            pool.submit(
                _practice_episode,
                group_idx,
                attempt,
                guidance_data[group_idx],
                guidance_strs[group_idx],
                game,
                model_name,
                vlm_kind,
                executor_class,
                max_steps,
                n_random_actions,
                lookback,
                controller_variant,
                checker_max_new_tokens,
                max_new_tokens,
                vlm,
                base_seed,
                parameters,
                verbose,
            ): (group_idx, attempt)
            for group_idx, attempt in jobs
        }

        for future in tqdm(as_completed(future_to_job), total=len(future_to_job), desc="practicing"):
            outcome = future.result()
            if outcome is None:
                continue
            row, vlm_call_log = outcome

            pkl_name = f"{row['group_idx']}_{row['attempt']}.pkl"
            with open(os.path.join(out_dir, pkl_name), "wb") as f:
                pickle.dump(vlm_call_log, f)

            rows.append(row)

            tmp_path = checkpoint_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(rows, f, indent=2)
            os.replace(tmp_path, checkpoint_path)

    group_order = {g: i for i, g in enumerate(guidance_data)}
    rows.sort(key=lambda r: (group_order[r["group_idx"]], r["attempt"]))

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    log_info(f"Saved results      -> {csv_path}")
    log_info(f"Saved episode pkls -> {out_dir}/")

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
