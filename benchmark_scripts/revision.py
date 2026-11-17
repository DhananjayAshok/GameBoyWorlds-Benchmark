from __future__ import annotations

import json

import click

from gameboy_worlds import get_benchmark_tasks

from execution.registry import AVAILABLE_SUPERVISORS

from utils import log_info

from benchmark_scripts import common
from python_scripts import paths

SUMMARY_COLUMNS = ["n_attempts", "n_supervisor_calls"]

_EMPTY_SUMMARY = {key: 0 for key in SUMMARY_COLUMNS}


def _summary(result: dict, report) -> dict:
    attempts = [a for record in result["step_log"] for a in record["attempts"]]
    return {"n_attempts": len(attempts),
            "n_supervisor_calls": len(report.supervisor_calls)}


@click.command(name="revision")
@click.option("--max_leg_steps", default=5, show_default=True, type=int,
              help="Env-step cap for ONE executor attempt. Internal to the supervisor: it "
                   "decides how often the supervisor gets to look, not the episode budget.")
@click.option("--max_frames_per_slice", default=8, show_default=True, type=int,
              help="Trajectory frames per critique call.")
@click.option("--executor_max_new_tokens", default=8000, show_default=True, type=int,
              help="Token budget per executor action call. Overrides the project-wide "
                   "executor_vlm_max_new_tokens for this process only.")
@click.pass_obj
def revision_cmd(obj, max_leg_steps, max_frames_per_slice, executor_max_new_tokens):
    """Benchmark with a hint revised between short executor legs, and no plan."""
    parameters = obj["parameters"]
    previous = parameters.get("executor_vlm_max_new_tokens")
    parameters["executor_vlm_max_new_tokens"] = executor_max_new_tokens
    log_info(f"Executor token budget: {previous} -> {executor_max_new_tokens} "
             f"(this run only). Supervisor calls: {obj['supervisor_max_new_tokens']}.")
    game = obj["game"]
    controller_variant = obj["controller_variant"]
    extra_name = obj["extra_name"]
    executor = obj["executor"]
    executor_class = obj["executor_class"]
    model_save_name = obj["model_save_name"]
    supervisor_name = "revision"
    supervisor_class = AVAILABLE_SUPERVISORS[supervisor_name]

    emulator_kwargs = {
        "headless": True,
        "save_video": obj["save_video"],
        "session_name": paths.benchmark_session_name(supervisor=supervisor_name, executor=executor, controller_variant=controller_variant,
                                                     model=model_save_name,
                                                     extra_name=extra_name),
        "max_steps": obj["max_steps"],
    }

    columns = common.COMMON_COLUMNS + [*SUMMARY_COLUMNS, "step_log", common.SESSION_COLUMN]
    tasks = common.select_tasks(get_benchmark_tasks(game=game), obj["n_tasks"])
    save_path = common.results_path(parameters, game, supervisor=supervisor_name, executor=executor, controller_variant=controller_variant,
                                    model=model_save_name, extra_name=extra_name,
                                    n_tasks=obj["n_tasks"])
    results, n_completed = common.load_checkpoint(save_path, obj["regenerate"],
                                                  columns, parameters)

    def run_one(row):
        def play(environment):
            supervisor = supervisor_class(
                task=row["task"],
                executor_class=executor_class,
                env=environment,
                game=row["game"],
                max_steps=obj["max_steps"],
                supervisor_vlm_model=obj["supervisor_vlm_model"],
                supervisor_vlm_kind=obj["supervisor_vlm_kind"],
                max_new_tokens=obj["supervisor_max_new_tokens"],
                max_leg_steps=max_leg_steps,
                max_frames_per_slice=max_frames_per_slice,
                verbose=obj["verbose"],
                parameters=parameters,
                vlm_model=obj["executor_vlm_model"],
                vlm_kind=obj["executor_vlm_kind"],
            )
            result = supervisor.evaluate()
            report = result["report"]
            if obj["verbose"]:
                log_info("\n----- trajectory " + "-" * 44 + "\n" + str(report), parameters)
            return common.PlayResult(
                report=report,
                extras={"summary": _summary(result, report),
                        "step_log": result["step_log"]},
            )

        return common.run_episode(
            row, play,
            supervisor=supervisor_name,
            controller_variant=obj["controller_variant"],
            executor_name=executor_class.__name__,
            model=model_save_name,
            extra_name=extra_name,
            **emulator_kwargs,
        )

    def build_row(row, outcome):
        extras = outcome.extras
        summary = extras.get("summary", _EMPTY_SUMMARY)
        return common.common_row(row, outcome) + [
            *[summary[key] for key in SUMMARY_COLUMNS],
            json.dumps(extras.get("step_log", []), default=str),
            json.dumps(outcome.session_dirs),
        ]

    def on_episode(row, outcome):
        summary = outcome.extras.get("summary", _EMPTY_SUMMARY)
        log_info(f"  -> success={outcome.success}  steps={outcome.n_steps}  "
                 f"{summary['n_attempts']} leg(s), "
                 f"{summary['n_supervisor_calls']} supervisor calls", parameters)

    common.run_sweep(
        tasks,
        columns=columns,
        save_path=save_path,
        results=results,
        n_completed=n_completed,
        run_one=run_one,
        build_row=build_row,
        on_episode=on_episode,
    )
