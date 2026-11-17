from __future__ import annotations

import json

import click

from gameboy_worlds import get_benchmark_tasks

from execution.registry import AVAILABLE_SUPERVISORS
from execution.supervisors import PLAN_SEPARATOR

from utils import log_info

from benchmark_scripts import common
from python_scripts import paths

SUMMARY_COLUMNS = ["planned", "n_plan_steps", "n_original_plan_steps", "n_slots_attempted",
                   "n_steps_cleared", "n_attempts", "n_replans", "n_supervisor_calls"]

_EMPTY_SUMMARY = {key: 0 for key in SUMMARY_COLUMNS}


def _summary(result: dict, report) -> dict:
    # n_steps_cleared counts slots: its denominator is n_slots_attempted, not n_plan_steps.
    step_log = result["step_log"]
    attempts = [a for record in step_log for a in record["attempts"]]
    return {
        "planned": result["planned"],
        "n_plan_steps": len(result["plan"]),
        "n_original_plan_steps": len(result["original_plan"]),
        "n_slots_attempted": len(step_log),
        "n_steps_cleared": sum(1 for r in step_log if r["cleared"]),
        "n_attempts": len(attempts),
        "n_replans": result["n_replans"],
        "n_supervisor_calls": len(report.supervisor_calls),
    }


@click.command(name="subgoal")
@click.option("--max_leg_steps", default=5, show_default=True, type=int,
              help="Env-step cap for ONE executor attempt. Internal to the supervisor: it "
                   "decides how often the supervisor gets to look, not the episode budget.")
@click.option("--max_attempts_per_step", default=3, show_default=True, type=int,
              help="Failed attempts at one step before the supervisor moves on. The final "
                   "step ignores it — there is nothing to move on to.")
@click.option("--max_replans", default=2, show_default=True, type=int,
              help="Times the plan may be rewritten in one episode. Bounds both cost and "
                   "the risk of thrashing between two readings of the same screen.")
@click.option("--max_frames_per_slice", default=8, show_default=True, type=int,
              help="Trajectory frames per judging call.")
@click.option("--executor_max_new_tokens", default=8000, show_default=True, type=int,
              help="Token budget per executor action call. Overrides the project-wide "
                   "executor_vlm_max_new_tokens for this process only.")
@click.pass_obj
def subgoal_cmd(obj, max_leg_steps, max_attempts_per_step, max_replans,
                max_frames_per_slice, executor_max_new_tokens):
    """Benchmark with a plan written from the task alone, supervised step by step."""
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
    supervisor_name = "subgoal"
    supervisor_class = AVAILABLE_SUPERVISORS[supervisor_name]

    emulator_kwargs = {
        "headless": True,
        "save_video": obj["save_video"],
        "session_name": paths.benchmark_session_name(supervisor=supervisor_name, executor=executor, controller_variant=controller_variant,
                                                     model=model_save_name,
                                                     extra_name=extra_name),
        "max_steps": obj["max_steps"],
    }

    columns = common.COMMON_COLUMNS + [
        "hint", "original_plan", *SUMMARY_COLUMNS, "step_log", common.SESSION_COLUMN]
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
                max_attempts_per_target=max_attempts_per_step,
                max_replans=max_replans,
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
                extras={
                    "plan": PLAN_SEPARATOR.join(result["plan"]) if result["plan"] else None,
                    "original_plan": (PLAN_SEPARATOR.join(result["original_plan"])
                                      if result["original_plan"] else None),
                    "summary": _summary(result, report),
                    "step_log": result["step_log"],
                },
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
            extras.get("plan"),
            extras.get("original_plan"),
            *[summary[key] for key in SUMMARY_COLUMNS],
            json.dumps(extras.get("step_log", []), default=str),
            json.dumps(outcome.session_dirs),
        ]

    def on_episode(row, outcome):
        summary = outcome.extras.get("summary", _EMPTY_SUMMARY)
        log_info(f"  -> success={outcome.success}  steps={outcome.n_steps}  "
                 f"plan={summary['n_steps_cleared']}/{summary['n_slots_attempted']} steps "
                 f"cleared over {summary['n_attempts']} attempt(s), "
                 f"{summary['n_replans']} replan(s)"
                 f"{'' if summary['planned'] else '  [UNPLANNED]'}  "
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
