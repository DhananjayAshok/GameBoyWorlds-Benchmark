"""
Entry point for a strategist playthrough. Use --help for CLI options.

Not a benchmark arm. Every arm in ``run_benchmark.py`` sweeps a fixed list of short tasks
and scores each independently; a strategist run is the opposite shape -- one long-horizon
playthrough, one emulator session, and a sequence of tasks the strategist chooses as it goes.
There is no task list to sweep and no per-task pass rate to average.

    python run_strategist.py --name gemma_starter --init_state starter \\
        --model google/gemma-4-31b-it --vlm_kind vllm --max_episodes 400

State (tiles) persists under
``Paths.strategist_dir(name)`` and is resumed by default, so re-running the same --name
continues the same playthrough. Each process writes its own ``provenance_<stamp>.json``
there recording the settings it ran under.
"""

from __future__ import annotations

import os
from datetime import datetime

import click

from gameboy_worlds import get_environment

from execution.strategist.pokemon.strategist import PokemonStrategist
from python_scripts import paths
from utils import load_parameters, log_info

#: As high as the emulator allows, so the playthrough ends on its episode count or the wall
#: clock, never on the emulator truncating underneath it. This IS the ceiling:
#: ``gameboy_hard_max_steps`` in GameBoyWorlds' config is 1,000,000 and anything larger is
#: silently clamped to it.
DEFAULT_ENV_MAX_STEPS = 1_000_000


@click.command()
@click.option("--game", default="pokemon_red", type=str)
@click.option("--init_state", default="initial", show_default=True, type=str,
              help="Where the playthrough begins. 'starter' is the point where the first "
                   "party member is chosen; 'initial' is the true start of the game.")
@click.option("--name", required=True, type=str,
              help="Identity of this playthrough. Keys the state directory AND the tile "
                   "database, so re-running the same name continues where it left off.")
@click.option("--model", required=True, type=str,
              help="The model every layer uses, unless overridden per layer below.")
@click.option("--vlm_kind", required=True, type=str)
@click.option("--strategist_vlm_model", default=None, type=str, help="Defaults to --model.")
@click.option("--strategist_vlm_kind", default=None, type=str)
@click.option("--supervisor_vlm_model", default=None, type=str, help="Defaults to --model.")
@click.option("--supervisor_vlm_kind", default=None, type=str)
@click.option("--executor_vlm_model", default=None, type=str, help="Defaults to --model.")
@click.option("--executor_vlm_kind", default=None, type=str)
@click.option("--max_episodes", default=400, show_default=True, type=int,
              help="Episodes this run may take. One episode is a whole supervised run, so "
                   "this multiplies the cost of everything else. Set it ABOVE what the "
                   "wall clock allows, so the job ends on time rather than on the counter.")
@click.option("--supervisor_max_steps", default=10, show_default=True, type=int,
              help="Executor calls the supervisor may spend per episode.")
@click.option("--env_max_steps", default=DEFAULT_ENV_MAX_STEPS, show_default=True, type=int)
@click.option("--strategist_max_new_tokens", default=4800, show_default=True, type=int)
@click.option("--supervisor_max_new_tokens", default=4800, show_default=True, type=int)
@click.option("--subgoal_every", default=5, show_default=True, type=int,
              help="Re-plan the current goal's subgoals every this many episodes it stays unchanged.")
@click.option("--report_detail", default="strategist", show_default=True,
              type=click.Choice(["strategist", "supervisor", "executor"]),
              help="How much of each episode is archived to episode_<n>/report.pkl.gz. "
                   "'strategist' keeps the strategist's own calls and the episode record, "
                   "'supervisor' adds the supervisor's calls, 'executor' keeps everything "
                   "including every frame the executors saw.")
@click.option("--controller_variant", default="state_wise", show_default=True, type=str,
              help="The Pokemon executors act through state-wise actions, not low-level "
                   "button presses.")
@click.option("--save_video", default=True, show_default=True, type=bool)
@click.option("--global_map/--no_global_map", default=False, show_default=True,
              help="Build a global grid map from every screen seen and show the strategist where it "
                   "has not been yet. Off reproduces the strategist exactly as before.")
@click.option("--global_map_shadow", is_flag=True, default=False,
              help="Build the global map for measurement only; the strategist never sees it. "
                   "The control arm of a with/without comparison.")
@click.option("--resume_run", default=None, type=str,
              help="Fork another run's latest state (artifacts and emulator save) into --name and "
                   "continue from it, e.g. to recreate a failure from a past playthrough.")
@click.option("--resume/--fresh", default=True, show_default=True,
              help="--fresh starts the tile database "
                   "over, discarding what is on disk under this --name. It does NOT reset "
                   "the emulator to a later point: the run still starts at --init_state.")
def main(game, init_state, name, model, vlm_kind, strategist_vlm_model, strategist_vlm_kind,
         supervisor_vlm_model, supervisor_vlm_kind, executor_vlm_model, executor_vlm_kind,
         max_episodes, supervisor_max_steps, env_max_steps, strategist_max_new_tokens,
         supervisor_max_new_tokens, subgoal_every, report_detail, controller_variant, save_video,
         global_map, global_map_shadow, resume_run, resume):
    """Play one long-horizon playthrough with the strategist."""
    parameters = load_parameters()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = paths.strategist_dir(parameters, game=game, name=name)
    # The strategist makes its own directory. Making it here as well breaks --resume_run, which
    # refuses to fork into a directory that already exists.
    if resume_run is None:
        os.makedirs(out_dir, exist_ok=True)

    log_info(f"Strategist playthrough {name!r} on {game} from {init_state!r}", parameters)
    log_info(f"State: {out_dir}", parameters)

    environment = get_environment(
        game=game,
        environment_variant="default",
        controller_variant=controller_variant,
        init_state=init_state,
        max_steps=env_max_steps,
        headless=True,
        save_video=save_video,
        # depathify'd for the same reason strategist_dir is: a name with spaces or
        # slashes in it becomes a directory name here, on GameBoyWorlds' storage.
        # No game segment: GameBoyWorlds roots every session at sessions/<game>/
        # already, so putting it here would repeat it one level down.
        session_name=f"strategist/{paths.depathify(name)}/{stamp}/",
    )
    try:
        environment.reset()
        strategist = PokemonStrategist(
            env=environment,
            game=game,
            name=name,
            max_episodes=max_episodes,
            supervisor_max_steps=supervisor_max_steps,
            strategist_vlm_model=strategist_vlm_model or model,
            strategist_vlm_kind=strategist_vlm_kind or vlm_kind,
            max_new_tokens=strategist_max_new_tokens,
            supervisor_max_new_tokens=supervisor_max_new_tokens,
            subgoal_every=subgoal_every,
            report_detail=report_detail,
            resume=resume,
            resume_run=resume_run,
            global_map=global_map,
            global_map_shadow=global_map_shadow,
            parameters=parameters,
            # Forwarded to each episode's PokemonPlayThroughSupervisor, which forwards the
            # vlm_* pair on to every executor it spawns.
            supervisor_vlm_model=supervisor_vlm_model or model,
            supervisor_vlm_kind=supervisor_vlm_kind or vlm_kind,
            vlm_model=executor_vlm_model or model,
            vlm_kind=executor_vlm_kind or vlm_kind,
        )
        report = strategist.run()
    finally:
        # An emulator left open holds its session directory and its video handle, so a
        # crashed run that keeps them is a run whose partial video cannot be read.
        environment.close()

    log_info("\n" + str(report) + "\n", parameters)


if __name__ == "__main__":
    main()
