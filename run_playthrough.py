from __future__ import annotations

from datetime import datetime

import click

from execution.strategist.pokemon.strategist import PokemonStrategist
from python_scripts import paths
from utils import load_parameters, log_info

# GameBoyWorlds' gameboy_hard_max_steps: the emulator clamps anything larger.
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
              help="Episodes this process may run. One episode is one task handed to a supervisor.")
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
@click.option("--npc_registry/--no_npc_registry", default=True, show_default=True,
              help="Record every person seen on screen (their look, where, when, and what they said).")
@click.option("--override_goal", default=None, type=str,
              help="On a new --name, replace the default goal tree with this single goal; subgoals are planned under it.")
@click.option("--resume_run", default=None, type=str,
              help="Fork another run's latest state (artifacts and emulator save) into --name and "
                   "continue from it, e.g. to recreate a failure from a past playthrough.")
@click.option("--remote", is_flag=True, default=False,
              help="Play the environment server session in $ENVCTL_SESSION instead of a local emulator. "
                   "The server decides the game and start; --game, --init_state, --env_max_steps, "
                   "--controller_variant and --save_video do not apply.")
def main(game, init_state, name, model, vlm_kind, strategist_vlm_model, strategist_vlm_kind,
         supervisor_vlm_model, supervisor_vlm_kind, executor_vlm_model, executor_vlm_kind,
         max_episodes, supervisor_max_steps, env_max_steps, strategist_max_new_tokens,
         supervisor_max_new_tokens, subgoal_every, report_detail, controller_variant, save_video,
         npc_registry, override_goal, resume_run, remote):
    """Play one long playthrough; re-running the same --name continues it."""
    parameters = load_parameters()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if remote:
        if resume_run is not None:
            raise click.UsageError("--resume_run forks a local emulator save; a remote session cannot load one")
        from execution.remote_env import RemoteEnvironment
        environment = RemoteEnvironment.attach()
        game = environment.game
        log_info(f"Playthrough {name!r} on server session {environment.described['session']} ({game}), "
                 f"step {environment.steps}", parameters)
    else:
        from gameboy_worlds import get_environment
        log_info(f"Playthrough {name!r} on {game} from {init_state!r}", parameters)
        environment = get_environment(
            game=game,
            environment_variant="default",
            controller_variant=controller_variant,
            init_state=init_state,
            max_steps=env_max_steps,
            headless=True,
            save_video=save_video,
            # Relative to GameBoyWorlds' sessions/<game>/; read back by paths.playthrough_sessions_root.
            session_name=f"{paths.PLAYTHROUGH_SEGMENT}/{paths.depathify(name)}/{stamp}/",
        )
    log_info(f"State: {paths.playthrough_dir(parameters, game=game, name=name)}", parameters)
    try:
        if not remote:
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
            resume_run=resume_run,
            override_goal=override_goal,
            npc_registry=npc_registry,
            parameters=parameters,
            # Forwarded to each episode's supervisor; vlm_model/vlm_kind go on to its executors.
            supervisor_vlm_model=supervisor_vlm_model or model,
            supervisor_vlm_kind=supervisor_vlm_kind or vlm_kind,
            vlm_model=executor_vlm_model or model,
            vlm_kind=executor_vlm_kind or vlm_kind,
        )
        report = strategist.run()
    finally:
        # Closing finalises the video, so a crashed run's partial recording stays readable.
        environment.close()

    log_info("\n" + str(report) + "\n", parameters)


if __name__ == "__main__":
    main()
