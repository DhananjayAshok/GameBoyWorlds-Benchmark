import click

from gameboy_worlds import AVAILABLE_GAMES

from benchmark_scripts import (
    baseline,
    info_subgoal_parametric,
    info_subgoal_retrieval,
    revision,
    subgoal,
)
from execution.registry import (AVAILABLE_EXECUTORS, WORLD_MODEL_EXECUTOR,
                                make_world_model_executor_class)
from utils import load_parameters
from python_scripts.paths import model_save_name


def _absent(value) -> bool:
    return value in (None, "", "none")


def _resolve_executor(executor: str, world_model_run_name, world_model_game=None,
                      game: str = None):
    run_name = None if _absent(world_model_run_name) else world_model_run_name
    source_game = None if _absent(world_model_game) else world_model_game
    if executor == WORLD_MODEL_EXECUTOR:
        if run_name is None:
            raise click.UsageError(
                f"--executor {WORLD_MODEL_EXECUTOR} requires --world_model_run_name.")
        if source_game is not None and source_game not in AVAILABLE_GAMES:
            raise click.UsageError(
                f"--world_model_game {source_game!r} is not a known game.")
        if source_game == game:
            source_game = None
        return make_world_model_executor_class(run_name, source_game)
    for flag, value in (("--world_model_run_name", run_name),
                        ("--world_model_game", source_game)):
        if value is not None:
            raise click.UsageError(
                f"{flag} is only read by --executor {WORLD_MODEL_EXECUTOR}, not {executor}.")
    return AVAILABLE_EXECUTORS[executor]


@click.group()
@click.option("--game", default="pokemon_red", type=click.Choice(AVAILABLE_GAMES))
@click.option("--controller_variant", default="low_level", type=str)
@click.option("--executor", default="single_none",
              type=click.Choice(list(AVAILABLE_EXECUTORS.keys())))
@click.option("--executor_vlm_model", required=True, type=str,
              help="The model that plays. Required rather than falling back to the config's "
                   "executor_vlm_model: the supervisor's model defaults to THIS value, not to "
                   "the config, so an omitted flag left every supervised arm with no model at "
                   "all — and failed at the first supervisor call, after the emulator was up "
                   "and the CSV had been named from the config, which reads as a model "
                   "mismatch rather than a missing flag.")
@click.option("--executor_vlm_kind", required=True, type=str,
              help="VLM kind for --executor_vlm_model. Required for the same reason.")
@click.option("--supervisor_vlm_model", default=None, type=str,
              help="The one model every supervisor reasons with — selection, planning, "
                   "judging, hinting. Defaults to the executor's. A group option because "
                   "there are exactly two models anywhere in an arm, and this is the "
                   "second; the model that BUILT any knowledge read at test time is a "
                   "property of the document, recorded in its provenance, not a flag.")
@click.option("--supervisor_vlm_kind", default=None, type=str,
              help="VLM kind for --supervisor_vlm_model. Defaults to the executor's.")
@click.option("--supervisor_max_new_tokens", default=5000, show_default=True, type=int,
              help="Token budget for every supervisor call. One number rather than one per "
                   "stage: the stages used to differ and the only thing that bought was a "
                   "silent truncation when a reply outgrew its stage's allowance.")
@click.option("--save_video", type=bool, default=True)
@click.option("--max_steps", default=200, type=int,
              help="Emulator steps for the WHOLE episode, every internal retry included.")
@click.option("--n_tasks", type=int, default=None,
              help="Run the FIRST n tasks in benchmark order. A prefix rather than a "
                   "sample, so --n_tasks 5 is a subset of --n_tasks 10 and both are a "
                   "prefix of the full sweep. Omit to run every task.")
@click.option("--verbose", is_flag=True, default=False)
@click.option("--regenerate", is_flag=True, default=False,
              help="Ignore any existing CSV for this run and start over.")
@click.option("--extra_name", default=None, type=str,
              help="Free-text discriminator appended to the CSV and session names, for runs "
                   "this identity cannot otherwise tell apart — two info_subgoal_retrieval "
                   "runs over different --docs_mode being the case it exists for, since which "
                   "documents were read reaches no other part of the name. Omit and the name "
                   "is unchanged. 'none' is accepted as the shell's absent sentinel.")
@click.option("--world_model_run_name", default=None, type=str,
              help="RL run_name whose trained world model and observation encoder drive "
                   "the 'world_model' executor. Required by that executor and rejected by "
                   "the others. It becomes part of the executor's name — world_model_<run_name>"
                   " — so it names the CSV and session directory; pass that full name as "
                   "--executor to the debug tools. 'none' is accepted as the shell's absent "
                   "sentinel.")
@click.option("--world_model_game", default=None, type=str,
              help="Game whose checkpoint the 'world_model' executor plays with, for a title "
                   "with no world model of its own — e.g. --game pokemon_crystal "
                   "--world_model_game pokemon_red. Defaults to --game. A different game "
                   "becomes part of the executor's name, world_model_<game>_<run_name>. "
                   "'none' is accepted as the shell's absent sentinel.")
@click.pass_context
def main(ctx, game, controller_variant, executor, executor_vlm_model, executor_vlm_kind,
         supervisor_vlm_model, supervisor_vlm_kind, supervisor_max_new_tokens,
         save_video, max_steps, n_tasks,
         verbose, regenerate, extra_name, world_model_run_name, world_model_game):
    """Benchmark a frozen VLM on a game, with or without prebuilt knowledge."""
    parameters = load_parameters()
    executor_class = _resolve_executor(executor, world_model_run_name, world_model_game, game)
    ctx.obj = dict(
        parameters=parameters,
        game=game,
        controller_variant=controller_variant,
        extra_name=extra_name,
        # The class name, not the --executor word: the word would let two world-model checkpoints share a CSV.
        executor=executor_class.__name__,
        executor_class=executor_class,
        executor_vlm_model=executor_vlm_model,
        executor_vlm_kind=executor_vlm_kind,
        supervisor_vlm_model=supervisor_vlm_model or executor_vlm_model,
        supervisor_vlm_kind=supervisor_vlm_kind or executor_vlm_kind,
        supervisor_max_new_tokens=supervisor_max_new_tokens,
        model_save_name=model_save_name(executor_vlm_model),
        save_video=save_video,
        max_steps=max_steps,
        n_tasks=n_tasks,
        verbose=verbose,
        regenerate=regenerate,
    )


main.add_command(baseline, name="baseline")
main.add_command(revision, name="revision")
main.add_command(subgoal, name="subgoal")
main.add_command(info_subgoal_retrieval, name="info_subgoal_retrieval")
main.add_command(info_subgoal_parametric, name="info_subgoal_parametric")


if __name__ == "__main__":
    main()
