import click

from python_scripts import paths
from python_scripts.common import MAYBE_NONE
from python_scripts.paths import BENCHMARK_SUPERVISORS


def _game(f):
    return click.option("--game", required=True, help="Game name, e.g. deja_vu_1.")(f)


def _model(f):
    return click.option("--model_name", required=True,
                        help="Full VLM name, e.g. google/gemma-4-31b-it. Folded to a "
                             "save-name internally.")(f)


def _run(f):
    return click.option("--run_name", default="my_run", show_default=True,
                        help="Run name used by the RL/curiosity stages.")(f)


def _executor(f):
    return click.option("--executor", default="single_actions", show_default=True,
                        help="Executor short name.")(f)

def _controller(f):
    return click.option("--controller_variant", default="low_level", show_default=True,
                        help="Controller variant, e.g. low_level.")(f)


def _source(f):
    return click.option("--source", default="attempt", show_default=True,
                        help="Source vertical: attempt (aka zeroshot) or curiosity.")(f)


def _identity(f):
    return _game(_model(_run(_executor(_controller(f)))))


@click.command("model_save_name")
@_model
def model_save_name_cmd(model_name):
    """The lowercased basename the pipeline keys paths on."""
    click.echo(paths.model_save_name(model_name))


@click.command("finetuned_model_name")
@_game
@_model
@_run
@click.option("--mode", default="both", show_default=True)
def finetuned_model_name_cmd(game, model_name, run_name, mode):
    """Served-model name a fine-tuned checkpoint is benchmarked under."""
    click.echo(paths.finetuned_model_name(model_name=model_name, game=game,
                                          run_name=run_name, mode=mode))


@click.command("grouped_dir")
@_game
@_run
@click.pass_obj
def grouped_dir_cmd(obj, game, run_name):
    """<storage>/grouped_trajectories/<game>/<run_name>"""
    click.echo(paths.grouped_dir(obj["parameters"](), game=game, run_name=run_name))


@click.command("world_model_dir")
@_game
@_run
@click.pass_obj
def world_model_dir_cmd(obj, game, run_name):
    """<storage>/curiosity_buffers/world_model/<game>/<run_name>"""
    click.echo(paths.world_model_dir(obj["parameters"](), game=game, run_name=run_name))


@click.command("observation_embedder_dir")
@_game
@_run
@click.pass_obj
def observation_embedder_dir_cmd(obj, game, run_name):
    """<storage>/observation_embedders/<game>/<run_name>"""
    click.echo(paths.observation_embedder_dir(obj["parameters"](), game=game,
                                              run_name=run_name))


@click.command("grouped_file")
@_game
@_run
@click.option("--init_state", required=True)
@click.pass_obj
def grouped_file_cmd(obj, game, run_name, init_state):
    """The grouped-trajectory pkl for one init_state."""
    click.echo(paths.grouped_file(obj["parameters"](), game=game, run_name=run_name,
                                  init_state=init_state))


@click.command("init_states")
@_game
@_run
@click.pass_obj
def init_states_cmd(obj, game, run_name):
    """Every init_state with a grouped-trajectory file, one per line."""
    for state in paths.init_states(obj["parameters"](), game=game, run_name=run_name):
        click.echo(state)


@click.command("proposed_tasks_dir")
@_game
@_model
@click.pass_obj
def proposed_tasks_dir_cmd(obj, game, model_name):
    """<storage>/proposed_tasks/<game>/<model>"""
    click.echo(paths.proposed_tasks_dir(obj["parameters"](), game=game, model_name=model_name))


@click.command("curiosity_dir")
@_game
@_model
@_run
@click.pass_obj
def curiosity_dir_cmd(obj, game, model_name, run_name):
    """<proposed_tasks>/curiosity/<run_name>"""
    click.echo(paths.curiosity_dir(obj["parameters"](), game=game, model_name=model_name,
                                   run_name=run_name))


@click.command("curiosity_annotation")
@_game
@_model
@_run
@click.pass_obj
def curiosity_annotation_cmd(obj, game, model_name, run_name):
    """The curiosity vertical's trajectory_annotation.json."""
    click.echo(paths.curiosity_annotation(obj["parameters"](), game=game,
                                          model_name=model_name, run_name=run_name))


@click.command("zeroshot_dir")
@_game
@_model
@click.pass_obj
def zeroshot_dir_cmd(obj, game, model_name):
    """<proposed_tasks>/zeroshot"""
    click.echo(paths.zeroshot_dir(obj["parameters"](), game=game, model_name=model_name))


@click.command("tasks_file")
@_game
@_model
@click.pass_obj
def tasks_file_cmd(obj, game, model_name):
    """The proposed-tasks JSONL that attempt_tasks.py consumes."""
    click.echo(paths.tasks_file(obj["parameters"](), game=game, model_name=model_name))


@click.command("attempts_dir")
@_game
@_model
@_executor
@_controller
@click.pass_obj
def attempts_dir_cmd(obj, game, model_name, executor, controller_variant):
    """Where attempt_tasks.py writes for this identity."""
    click.echo(paths.attempts_dir(obj["parameters"](), game=game, model_name=model_name,
                                  executor=executor, controller_variant=controller_variant))


@click.command("all_trajectories_csv")
@_game
@_model
@_executor
@_controller
@click.pass_obj
def all_trajectories_csv_cmd(obj, game, model_name, executor, controller_variant):
    """<attempts_dir>/all_trajectories.csv"""
    click.echo(paths.all_trajectories_csv(obj["parameters"](), game=game,
                                          model_name=model_name, executor=executor,
                                          controller_variant=controller_variant))


@click.command("success_trajectories_stem")
@_game
@_model
@_executor
@_controller
@click.pass_obj
def success_trajectories_stem_cmd(obj, game, model_name, executor, controller_variant):
    """The zeroshot vertical's build_info --trajectory_path stem."""
    click.echo(paths.success_trajectories_stem(obj["parameters"](), game=game,
                                               model_name=model_name, executor=executor,
                                               controller_variant=controller_variant))


@click.command("info_source_stem")
@_identity
@_source
@click.pass_obj
def info_source_stem_cmd(obj, game, model_name, run_name, executor, controller_variant, source):
    """The build_info --trajectory_path stem for one source. Replaces the Bash function."""
    click.echo(paths.info_source_stem(obj["parameters"](), game=game, model_name=model_name,
                                      run_name=run_name, executor=executor, controller_variant=controller_variant, source=source))


@click.command("source_info_dir")
@_identity
@_source
@click.pass_obj
def source_info_dir_cmd(obj, game, model_name, run_name, executor, controller_variant, source):
    """Where build_info writes for one source. Replaces Bash's info_dir_for_stem."""
    click.echo(paths.source_info_dir(obj["parameters"](), game=game, model_name=model_name,
                                     run_name=run_name, executor=executor, controller_variant=controller_variant, source=source))


@click.command("info_doc")
@_identity
@_source
@click.pass_obj
def info_doc_cmd(obj, game, model_name, run_name, executor, controller_variant, source):
    """<source_info_dir>/info.json"""
    click.echo(paths.info_doc(obj["parameters"](), game=game, model_name=model_name,
                              run_name=run_name, executor=executor, controller_variant=controller_variant, source=source))


@click.command("insights_jsonl")
@_identity
@_source
@click.pass_obj
def insights_jsonl_cmd(obj, game, model_name, run_name, executor, controller_variant, source):
    """<source_info_dir>/insights.jsonl"""
    click.echo(paths.insights_jsonl(obj["parameters"](), game=game, model_name=model_name,
                                    run_name=run_name, executor=executor, controller_variant=controller_variant, source=source))


@click.command("parametric_doc")
@_game
@_model
@click.pass_obj
def parametric_doc_cmd(obj, game, model_name):
    """<storage>/parametric_docs/<game>/<model>/info.json"""
    click.echo(paths.parametric_doc(obj["parameters"](), game=game, model_name=model_name))


@click.command("benchmark_dir")
@_game
@click.pass_obj
def benchmark_dir_cmd(obj, game):
    """<results>/benchmark/<game>"""
    click.echo(paths.benchmark_dir(obj["parameters"](), game=game))


@click.command("benchmark_csv")
@_game
@_executor
@_controller
@click.option("--supervisor", required=True, type=click.Choice(BENCHMARK_SUPERVISORS),
              help="Supervisor key, as in execution.registry.AVAILABLE_SUPERVISORS. Names "
                   "the CSV and the session dir.")
@click.option("--model", required=True,
              help="Model SAVE name (already folded), or a finetuned_model_name.")
@click.option("--extra_name", default=None, type=MAYBE_NONE,
              help="Free-text discriminator the run used, if any. Must match, or this names a "
                   "file nothing wrote.")
@click.option("--n_tasks", default=None, type=MAYBE_NONE,
              help="Subset size, if the run used --n_tasks. Adds the _firstN suffix.")
@click.pass_obj
def benchmark_csv_cmd(obj, game, executor, controller_variant, supervisor, model, extra_name,
                      n_tasks):
    """<results>/benchmark/<game>/<supervisor>_<executor>_<controller_variant>_<model>[_<extra>][_firstN].csv"""
    click.echo(paths.benchmark_csv(obj["parameters"](), game=game, supervisor=supervisor,
                                   executor=executor, controller_variant=controller_variant,
                                   model=model, extra_name=extra_name,
                                   n_tasks=int(n_tasks) if n_tasks is not None else None))


@click.command("benchmark_series_csv")
@_game
@click.pass_obj
def benchmark_series_csv_cmd(obj, game):
    """The GameBoyWorlds series CSV whose game column contains --game."""
    click.echo(paths.benchmark_series_csv(obj["parameters"](), game=game))


@click.command("train_games")
@_game
@click.pass_obj
def train_games_cmd(obj, game):
    """The games in --game's series that source data can be collected from, one per line."""
    for name in paths.train_games(obj["parameters"](), game=game):
        click.echo(name)


@click.command("gameboy_worlds_storage")
def gameboy_worlds_storage_cmd():
    """GameBoyWorlds' own storage_dir — a DIFFERENT tree from this project's."""
    click.echo(paths.gameboy_worlds_storage())


@click.command("sessions_dir")
@_game
def sessions_dir_cmd(game):
    """<GameBoyWorlds storage>/sessions/<game>"""
    click.echo(paths.sessions_dir(game=game))


@click.command("model_checkpoint_dir")
@_game
@_model
@_run
@click.option("--mode", default="both", show_default=True)
@click.pass_obj
def model_checkpoint_dir_cmd(obj, game, model_name, run_name, mode):
    """Where train_vlm.sh writes a fine-tuned checkpoint."""
    click.echo(paths.model_checkpoint_dir(obj["parameters"](), game=game, run_name=run_name,
                                          mode=mode, model_name=model_name))


@click.command("debug_dir")
@_game
@click.option("--stage", required=True, help="Report stage, e.g. info, benchmark, attempt.")
@click.option("--output_dir", default=None, type=MAYBE_NONE,
              help="Report root override. Defaults to <results>/debug/<game>.")
@click.pass_obj
def debug_dir_cmd(obj, game, stage, output_dir):
    """<results>/debug/<game>/<stage>, created on demand."""
    click.echo(paths.debug_dir(obj["parameters"](), game=game, stage=stage,
                               output_dir=output_dir))


@click.command("debug_frames_dir")
@_game
@click.option("--stage", required=True)
@click.pass_obj
def debug_frames_dir_cmd(obj, game, stage):
    """<storage>/tmp/debug_frames/<game>/<stage>, created on demand."""
    click.echo(paths.debug_frames_dir(obj["parameters"](), game=game, stage=stage))


PATH_COMMANDS = [
    model_save_name_cmd,
    finetuned_model_name_cmd,
    grouped_dir_cmd,
    world_model_dir_cmd,
    observation_embedder_dir_cmd,
    grouped_file_cmd,
    init_states_cmd,
    proposed_tasks_dir_cmd,
    curiosity_dir_cmd,
    curiosity_annotation_cmd,
    zeroshot_dir_cmd,
    tasks_file_cmd,
    attempts_dir_cmd,
    all_trajectories_csv_cmd,
    success_trajectories_stem_cmd,
    info_source_stem_cmd,
    source_info_dir_cmd,
    info_doc_cmd,
    insights_jsonl_cmd,
    parametric_doc_cmd,
    benchmark_dir_cmd,
    benchmark_csv_cmd,
    benchmark_series_csv_cmd,
    train_games_cmd,
    gameboy_worlds_storage_cmd,
    sessions_dir_cmd,
    model_checkpoint_dir_cmd,
    debug_dir_cmd,
    debug_frames_dir_cmd,
]
