"""
CLI for syncing project data with the Hugging Face Hub. Supports creating dataset repos,
downloading a repo snapshot to the local sync directory, and uploading the sync directory
to the Hub. Use --help for CLI options.
"""
from utils import (
    load_parameters,
    compute_secondary_parameters,
    log_error,
    log_info,
    log_warn,
)
import click
from huggingface_hub import HfApi
import os
import shlex
import sys

from python_scripts import paths

loaded_parameters = load_parameters()

GLOBAL_IGNORE_PATTERNS = [
    "*.gb",
    "*.gbc",
    ".cache/*",
    ".DS_Store",
    "tmp/*",
]

ALLOWED_SETS = {
    "curiosity": {
        "root": "storage_dir",
        "subdir": "proposed_tasks",
        "allow_patterns": [
            "*/curiosity/*trajectory_annotation.json",
            "*/curiosity/*trajectory_annotation.pkl",
            "*/curiosity/*/info_docs/*",
        ],
        "ignore_patterns": [],
    },
    "results": {
        "root": "results_dir",
        "subdir": "",
        "allow_patterns": [
            "benchmark/*curiosity_only*.csv",
            "benchmark/*subgoal_single_visual_low_level*.csv",
        ],
        "ignore_patterns": ["*info_subgoal_parametric_*"],
    },
    # The three trees one playthrough writes to. Kept as separate sets because they sit
    # under two different storage roots and differ by orders of magnitude in size.
    # The set names say "playthrough"; the patterns still say "strategist" because that is
    # the on-disk segment, which was deliberately left alone when the code was renamed.
    "playthrough": {
        "root": "storage_dir",
        "subdir": "playthrough_artifacts",
        "allow_patterns": [
            "*/strategist/*/episode_*/*.pkl",
            "*/strategist/*/episode_*/report.pkl.gz",
            "*/strategist/*/provenance_*.json",
        ],
        # The */strategist/* prefix already excludes <game>/perception/tile_recognizer/,
        # the shared tile database, which is not a per-playthrough artifact.
        "ignore_patterns": [],
    },
    "playthrough_videos": {
        "root": "gameboy_worlds_storage_dir",
        "subdir": "sessions",
        "allow_patterns": ["*/strategist/*/videos/*.mp4"],
        # fnmatch's * crosses /, so the allow pattern means "a strategist segment somewhere",
        # not "at depth 1". A benchmark task whose mission depathifies to exactly
        # "strategist" would otherwise match. Depth cannot be pinned without naming every
        # game, so the other run kinds are excluded by name instead.
        "ignore_patterns": ["*/benchmark_*/*", "*/tmp_sessions/*", "*/random_sweep/*"],
    },
    # Rooted inside rom_data, which also holds the copyrighted ROMs and the 1200+ handmade
    # benchmark start states. The allow pattern takes only states the strategist itself
    # wrote -- custom_ is Environment.save_custom_state's prefix, strategist is
    # PokemonStrategist._emulator_state_name's -- so the uuid-named temporaries that
    # Environment._simulate leaves behind when a job is killed stay out. The ignores restate
    # the ROM exclusion locally so that widening the allow list cannot leak one.
    "playthrough_states": {
        "root": "gameboy_worlds_storage_dir",
        "subdir": "rom_data",
        "allow_patterns": ["*/states/custom_strategist*.state"],
        "ignore_patterns": ["*.gb", "*.gbc", "*.gba", "*.sav", "*.ips", "*.bps"],
    },
}


def set_root(set_name, parameters):
    """
    Resolve the local directory one set is rooted at. Repo paths are relative to it, and it
    is also the directory the Hub API scans, so it is kept as narrow as the patterns allow.
    """
    spec = ALLOWED_SETS[set_name]
    root_key = spec["root"]
    if root_key not in parameters:
        log_error(
            f"Set '{set_name}' is rooted at '{root_key}', which is not in the parameters. Check your configuration.",
            parameters,
        )
    return os.path.abspath(os.path.join(parameters[root_key], spec["subdir"]))


def check_set(file_set, parameters):
    """
    Validate the --set option and expand it into the list of set names to act on.
    """
    file_set = file_set.lower()
    if file_set not in list(ALLOWED_SETS) + ["all"]:
        log_error(
            f"Set {file_set} is not in the list of allowed sets: {list(ALLOWED_SETS)}. Add it to ALLOWED_SETS in sync_data.py before you proceed.",
            parameters,
        )
    return list(ALLOWED_SETS) if file_set == "all" else [file_set]


def set_patterns(set_name):
    """
    Resolve one set name into the allow and ignore pattern lists for the Hub API.
    """
    spec = ALLOWED_SETS[set_name]
    allow_patterns = list(spec["allow_patterns"])
    ignore_patterns = list(spec["ignore_patterns"]) + GLOBAL_IGNORE_PATTERNS
    return allow_patterns, ignore_patterns


def no_dry_run_command():
    """
    Rebuild the command the user just ran, switched over to --no_dry_run.
    """
    argv = [arg for arg in sys.argv[1:] if arg != "--dry_run"]
    parts = ["python", os.path.basename(sys.argv[0])] + argv + ["--no_dry_run"]
    return " ".join(shlex.quote(part) for part in parts)


def set_option(all_help):
    return click.option(
        "--set",
        "file_set",
        type=str,
        default="all",
        show_default=True,
        help=f"Which file set to act on. One of {list(ALLOWED_SETS)}. {all_help}",
    )



@click.command()
@click.option("--private", is_flag=True, default=False, help="Whether the repo should be private.")
@click.pass_obj
def create_hub_repo(parameters, private):
    """
    Create a new repo on the Hugging Face Hub.
    """
    repo_namespace = parameters["huggingface_repo_namespace"]
    repo_name = parameters["huggingface_repo_name"]
    repo_id = f"{repo_namespace}/{repo_name}"
    api = parameters["api"]
    api.create_repo(
    repo_id=repo_id,
    repo_type="dataset",
    exist_ok=True, # Won't raise an error if the repo already exists
    private=private)
    log_info(f"Successfully created repo {repo_id} on the Hugging Face Hub.", parameters)


@click.command()
@set_option("'all' loops over every allowed set.")
@click.option(
    "--dry_run/--no_dry_run",
    default=True,
    show_default=True,
    help="With --dry_run only report what would be written and download nothing. With --no_dry_run report the same thing first, then actually download.",
)
@click.pass_obj
def setup_sync(parameters, file_set, dry_run):
    """
    Set up the local sync directory to sync with the specified repo on the Hugging Face Hub.
    """
    repo_namespace = parameters["huggingface_repo_namespace"]
    repo_name = parameters["huggingface_repo_name"]
    repo_id = f"{repo_namespace}/{repo_name}"
    set_names = check_set(file_set, parameters)
    api = parameters["api"]
    for set_name in set_names:
        local_dir = set_root(set_name, parameters)
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
            log_info(f"Created root directory {local_dir} for set '{set_name}'.", parameters)
        allow_patterns, ignore_patterns = set_patterns(set_name)
        result = api.snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            dry_run=True,
        )
        incoming = [f for f in result if f.will_download]
        to_overwrite = [
            f
            for f in incoming
            if f.local_path is not None and os.path.exists(f.local_path)
        ]
        to_create = [f for f in incoming if f not in to_overwrite]
        log_info(
            f"[dry run] set '{set_name}' of {repo_id} into {local_dir}: {len(result)} file(s) matched, {len(to_create)} new, {len(to_overwrite)} would be overwritten, {len(result) - len(incoming)} already up to date.",
            parameters,
        )
        for f in to_overwrite:
            log_warn(f"[dry run] would overwrite {f.local_path}", parameters)
        for f in to_create:
            log_info(f"[dry run] would create {f.filename}", parameters)
        if dry_run:
            log_info(
                f"[dry run] nothing written for set '{set_name}'.",
                parameters,
            )
            continue
        api.snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
        log_info(f"Tried to sync set '{set_name}' of repo at {repo_namespace}/{repo_name} with directory {local_dir}. Check output above for success", parameters)
    if dry_run:
        log_info(
            f"[dry run] no files were written. To actually sync, run:\n    {no_dry_run_command()}",
            parameters,
        )


@click.command()
@set_option("'all' is rejected on push, you must name a single set.")
@click.pass_obj
def push_data_to_hub(parameters, file_set):
    """
    In general how this works is:
        1. Check if repo exists. If it does, first pull and write it to sync directory (i.e. root_dir)
    """
    repo_namespace = parameters["huggingface_repo_namespace"]
    repo_name = parameters["huggingface_repo_name"]
    api = parameters["api"]
    repo_id = f"{repo_namespace}/{repo_name}"
    if file_set.lower() == "all":
        log_error(
            f"Pushing set 'all' is unsafe. Please specify one of {list(ALLOWED_SETS)} to push.",
            parameters,
        )
    set_names = check_set(file_set, parameters)
    for set_name in set_names:
        folder_path = set_root(set_name, parameters)
        if not os.path.exists(folder_path):
            log_error(f"Root directory {folder_path} for set '{set_name}' does not exist. Cannot push to hub.", parameters)
        allow_patterns, ignore_patterns = set_patterns(set_name)
        api.upload_large_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=folder_path,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
        log_info(f"Successfully pushed set '{set_name}' from {folder_path} to repo {repo_namespace}/{repo_name} on the Hugging Face Hub.", parameters)


@click.group()
@click.option("--huggingface_repo_namespace", type=str, default=None, help="The namespace (user or org) on the hub where the repo is located.")
@click.option("--huggingface_repo_name", type=str, default=None, help="The name of the repo on the hub.")
@click.pass_context
def main(ctx, **input_parameters):
    if "huggingface_repo_namespace" not in loaded_parameters:
        if input_parameters["huggingface_repo_namespace"] is not None:
            loaded_parameters["huggingface_repo_namespace"] = input_parameters["huggingface_repo_namespace"]
        else:
            log_error("huggingface_repo_namespace must be specified either in the config file or as a command line argument.", loaded_parameters)
    if "huggingface_repo_name" not in loaded_parameters:
        if input_parameters["huggingface_repo_name"] is not None:
            loaded_parameters["huggingface_repo_name"] = input_parameters["huggingface_repo_name"]
        else:
            log_error("huggingface_repo_name must be specified either in the config file or as a command line argument.", loaded_parameters)
    compute_secondary_parameters(loaded_parameters)
    # A distinct key, never storage_dir: the strategist's videos and save states live on the
    # GameBoyWorlds submodule's storage, its state and reports on this project's, and a set
    # rooted at one must not silently resolve against the other. Injected here rather than in
    # compute_secondary_parameters so that every script loading parameters does not end up
    # importing gameboy_worlds and running its directory-creating config loader.
    loaded_parameters["gameboy_worlds_storage_dir"] = paths.gameboy_worlds_storage()
    api = HfApi()
    loaded_parameters["api"] = api
    ctx.obj = loaded_parameters


main.add_command(create_hub_repo, name="init")
main.add_command(setup_sync, name="pull")
main.add_command(push_data_to_hub, name="push")

if __name__ == "__main__":
    main()
