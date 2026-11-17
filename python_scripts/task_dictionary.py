import os

import click

from python_scripts.common import log

# TODO: print only the benchmark init states, as one comma-separated list per game.

OUTPUT_RELPATH = os.path.join("scripts", "core", "all_train_states.sh")


@click.command(name="task_dictionary")
@click.pass_obj
def task_dictionary_cmd(obj):
    """Regenerate scripts/core/all_train_states.sh from GameBoyWorlds' training states."""
    from gameboy_worlds import get_all_training_states

    all_training_states = get_all_training_states()

    output_path = os.path.join(obj["parameters"]()["project_root"], OUTPUT_RELPATH)
    lines = [
        "# AUTO-GENERATED — do not edit by hand.",
        "# Maps (game, init_state_group) keys to comma-separated init_state names for all",
        "# supported games. Sourced by create_all_traj.sh and propose_all_zeroshot.sh to",
        "# iterate over every training init_state for a given game.",
        "# Regenerate by running: python python_funcs.py task_dictionary",
        "",
        "declare -A TRAIN_STATES",
        "",
    ]

    for game, states_list in all_training_states.items():
        states_str = ",".join(states_list)
        lines.append(f'TRAIN_STATES["{game}"]="{states_str}"')

    lines += ["", "export TRAIN_STATES", ""]
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    log(f"Saved task dictionary to {output_path}")
