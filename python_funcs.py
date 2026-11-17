# stdout is a data channel captured by Bash: every diagnostic must go to stderr.

import functools

import click

from python_scripts.combine_trajectories import combine_trajectories_cmd
from python_scripts.embed_markdown import embed_markdown_cmd
from python_scripts.paths_cli import PATH_COMMANDS
from python_scripts.strings import strings_cmd
from python_scripts.task_dictionary import task_dictionary_cmd


@click.group()
@click.pass_context
def main(ctx):
    """Path lookups and small helpers shared by the Python and Bash halves of the pipeline."""
    from utils.parameter_handling import load_parameters

    ctx.obj = {"parameters": functools.cache(load_parameters)}


main.add_command(strings_cmd)
main.add_command(task_dictionary_cmd)
main.add_command(combine_trajectories_cmd)
main.add_command(embed_markdown_cmd)
for _command in PATH_COMMANDS:
    main.add_command(_command)


if __name__ == "__main__":
    main()
