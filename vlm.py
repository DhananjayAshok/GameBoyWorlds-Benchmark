from vlm_scripts import (infer_tasks, propose_tasks_zeroshot, attempt_tasks, build_info,
                         infer_guidance, practice_tasks, clean_practice)
import numpy as np
import click
from utils import load_parameters


@click.group()
@click.option("--model_name", required=True, help="VLM model name (e.g. gpt-4o)")
@click.option(
    "--vlm_kind",
    required=True,
    type=click.Choice(["openai", "anthropic", "openrouter", "huggingface", "vllm"]),
    help="VLM backend kind",
)
@click.option(
    "--game", required=True, help="Game name used in prompts (e.g. 'Pokemon Red')"
)
@click.option(
    "--overwrite", is_flag=True, default=False, help="Overwrite existing output files."
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Print prompts and outputs during annotation.",
)
@click.option(
    "--max_new_tokens",
    default=2000,
    show_default=True,
    help="Max tokens for each VLM call",
)
@click.pass_context
def main(ctx, model_name, vlm_kind, game, overwrite, verbose, max_new_tokens):
    parameters = load_parameters()
    np.random.seed(parameters["random_seed"])
    ctx.obj = dict(
        vlm_kind=vlm_kind,
        game=game,
        parameters=parameters,
        model_name=model_name,
        overwrite=overwrite,
        verbose=verbose,
        max_new_tokens=max_new_tokens,
    )


main.add_command(infer_tasks, name="infer_tasks")
main.add_command(propose_tasks_zeroshot, name="propose_tasks_zeroshot")
main.add_command(attempt_tasks, name="attempt_tasks")
main.add_command(build_info, name="build_info")
main.add_command(infer_guidance, name="infer_guidance")
main.add_command(practice_tasks, name="practice_tasks")
main.add_command(clean_practice, name="clean_practice")

if __name__ == "__main__":
    main()
