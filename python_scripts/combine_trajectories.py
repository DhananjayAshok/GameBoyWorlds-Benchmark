import os
import pickle

import click

from python_scripts.common import log


@click.command(name="combine_trajectories")
@click.option(
    "--input_paths",
    required=True,
    help="Comma-separated grouped-trajectory pkl paths to concatenate.",
)
@click.option(
    "--save_path",
    required=True,
    help="Directory to write the combined pkl into (created if absent).",
)
@click.option(
    "--z_kind",
    default="global",
    show_default=True,
    help="z_kind used in the output filename, matching group_trajectories.py.",
)
def combine_trajectories_cmd(input_paths, save_path, z_kind):
    """Write a manifest pkl referencing several grouped-trajectory pkls."""
    paths = [p.strip() for p in input_paths.split(",") if p.strip()]

    manifest = []
    for path in paths:
        if not os.path.exists(path):
            log(f"WARN: skipping missing grouped trajectory pkl: {path}")
            continue
        manifest.append(os.path.abspath(path))

    if not manifest:
        log(
            f"WARN: no input paths exist across {len(paths)} path(s); writing empty manifest."
        )

    os.makedirs(save_path, exist_ok=True)
    out_path = os.path.join(save_path, f"grouped_{z_kind}_high_reward_trajectories.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(manifest, f)  # the manifest holds input pkl paths, not the trajectories
    log(f"Wrote manifest of {len(manifest)} file(s) -> {out_path}")
