# observations has shape (num_frames, 144, 160, 1); actions and rewards have length num_frames-1.

import os
import pickle
from tqdm import tqdm
import random
import numpy as np
import click
from PIL import ImageDraw, ImageFont

from python_scripts import paths
from utils import load_parameters, log_info, convert_numpy_greyscale_to_pillow


try:
    _FONT = ImageFont.load_default(size=14)
except TypeError:
    _FONT = ImageFont.load_default()


def plot_transitions(observations, save_name, high_level_actions=None):
    total_height = observations[0].shape[0]
    total_width = sum(obs.shape[1] for obs in observations)
    single_image = np.zeros((total_height, total_width), dtype=observations[0].dtype)
    current_x = 0
    for obs in observations:
        w = obs.shape[1]
        single_image[:, current_x:current_x + w] = obs[:, :]
        current_x += w
    img = convert_numpy_greyscale_to_pillow(single_image)
    draw = ImageDraw.Draw(img)
    current_x = 0
    for i, obs in enumerate(observations):
        w = obs.shape[1]
        bg_top = np.mean(obs[:12, :max(w, 12)])
        fill = 0 if bg_top > 127 else 255
        draw.text((current_x + 2, 2), str(i), fill=fill, font=_FONT)
        if high_level_actions is not None and i < len(high_level_actions):
            label = str(high_level_actions[i][1]['low_level_action']).replace("LowLevelActions.PRESS_BUTTON_", "").replace("LowLevelActions.PRESS_ARROW_", "")
            bg_bot = np.mean(obs[-12:, :max(w, 12)])
            fill_bot = 0 if bg_bot > 127 else 255
            draw.text((current_x + 2, total_height - 16), label, fill=fill_bot, font=_FONT)
        current_x += w
    img.save(save_name)


@click.command()
@click.option("--name", required=True, help="Name to save the images under (e.g. game name or group name)")
@click.option("--trajectory_path", required=True, help="Path to grouped_high_reward_trajectories.pkl")
@click.option("--frac", default=1.0, show_default=True, help="Fraction of observation transitions to sample")
@click.option("--group_idx", default=None, type=int, help="If specified, only plot this group.")
@click.option("--traj_idx", default=None, type=int, help="If specified, only plot this trajectory (requires --group_idx).")
@click.option("--max_per_group", default=50, type=int, help="Maximum number of trajectories to plot per group (only applies if --traj_idx is not specified).")
def show(name, trajectory_path, frac, group_idx, traj_idx, max_per_group):
    """Randomly sample a fraction of observation transitions and save them as images."""
    assert traj_idx is None or group_idx is not None, "--traj_idx requires --group_idx to be specified."
    parameters = load_parameters()
    with open(trajectory_path, "rb") as f:
        grouped_trajectories = pickle.load(f)

    render_root = paths.trajectory_render_dir(parameters, name=name)
    os.makedirs(render_root, exist_ok=True)

    if group_idx is not None:
        indexes = [group_idx]
    else:
        indexes = list(range(len(grouped_trajectories)))
        random.shuffle(indexes)
        indexes = indexes[:int(len(grouped_trajectories) * frac)]

    for i, trajectory_group in tqdm(enumerate(grouped_trajectories), desc="Plotting observation transitions", total=len(grouped_trajectories)):
        if i not in indexes:
            continue
        group_dir = paths.trajectory_render_dir(parameters, name=name, group=i)
        os.makedirs(group_dir, exist_ok=True)
        if traj_idx is not None:
            trajectory = trajectory_group[traj_idx]
            observations, _, high_level_actions, _, init_state = trajectory
            save_path = group_dir + f"/traj_{traj_idx}.png"
            plot_transitions(observations, save_path, high_level_actions=high_level_actions)
            log_info(f"Saved {save_path}")
        else:
            for j, trajectory in tqdm(enumerate(trajectory_group), leave=False, total=len(trajectory_group)):
                if j >= max_per_group:
                    break
                observations, _, high_level_actions, _, init_state = trajectory
                plot_transitions(observations, group_dir + f"/traj_{j}.png", high_level_actions=high_level_actions)
            if group_idx is not None:
                log_info(f"Saved {group_dir}/")
    if group_idx is None and traj_idx is None:
        log_info(f"Saved {render_root}/")


if __name__ == "__main__":
    show()
