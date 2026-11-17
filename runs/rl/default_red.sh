#!/usr/bin/env bash
# The pokemon_red default sweep: collect trajectories, group them, render the high-reward set.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }
source configs/config.env || { echo "Could not source configs/config.env"; exit 1; }

init_state_group="default"
game="pokemon_red"
init_states="default,train_0,train_1,train_2,train_3,train_4"
run_name="default_sweep"
z_min=4.5

bash scripts/rl/create_traj.sh --game "$game" --init_state_group "$init_state_group" --init_states "$init_states" --run_name "$run_name" --sweep true


bash scripts/rl/group_trajectories.sh --game "$game" --replay_buffer_folder $run_name/ --save_path $storage_dir/grouped_trajectories/$game/$run_name/ --z_min $z_min

python show_trajectories.py --name pokemon_red_default_sweep --trajectory_path $storage_dir/grouped_trajectories/$game/$run_name/grouped_global_high_reward_trajectories.pkl

